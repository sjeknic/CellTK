"""Post-processing: Adds dummy masks for missing segm masks, unmerges objects"""
import sys
from itertools import combinations, chain

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from sklearn.neighbors import NearestNeighbors

from tracker.split_objects import compute_split, match_pred_succ, calc_seed_points
from tracker.tracking import CellTrack

np.random.seed(42)


def add_dummy_masks(all_tracks, img_shape):
    """
    Resolves False Negatives (missing segmentation masks) by adding
    same mask as at time point t at linearly interpolated positions
    between time points t and t+n
    Args:
        all_tracks: dict containing all trajectories
        img_shape: tuple of the image shape

    Returns: dict containing all trajectories

    """
    img_shape = np.array(img_shape).reshape(-1, 1)
    for track_id in all_tracks.keys():
        track = all_tracks[track_id]
        time_steps = np.array(sorted(track.masks.keys()))
        if len(time_steps) == (time_steps[-1] - time_steps[0] + 1):
            continue
        # add time steps
        if len(time_steps > 1):
            d_t = time_steps[1:] - time_steps[:-1]
            is_gap = d_t > 1
            t_start = time_steps[:-1][is_gap]
            t_end = time_steps[1:][is_gap]
            time_span = d_t[is_gap]
            # linear interpolation of start and end position where to place dummy mask
            dummy_masks = {(t_s+tt): (np.array(track.masks[t_s])
                                      + (np.median(track.masks[t_e], axis=-1) -
                                         np.median(track.masks[t_s], axis=-1)).reshape(-1, 1) * tt / t_span
                                      ).astype(int)
                           for t_s, t_e, t_span in zip(t_start, t_end, time_span)
                           for tt in range(1, t_span)}
            # mask pixels must lay in img 0...img_shape, neg indices lead to placement on other side of array
            dummy_masks = {k: tuple(mask[..., np.all((mask > 0) & (mask < np.repeat(img_shape,
                                                                                    mask.shape[-1],
                                                                                    axis=-1)),
                                                     axis=0)])
                           for k, mask in dummy_masks.items()}
            track.masks.update(dummy_masks)
    return all_tracks


def check_lineage(all_tracks):
    """
    Removes predecessor- successor links if track is linked to more than two successors.
    Args:
        all_tracks: dict containing all trajectories

    Returns: dict containing all trajectories

    """
    lineage = {}
    for track_id, track in all_tracks.items():
        for predecessor in track.pred_track_id:
            if predecessor != 0:
                if predecessor not in lineage:
                    lineage[predecessor] = set()
                lineage[predecessor].add(track_id)
        for successor in track.successors:
            if track_id not in lineage:
                lineage[track_id] = set()
            lineage[track_id].add(successor)
    for track_id, n_succ in lineage.items():
        if len(n_succ) > 2:
            for successor in n_succ:
                assert successor in all_tracks[track_id].successors
                all_tracks[track_id].successors.remove(successor)
                assert track_id in all_tracks[successor].pred_track_id
                all_tracks[successor].pred_track_id.remove(track_id)
                if len(all_tracks[successor].pred_track_id) == 0:
                    all_tracks[successor].pred_track_id = [0]
    return all_tracks


def untangle_tracks(all_tracks):
    """
    Untangles the trajectories by applying a minimum number of operations:
    splitting/merging of tracks or cutting edges on the lineage graph.
    Args:
        all_tracks: dict containing all trajectories

    Returns: dict containing all trajectories

    """
    costs, A_eq, b_eq, A_ieq, b_ieq = setup_constraints(all_tracks)
    result = solve_untangling_problem(costs, A_eq, b_eq, A_ieq, b_ieq)
    selected_operations = {var_name: value
                           for var_name, value in result.items()
                           if value > 0}
    # variable x_t1_t2..._tn : x: operation to apply on several tracks
    # or x_t1 : operation to apply on a single track
    tracks_to_remove = []
    # apply first edge cut, then merge, then split -> sort edge names
    graph_operations = list(selected_operations.keys())
    edge_operations = [grap_op for grap_op in graph_operations
                       if grap_op.startswith('e')]
    merge_operations = [graph_op for graph_op in graph_operations
                        if graph_op.startswith('m')]
    split_operations = [graph_op for graph_op in graph_operations
                        if graph_op.startswith('s')]
    # sort split operations by track start time so predecessor positions are known by then
    split_operations.sort(key=lambda x: min(all_tracks[int(x.split('_')[-1])].masks.keys()))
    graph_operations = edge_operations
    graph_operations.extend(merge_operations)
    graph_operations.extend(split_operations)

    all_splits = {int(split_op.split('_')[1]): selected_operations[split_op]+1
                  for split_op in split_operations}
    for graph_op in graph_operations:
        track_ids = [int(el) for el in graph_op.split('_')[1:]]
        if graph_op.startswith('e'):
            all_tracks = cut_edge(all_tracks, track_ids)

        elif graph_op.startswith('m'):
            # merge two or more tracks
            merge_track_ids = track_ids
            tracks_to_remove.extend(merge_track_ids[1:])
            all_tracks = merge_tracks(all_tracks, merge_track_ids)

        elif graph_op.startswith('s'):
            # split a track
            split_track_id = track_ids[0]
            tracks_to_remove.append(split_track_id)
            all_tracks = split_tracks(all_tracks, split_track_id,
                                      selected_operations[graph_op]+1, all_splits)

        else:
            raise AssertionError('unknown untangling operation: ', graph_op)

    all_tracks = remove_tracks(all_tracks, tracks_to_remove)
    all_tracks = stack_tracks(all_tracks)
    all_tracks = check_lineage(all_tracks)

    return all_tracks


def cut_edge(all_tracks, track_ids):
    """
    Removes a predecessor-successor link.
    Args:
        all_tracks: dict containing all trajectories
        track_ids: tuple of tracks (predecessor id, successor id)

    Returns: dict containing all trajectories

    """
    succ_track_id = track_ids[1]
    pred_track_id = track_ids[0]
    if pred_track_id in all_tracks:
        pred_track = all_tracks[pred_track_id]
        if succ_track_id in pred_track.successors:
            pred_track.successors.remove(succ_track_id)
    if succ_track_id in all_tracks:
        successor_track = all_tracks[succ_track_id]
        if pred_track_id in successor_track.pred_track_id:
            successor_track.pred_track_id.remove(pred_track_id)
        if len(successor_track.pred_track_id) == 0:
            successor_track.pred_track_id = [0]
    return all_tracks


def remove_tracks(all_tracks, track_ids_to_remove):
    """
    Removes selected track ids from the set of all tracks.
    Args:
        all_tracks: dict containing all trajectories
        track_ids_to_remove: list of track ids to remove from all tracks

    Returns: dict containing all trajectories

    """
    for track_id in set(track_ids_to_remove):
        track = all_tracks.pop(track_id)
        if track.pred_track_id not in [[0], []]:
            # remove as succ in pred tracks
            for predecessor in track.pred_track_id:
                if predecessor in all_tracks:
                    if track_id in all_tracks[predecessor].successors:
                        all_tracks[predecessor].successors.remove(track_id)
        if len(track.successors) > 0:
            # remove as predecessor in succ tracks
            for successor in track.successors:
                if successor in all_tracks:
                    if track_id in all_tracks[successor].pred_track_id:
                        all_tracks[successor].pred_track_id.remove(track_id)
    return all_tracks


def stack_tracks(all_tracks):
    """
    Stacks track with successor track to single track if track has only one successor.
    Args:
        all_tracks: dict containing all trajectories

    Returns: dict containing all trajectories

    """
    indices = [(track_id, min(track.masks.keys()))
               for track_id, track in all_tracks.items()]
    indices.sort(key=lambda x: x[1], reverse=True)
    track_ids, _ = list(zip(*indices))
    for track_id in track_ids:
        # check whether predecessor has only one succ - link succ to predecessor
        predecessor = all_tracks[track_id].pred_track_id
        if (len(predecessor) == 1) and (predecessor[0] != 0):
            predecessor = predecessor[0]
            pred_successor = all_tracks[predecessor].successors
            if len(pred_successor) == 1:
                assert list(pred_successor)[0] == track_id, \
                    f'predecessor successor {pred_successor} and track id {track_id} not matching'
                track = all_tracks.pop(track_id)
                all_tracks[predecessor].masks.update(track.masks)
                all_tracks[predecessor].successors = track.successors
                for successor in track.successors:
                    all_tracks[successor].pred_track_id = [predecessor]
    return all_tracks


def merge_tracks(all_tracks, merge_track_ids):
    """
    Merges several tracks into a single track.
    Args:
        all_tracks: dict containing all trajectories
        merge_track_ids: list of track ids to merge

    Returns: dict containing all trajectories

    """
    tracks_to_merge = {t_id: all_tracks[t_id] for t_id in merge_track_ids[1:]}

    # add masks of other tracks to first track
    merged_track = all_tracks[merge_track_ids[0]]
    for track_id, track in tracks_to_merge.items():
        for time_step, mask in track.masks.items():
            if time_step not in merged_track.masks:  # missing mask possible due to skip
                merged_track.masks[time_step] = mask
            else:
                merged_track.masks[time_step] = tuple(np.hstack([merged_track.masks[time_step], mask]))
        # remove pred,succ, remove tracks
        for predecessor in track.pred_track_id:
            if predecessor == 0:
                continue
            if track_id in all_tracks[predecessor].successors:
                all_tracks[predecessor].successors.remove(track_id)
        for successor in track.successors:
            if track_id in all_tracks[successor].pred_track_id:
                all_tracks[successor].pred_track_id.remove(track_id)
    all_tracks[merge_track_ids[0]] = merged_track
    return all_tracks


def split_tracks(all_tracks, split_track_id, n_splits, all_split_track_ids):
    """
    Splits track into several tracks and adds links.
    Args:
        all_tracks: dict containing all trajectories
        split_track_id: int indicating which track id to split
        n_splits: int, number of tracks the track is split into
        all_split_track_ids: list containing all track ids that will be split

    Returns: dict containing all trajectories

    """
    track = all_tracks[split_track_id]
    predecessors = [predecessor
                    for predecessor in track.pred_track_id
                    if predecessor != 0]
    successors = list(set(track.successors))
    new_track_ids = [max(all_tracks.keys()) + i + 1 for i in range(n_splits)]
    if successors:
        successor_seeds = []
        for s in successors:
            n_succ = all_split_track_ids[s] if s in all_split_track_ids else 1
            successor_seeds.extend([(s, all_tracks[s].get_first_position()) for _ in range(n_succ)])
        successors, succ_positions = list(zip(*successor_seeds))

        if len(successors) < len(predecessors):
            # remove successor info
            for s in successors:
                if split_track_id in all_tracks[s].pred_track_id:
                    all_tracks[s].pred_track_id.remove(split_track_id)
                if s in track.successors:
                    track.successors.remove(s)
            successors = []

    for track_id in new_track_ids:
        all_tracks[track_id] = CellTrack(track_id, [])
        all_tracks[track_id].successors = {}

    # if segmentation mask (n pixels) smaller than number of objects to split into don't split
    segm_masks = track.masks
    is_too_small_mask = [len(mask[0]) < n_splits for mask in segm_masks.values()]
    if np.any(is_too_small_mask):
        for p in predecessors:
            if p != 0:
                if split_track_id in all_tracks[p].successors:
                    all_tracks[p].successors.remove(split_track_id)
        for s in successors:
            if split_track_id in all_tracks[s].pred_track_id:
                all_tracks[s].pred_track_id.remove(split_track_id)
        return all_tracks

    if len(track.successors) == n_splits:
        seed_positions = [all_tracks[s].get_first_position() for s in track.successors]
    elif len(predecessors) == n_splits:
        seed_positions = [all_tracks[p].get_last_position() for p in predecessors]
    else:  # sequence of merge errors or cell division
        mask = np.array(segm_masks[min(segm_masks.keys())])
        seeds = calc_seed_points(mask, n_splits)
        random_ind = np.random.choice(seeds.shape[1], n_splits, replace=False)
        seed_positions = [mask[:, rand_i] for rand_i in random_ind]

    # split mask and assign mask parts to new tracks
    for time_step in sorted(segm_masks.keys()):
        mask = np.array(segm_masks[time_step])
        centers = compute_split(mask, seed_positions)
        nn = NearestNeighbors(n_neighbors=1).fit(centers.T)
        _, indices = nn.kneighbors(mask.T)
        for i, track_id in enumerate(new_track_ids):
            all_tracks[track_id].masks[time_step] = tuple(mask.T[indices.reshape(-1) == i].T)

    for p in predecessors:
        if p != 0:
            if split_track_id in all_tracks[p].successors:
                all_tracks[p].successors.remove(split_track_id)
    for s in successors:
        if split_track_id in all_tracks[s].pred_track_id:
            all_tracks[s].pred_track_id.remove(split_track_id)

    pred_positions = [np.array(all_tracks[p].get_last_position()) for p in predecessors]
    start_position_split_tracks = [np.array(all_tracks[track_id].get_first_position())
                                   for track_id in new_track_ids]
    end_position_split_tracks = [np.array(all_tracks[track_id].get_last_position())
                                 for track_id in new_track_ids]

    # match predecessor tracks of split track to new tracks
    if not predecessors or len(start_position_split_tracks) < len(predecessors):
        for track_id in new_track_ids:
            all_tracks[track_id].pred_track_id.append(0)
    else:
        matching_indices = match_pred_succ(pred_positions, start_position_split_tracks)
        for index_s, index_p in matching_indices:
            pred_id = predecessors[index_p]
            succ_id = new_track_ids[index_s]
            all_tracks[pred_id].successors = set(all_tracks[pred_id].successors)
            all_tracks[pred_id].successors.add(succ_id)
            all_tracks[succ_id].pred_track_id.append(pred_id)
            all_tracks[succ_id].pred_track_id = list(set(all_tracks[succ_id].pred_track_id))

    # match successor tracks of split track to new tracks
    if successors and len(succ_positions) >= len(end_position_split_tracks):
        matching_indices = match_pred_succ(end_position_split_tracks, succ_positions)
        for index_s, index_p in matching_indices:
            pred_id = new_track_ids[index_p]
            succ_id = successors[index_s]
            all_tracks[pred_id].successors = set(all_tracks[pred_id].successors)
            all_tracks[pred_id].successors.add(succ_id)
            all_tracks[succ_id].pred_track_id.append(pred_id)
            all_tracks[succ_id].pred_track_id = list(set(all_tracks[succ_id].pred_track_id))

    return all_tracks


def calc_merge_sets(all_tracks, track_id):
    """
    Computes all sets of tracks a track can be merged with.
    Args:
        all_tracks: dict containing all trajectories
        track_id: int indicating the track id

    Returns: set containing all sets of tracks a track can be merged with, a list of merge costs

    """
    predecessors = all_tracks[track_id].pred_track_id
    successors = all_tracks[track_id].successors

    if successors or predecessors != [0]:
        parallel_tracks = [p for succ in successors for p in all_tracks[succ].pred_track_id]
        temp = [list(all_tracks[p].successors) for p in predecessors if p != 0]
        if temp:
            temp = np.concatenate(temp)
            parallel_tracks.extend(list(temp))
    else:
        parallel_tracks = [track_id]
    predecessors = list(set(predecessors))

    parallel_tracks = set(parallel_tracks)
    parallel_tracks.remove(track_id)

    parallel_combinations = []
    costs = []
    for select_n_tracks in range(1, len(parallel_tracks)+1):
        for set_of_tracks in combinations(parallel_tracks, select_n_tracks):
            set_of_tracks = list(set_of_tracks)
            set_of_tracks.append(track_id)
            same_predecessors = []
            same_successors = []
            for t_id in set_of_tracks:
                other_predecessors = list(set([predecessor for other_track_id in set_of_tracks
                                               for predecessor in all_tracks[other_track_id].pred_track_id
                                               if other_track_id != t_id]))
                other_successors = list(set([successor for other_track_id in set_of_tracks
                                             for successor in all_tracks[other_track_id].successors
                                             if other_track_id != t_id]))
                curr_predecessor = all_tracks[t_id].pred_track_id
                curr_successors = list(all_tracks[t_id].successors)
                if (other_predecessors == [0]) or (curr_predecessor == [0]) or (other_predecessors == curr_predecessor):
                    same_predecessors.append(True)
                else:
                    same_predecessors.append(False)
                if (not other_successors) or (not curr_successors) or (other_successors == curr_successors):
                    same_successors.append(True)
                else:
                    same_successors.append(False)

            t_start, t_end = list(zip(*[(min(all_tracks[t_id].masks.keys()), max(all_tracks[t_id].masks.keys()))
                                        for t_id in set_of_tracks]))
            # no successor t_end <= t_end track
            all_predecessors = list(set([predecessor for t_id in set_of_tracks
                                         for predecessor in all_tracks[t_id].pred_track_id]))
            all_successors = list(set([successor for t_id in set_of_tracks
                                       for successor in all_tracks[t_id].successors]))
            if min(same_successors) and min(same_predecessors):
                if successors and (predecessors != [0]):
                    if (min(t_start[:-1]) >= t_start[-1]) and (max(t_end[:-1]) <= t_end[-1]):
                        dt = max(t_end) - min(t_start) + 1
                        parallel_combinations.append(sorted(set_of_tracks))
                        costs.append(dt)

                elif successors:
                    if all_predecessors != [0]:
                        t_end_pred_max = max([max(all_tracks[p].masks.keys()) for p in all_predecessors if p!=0])
                        if t_end_pred_max < min(t_start):
                            dt = max(t_end) - min(t_start) + 1
                            parallel_combinations.append(sorted(set_of_tracks))
                            costs.append(dt)
                    else:
                        dt = max(t_end) - min(t_start) + 1
                        parallel_combinations.append(sorted(set_of_tracks))
                        costs.append(dt)

                else:
                    if all_successors:
                        t_start_succ_min = min([min(all_tracks[s].masks.keys()) for s in all_successors])
                        if t_start_succ_min > max(t_end):
                            dt = max(t_end) - min(t_start) + 1
                            parallel_combinations.append(sorted(set_of_tracks))
                            costs.append(dt)
                    else:
                        dt = max(t_end) - min(t_start) + 1
                        parallel_combinations.append(sorted(set_of_tracks))
                        costs.append(dt)

    if not parallel_combinations:
        parallel_combinations = [[track_id]]
        costs = [1]

    return parallel_combinations, costs


def coupling_constraints(edge_names, merged_tracks):
    """
    Adds edge coupling constraints for tracks to be merged.
    Args:
        edge_names: list of variable names
        merged_tracks: set of track ids

    Returns: list of inequality constraint, list of right hand side of ieq constraint

    """
    ieq = []
    b_ieq = []
    for edge_pair in combinations(edge_names, 2):
        constraint = dict()
        constraint[edge_pair[0]] = -1
        constraint[edge_pair[1]] = 1
        constraint[merged_tracks] = 1
        ieq.append(constraint)
        b_ieq.append(1)

        # swap signs of edges
        constraint = dict()
        constraint[edge_pair[0]] = 1
        constraint[edge_pair[1]] = -1
        constraint[merged_tracks] = 1
        ieq.append(constraint)
        b_ieq.append(1)
    return ieq, b_ieq


def setup_constraints(all_tracks):
    """
    Defines constraints for untangling operations: remove predecessor-successor link, split track, merge track
    Args:
        all_tracks: dict containing all trajectories

    Returns: variables with assigned costs, inequality and equality constraints

    """
    # limitations: no handling of nested segmentation errors, switching segmentation errors
    # the graph can be adapted using the options:
    #   - cut edges between track and its predeccessors
    #   - cut edges between track and its successors
    #   - split track in several parts
    #   - merge a set of tracks
    graph_variables = {}  # var name: cost
    ieq_constraints = []
    eq_constraints = []  # list of dicts {varname:factor, ...}
    b_eq = []
    b_ieq = []  # list of ints
    merge_sets = {}  # {track_id: [[sets]}
    track_length = []
    n_edges = []
    for t_id, track in all_tracks.items():
        track_length.append(max(track.masks.keys()) - min(track.masks.keys()) + 1)
        n_edges.append(max(len(track.pred_track_id), len(track.successors)))
    # assume errors occur only for short time spans, leading to few short track fragments (errors)
    # many longer correct but fragmented tracks, very few full tracks
    median_track_length = np.ceil(np.quantile(track_length, 0.3))  # assuming shortest n% of tracks are error tracks
    n_edges_quantile = np.ceil(np.quantile(n_edges, 0.99))
    edge_cost = 2 * int(median_track_length * n_edges_quantile)

    print('detangling - set up equations')
    for t_id, track in all_tracks.items():
        pred_ieq = {}
        succ_ieq = {}

        n_pred_successors = len(set(chain.from_iterable([all_tracks[p].successors
                                                         for p in track.pred_track_id if p != 0])))
        n_succ_predecessors = len(set(chain.from_iterable([list(all_tracks[s].pred_track_id)
                                                           for s in track.successors])))

        merged_track_sets = calc_merge_sets(all_tracks, t_id)
        for m_set, dt_cost in zip(*merged_track_sets):
            merge_var_name = [str(m) for m in m_set]
            # add merge tracks
            merge_current = '_'.join(['m', *merge_var_name])
            if len(m_set) > 1:
                if merge_current not in graph_variables:
                    graph_variables[merge_current] = dt_cost * (len(m_set) - 1)
                if t_id not in merge_sets:
                    merge_sets[t_id] = []
                if m_set not in merge_sets[t_id]:
                    merge_sets[t_id].append(merge_var_name)

                same_pred = [i for i in m_set if all_tracks[i].pred_track_id == track.pred_track_id]
                same_succ = [i for i in m_set if all_tracks[i].successors == track.successors]
                pred_ieq[merge_current] = len(same_pred) - 1  # this does not effect pred/successors
                succ_ieq[merge_current] = len(same_succ) - 1

            # add edge cuts / split tracks
            for m in m_set:
                split_id = 's_' + str(m)
                if split_id not in graph_variables:
                    graph_variables[split_id] = max(all_tracks[m].masks.keys()) \
                                                - min(all_tracks[m].masks.keys()) + 1
                succ_ieq[split_id] = -1
                pred_ieq[split_id] = -1
                pred_cuts = {'e_'+str(pred) + '_' + str(m): edge_cost
                             for pred in all_tracks[m].pred_track_id if pred in all_tracks.keys()}
                succ_cuts = {'e_' + str(m) + '_' + str(succ): edge_cost
                             for succ in all_tracks[m].successors}

                graph_variables.update(pred_cuts)
                graph_variables.update(succ_cuts)

                succ_ieq.update({k: -1 for k in succ_cuts.keys()})
                pred_ieq.update({k: -1 for k in pred_cuts.keys()})

        # add predecessors
        for pred in track.pred_track_id:
            # add split predecessor track
            if pred != 0:
                split_pred = 's_' + str(pred)
                if split_pred not in graph_variables:
                    graph_variables[split_pred] = max(all_tracks[pred].masks.keys()) \
                                                  - min(all_tracks[pred].masks.keys()) + 1
                pred_ieq[split_pred] = 1  # positive! - as more predecessors need to be resolved
                # add merge predecessor tracks
                merged_track_sets = calc_merge_sets(all_tracks, pred)
                for m_set, dt_cost in zip(*merged_track_sets):
                    if len(m_set) > 1:
                        merge_var_name = [str(m) for m in m_set]
                        merge_pred = '_'.join(['m', *merge_var_name])
                        if merge_pred not in graph_variables:
                            graph_variables[merge_pred] = dt_cost * (len(m_set) - 1)
                        if pred not in merge_sets:
                            merge_sets[pred] = []
                        if merge_var_name not in merge_sets[pred]:
                            merge_sets[pred].append(merge_var_name)
                        same_succ = [i for i in m_set if t_id in all_tracks[i].successors]

                        pred_ieq[merge_pred] = min(-len(same_succ) + 1, 0)

        # add successors
        for succ in track.successors:
            # add split successor track
            split_succ = 's_' + str(succ)
            if split_succ not in graph_variables:
                graph_variables[split_succ] = max(all_tracks[succ].masks.keys()) \
                                              - min(all_tracks[succ].masks.keys()) + 1
            succ_ieq[split_succ] = 1  # positive! - as more successors need to be resolved

            # add merge successor tracks
            merged_track_sets = calc_merge_sets(all_tracks, succ)
            for m_set, dt_cost in zip(*merged_track_sets):
                if len(m_set) > 1:
                    merge_var_name = [str(m) for m in m_set]
                    merge_succ = '_'.join(['m', *merge_var_name])
                    if merge_succ not in graph_variables:
                        graph_variables[merge_succ] = dt_cost * (len(m_set) - 1)
                    if succ not in merge_sets:
                        merge_sets[succ] = []
                    if merge_var_name not in merge_sets[succ]:
                        merge_sets[succ].append(merge_var_name)
                    same_pred = [i for i in m_set if t_id in all_tracks[i].pred_track_id]

                    succ_ieq[merge_succ] = min(-len(same_pred) + 1, 0)
        if track.pred_track_id != [0]:
            ieq_constraints.append(pred_ieq)
            b_ieq.append(-len(track.pred_track_id) + n_pred_successors)
        if track.successors:
            ieq_constraints.append(succ_ieq)
            b_ieq.append(-len(track.successors) + max(2*n_succ_predecessors, 1))

    mutual_exclusive_ieq = []
    b_me = []
    for t_id, m_set in merge_sets.items():
        # merge sets mutually exclusive
        constraint = {'_'.join(['m', *e]): 1 for e in m_set}
        mutual_exclusive_ieq.append(constraint)
        b_me.append(1)
    ieq_constraints.extend(mutual_exclusive_ieq)
    b_ieq.extend(b_me)

    # coupling cut edge for merge sets
    for t_id, m_sets in merge_sets.items():
        for merge_track_ids in m_sets:
            merge_var_name = '_'.join(['m', *merge_track_ids])
            m_tracks = [int(m) for m in merge_track_ids]

            pred_cuts = [(str(p), str(m_id)) for m_id in m_tracks for p in all_tracks[m_id].pred_track_id if p != 0]
            succ_cuts = [(str(m_id), str(s)) for m_id in m_tracks for s in all_tracks[m_id].successors]

            pred = {p_cut[0] for p_cut in pred_cuts}
            succ = {s_cut[1] for s_cut in succ_cuts}
            for p in pred:
                p_cuts = ['_'.join(['e', *p_cut]) for p_cut in pred_cuts if p == p_cut[0]]
                coupling_ieq, coupling_b_ieq = coupling_constraints(p_cuts, merge_var_name)
                ieq_constraints.extend(coupling_ieq)
                b_ieq.extend(coupling_b_ieq)
            for s in succ:
                s_cuts = ['_'.join(['e', *s_cut]) for s_cut in succ_cuts if s == s_cut[1]]
                coupling_ieq, coupling_b_ieq = coupling_constraints(s_cuts, merge_var_name)
                ieq_constraints.extend(coupling_ieq)
                b_ieq.extend(coupling_b_ieq)

    return graph_variables, eq_constraints, b_eq, ieq_constraints, b_ieq


def print_const(ieq, b):
    """Prints constraints to console."""
    for aa, bb in zip(ieq, b):
        a_part = [str(v)+'*'+str(k) for k, v in aa.items()]
        print(' + '.join(a_part), ' <= ', bb)


def solve_untangling_problem(costs, A_eq, b_eq, A_ieq, b_ieq):
    """
    Solves untangling problem as integer linear program.
    Args:
        costs: dict of cost terms
        A_eq: list of equality constraints
        b_eq: list of right hand side of eq constraints
        A_ieq: list of inequality constraints
        b_ieq: list of right hand side of ieq constraints

    Returns:

    """
    print('detangling - solve optim problem')
    index_model_vars = {i: k for i, k in enumerate(costs.keys())}
    var_name_to_index = {v: k for k, v in index_model_vars.items()}

    model = gp.Model('detangle')
    m_costs = [costs[index_model_vars[i]] for i in range(len(index_model_vars))]
    v_type = []
    for k in sorted(index_model_vars.keys()):
        if index_model_vars[k].startswith('s'):
            v_type.append(GRB.INTEGER)
        else:
            v_type.append(GRB.BINARY)

    model_vars = model.addVars(range(len(index_model_vars)), vtype=v_type, obj=m_costs)
    model.modelSense = GRB.MINIMIZE

    model.addConstrs((gp.LinExpr([(factor, model_vars[var_name_to_index[var_name]])
                                  for var_name, factor in A_ieq[i_index].items()]) <= b_ieq[i_index]
                      for i_index in range(len(A_ieq))))

    model.addConstrs((gp.LinExpr([(factor, model_vars[var_name_to_index[var_name]])
                                  for var_name, factor in A_eq[i_index].items()]) == b_eq[i_index]
                      for i_index in range(len(A_eq))))
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print('Optimal objective: %g' % model.objVal)
    elif model.status == GRB.INF_OR_UNBD:
        print('Model is infeasible or unbounded')
        sys.exit(0)
    elif model.status == GRB.INFEASIBLE:
        print('Model is infeasible')
        sys.exit(0)
    elif model.status == GRB.UNBOUNDED:
        print('Model is unbounded')
        sys.exit(0)
    else:
        print('Optimization ended with status %d' % model.status)
        sys.exit(0)
    optim_result = {index_model_vars[f_var.index]: int(np.rint(f_var.X))
                    for f_var in model.getVars()}
    return optim_result


def no_untangling(all_tracks):
    """
    Removes predecessor- successor links if tracks have more than one predecessor/
    more than two successors. To evaluate influence of untangling step.
    Args:
        all_tracks: dict containing all trajectories

    Returns: dict containing all trajectories

    """
    for t_id in list(all_tracks.keys()):
        track = all_tracks[t_id]
        if len(track.pred_track_id) > 1:
            pred = track.pred_track_id
            track.pred_track_id = [0]
            for p in pred:
                if t_id in all_tracks[p].successors:
                    all_tracks[p].successors.remove(t_id)
        if len(track.successors) > 2:
            succ = track.successors
            track.successors = set()
            for s in succ:
                if t_id in all_tracks[s].pred_track_id:
                    all_tracks[s].pred_track_id.remove(t_id)
    for t_id in list(all_tracks.keys()):
        if not all_tracks[t_id].pred_track_id:
            all_tracks[t_id].pred_track_id = [0]

    return all_tracks


def no_fn_correction(all_tracks, keep_link=True):
    """
    Splits track if a segmentation mask is missing.
    To evaluate influence of adding missing segmentation masks.
    Args:
        all_tracks: dict containing all trajectories
        keep_link: boolean whether to keep link of fragmented track

    Returns: dict containing all trajectories

    """
    max_id = max(all_tracks.keys())
    new_tracks = {}
    for t_id in all_tracks.keys():
        track = all_tracks[t_id]
        time_steps = np.array(sorted(track.masks.keys()))
        if len(time_steps) == (time_steps[-1] - time_steps[0] + 1):
            continue
        d_t = time_steps[1:] - time_steps[:-1]
        is_gap = d_t > 1
        t_start_tracklet = time_steps[1:][is_gap]

        pred_track = t_id
        succ_tracks = track.successors
        track_masks = {}
        for t in list(sorted(track.masks.keys())):
            if t >= t_start_tracklet[0]:
                track_masks[t] = track.masks.pop(t)
        track.successors = set()
        for t in sorted(track_masks.keys()):
            m = track_masks[t]
            if t in t_start_tracklet:
                max_id += 1
                if not keep_link:
                    new_tracks[max_id] = CellTrack(max_id, pred_track_id=[0])
                else:
                    if pred_track in new_tracks:
                        new_tracks[pred_track].successors = {max_id}
                    else:
                        all_tracks[pred_track].successors = {max_id}
                    new_tracks[max_id] = CellTrack(max_id, pred_track_id=[pred_track])
                    pred_track = max_id
            new_tracks[max_id].masks[t] = m
        new_tracks[max_id].successors = succ_tracks
        for s in succ_tracks:
            if t_id in all_tracks[s].pred_track_id:
                all_tracks[s].pred_track_id.remove(t_id)
                all_tracks[s].pred_track_id.append(max_id)
                all_tracks[s].pred_track_id = list(set(all_tracks[s].pred_track_id))

    all_tracks.update(new_tracks)
    return all_tracks
