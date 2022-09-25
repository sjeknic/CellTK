"""Unmerges masks of multiple merged objects"""
from itertools import product

# sjeknic(20220924 changed to kvxopt instead of cvxopt)
import kvxopt
import numpy as np
from kvxopt import glpk
from kvxopt.glpk import ilp as int_lin_prog
from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import disk
from skimage.segmentation import watershed

glpk.options['msg_lev'] = 'GLP_MSG_OFF'


# fixme: replace glpk with gurobi here
def compute_split(merged_object_ind, object_centers, plot=False):
    """
    Computes a potential split for a merged object.
    Args:
        merged_object_ind: array of mask indices of a segmentation mask
        object_centers: array of object center positions
        plot: boolean whether to show result

    Returns: array of seed points

    """
    assert len(object_centers) <= len(merged_object_ind[0])
    seed_points = calc_seed_points(merged_object_ind, len(object_centers), plot)
    matching = find_best_matches(list(zip(*seed_points)), object_centers)
    object_indices, point_indices = list(zip(*matching))
    selected_points = seed_points[:, tuple(point_indices)]
    return selected_points


def calc_seed_points(merged_object_ind, min_num_seeds, plot=False):
    """
    Calculates a set of seed points given a segmentation mask.
    Args:
        merged_object_ind:  array of mask indices of a segmentation mask
        min_num_seeds: int indicating minimum number of seed points needed
        plot: boolean whether to show result

    Returns:

    """
    # add background border - needed for distance trafo
    box_shape = np.max(merged_object_ind, axis=1) - np.min(merged_object_ind, axis=1) + 3
    mask = np.zeros(tuple(box_shape))
    mask[tuple(merged_object_ind - np.min(merged_object_ind, axis=1).reshape(-1, 1) + 1)] = 1
    dist_trafo = distance_transform_edt(mask)
    gradient_dist = np.stack(np.gradient(dist_trafo))
    abs_grad = np.sum(gradient_dist**2, axis=0)
    seed_points = np.where((abs_grad < 0.1) & (dist_trafo > 0))
    if plot:
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(abs_grad)
        ax[1].imshow((abs_grad < 0.1) & (dist_trafo > 0))
    if len(seed_points[0]) < min_num_seeds:
        indices = np.random.choice(merged_object_ind.shape[-1],
                                   min(merged_object_ind.shape[-1], min_num_seeds*3),
                                   replace=False)
        seed_points = merged_object_ind[:, tuple(indices)]
    else:
        # compute non shifted position
        seed_points = np.array(seed_points) + \
                      np.min(merged_object_ind, axis=1).reshape(-1, 1) - 1
    return seed_points


def find_best_matches(seed_points, object_centers):
    """
    Matches object centers to closest seed points.
    Args:
        seed_points: list of mask point positions
        object_centers: list of object centers positions

    Returns:
        list of tuples that indicate the mapping of seed points to object centers
    """
    edge_variables = {i: ((indices_combi[0], indices_combi[1]),
                          np.linalg.norm(object_centers[indices_combi[0]] - seed_points[indices_combi[1]]))
                      for i, indices_combi in enumerate(product(np.arange(len(object_centers)),
                                                                np.arange(len(seed_points))))}
    index_edge_variables = {v[0]: k for k, v in edge_variables.items()}
    n_variables = len(edge_variables)
    costs = kvxopt.matrix([edge_variables[index_key][1]
                           for index_key in sorted(edge_variables.keys())])

    # optim problem: each object center needs to be assigned to a seed point
    # and at most one object center is assigned to a seed point

    # eq constraints: assign each object center to a seed point
    matrix_indices = [(i_obj, index_edge_variables[(i_obj, j_point)])
                      for i_obj in range(len(object_centers))
                      for j_point in range(len(seed_points))]
    ind_i, ind_j = list(zip(*matrix_indices))
    eq_constraints = kvxopt.spmatrix(1, ind_i, ind_j,
                                     (len(object_centers),
                                      len(index_edge_variables)))
    b_eq = len(object_centers) * [1]

    # ieq constraints: maximum one object center assigned to a seed point
    ieq_constraints = []
    matrix_indices = [(i_point, index_edge_variables[(j_obj, i_point)])
                      for i_point in range(len(seed_points))
                      for j_obj in range(len(object_centers))]
    ind_i, ind_j = list(zip(*matrix_indices))
    in_flow = kvxopt.spmatrix(1, ind_i, ind_j,
                              (len(seed_points),
                               len(index_edge_variables)))
    ieq_constraints.append(in_flow)
    b_ieq = [1] * len(seed_points)

    upper_border = kvxopt.spmatrix(1, range(n_variables),
                                   range(n_variables), (n_variables, n_variables))
    ieq_constraints.append(upper_border)
    b_ieq.extend([1] * n_variables)
    lower_border = kvxopt.spmatrix(-1, range(n_variables),
                                   range(n_variables), (n_variables, n_variables))
    ieq_constraints.append(lower_border)
    b_ieq.extend([0] * n_variables)

    integer_vars = set(edge_variables.keys())
    status, x = int_lin_prog(costs, kvxopt.sparse(ieq_constraints),
                             kvxopt.matrix(b_ieq, tc='d'),
                             kvxopt.sparse(eq_constraints),
                             kvxopt.matrix(b_eq, tc='d'), integer_vars)

    result = [edge_variables[id_variable][0]
              for id_variable, var_value in enumerate(x) if var_value > 0]
    return result


def match_pred_succ(predecessors, successors):
    """
    Matches n predecessors to k successors, where k <= n
    Args:
        predecessors: list of object center positions
        successors: list of object center positions

    Returns: list of tuples that indicate the mapping of predecessors to successors

    """
    edge_variables = {i: ((indices_combi[0], indices_combi[1]),
                          np.linalg.norm(successors[indices_combi[0]] - predecessors[indices_combi[1]]))
                      for i, indices_combi in enumerate(product(np.arange(len(successors)),
                                                                np.arange(len(predecessors))))}
    index_edge_variables = {v[0]: k for k, v in edge_variables.items()}
    n_variables = len(edge_variables)
    costs = kvxopt.matrix([edge_variables[index_key][1]
                           for index_key in sorted(edge_variables.keys())])
    # optim problem: each successor needs to be assigned to a predecessor
    # and at most two successors are assigned to a predecessor,
    # however at least one successor is assigned to a predecessor

    # eq constraints: assign each successor to exactly one predecessor
    matrix_indices = [(i_obj, index_edge_variables[(i_obj, j_point)])
                      for i_obj in range(len(successors))
                      for j_point in range(len(predecessors))]
    ind_i, ind_j = list(zip(*matrix_indices))
    eq_constraints = kvxopt.spmatrix(1, ind_i, ind_j, (len(successors), len(index_edge_variables)))
    b_eq = len(successors) * [1]

    # ieq constraints: assign predecessors at least one, at most two successors
    ieq_constraints = []
    matrix_indices = [(i_point, index_edge_variables[(j_obj, i_point)])
                      for i_point in range(len(predecessors))
                      for j_obj in range(len(successors))]
    ind_i, ind_j = list(zip(*matrix_indices))
    in_flow_min = kvxopt.spmatrix(-1, ind_i, ind_j, (len(predecessors), len(index_edge_variables)))
    ieq_constraints.append(in_flow_min)
    b_ieq = [-1] * len(predecessors)

    upper_border = kvxopt.spmatrix(1, range(n_variables), range(n_variables), (n_variables, n_variables))
    ieq_constraints.append(upper_border)
    b_ieq.extend([1] * n_variables)
    lower_border = kvxopt.spmatrix(-1, range(n_variables), range(n_variables), (n_variables, n_variables))
    ieq_constraints.append(lower_border)
    b_ieq.extend([0] * n_variables)

    integer_vars = set(edge_variables.keys())
    status, x = int_lin_prog(costs, kvxopt.sparse(ieq_constraints),
                             kvxopt.matrix(b_ieq, tc='d'),
                             kvxopt.sparse(eq_constraints),
                             kvxopt.matrix(b_eq, tc='d'), integer_vars)

    result = [edge_variables[id_variable][0]
              for id_variable, var_value in enumerate(x)
              if var_value > 0]
    return result


def dummy_objects(img_size=(200, 200), max_n_obj=5):
    """Creates dummy objects for testing."""
    mask = np.zeros([img_s + img_s // 10 * 2 + 2
                     for img_s in img_size])
    merged_mask = mask.copy()
    coord_offset = [(img_s // 10 * 2) + 1
                    for img_s in img_size]
    n_obj = np.random.randint(4, max_n_obj+1)
    obj_positions = []
    for i_obj in range(n_obj):
        obj_disk = disk(img_size[0] / 10)
        obj_position = np.random.randint(coord_offset, img_size)
        obj_positions.append(obj_position)
        offset = np.random.randint(0, [min(20, img_s-obj_pos+1)
                                       for obj_pos, img_s in zip(obj_position, img_size)])
        off_position = [obj_pos + shift for obj_pos, shift in zip(obj_position, offset)]
        mask[obj_position[0] - obj_disk.shape[0] // 2:
             obj_position[0] + obj_disk.shape[0] // 2 + 1,
             obj_position[1] - obj_disk.shape[1] // 2:
             obj_position[1] + obj_disk.shape[1] // 2 + 1][obj_disk > 0] = (i_obj + 1)
        merged_mask[off_position[0] - obj_disk.shape[0] // 2:
                    off_position[0] + obj_disk.shape[0] // 2 + 1,
                    off_position[1] - obj_disk.shape[1] // 2:
                    off_position[1] + obj_disk.shape[1] // 2 + 1] += obj_disk
    return mask, (merged_mask > 0).astype(np.int), obj_positions


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    for i_run in range(1000):
        single_objects, merged_objects, obj_center = dummy_objects(max_n_obj=7)
        seeds = compute_split(np.where(merged_objects > 0), obj_center)
        unmerged = np.zeros_like(merged_objects)
        seeds_mask = np.zeros_like(unmerged)
        sm = np.zeros(merged_objects.shape)
        for i_seed, seed in enumerate(np.array(seeds).T):
            seeds_mask[tuple(seed)] = i_seed + 1
        unmerged = watershed(-distance_transform_edt(merged_objects),
                             seeds_mask, mask=merged_objects)
        f, ax = plt.subplots(1, 4)
        ax[0].imshow(single_objects, cmap='magma')
        ax[0].set_title('t')
        ax[1].imshow(merged_objects, cmap='magma')
        ax[1].set_title('segmentation t+1')
        ax[2].imshow(unmerged, cmap='magma')
        ax[2].set_title('unmerged t+1')
        ax[3].imshow(seeds_mask > 0, cmap='magma')
        ax[3].set_title('seeds t+1')
        plt.show()
