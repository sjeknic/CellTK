"""Utilities to construct a graph from vertices and find the best paths through the graph with min
cost flow."""
import sys
from itertools import chain
from itertools import product, combinations

import gurobipy as gp
import numpy as np
from gurobipy import GRB


class Edge:
    def __init__(self, start_vertex, end_vertex, cost, edge_capacity):
        """
        Defines directed edges between a pair of vertices.
        Args:
            start_vertex: the Vertex instance the edge starts from
            end_vertex: the Vertex instance the edge ends at
            cost: cost of the edge
            edge_capacity: capacity of the edge (see min cost flow programming)
        """
        self.check_edge(start_vertex, end_vertex)
        self.cost = cost
        self.capacity = edge_capacity
        self.start_vertex = start_vertex
        self.end_vertex = end_vertex
        self._string_id = construct_edge_id(self.start_vertex, self.end_vertex)

    def check_edge(self, start_vertex, end_vertex):
        assert start_vertex != end_vertex, 'No self loops allowed'
        assert start_vertex.id.time <= end_vertex.id.time, 'no backwards connections allowed'

    def __eq__(self, other):
        if isinstance(other, Edge):
            if self.start_vertex.id == other.start_vertex.id and self.end_vertex.id == other.end_vertex.id:
                return True
        return False

    def string_id(self):
        """Id of the edge"""
        return self._string_id


class CoupledEdge(Edge):
    def __init__(self, start_vertex, end_vertex, cost, edge_capacity, edge_name):
        """
        Defines coupled,directed edges.
        Args:
            start_vertex: the Vertex instance the edge starts from
            end_vertex: the Vertex instance the edge ends at
            cost: cost of the edge
            edge_capacity: capacity of the edge (see min cost flow programming)
            edge_name: name of the edge this edge is coupled with
        """
        super().__init__(start_vertex, end_vertex, cost, edge_capacity)
        self._string_id = edge_name
        self.coupled_edges = set()

    def add_coupling(self, edge_name):
        self.coupled_edges.add(edge_name)


def construct_edge_id(start_vertex, end_vertex):
    if isinstance(end_vertex, Vertex):
        return ''.join([start_vertex.id.string_id(), 'x', end_vertex.id.string_id()])
    else:
        end_v = 'x'.join([v.id.string_id() for i, v in enumerate(end_vertex)])
        edge_name = ''.join([start_vertex.id.string_id(), 'x', end_v])
        return [edge_name+'x'+str(i) for i in range(1, len(end_vertex) + 1)]


class VertexId:
    def __init__(self, time, index_id):
        """
        Provides a unique identifier for each vertex in a graph.
        Args:
            time: an integer defining the time point of the vertex in the graph
            index_id: identifier of the vertex within the considered time point
        """
        assert isinstance(time, int), 'time should be an int'
        self.time = time
        self.index_id = index_id
        self._string_id = str(self.time) + '_' + str(self.index_id)

    def __eq__(self, other):
        if self.time == other.time and self.index_id == other.index_id:
            return True
        return False

    def string_id(self):
        return self._string_id


class Vertex:
    def __init__(self, time, index_id, edge_capacity=1, features=None):
        """
        Defines a "normal" vertex in a graph. Those vertices correspond to segmented objects.
        Args:
            time: an int defining the time point where the vertex exists
            index_id: index of the vertex within the considered time point
            edge_capacity: an int specifying the maximum capacity of the in/out edges
            features: a np.array of features
        """
        self.id = VertexId(time, index_id)
        self.type = 'normal'
        self.features = features
        self.edge_capacity = edge_capacity  # maximum capacity of edges ending at vertex
        self.in_edges = {}
        self.out_edges = {}
        self.next_vertices = set()

    def __eq__(self, other):
        if self.id == other.id:
            if self.type == other.type:
                return True
        return False

    def add_edge(self, edge):
        """Add incoming/ outgoing edges to vertex"""
        if edge.start_vertex == self:
            assert edge.string_id() not in self.out_edges.keys(), 'edge with same id already added to vertex'
            self.out_edges[edge.string_id()] = edge
        elif edge.end_vertex == self:
            assert edge.string_id() not in self.in_edges, \
                f'edge {edge.string_id()} already added to vertex'
            self.in_edges[edge.string_id()] = edge


class SplitVertex(Vertex):
    def __init__(self, time, index_id, edge_capacity=1, features=None):
        """Defines a split vertex in a graph, which
        allows modelling of splitting objects due to over-segmentation or cell division."""
        super().__init__(time, index_id, edge_capacity, features)
        self.type = 'split'


class MergeVertex(Vertex):
    """Defines a merge vertex in a graph, which allows
     modelling of merging objects due to under-segmentation."""
    def __init__(self, time,  index_id, edge_capacity=1, features=None):
        super().__init__(time, index_id, edge_capacity, features)
        self.type = 'merge'


class AppearVertex(Vertex):
    """Defines an appear vertex in a graph, which
    allows modelling of appearing objects."""
    def __init__(self, time, index_id, edge_capacity, features=None):
        super().__init__(time, index_id, edge_capacity, features)
        self.type = 'appear'


class DeleteVertex(Vertex):
    """Defines a delete vertex in a graph , which
    allows modelling of disappearing objects."""
    def __init__(self, time, index_id, edge_capacity, features=None):
        super().__init__(time, index_id, edge_capacity, features)
        self.type = 'delete'


class SourceVertex(Vertex):
    """Defines the source of all flow."""
    def __init__(self, time, index_id, edge_capacity, features=None):
        super().__init__(time, index_id, edge_capacity, features)
        self.type = 'source'


class SinkVertex(Vertex):
    """Defines the sink of all flow."""
    def __init__(self, time, index_id, edge_capacity, features=None):
        super().__init__(time, index_id, edge_capacity, features)
        self.type = 'sink'


class SkipVertex(Vertex):
    """Defines a skip connection between two successive time steps, which
     allows modelling of missing segmentation masks."""
    def __init__(self, time, index_id, edge_capacity=1, features=None):
        super().__init__(time, index_id, edge_capacity, features)
        self.type = 'skip'


class VerticesDict:
    def __init__(self):
        """
        Contains a set of vertices.
        data: a dict storing all vertices by their unique identifier
        data_time_points: a dist storing all vertices by the time step they exist in
        """
        self.data = {}
        self.data_time_points = {}

    def __getitem__(self, data_index):
        """
        Returns vertex by their identifier.
        Args:
            data_index: instance of VertexId which indexes a vertex

        Returns: instance of Vertex

        """
        if isinstance(data_index, tuple):
            data_key = VertexId(*data_index).string_id()
            assert data_key in self.data.keys(), f'vertex id {data_index} not in dictionary'
            return self.data[data_key]
        if isinstance(data_index, str):
            assert data_index in self.data.keys(), f'vertex id {data_index} not in dictionary'
            return self.data[data_index]
        assert isinstance(data_index, int), f'{data_index} is not an integer time step'
        assert data_index in self.data_time_points.keys(), \
            f'time step {data_index} not in time steps. Available time steps: {self.data_time_points.keys()}'
        return self.data_time_points[data_index]

    def add_vertex(self, vertex):
        """
        Adds a vertex to the dict.
        Args:
            vertex: instance of Vertex
        """
        assert vertex.id.string_id() not in self.data.keys(), 'vertex with same id already exists'
        self.data.update({vertex.id.string_id(): vertex})
        if vertex.id.time not in self.data_time_points.keys():
            self.data_time_points[vertex.id.time] = [vertex]
        else:
            self.data_time_points[vertex.id.time].append(vertex)

    def get_time_steps(self):
        return sorted(self.data_time_points.keys())

    def get_vertices_by_type(self, vertex_type, time=None):
        """Returns all vertices of a specific vertex type."""
        if time is None:
            return filter(lambda x: x.type == vertex_type, self.data.values())
        return filter(lambda x: x.type == vertex_type, self.data_time_points[time])

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.data_keys = iter(list(self.data.keys()))
        return self

    def __next__(self):
        return self.data[next(self.data_keys)]

    def __contains__(self, item):
        return item in self.data.keys()


class SparseGraph:
    def __init__(self, ad_costs, img_shape, allow_cell_division=True, n_mitosis_pairs=10):
        """
        Constructs a graph from a set of vertices and finds optimal paths between vertices by minimizing
                a coupled min cost flow problem.
        Args:
            ad_costs: a tuple providing the cost for appearance/disappearance of objects
            img_shape: a tuple providing the img shape
            allow_cell_division: a boolean indicating whether cell divisions shall be considered or not
            n_mitosis_pairs: an integer to prune the number of mitosis edges considering only the n mitosis pairs
            with the smallest cost for each split node
        """
        self.vertices = VerticesDict()
        self.edges = {}
        # directed edges start_vertex: [possible end vertices]
        self.valid_edges = {'source': ['appear', 'normal'],
                            'sink': [],
                            'merge': ['normal', 'delete'],
                            'split': ['normal'],
                            'appear': ['split', 'delete', 'normal'],
                            'delete': ['sink'],
                            'normal': ['merge', 'split', 'skip', 'delete', 'sink', 'normal'],
                            'skip': ['merge', 'split', 'skip', 'sink', 'normal'],
                            }
        self.result = None
        if isinstance(ad_costs, tuple):
            self.appear_cost = ad_costs[0]  # e.g. average distance an object moves between 2 time steps
            self.delete_cost = ad_costs[1]  # e.g. average distance an object moves between 2 time steps
        else:
            self.appear_cost = ad_costs
            self.delete_cost = ad_costs
        self.img_shape = np.array(img_shape)
        self.cell_division = allow_cell_division
        self.n_mitosis_pairs = n_mitosis_pairs

    def get_vertex(self, vertex_id):
        return self.vertices[vertex_id]

    def add_vertex(self, vertex):
        self.vertices.add_vertex(vertex)

    def construct_graph(self):
        """Sets up the graph from a set of provided vertices."""
        all_time_points = self.vertices.get_time_steps()
        assert all_time_points, 'empty vertices list'
        assert len(set(all_time_points)) > 1, 'all vertices in same time step'
        start_time = all_time_points[0]
        end_time = all_time_points[-1]

        print('Add vertices to graph')
        for i, time_point in enumerate(all_time_points[1:]):
            normal_vertices_current_step = list(self.vertices.get_vertices_by_type('normal', all_time_points[i]))
            normal_vertices_next_step = list(self.vertices.get_vertices_by_type('normal', time_point))

            # create merge vertices at t for each object at t+1
            for n_vertex in normal_vertices_next_step:
                m_vertex = MergeVertex(all_time_points[i], 'merge_' + n_vertex.id.string_id())
                m_vertex.next_vertices.add(n_vertex.id.string_id())
                self.add_vertex(m_vertex)
            merge_vertices_current_step = list(self.vertices.get_vertices_by_type('merge', all_time_points[i]))

            # add appear vertex
            appear_vertex = AppearVertex(all_time_points[i], 'a', len(normal_vertices_next_step))
            self.add_vertex(appear_vertex)

            # add delete vertex
            # capacity: sum of normal vertices at t-1 and t, because a and d are connected!
            delete_vertex = DeleteVertex(time_point, 'd', len(normal_vertices_current_step)
                                         + len(normal_vertices_next_step))
            self.add_vertex(delete_vertex)

            # create split vertices at t+1 for each object at t
            if self.cell_division:
                for n_vertex in normal_vertices_current_step:
                    # next vertices split vertex is connected to only normal vertices
                    # - check with .islower to sort out special nodes
                    n_neighbors = list(filter(lambda x: ~x.islower(), n_vertex.next_vertices))
                    if len(n_neighbors) > 1:
                        split_vertex = SplitVertex(time_point, 'split_' + n_vertex.id.string_id())
                        self.add_vertex(split_vertex)
                        n_vertex.next_vertices.add(split_vertex.id.string_id())
                for n_vertex in self.vertices.get_vertices_by_type('skip', all_time_points[i]):
                    n_neighbors = list(filter(lambda x: ~x.islower(), n_vertex.next_vertices))
                    if len(n_neighbors) > 1:
                        split_vertex = SplitVertex(time_point, 'split_' + n_vertex.id.string_id())
                        self.add_vertex(split_vertex)
                        n_vertex.next_vertices.add(split_vertex.id.string_id())

            # connections from appear vertex to split/normal and from normal/merge to delete vertex
            # and from appear to delete
            split_vertices_next_step = list(self.vertices.get_vertices_by_type('split', time_point))
            appear_vertex.next_vertices.update({vertex.id.string_id()
                                                for vertex in normal_vertices_next_step})
            appear_vertex.next_vertices.update({vertex.id.string_id()
                                                for vertex in split_vertices_next_step})
            for merge_vertex in merge_vertices_current_step:
                merge_vertex.next_vertices.add(delete_vertex.id.string_id())
            for n_vertex in normal_vertices_current_step:
                n_vertex.next_vertices.add(delete_vertex.id.string_id())
            appear_vertex.next_vertices.add(delete_vertex.id.string_id())

        # add edges
        vertex_pairs = chain(*[product([vertex], vertex.next_vertices)
                               for vertex in self.vertices])
        for start_vertex, end_vertex_id in vertex_pairs:
            end_vertex = self.vertices[end_vertex_id]
            if start_vertex.type in ['normal', 'skip'] and end_vertex.type == 'normal':
                merge_vertex = self.vertices[str(start_vertex.id.time)+'_merge_'+end_vertex.id.string_id()]
                self.construct_edge(start_vertex, merge_vertex)
                self.construct_edge(start_vertex, end_vertex)
                if self.cell_division:
                    split_id = str(end_vertex.id.time)+'_split_'+start_vertex.id.string_id()
                    if split_id in self.vertices:
                        split_vertex = self.vertices[split_id]
                        self.construct_edge(start_vertex, split_vertex)
                        self.construct_edge(split_vertex, end_vertex)

            else:
                self.construct_edge(start_vertex, self.vertices[end_vertex_id])

        # add coupled edges for cell division
        if self.cell_division:
            for split_vertex in self.vertices.get_vertices_by_type('split'):
                end_v = [edge.end_vertex for edge in split_vertex.out_edges.values()]
                if len(end_v) > 1:
                    daughter_pairs = combinations(end_v, 2)
                    mitosis_pairs = [(pair, self.calc_mitosis_cost(split_vertex, pair)) for pair in daughter_pairs]
                    mitosis_pairs.sort(key=lambda x: x[1])
                    best_pairs = mitosis_pairs[:self.n_mitosis_pairs]
                    for pair in best_pairs:
                        daughter_vertices, cost = pair
                        self.construct_edge(split_vertex, daughter_vertices)

        print('Add sink and source vertex to graph')
        normal_vertices = self.vertices.get_vertices_by_type('normal')

        n_normal_vertices = sum(1 for _ in normal_vertices)
        source_vertex = SourceVertex(start_time - 1, 'source', n_normal_vertices)
        sink_vertex = SinkVertex(end_time + 1, 'sink', n_normal_vertices)
        self.add_vertex(source_vertex)
        self.add_vertex(sink_vertex)

        for appear_vertex in self.vertices.get_vertices_by_type('appear'):
            self.construct_edge(source_vertex, appear_vertex)
        for delete_vertex in self.vertices.get_vertices_by_type('delete'):
            self.construct_edge(delete_vertex, sink_vertex)

        # connect all vertices of first time/last time point to source/ sink
        for vertex in self.vertices[start_time]:
            self.construct_edge(source_vertex, vertex)
        for vertex in self.vertices[end_time]:
            if vertex.type != 'skip':
                self.construct_edge(vertex, sink_vertex)

        # add for each last skip vertex connection to sink
        all_skip_vertices = [vertex.id.string_id()
                             for vertex in self.vertices.get_vertices_by_type('skip')]
        if len(all_skip_vertices) > 0:
            for vertex in self.vertices.get_vertices_by_type('normal'):
                v_id = vertex.id.string_id()
                skip_vertices = filter(lambda x: v_id == x.split('skip_')[-1], all_skip_vertices)
                time_point_skip = [int(skip_vertex.split('_')[0])
                                   for skip_vertex in skip_vertices]
                if len(time_point_skip) > 0:
                    last_time_skip = np.max(time_point_skip)
                    last_skip_vertex = self.get_vertex(str(last_time_skip)+'_skip_'+v_id)
                    self.construct_edge(last_skip_vertex, sink_vertex)

        # set edge capacity for appear-split, merge-delete edges,
        # as now all in going/ out going edges known
        for edge_key in self.edges.keys():
            edge = self.edges[edge_key]
            if edge.start_vertex.type == 'merge' and edge.end_vertex.type == 'delete':
                n_edges = [1 for e_in in edge.start_vertex.in_edges.values()
                           if e_in.start_vertex.type in ['normal', 'skip']]
                edge.capacity = len(n_edges)
            elif edge.start_vertex.type == 'appear' and edge.end_vertex.type == 'split':
                n_edges = [1 for e_out in edge.end_vertex.out_edges.values() if
                           e_out.end_vertex.type in ['normal', 'skip']]
                edge.capacity = len(n_edges)

    def compute_constraints(self):
        """Sets up the equality constraints for optimization."""
        print('Set up constraints')
        flow_variables = {key: i
                          for i, key in enumerate(self.edges.keys())}
        # define equality constraints A_eq*x = b_eq
        A_eq = []
        b_eq = []
        ####################
        # flow conservation constraints
        # flow_in == flow_out for all non source/sink vertices
        # ##################
        A_flow_conservation = []
        b_flow_conservation = []
        for vertex in self.vertices:
            if not (vertex.type == 'sink' or vertex.type == 'source'):
                in_keys = vertex.in_edges.keys()
                out_keys = vertex.out_edges.keys()
                #  in_keys-out_keys = x
                constraint = dict()
                for edge_key in in_keys:
                    constraint[flow_variables[edge_key]] = 1
                for edge_key in out_keys:
                    constraint[flow_variables[edge_key]] = -1
                A_flow_conservation.append(constraint)
                b_flow_conservation.append(0)
        A_eq.extend(A_flow_conservation)
        b_eq.extend(b_flow_conservation)

        ##################
        # flow out of source == n units
        ##################
        source_vertex = list(self.vertices.get_vertices_by_type('source'))[0]
        A_source_constraint = dict()
        for edge_key in source_vertex.out_edges.keys():
            A_source_constraint[flow_variables[edge_key]] = 1
        b_source_constraint = source_vertex.edge_capacity
        A_eq.append(A_source_constraint)
        b_eq.append(b_source_constraint)

        ##################
        # constraint input==1, output==1 to all normal vertices
        ##################
        A_input_constraint = []
        b_input_constraint = []
        for vertex in self.vertices.get_vertices_by_type('normal'):
            constraint = dict()
            for edge_key in vertex.in_edges.keys():
                constraint[flow_variables[edge_key]] = 1
            A_input_constraint.append(constraint)
            b_input_constraint.append(vertex.edge_capacity)
            constraint = dict()
            for edge_key in vertex.out_edges.keys():
                constraint[flow_variables[edge_key]] = 1
            A_input_constraint.append(constraint)
            b_input_constraint.append(vertex.edge_capacity)
        A_eq.extend(A_input_constraint)
        b_eq.extend(b_input_constraint)

        ###############################
        # n input/ output units to appear
        ###############################
        A_input_constraint = []
        b_input_constraint = []
        for vertex in self.vertices.get_vertices_by_type('appear'):
            constraint = dict()
            for edge_key in vertex.out_edges.keys():
                constraint[flow_variables[edge_key]] = 1
            A_input_constraint.append(constraint)
            b_input_constraint.append(vertex.edge_capacity)
        A_eq.extend(A_input_constraint)
        b_eq.extend(b_input_constraint)

        ########################
        # coupling of split vertices with appear vertices
        #########################
        A_ieq = []
        b_ieq = []
        A_split_coupling_constraint = []
        b_split_coupling_constraint = []
        for vertex in self.vertices.get_vertices_by_type('split'):
            in_edges = vertex.in_edges
            out_edges = vertex.out_edges
            coupling_constraint = dict()
            flow_constraint = dict()
            for edge_key in in_edges.keys():  # 2 edges in: axs, nxs
                flow_constraint[flow_variables[edge_key]] = 1
                if isinstance(in_edges[edge_key].start_vertex, AppearVertex):
                    coupling_constraint[flow_variables[edge_key]] = -1
                else:
                    coupling_constraint[flow_variables[edge_key]] = 1
            # axs>=nxs, input to s only if axs set
            A_ieq.append(coupling_constraint)
            b_ieq.append(0)
            in_edge_mother_vertex = [edge_key for edge_key in in_edges.keys()
                                     if self.edges[edge_key].start_vertex.type
                                     in ['normal', 'skip']]
            assert len(in_edge_mother_vertex) == 1, 'more than one mother vertex assigned to split vertex'
            in_edge_mother_vertex = in_edge_mother_vertex[0]
            for edge_key in out_edges.keys():  # k sxn edges out
                flow_constraint[flow_variables[edge_key]] = -1
                coupling_in_out = dict()
                coupling_in_out[flow_variables[edge_key]] = 1
                coupling_in_out[flow_variables[in_edge_mother_vertex]] = -1
                # nxs => sxn, output from split node to 'daughter' node
                # only if flow from 'mother' node
                A_ieq.append(coupling_in_out)
                b_ieq.append(0)
            # sum in edges == flow out edges
            A_split_coupling_constraint.append(flow_constraint)
            b_split_coupling_constraint.append(0)

            # add coupled mitosis edges
            mitosis_constraint = dict()
            for edge_key, edge in out_edges.items():
                if isinstance(edge, CoupledEdge):
                    constraint = dict()
                    # e_daughter_1==e_daughter_2
                    constraint[flow_variables[edge_key]] = 1
                    constraint.update({flow_variables[coupled_edge_name]: -1
                                       for coupled_edge_name in edge.coupled_edges})
                    A_eq.append(constraint)
                    b_eq.append(0)
                    # sum of mitosis edges <=2
                    mitosis_constraint[flow_variables[edge_key]] = 1
            A_ieq.append(mitosis_constraint)
            b_ieq.append(2)
            # either mitosis or over-segmentation
            coupled_edges = [k for k, v in out_edges.items()
                             if isinstance(v, CoupledEdge)]
            not_coupled_edges = [k for k, v in out_edges.items()
                                 if not isinstance(v, CoupledEdge)]
            for c_edge, e in product(coupled_edges, not_coupled_edges):
                constraint = dict()
                constraint[flow_variables[c_edge]] = 1
                constraint[flow_variables[e]] = 1
                A_ieq.append(constraint)
                b_ieq.append(1)

        A_eq.extend(A_split_coupling_constraint)
        b_eq.extend(b_split_coupling_constraint)

        ##########################
        # coupling of delete vertex with merge vertices
        ##########################
        A_merge_coupling_constraint = []
        b_merge_coupling_constraint = []
        for vertex in self.vertices.get_vertices_by_type('merge'):
            in_edges = vertex.in_edges
            out_edges = vertex.out_edges
            coupling_constraint = dict()
            flow_constraint = dict()
            for edge_key in out_edges.keys():  # 2 edges out: m->n, m->d
                flow_constraint[flow_variables[edge_key]] = -1
                if isinstance(out_edges[edge_key].end_vertex, DeleteVertex):
                    coupling_constraint[flow_variables[edge_key]] = -1

                else:
                    coupling_constraint[flow_variables[edge_key]] = 1
            # mxn<=mxd, output from m->n only if m->d is set
            A_ieq.append(coupling_constraint)
            b_ieq.append(0)
            out_edge_merged_vertex = [edge_key for edge_key in out_edges.keys() if
                                      self.edges[edge_key].end_vertex.type in ['normal', 'skip']]
            assert len(out_edge_merged_vertex) == 1, 'more than one merged vertex assigned to merge vertex'
            out_edge_merged_vertex = out_edge_merged_vertex[0]
            for edge_key in in_edges.keys():  # k mxn  edges in
                flow_constraint[flow_variables[edge_key]] = 1
                coupling_in_out = dict()
                coupling_in_out[flow_variables[edge_key]] = 1
                coupling_in_out[flow_variables[out_edge_merged_vertex]] = -1
                # input to merge node only if output m->n is set
                A_ieq.append(coupling_in_out)
                b_ieq.append(0)
            # sum in edges == flow out edges
            A_merge_coupling_constraint.append(flow_constraint)
            b_merge_coupling_constraint.append(0)

        A_eq.extend(A_merge_coupling_constraint)
        b_eq.extend(b_merge_coupling_constraint)

        return A_eq, b_eq, A_ieq, b_ieq, flow_variables

    def compute_edge_cost(self, start_vertex, end_vertex):
        """Computes the edge cost between two vertices."""
        if start_vertex.type == 'appear' and end_vertex.type == 'normal':
            return max(1, self.calc_vertex_appear_cost(start_vertex, end_vertex))

        if (start_vertex.type in ['normal', 'skip']) and end_vertex.type == 'delete':
            return max(1, self.calc_vertex_delete_cost(start_vertex, end_vertex))

        if start_vertex.type == 'split' and end_vertex.type == 'normal':
            return self.calc_vertex_split_cost(start_vertex, end_vertex)

        if (start_vertex.type in ['normal', 'skip']) and end_vertex.type == 'merge':
            return self.calc_vertex_merge_cost(start_vertex, end_vertex)

        if (start_vertex.type in ['normal', 'skip']) and (end_vertex.type in ['normal', 'skip']):
            dist = compute_distance(start_vertex.features[-1], end_vertex.features[2])  # p_t+1_est and p_t+1
            if end_vertex.type == 'skip':
                dist = self.calc_vertex_skip_cost(start_vertex, end_vertex)
            return dist
        # all other edges have zero cost
        return 0

    def calc_vertex_skip_cost(self, start_vertex, end_vertex):
        """Calculates cost between normal/skip vertex to skip vertex."""
        dist = compute_distance(start_vertex.features[2], end_vertex.features[2])
        # allow skip only if no overlap of estimated seeds in next frame
        # no overlap bbox(t+1) and hat(seeds)(t+1)
        has_overlap = [seeds_in_bounding_box(self.get_vertex(v).features[0], end_vertex.features[1])
                       for v in start_vertex.next_vertices
                       if self.get_vertex(v).type == 'normal']
        if len(has_overlap) == 0:
            return dist
        if np.any(has_overlap):
            dist = self.delete_cost * 1000
        return dist

    def calc_vertex_appear_cost(self, start_vertex, end_vertex):
        """Calculates appearance costs."""
        if len(self.img_shape) == 3:
            border_dist_top_left = np.min(end_vertex.features[2][1:])
            border_dist_bottom_right = np.min(self.img_shape[1:] - np.stack(end_vertex.features[2])[1:])
        else:
            border_dist_top_left = np.min(end_vertex.features[2])
            border_dist_bottom_right = np.min(self.img_shape - np.stack(end_vertex.features[2]))
        border_dist = np.min(np.stack([border_dist_top_left, border_dist_bottom_right]))
        return min(self.appear_cost, border_dist)

    def calc_vertex_delete_cost(self, start_vertex, end_vertex):
        """Calculate disappearance costs."""
        if len(self.img_shape) == 3:
            border_dist_top_left = np.min(start_vertex.features[2][1:])
            border_dist_bottom_right = np.min(self.img_shape[1:] - np.stack(start_vertex.features[2])[1:])
        else:
            border_dist_top_left = np.min(start_vertex.features[2])
            border_dist_bottom_right = np.min(self.img_shape - np.stack(start_vertex.features[2]))
        border_dist = np.min(np.stack([border_dist_top_left, border_dist_bottom_right]))
        return min(self.delete_cost, border_dist)

    def calc_vertex_split_cost(self, start_vertex, end_vertex):
        """Calculates over-segmentation costs."""
        n_vertex_id = start_vertex.id.string_id().split('split_')[1]
        mother_vertex = self.vertices[n_vertex_id]
        seed_points = np.array(mother_vertex.features[1])
        # dist = ||seed_points(t) - p(t+1)||_2
        cost = np.min(np.linalg.norm(seed_points - end_vertex.features[2].reshape(-1, 1), axis=0))
        has_overlap = seeds_in_bounding_box(mother_vertex.features[0], end_vertex.features[1])
        if not has_overlap:
            cost = self.appear_cost * 1000
        return cost

    def calc_vertex_merge_cost(self, start_vertex, end_vertex):
        """Calculates under-segmentation costs."""
        n_vertex_id = end_vertex.id.string_id().split('merge_')[1]
        merged_vertex = self.vertices[n_vertex_id]
        seed_points = np.array(merged_vertex.features[1])
        # dist =  ||hat(p)(t+1) - seed_points(t+1)||_2
        cost = np.min(np.linalg.norm(seed_points - start_vertex.features[-1].reshape(-1, 1), axis=0))
        has_overlap = seeds_in_bounding_box(merged_vertex.features[0], start_vertex.features[1])
        if not has_overlap:
            cost = self.delete_cost * 1000
        return cost

    def prune_edges(self):
        """Remove edges with large costs."""
        # fix edge variables by setting their max capacity to one
        prune_cost = max(self.appear_cost, self.delete_cost) * 10
        for edge_name in self.edges.keys():
            edge = self.edges[edge_name]
            if edge.cost > prune_cost:
                edge.capacity = 0

    def gurobi_optimize(self):
        """Solves the optimization problem using commercial gurobi solver."""
        eq_constraints, b_eq, ieq_constraints, b_ieq, index_flow_vars = self.compute_constraints()
        # flow var names is dict with keys 0...N
        self.prune_edges()
        flow_variable_names = {v: k for k, v in index_flow_vars.items()}
        print('Add Equations')
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as model:

                # flow var names is dict with keys 0...N, no sorting of keys necessary
                costs = [self.edges[flow_variable_names[id_flow_var]].cost
                         for id_flow_var in range(len(flow_variable_names))]

                # lower / upper bound
                upper_bound = [self.edges[flow_variable_names[id_flow_var]].capacity
                               for id_flow_var in range(len(flow_variable_names))]
                flow_vars = model.addVars(range(len(flow_variable_names)), vtype=GRB.INTEGER,
                                          ub=upper_bound, obj=costs)
                model.modelSense = GRB.MINIMIZE

                # add equality and inequality constraints to the model
                model.addConstrs((gp.LinExpr([(factor, flow_vars[v_index])
                                              for v_index, factor in eq_constraints[i_index].items()]) == b_eq[i_index]
                                  for i_index in range(len(eq_constraints))))
                model.addConstrs((gp.LinExpr([(factor, flow_vars[v_index])
                                              for v_index, factor in ieq_constraints[i_index].items()]) <= b_ieq[i_index]
                                  for i_index in range(len(ieq_constraints))))
                print('Optimize')
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
                self.result = {flow_variable_names[f_var.index]: int(np.rint(f_var.X))
                               for f_var in model.getVars()}
        return self.result

    def construct_edge(self, start_vertex, end_vertex):
        """Constructs edges between two vertices"""
        if isinstance(end_vertex, Vertex):
            self._add_single_edge(start_vertex, end_vertex)
        elif start_vertex.type == 'split' and isinstance(end_vertex, tuple):
            self._add_coupled_edge(start_vertex, end_vertex)
        else:
            raise AssertionError('unknown inputs for edge')

    def _add_coupled_edge(self, start_vertex, end_vertices):
        """Constructs coupled edges betwenn pairs of end vertices."""
        cost = self.calc_mitosis_cost(start_vertex, end_vertices)
        edge_ids = construct_edge_id(start_vertex, end_vertices)
        if edge_ids[0] in self.edges.keys():
            return
        edges = {e_id: CoupledEdge(start_vertex, end_v, cost, 1, e_id) for e_id, end_v in
                 zip(*[edge_ids, end_vertices])}
        edges[edge_ids[0]].add_coupling(edge_ids[1])
        edges[edge_ids[1]].add_coupling(edge_ids[0])

        self.edges.update(edges)
        for edge in edges.values():
            start_vertex.add_edge(edge)
            edge.end_vertex.add_edge(edge)

    def _add_single_edge(self, start_vertex, end_vertex):
        """Constructs edges between two vertices"""
        if end_vertex.type not in self.valid_edges[start_vertex.type]:
            return
        edge_id = construct_edge_id(start_vertex, end_vertex)
        if edge_id in self.edges.keys():
            return
        edge_capacity = min(start_vertex.edge_capacity, end_vertex.edge_capacity)
        edge = Edge(start_vertex, end_vertex, self.compute_edge_cost(start_vertex, end_vertex),
                    edge_capacity)
        self.edges[edge.string_id()] = edge
        start_vertex.add_edge(edge)
        end_vertex.add_edge(edge)

    def calc_mitosis_cost(self, start_vertex, end_vertices):
        """Calculates the mitosis costs."""
        mother_vertex = [edge.start_vertex
                         for _, edge in start_vertex.in_edges.items()
                         if edge.start_vertex.type in ['normal', 'skip']][0]
        daugther_vertices = [vertex.features[2] for vertex in end_vertices]
        dist1 = 0.5 * (daugther_vertices[0] + daugther_vertices[1])
        dist1 = np.linalg.norm(dist1 - mother_vertex.features[2])

        dist2 = np.abs(np.linalg.norm(daugther_vertices[0] - mother_vertex.features[2].reshape(1, -1)) \
                       - np.linalg.norm(daugther_vertices[1] - mother_vertex.features[2]))

        dist_3 = np.linalg.norm(daugther_vertices[0] - daugther_vertices[1])

        size_mother = np.linalg.norm(mother_vertex.features[0][1] - mother_vertex.features[0][0])
        if dist_3 < 1.5*size_mother:
            dist = dist1 + dist2
        else:
            return 1.000*self.appear_cost
        return dist

    def print_graph(self):
        """Prints for each vertex the vertices which provide input to the vertex/ the vertex outputs to."""
        for vertex in self.vertices:
            print('___' * 4)
            print('vertex:', vertex.id.string_id())
            in_vertex = [egde.start_vertex.id.string_id() for egde in vertex.in_edges.values()]
            out_vertex = [edge.end_vertex.id.string_id() for edge in vertex.out_edges.values()]
            print('in:', in_vertex)
            print('out:', out_vertex)

    def calc_trajectories(self):
        """Computes the trajectories given by the flow variables of the edges."""
        trajectories = {}
        flow_edges = list(filter(lambda x: self.result[x] > 0, self.result.keys()))
        flow_edges.sort(key=lambda x: tuple([int(sub.split('_')[0]) for sub in x.split('x')]))
        for edge_name in flow_edges:
            # source/appear -> normal : create new trajectory
            # split -> normal : create new trajectory (start_vertex: n), add add id as successor to mother track
            # skip/normal -> merge : create new trajectory,
            #                  add all predecessors for merged node, add id as successor to predecessor tracks
            # skip/normal -> normal: add to trajectory
            # else: ignore

            start_vertex = self.edges[edge_name].start_vertex
            end_vertex = self.edges[edge_name].end_vertex
            if (start_vertex.type in ['source', 'appear']) and (end_vertex.type == 'normal'):
                track_id = len(trajectories)
                end_vertex.track_id = track_id
                trajectories[len(trajectories)] = {'predecessor': [start_vertex],
                                                   'track': [end_vertex],
                                                   'successor': []}
            elif (start_vertex.type == 'split') and (end_vertex.type == 'normal'):
                mother_vertex = [in_edge.start_vertex
                                 for in_edge in start_vertex.in_edges.values()
                                 if (self.result[in_edge.string_id()] > 0) and
                                 (in_edge.start_vertex.type in ['normal', 'skip'])]
                assert len(mother_vertex) == 1, ' split vertex has exactly one predecessor'
                end_vertex.track_id = len(trajectories)
                if mother_vertex[0].type == 'skip':
                    v_id = start_vertex.id.string_id().split('skip_')[-1]
                    mother_vertex = [self.get_vertex(v_id)]

                trajectories[len(trajectories)] = {'predecessor': mother_vertex,
                                                   'track': [end_vertex],
                                                   'successor': []}
                for m_vertex in mother_vertex:
                    trajectories[m_vertex.track_id]['successor'].append(end_vertex)

            elif (start_vertex.type == 'merge') and (end_vertex.type == 'normal'):
                mother_vertices = [in_edge.start_vertex
                                   for in_edge in start_vertex.in_edges.values()
                                   if self.result[in_edge.string_id()] > 0
                                   ]
                end_vertex.track_id = len(trajectories)
                merging_vertices = []
                for vertex in mother_vertices:
                    if vertex.type == 'skip':
                        v_id = vertex.id.string_id().split('skip_')[-1]
                        merging_vertices.append(self.get_vertex(v_id))
                    else:
                        merging_vertices.append(vertex)
                trajectories[len(trajectories)] = {'predecessor': merging_vertices,
                                                   'track': [end_vertex],
                                                   'successor': []}
                for m_vertex in mother_vertices:
                    trajectories[m_vertex.track_id]['successor'].append(end_vertex)

            elif (start_vertex.type in ['normal', 'skip']) and (end_vertex.type in ['normal', 'skip']):
                if start_vertex.type == 'skip':
                    v_id = start_vertex.id.string_id().split('skip_')[-1]
                    start_vertex = self.get_vertex(v_id)
                end_vertex.track_id = start_vertex.track_id
                if end_vertex.type == 'normal':
                    trajectories[start_vertex.track_id]['track'].append(end_vertex)

        return trajectories


def get_id_representation(trajectories):
    """Extracts for each track its string id. """
    string_trajectories = {}
    for key, track in trajectories.items():
        string_trajectories[key] = {}
        for sub_key, vertices in track.items():
            string_ids = [vertex.id.string_id() for vertex in vertices]
            string_trajectories[key][sub_key] = string_ids
            if sub_key == 'predecessor':
                string_trajectories[key]['pred_track_id'] = sorted([vertex.track_id
                                                                    for vertex in vertices
                                                                    if hasattr(vertex, 'track_id')])
            if sub_key == 'successor':
                string_trajectories[key]['succ_track_id'] = sorted([vertex.track_id
                                                                    for vertex in vertices
                                                                    if hasattr(vertex, 'track_id')])
    return string_trajectories


def compute_distance(vec_a, vec_b):
    # note : for large vector components
    # overflow possible alternative for |x| > |y|: |x|*sqrt(1 + (y/x)**2)
    return np.linalg.norm(vec_a-vec_b, axis=-1)


def graph_tracking(all_tracklets, all_maching_candidates, img_shape,
                   cutoff_distance=float('inf'), allow_cell_division=True):
    """Computes for a set of tracks with a set of potential matching candidates the best matching
        based on coupled min cost flow problem."""
    if all_maching_candidates:
        graph = SparseGraph(cutoff_distance, img_shape, allow_cell_division=allow_cell_division)
        for tracklet_id, tracklet_features in all_tracklets.items():
            init_features = tracklet_features[0]
            estimated_features = tracklet_features[1]
            matching_candidates = all_maching_candidates[tracklet_id]
            time, t_id = tracklet_id
            vertex_id = VertexId(time, t_id).string_id()
            last_skip_vertex = None
            curr_skip_vertex = None

            t_next_steps = sorted(list(estimated_features.keys()))
            if vertex_id not in graph.vertices:
                if t_next_steps:
                    vertex = Vertex(time, t_id, features=estimated_features[t_next_steps[0]])
                else:
                    vertex = Vertex(time, t_id, features=init_features)
                graph.add_vertex(vertex)
            else:
                vertex = graph.get_vertex(vertex_id)
            for i, t_next in enumerate(t_next_steps):
                if i > 0:  # from t+n -> t+n+1 . skip connection at t+n, neighbors next step at t+n+1
                    skip_vertex_id = VertexId(t_next_steps[i - 1], 'skip_' + vertex_id).string_id()
                    if skip_vertex_id not in graph.vertices:
                        curr_skip_vertex = SkipVertex(t_next_steps[i - 1], 'skip_' + vertex_id,
                                                      features=estimated_features[t_next])
                        graph.add_vertex(curr_skip_vertex)
                    else:
                        curr_skip_vertex = graph.get_vertex(skip_vertex_id)
                matching_cand_keys = filter(lambda x, t=t_next: x[0] == t, matching_candidates)
                if i == 1:
                    vertex.next_vertices.add(skip_vertex_id)
                elif i > 1:
                    last_skip_vertex.next_vertices.add(skip_vertex_id)

                for candidate_key in matching_cand_keys:
                    candidate_time, candidate_id = candidate_key

                    n_vertex_id = VertexId(candidate_time, candidate_id).string_id()
                    if n_vertex_id not in graph.vertices:
                        n_vertex = Vertex(candidate_time, candidate_id, features=all_tracklets[candidate_key][0])
                        graph.add_vertex(n_vertex)
                    if i == 0:
                        vertex.next_vertices.add(n_vertex_id)
                    else:
                        curr_skip_vertex.next_vertices.add(n_vertex_id)

                last_skip_vertex = curr_skip_vertex
        graph.construct_graph()
        graph.gurobi_optimize()
        track_data = graph.calc_trajectories()
        track_data = get_id_representation(track_data)
        return track_data


def seeds_in_bounding_box(bounding_box, seed_points):
    """Computes overlap between a bounding box and seed points."""
    seed_points = np.array(seed_points)
    if len(seed_points.shape) == 1:
        seed_points = seed_points.reshape(-1, 1)
    upper_left = np.array(bounding_box[0])
    lower_right = np.array(bounding_box[1])
    dist_upper = seed_points - upper_left.reshape(-1, 1)
    dist_lower = lower_right.reshape(-1, 1) - seed_points
    below_upper_left = np.all(dist_upper > 0, axis=0)
    above_lower_right = np.all(dist_lower > 0, axis=0)
    in_box = below_upper_left & above_lower_right
    if np.any(in_box):
        return True
    return False


def print_constraints(A, b, variable_names, eq_sign='<='):
    """Prints constraints in readable format."""
    # constraints have the form Ax <= b or Ax == b
    for a, bb in zip(A, b):
        eq = ' + '.join([str(f) + '*' + variable_names[el] for el, f in a.items()])
        print(eq + eq_sign + str(bb))
