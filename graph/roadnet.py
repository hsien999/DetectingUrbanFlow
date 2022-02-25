from typing import Dict, List, Tuple, Set, Optional
import numpy as np
from sklearn.neighbors import KDTree
from graph.linear import euclidean_distance, bernoulli_lambda, add, sub
import random


# base node
class _Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(self.x) ^ hash(self.y)

    def __repr__(self):
        return "Node({},{})".format(self.x, self.y)

    def __getitem__(self, item):
        return self.x if item == 0 else (self.y if item == 1 else None)


# base edge
class _Edge:
    def __init__(self, node1, node2):
        self.node1: _Node = node1
        self.node2: _Node = node2

    def __getitem__(self, item):
        return self.node1 if item == 0 else (self.node2 if item == 1 else None)


# maintaining consistency of basic data types in data structures
float_ = np.float32
int_ = np.int32
bool_ = np.bool8


class RoadNetWork:
    """
    Data Structure for road network
    """

    def __init__(self) -> None:
        self.edges: Dict[int_, ((float_, float_), (float_, float_))] = {}  # road id -> (node1, node2)
        self.adjacency: Dict[(float_, float_), List[int_]] = {}  # node -> road id
        self.matches: Dict[int_, List[(float_, float_)]] = {}  # road id -> od points
        self.od_flags: Dict[int_, List[bool_]] = {}  # road id -> od flag (origin 1, dest 0)
        self.road_mark: Dict[(float_, float_), int_] = {}  # od point -> road id
        self.kd_trees = {}  # kd trees base on matches
        self.od_count, self.o_count, self.d_count = 0, 0, 0  # od points count

    def __add_node(self, x, y) -> (float_, float_):
        """
        Create nodes and generate new neighbouring edges for new nodes
        """
        new_n = (x, y)
        if new_n not in self.adjacency:
            self.adjacency[new_n] = []
        return new_n

    def build_kd_trees(self) -> None:
        """
        Build knn tree for each road
        """
        for road_id, points in self.matches.items():
            if len(points) > 0:
                self.kd_trees[road_id] = KDTree(points)
        return

    def clean_matches(self) -> None:
        """
        Clear repeated matches on the same road
        """
        # the same results can be guaranteed for the same points
        for road_id, matches in self.matches.items():
            self.matches[road_id] = list(set(matches))
        return

    def add_edge(self, road_id, x1, y1, x2, y2) -> None:
        """
        Add a new edge on the network
        """
        road_id = int_(road_id)
        node1 = self.__add_node(x1, y1)
        node2 = self.__add_node(x2, y2)
        self.edges[road_id] = (node1, node2)
        self.adjacency[node1].append(road_id)
        self.adjacency[node2].append(road_id)

    def add_matches(self, road_id, x, y, o_d) -> None:
        """
        Add a new match on the network
        """
        road_id = int_(road_id)
        o_d = bool_(o_d)
        new_n = (x, y)
        if road_id not in self.matches:
            self.matches[road_id] = []
        if road_id not in self.od_flags:
            self.od_flags[road_id] = []
        self.matches[road_id].append(new_n)
        self.od_flags[road_id].append(o_d)
        self.road_mark[new_n] = road_id
        self.od_count += 1
        if o_d:
            self.o_count += 1
        else:
            self.d_count += 1

    def network_constrained_neighbors(self, epsilon, pi) -> Tuple[Set, int, int]:
        """
        Construct a network constrained neighborhood based on the edge-expansion method
        by specifying a point pi and the radius of the neighborhood ε

        Parameters
        ----------
        epsilon : float
            the neighbourhood cutoff radius
        pi : array-like
            a point specified to search for

        Returns
        ----------
        neighbourhood : set
            a neighbourhood set for the given target point
        ori_cnt : int
            number of origin points inside region
        des_cnt : int
            number of destination points inside region
        """
        assert len(self.kd_trees) > 0, 'kd tree should be built firstly'
        assert pi in self.road_mark and pi is not None, 'pi should be on the network'
        road_id = self.road_mark[pi]
        edge = self.edges[road_id]
        st_node, ed_node = edge[0], edge[1]
        point = pi
        dist_st, dist_ed = euclidean_distance(point, st_node), euclidean_distance(point, ed_node)
        # results to return
        neighborhood = set()
        ori_cnt, des_cnt = 0, 0
        # search the matches on the matched road of pi firstly
        matches, flags, kd_tree = self.matches[road_id], self.od_flags[road_id], self.kd_trees[road_id]
        idxes = kd_tree.query_radius([point], epsilon)
        idxes = idxes[0] if len(idxes) > 0 else []
        for idx in idxes:
            if matches[idx] != point and matches[idx] not in neighborhood:
                neighborhood.add(matches[idx])
                if flags[idx]:
                    ori_cnt += 1
                else:
                    des_cnt += 1
        # then, expand the edges base on edge-expansion (using bfs)
        explored = {road_id}
        queue = [(st_node, epsilon - dist_st), (ed_node, epsilon - dist_ed)]
        while queue:
            n, d = queue.pop(0)
            if d <= 0:
                continue
            for r_id in self.adjacency[n]:
                if r_id in explored:
                    continue
                ed = self.edges[r_id]
                n2 = ed[0] if n != ed[0] else ed[1]
                dis = d - euclidean_distance(n, n2)
                if r_id in self.matches:
                    matches, flags = self.matches[r_id], self.od_flags[r_id]
                else:
                    matches, flags = [], []
                if dis >= 0:
                    # add all matches on the searching road if dis >= 0
                    for idx, mat in enumerate(matches):
                        if mat not in neighborhood:
                            neighborhood.add(mat)
                            if flags[idx]:
                                ori_cnt += 1
                            else:
                                des_cnt += 1
                    explored.add(r_id)  # possible circuits on road: to be optimised
                    queue.append((n2, dis))  # append to searching queue
                elif r_id in self.kd_trees:
                    # query kd tree of the current road if matched point existed
                    idxes = self.kd_trees[r_id].query_radius([n], d)
                    idxes = idxes[0] if len(idxes) > 0 else []
                    for idx in idxes:
                        if matches[idx] not in neighborhood:
                            neighborhood.add(matches[idx])
                            if flags[idx]:
                                ori_cnt += 1
                            else:
                                des_cnt += 1
                # if dis < 0 and none matched point existed, continue
        return neighborhood, ori_cnt, des_cnt

    def calc_test_statistics(self, epsilon, pi) -> Optional[Tuple[float, int, int, Set]]:
        """
        Calculate the test statistics based for the given pi and epsilon on the road network
        See docs in network_constrained_neighbors()

        Parameters
        ----------
        epsilon : int
            the neighbourhood cutoff radius
        pi : array-like
            a point specified to search for

        Returns
        ----------
        None : If unreachable
        bernoulli_lambda : float
            the lambada value
        n_o_r : int
            the number of origin points in neighbourhood region of pi
        n_d_r : int
            the number of destination points in neighbourhood region of pi
        neighbourhood : set
            a neighbourhood set for the given point pi
        """
        neighbour, n_o_r, n_d_r = self.network_constrained_neighbors(epsilon, pi)
        n_o, n_d = self.o_count, self.d_count
        if add(n_o_r, n_d_r) == 0 or add(n_o, n_d) == 0 or add(n_o, sub(n_d, add(n_o_r, n_d_r))) == 0:
            return None
        return bernoulli_lambda(n_o_r, n_d_r, n_o, n_d), n_o_r, n_d_r, neighbour


def generate_random_network(net: RoadNetWork, seed=2021) -> RoadNetWork:
    """
    Generate the same numbers of OD points as the observed dataset randomly
    on the road network following complete spatial randomness

    Parameters
    ----------
    net : RoadNetWork
        the observed road network
    seed : int or str, optional
        random seed, default = 2021

    Returns
    ----------
    generated random road network
    """
    if seed is not None:
        random.seed(seed)
    random_net = RoadNetWork()
    random_net.edges = net.edges
    random_net.adjacency = net.adjacency
    edges = net.edges
    o_cnt = net.o_count
    for i in range(net.od_count):
        road_id = random.sample(net.matches.keys(), 1)[0]
        edge = edges[road_id]
        p1, p2 = edge[0], edge[1]
        p_x = p1[0] + random.random() * (p2[0] - p1[0])
        diff_y, diff_x = p2[1] - p1[1], p2[0] - p1[0]
        not_ver = diff_x != 0
        a = diff_y / diff_x if not_ver else np.inf
        b = p1[1] - a * p1[0] if not_ver else (p1[1] + p2[1]) / 2
        p_y = (a * p_x if not_ver else 0) + b
        random_net.add_matches(road_id, p_x, p_y, True if i < o_cnt else False)
    random_net.build_kd_trees()
    return random_net


def identify_subareas(net: RoadNetWork, ran_net: RoadNetWork, test_pi, r_time, alpha, epsilon) \
        -> Optional[Tuple[bool, float, Set]]:
    """
    Identification of subareas of urban black holes and volcanoes

    Parameters
    ----------
    net : RoadNetWork
        the observed road network
    ran_net : RoadNetWork
        the random road network
    test_pi : tuple
        a OD point on 'net'
    r_time : int
        the number of repetitions of Monte Carlo simulation
    alpha : float
        a significance level
    epsilon : int or float
        the neighbourhood cutoff radius

    Returns
    ----------
    None : If unreachable
    flag : bool
        flag indicated of a volcano or black hole
    lambda_obs : float
        lambda value of observed road network
    neighbour : set
        a neighbourhood set for the given point test_pi
    """
    res_obs = net.calc_test_statistics(epsilon, pi=test_pi)
    if res_obs is None:
        return None
    lambda_obs, o_cnt, d_cnt, neighbour = res_obs
    p_value = 0
    for __ in range(r_time):
        road = random.sample(ran_net.matches.keys(), 1)[0]
        ran_pi = random.sample(ran_net.matches[road], 1)[0]
        res = ran_net.calc_test_statistics(epsilon, pi=ran_pi)
        if res is None:
            continue
        lambda_j, _, _, _ = res
        if lambda_j > lambda_obs:
            p_value += 1
    p_value /= (1.0 + r_time)
    if p_value > alpha or o_cnt == d_cnt:
        return None
    return d_cnt > o_cnt, lambda_obs, neighbour
