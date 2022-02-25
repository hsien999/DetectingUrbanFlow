from typing import Generic, TypeVar, Dict, Set
import heapq

T = TypeVar("T")
W = TypeVar("W")


class GraphUndirectedWeighted(Generic[T, W]):
    """
    Data structure to store an undirected weighted graph (based on adjacency lists)
    """

    def __init__(self) -> None:
        # adjacency: map from the vertex to the neighbouring vertexes (with weights)
        self.adjacency: Dict[T, Dict[T, W]] = {}

    def add_vertex(self, vertex: T) -> None:
        """
        add a vertex ONLY if its not present in the graph
        """
        if vertex not in self.adjacency:
            self.adjacency[vertex] = {}

    def add_edge(self, vertex1: T, vertex2: T, weight: W) -> None:
        """
        add an edge with the given weight
        """
        if vertex1 == vertex2:
            return
        self.add_vertex(vertex1)
        self.add_vertex(vertex2)
        self.adjacency[vertex1][vertex2] = weight
        self.adjacency[vertex2][vertex1] = weight

    def get_edges(self):
        """
        Returns all edges in the graph (generator)
        """
        for v1 in self.adjacency:
            for v2 in self.adjacency[v1]:
                yield v1, v2, self.adjacency[v1][v2]

    def get_vertices(self):
        """
        Returns all vertices in the graph (tuple)
        """
        return self.adjacency.keys()

    def __str__(self):
        """
        Returns string representation of the graph
        """
        string = ""
        for v1, v2, w in self.get_edges():
            string += "%s -> %s == %s\n" % (v1, v2, w)
        return string.rstrip("\n")

    def __getitem__(self, item):
        """
        Returns adjacency of given node
        """
        return self.adjacency[item]

    @staticmethod
    def build(vertices=None, edges=None):
        """
        Builds a graph from the given set of vertices and edges
        """
        g = GraphUndirectedWeighted()
        if vertices is None:
            vertices = []
        if edges is None:
            edges = []
        for vertex in vertices:
            g.add_vertex(vertex)
        for edge in edges:
            g.add_edge(*edge)
        return g


class DisjointSetTreeNode(Generic[T]):
    """
    Disjoint Set Node to store the parent and rank
    """

    def __init__(self, data: T) -> None:
        self.data = data
        self.parent = self
        self.rank = 0


class DisjointSetTree(Generic[T]):
    """
    Disjoint Set DataStructure
    """

    def __init__(self) -> None:
        """
        map from node name to the node object
        """
        self.map: Dict[T, DisjointSetTreeNode[T]] = {}

    def make_set(self, data: T) -> None:
        """
        create a new set with x as its member
        """
        self.map[data] = DisjointSetTreeNode(data)

    def find_set(self, data: T) -> DisjointSetTreeNode[T]:
        """
        find the set x belongs to (with path-compression)
        """
        elem_ref = self.map[data]
        if elem_ref != elem_ref.parent:
            elem_ref.parent = self.find_set(elem_ref.parent.data)
        return elem_ref.parent

    def link(self, node1: DisjointSetTreeNode[T], node2: DisjointSetTreeNode[T]) -> None:
        """
        helper function for union operation
        """
        if node1.rank > node2.rank:
            node2.parent = node1
        else:
            node1.parent = node2
            if node1.rank == node2.rank:
                node2.rank += 1

    def union(self, data1: T, data2: T) -> None:
        """
        merge 2 disjoint sets
        """
        self.link(self.find_set(data1), self.find_set(data2))

    def group(self) -> Dict[T, list]:
        """
        returns a dict of different groups in disjoint set
        """
        mem_group = {}
        for member in self.map:
            self.find_set(member)
        for member in self.map:
            parent = self.find_set(member).data
            if parent not in mem_group:
                mem_group[parent] = []
            mem_group[parent].append(member)
        return mem_group


class MaxHeap(object):
    """
    Max heap based on native heap implementation
    """

    def __init__(self, x=None):
        if x is None:
            x = ()
        self.heap = [-e for e in x]
        heapq.heapify(self.heap)

    def push(self, value):
        heapq.heappush(self.heap, -value)

    def pop(self):
        return -heapq.heappop(self.heap)


def breadth_first_search(graph: GraphUndirectedWeighted, start: T) -> Set[T]:
    """
    bfs test
    """
    explored, queue = {start}, [start]
    while queue:
        v = queue.pop(0)
        for w in graph[v]:
            if w not in explored:
                explored.add(w)
                queue.append(w)
    return explored


def depth_first_search(graph: GraphUndirectedWeighted, start: T) -> Set[T]:
    """
    dfs test
    """
    explored, stack = {start}, [start]
    while stack:
        v = stack.pop()
        explored.add(v)
        for adj in graph[v]:
            if adj not in explored:
                stack.append(adj)
    return explored
