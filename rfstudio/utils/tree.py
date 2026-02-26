from __future__ import annotations

from typing import Generic, List, Tuple, TypeVar

T = TypeVar('T')

class tree(Generic[T]):

    def __init__(self) -> None:
        self._edges: List[Tuple[int, int, T]] = []
        self._nodes = None
        self._links = None
        self._root = None

    def define(self: tree[T], node_from: int, node_to: int, edge_value: T) -> tree[T]:
        if self is tree:
            result = tree()
            result._edges.append((node_from, node_to, edge_value))
            return result
        self._edges.append((node_from, node_to, edge_value))
        return self

    def _validate(self) -> None:
        if self._root is not None:
            return
        self._nodes = {}
        for node_from, node_to, _ in self._edges:
            self._nodes.setdefault(node_from, len(self._nodes))
            self._nodes.setdefault(node_to, len(self._nodes))
        self._links = [[] for _ in self._nodes]
        out_degree = [0 for _ in self._nodes]
        for eid, (node_from, node_to, _) in enumerate(self._edges):
            self._links[self._nodes[node_from]].append((self._nodes[node_to], eid))
            out_degree[self._nodes[node_to]] += 1
        root = [idx for idx, deg in enumerate(out_degree) if deg == 0]
        assert len(root) == 1
        self._root = root[0]

    @property
    def num_nodes(self) -> int:
        self._validate()
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        self._validate()
        return len(self._edges)

    def edge(self, edge_index: int) -> T:
        self._validate()
        return self._edges[edge_index][2]

    @property
    def root(self) -> int:
        self._validate()
        return self._root

    def children(self, node_idx: int) -> List[Tuple[int, int]]:
        return self._links[node_idx]

    # def dfs(self) -> Iterator[Tuple[int, T, int]]:
    #     self._validate()
    #     stack = [self._root]
    #     while stack:
    #         curr = stack.pop(-1)
    #         for next, eid in self._links[curr]:
    #             stack.append(next)
    #             yield curr, self._edges[eid][2], next
