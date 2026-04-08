"""
graph/fall_graph.py
====================
Graph 17 keypoint COCO untuk fall detection.
Letakkan di: BlockGCN/graph/fall_graph.py

Interface identik dengan graph asli BlockGCN:
    g = Graph(labeling_mode='spatial')
    A = g.A   # shape (3, 17, 17)
"""

import numpy as np


class Graph:
    NUM_NODES    = 17
    CENTER_JOINT = 11   # left_hip — proxy torso center

    COCO_PAIRS = [
        (0,1),(0,2),(1,3),(2,4),
        (0,5),(0,6),(5,6),
        (5,7),(7,9),(6,8),(8,10),
        (5,11),(6,12),(11,12),
        (11,13),(13,15),(12,14),(14,16),
    ]

    def __init__(self, labeling_mode: str = "spatial"):
        self.num_node      = self.NUM_NODES
        self.self_link     = [(i, i) for i in range(self.NUM_NODES)]
        self.neighbor_link = self.COCO_PAIRS
        self.A = self._build()

    def _hop(self):
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(self.NUM_NODES))
        G.add_edges_from(self.neighbor_link)
        return {i: dict(nx.single_source_shortest_path_length(G, i))
                for i in range(self.NUM_NODES)}

    def _build(self):
        hop  = self._hop()
        dist = hop[self.CENTER_JOINT]
        A    = np.zeros((3, self.NUM_NODES, self.NUM_NODES), np.float32)
        for i in range(self.NUM_NODES):
            for j, d in hop[i].items():
                if i == j:
                    A[0, i, j] = 1
                elif d == 1:
                    if   dist[j] < dist[i]: A[1, i, j] = 1
                    elif dist[j] > dist[i]: A[2, i, j] = 1
                    else:                   A[1, i, j] = 1
        for k in range(3):
            rs = A[k].sum(axis=1, keepdims=True)
            rs[rs == 0] = 1
            A[k] /= rs
        return A

    def get_adjacency_matrix(self):
        return self.A


if __name__ == "__main__":
    g = Graph()
    print(f"A shape: {g.A.shape}")
    print(f"Non-zero per partition: {[(g.A[k]>0).sum() for k in range(3)]}")