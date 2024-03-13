import sys 
import numpy as np
import numpy.linalg as npl
import heapq

class UnionFind:
    def __init__(self, G):
        self.parent = {p : p for p in G.keys()}

    def Union(self, a, b):
        parenta, parentb = self.Find(a), self.Find(b)
        if parenta == parentb:
            return False 
        self.parent[parentb] = parenta
        return True
    
    def Find(self, a):
        if self.parent[a] == a:
            return a 
        self.parent[a] = self.Find(self.parent[a])
        return self.parent[a]

def parse() -> set[tuple[int, int, int, int]]:
    n = int(sys.stdin.readline())
    V = set()
    for _ in range(n):
        point_vel = [int(i) for i in sys.stdin.readline().split()]
        assert len(point_vel) == 4
        V.add(tuple(point_vel))
    return V

# According to corollary 2 in bunch_of_people et. al., we can take 
# max_dist = max(dist_at_0, dist_at_1) (we normalize time interval to [0,1])
def max_dist(pv1, pv2):
    p1, p2 = pv1[:2], pv2[:2]
    v1, v2 = pv1[2:], pv2[2:]
    # pos at 0 and at 1 for p1
    pos_p1 = [
        np.array([p1[0], p1[1]]),
        np.array([p1[0] + v1[0], p1[1] + v1[1]])
    ]
    # pos at 0 and at 1 for p2
    pos_p2 = [
        np.array([p2[0], p2[1]]),
        np.array([p2[0] + v2[0], p2[1] + v2[1]])
    ]
    # Can replace npl.norm with whatever else we want to use
    return max([npl.norm(a - b) for a in pos_p1 for b in pos_p2])

def find_MBMST(V):
    G = { p : {} for p in V }
    for p1 in V:
        for p2 in V:
            if p1 != p2:
                w = max_dist(p1, p2)
                G[p1][p2] = w
    return mst(G)

def mst(G):
    heap = []
    for p1 in G.keys():
        for p2 in G[p1]:
            heap.append((G[p1][p2], p1, p2))
    heapq.heapify(heap)
    MBMST = set()
    union_find = UnionFind(G)
    while len(MBMST) < len(G.keys()) - 1:
        _, p1, p2 = heapq.heappop(heap)
        if (p2, p1) in MBMST:
            continue 
        if union_find.Union(p1, p2):
            MBMST.add((p1, p2))
    return MBMST

def main():
    V = parse()
    MBMST = find_MBMST(V)
    # print(MBMST)
    print("MBMST sum = ", sum([max_dist(*e) for e in MBMST]))


if __name__ == '__main__':
    main()
