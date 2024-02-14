import sys 
import numpy as np
import numpy.linalg as npl
import heapq

class UnionFind:
    def __init__(self, V):
        self.parent = {p : p for p in V.keys()}

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

def parse():
    n = int(sys.stdin.readline())
    V = {}
    for _ in range(n):
        x, y, vx, vy = [int(i) for i in sys.stdin.readline().split()]
        V[(x,y)] = (vx,vy)
    return V

# According to corollary 2 in bunch_of_people et. al., we can take 
# max_dist = max(dist_at_0, dist_at_1) (we normalize time interval to [0,1])
def max_dist(p1, v1, p2, v2):
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

def main():
    V = parse()
    G = { p : {} for p in V.keys()}
    heap = []
    for p1 in V.keys():
        for p2 in V.keys():
            if p1 != p2:
                w = max_dist(p1, V[p1], p2, V[p2])
                G[p1][p2] = w
                heap.append((w, p1, p2))

    heapq.heapify(heap)
    MBMST = set()
    union_find = UnionFind(V)
    while len(MBMST) < len(V.keys()) - 1:
        _, p1, p2 = heapq.heappop(heap)
        if (p2, p1) in MBMST:
            continue 
        if union_find.Union(p1, p2):
            MBMST.add((p1, p2))
    print(MBMST)


if __name__ == '__main__':
    main()