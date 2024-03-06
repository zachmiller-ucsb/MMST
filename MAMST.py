from MBMST import parse, mst
from scipy.integrate import quad
from math import sqrt

# Computes integral over t=[0,1] of ||p(t)-d(t)||
def avg_dist(p1, v1, p2, v2):
    vxd, vyd = v1[0] - v2[0], v1[1] - v2[1]
    px1, py1 = p1
    px2, py2 = p2
    return quad(lambda t: sqrt((px1 - px2 + (vxd * t))**2 + (py1 - py2 + (vyd * t))**2), 0, 1)

def find_MAMST(V) -> set:
    G = { p : {} for p in V.keys()}
    for p1 in V.keys():
        for p2 in V.keys():
            if p1 != p2:
                w = avg_dist(p1, V[p1], p2, V[p2])
                G[p1][p2] = w
    return mst(G)

if __name__ == "__main__":
    V = parse()
    MAMST = find_MAMST(V)
    print(MAMST)
