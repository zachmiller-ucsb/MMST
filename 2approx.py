from MBMST import parse 
from nettrees.snt import SNT
from nettrees.point import Point
from nettrees.metric import Moving 
from nettrees.sntpointlocation import SNTPointLocation
from random import shuffle
from MBMST import mst, max_dist 

tau = 5.
cp = 1.
cc = 1.

# For our MST approximation
epsilon = 0.5

# For t - spanner generation
c = 16
delta = epsilon / c

# computes a delta^-1 WSPD
def genWSPD(u, v):
    if u.level < v.level:
        u, v = v, u 
    elif u.level == v.level:
        if tuple(u.point.coords) > tuple(v.point.coords):
            u, v = v, u 
    if 16 * tau**(u.level + 1) / (tau - 1) <= (1 / delta) * Moving().dist(u.point, v.point):
        return set([(u,v)]) 
    ret = set()
    for child in u.ch:
        ret = ret.union(genWSPD(child, v))
    return ret

def genTSpanner(V):
    points = [Point([p[0], p[1], p[0] + v[0], p[0] + v[1]], Moving()) for p, v in V.items()]

    # For some nondeterminism
    shuffle(points)

    T = SNT(tau, cp, cc)
    T.construct(points, SNTPointLocation)
    WSPD = genWSPD(T.root, T.root)

    # G will be our t-spanner
    G = {p : {} for p in V.keys()}
    for u, v in WSPD:
        x = (u.point.coords[0], u.point.coords[1])
        y = (v.point.coords[0], v.point.coords[1])
        G[x][y] = Moving().dist(u.point, v.point)
    return G

def main():
    V = parse()
    G = genTSpanner(V)

    MST = mst(G)
    # print(MST)
    print("(2+epsilon)-approx MST sum = ", sum([max_dist(e[0], V[e[0]], e[1], V[e[1]]) for e in MST]))

if __name__ == '__main__':
    main()
