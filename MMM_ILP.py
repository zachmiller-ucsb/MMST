import numpy as np
from scipy.optimize import LinearConstraint
from scipy.optimize import milp

def weights(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

def weightsV(p1, v1, p2, v2):
    return [weights(p1,p2), weights((p1[0]+v1[0],p1[1]+v1[1]), (p2[0]+v2[0], p2[1] + v2[1]))]

def MMM(V) -> set:

    eps = np.finfo(float).eps
    
    P = list(V)
    edges = []
    for i in range(len(P)):
        for j in range(i+1,len(P)):
            edges += [(P[i], P[j])]

    constraints = []
    for i,v in enumerate(V):
        constraints += [LinearConstraint(np.array(
            [1 if v in [u,w] else 0 for [u,w] in edges] + [0,0]), 1,1)]

    # w0 - weights0 >= 0
    # w1 - weights1 >= 0
    # w1 >= w0
    constraints += [LinearConstraint(np.array(
        [-weightsV(p1, V[p1], p2, V[p2])[0] for (p1, p2) in edges] + [1,0]), 0)]
    constraints += [LinearConstraint(np.array(
        [-weightsV(p1, V[p1], p2, V[p2])[1] for (p1, p2) in edges] + [0,1]), 0)]
    constraints += [LinearConstraint(np.array([0] * len(edges) + [-1,1]), 0)]
    obj_fn = np.array([0]*(len(edges)+1) + [1])
    print(obj_fn)
    res = milp(obj_fn,
               integrality=np.array([1]*len(edges) + [0,0]),
               constraints=constraints)
    if res.x is None:
        return [res,None]
    mm_edges = []
    for i,flag in enumerate(res.x[:-2]):
        if abs(flag-1) < eps:
            mm_edges += [edges[i]]
    return [res,mm_edges]

# Test
# print(MMM({
#    # point : velocity
#    (0,0):(0,0),
#    (1,1):(0,0),
#    (-1,-1):(0,0),
#    (-2,-2):(0,0),
# }))
