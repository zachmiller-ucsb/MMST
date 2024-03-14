import numpy as np
from scipy.optimize import LinearConstraint
from scipy.optimize import milp
from scipy.optimize import OptimizeResult

def add(p, v):
    return (p_i + v_i for p_i,v_i in zip(p,v))
    
def weights(p1, p2):    
    return sum([(p1i - p2i)**2 for p1i,p2i in zip(p1,p2)])**0.5

def weightsV(p1, v1, p2, v2):
    return [weights(p1,p2), weights(add(p1,v1), add(p2,v2))]

# Finds a minimal moving matching.

# Returns a tuple of type (scipy.optimize.OptimizeResult, {set of
# edges}), where an edge is a tuple of two points at time zero.  Note
# that this does not distinguish the case where two distinct moving
# points have the same starting position.

# The parameter V is a map of points to velocities.  They can be of
# any identical dimension, but V must have an even number of points so
# that a matching exists.

def MMM(V, relax=False) -> (OptimizeResult, set):

    eps = np.finfo(float).eps
    
    P = list(V)
    assert len(P) > 0, "The set of points must be non-empty"
    assert len(P) % 2 == 0, "The set of points must be of even size"
    dim = len(P[0])
    for p in P:
        assert len(p) == len(V[p]), "Velocity dimension must match point dimension"
        assert len(p) == dim, "Dimension of all points must match"
        
    edges = []
    for i in range(len(P)):
        for j in range(i+1,len(P)):
            edges += [(P[i], P[j])]

    constraints = []
    for i,v in enumerate(V):
        constraints += [LinearConstraint(np.array(
            [1 if v in [u,w] else 0 for [u,w] in edges] + [0,0]), 1,1)]

    # w0 - (sum of weights at 0) >= 0
    # w1 - (sum of weights at 1) >= 0
    # w1 >= w0
    # minimize w1
    constraints += [LinearConstraint(np.array(
        [-weightsV(p1, V[p1], p2, V[p2])[0] for (p1, p2) in edges] + [1,0]), 0)]
    constraints += [LinearConstraint(np.array(
        [-weightsV(p1, V[p1], p2, V[p2])[1] for (p1, p2) in edges] + [0,1]), 0)]
    constraints += [LinearConstraint(np.array([0] * len(edges) + [-1,1]), 0)]
    obj_fn = np.array([0]*(len(edges)+1) + [1])
    res = milp(obj_fn,
               integrality=np.array(([0] if relax else [1])*len(edges) + [0,0]),
               constraints=constraints)
    if res.x is None:
        return [res,None]
    mm_edges = []
    enum_edges = [(-w,i) for (i,w) in enumerate(res.x[:-2])]
    enum_edges.sort()
    vertices = []
    for (w,i) in enum_edges:
        u,v = edges[i]
        if u not in vertices and v not in vertices:
            mm_edges += [edges[i]]
            vertices += [u, v]
    return [res,mm_edges]

S = {
    # point : velocity
    (0,0):(0,0),
    (1,1):(0,0),
    (-1,-1):(0,0),
    (-2,-2):(0,0),
    (1,-1):(-5,0),
    (2,-2):(-3,3),
}
