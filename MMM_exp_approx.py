import numpy as np

def add(p, v):
    return (p_i + v_i for p_i,v_i in zip(p,v))

def sub(p, v):
    return (p_i - v_i for p_i,v_i in zip(p,v))

def dist(p1, p2):
    return sum((p1i-p2i)**2 for p1i,p2i in zip(p1,p2))**0.5

# Insight: the average distance is invariant under frame of reference.
# Therefore, make one point be the origin with zero velocity
def average_distance(p1, v1, p2, v2):
    p = sub(p2,p1)
    v = sub(v2,v1)

    # f: t -> p + (t * v), 0 <= t <= 1
    # Integrate dist(f(t),0) over 0 <= t <= 1
    # \int_0^1 |f(t)| dt = \int_0^1
    # \sqrt{(p_1 + v_1 t)^2 + (p_2 + v_2 t)^2 + \cdots + (p_n + v_n t)^2}
    # Gather into a quadratic: \sqrt{at^2 + bt + c}
    # a = \sum_{i=1}^n v_i^2
    # b = 2\sum_{i=1}^n v_ip_i
    # c = \sum_{i=1}^n p_i^2
    a = sum(v_i*v_i for v_i in v)
    b = 2*sum(v_i*p_i for p_i,v_i in zip(p,v))
    c = sum(p_i*p_i for p_i in p)

    if a == 0:
        return dist(p1,p2)

    # Assume a != 0.
    # ax^2 + bx + c = a((x + b/2a)^2 + (c/a) - (b/2a)^2)

    # \sqrt{a((x + b/2a)^2 + (c/a) - (b/2a)^2)}
    # = \sqrt{a}\cdot\sqrt{(x + b/2a)^2 + (c/a) - (b/2a)^2)
    # Let p = b/2a, q = (c/a) - (b/2a)^2
    # \int_0^1 \sqrt{ax^2+bx+c} dx
    # = \sqrt{a}\int_0^1 \sqrt{(x+p)^2 + q}dx
    p = b/(2*a)
    q = (c/a) - p*p

    # Note that the last term is non-negative because q = c/a -
    # b^2/4a^2 >= 0 iff 4ac - b^2 >= 0 iff b^2 - 4ac <= 0 iff there is
    # one or zero real roots, which is our situation.

    # Goal: reduce to integrating \sqrt{x^2+1}
    # u = \frac{1}{\sqrt{q}}(x+p)
    # \sqrt{q}du = dx

    # = \sqrt{a}\int_{u(0)}^{u(1)}\sqrt{qu^2+q} \sqrt{q} du
    # = q\sqrt{a}\int_{u(0)}^{u(1)}\sqrt{u^2+1} du
    # = q\sqrt{a}[(1/2)u\sqrt{u^2+1}+\sinh^{-1}(u)]_{u(0)}^{u(1)}

    u = lambda t: (1/np.sqrt(q)) * (t + p)
    I = lambda t: q*np.sqrt(a) * (1/2) * u(t) * np.sqrt(u(t)*u(t) + 1) + np.arcsinh(u(t))
    return I(1) - I(0)

# Finds a minimal moving matching.

# Returns a tuple of type (scipy.optimize.OptimizeResult, {set of
# edges}), where an edge is a tuple of two points at time zero.  Note
# that this does not distinguish the case where two distinct moving
# points have the same starting position.

# The parameter V is a map of points to velocities.  They can be of
# any identical dimension, but V must have an even number of points so
# that a matching exists.

def MMM(V) -> (OptimizeResult, set):

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
               integrality=np.array([1]*len(edges) + [0,0]),
               constraints=constraints)
    if res.x is None:
        return [res,None]
    mm_edges = []
    for i,flag in enumerate(res.x[:-2]):
        if abs(flag-1) < eps:
            mm_edges += [edges[i]]
    return [res,mm_edges]

print(MMM({
   # point : velocity
   (0,0):(0,0),
   (1,1):(0,0),
   (-1,-1):(0,0),
   (-2,-2):(0,0),
}))
