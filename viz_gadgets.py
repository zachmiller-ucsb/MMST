import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import numpy.linalg as npl
from random import randint
from MBMST import find_MBMST

def gen(num_points, x_lim) -> list[int]:
    points = []
    for _ in range(num_points):
        points.append(randint(*x_lim))
    return points

def parse() -> tuple[int, int, list[int]]:
    """
    Expects format:
      num_points l
      x_min x_max
    """
    params = input().split()
    num_points, l = tuple(map(lambda p: int(p), params))
    x_lim = tuple(map(lambda c: int(c), input().split()))
    assert len(x_lim) == 2 and x_lim[0] <= x_lim[1]
    return num_points, l, gen(num_points, x_lim)

def run_viz(V, alg, frames, interval):
    MMST = alg(V)
    fig, ax = plt.subplots(2, 1)
    nodes = ax[0].scatter(*zip(*V.keys()))
    cost_x, cost_y = [],[]
    cost = ax[1].plot(cost_x, cost_y)[0]
    def curved_edge(x_coords, y_coords):
        return ax[0].annotate("",
            xy=(x_coords[0], y_coords[0]), xycoords='data',
            xytext=(x_coords[1], y_coords[1]), textcoords='data',
            arrowprops=dict(arrowstyle="-", color="0.5",
                            shrinkA=5, shrinkB=5,
                            patchA=None, patchB=None,
                            connectionstyle="angle3,angleA=45,angleB=135",
                            ),
        )
    edges = { e : curved_edge([e[0][0], e[1][0]], [e[0][1], e[1][1]]) for e in MMST }
    # Compute x and y axis lims
    def calc_lims(c, v):
        cx, cy = c
        vx, vy = v
        cx0, cx1 = cx, cx + vx * frames
        cy0, cy1 = cy, cy + vy * frames
        min_x, max_x = (cx0, cx1) if cx0 < cx1 else (cx1, cx0)
        min_y, max_y = (cy0, cy1) if cy0 < cy1 else (cy1, cy0)
        return ([min_x, max_x], [min_y, max_y])
    x_lim, y_lim = calc_lims(*list(V.items())[0])
    for c, v in V.items():
        x_lim_, y_lim_ = calc_lims(c, v)
        for lim, lim_ in [(x_lim, x_lim_), (y_lim, y_lim_)]:
            lim[0] = min(lim[0], lim_[0])
            lim[1] = max(lim[1], lim_[1])
    ax[0].set(xlim=x_lim, ylim=y_lim)
    # Animation frame update
    def update(frame):
        new_coords = { cv[0] : np.array(cv[0]) + np.array(cv[1]) * frame for cv in V.items() }
        curr_cost = 0
        for e, e_ann in edges.items():
            e_ann.remove()
            edges[e] = curved_edge([new_coords[e[0]][0], new_coords[e[1]][0]], [new_coords[e[0]][1], new_coords[e[1]][1]])
            curr_cost += npl.norm(new_coords[e[1]] - new_coords[e[0]])
            new_x = list(map(lambda nc: nc[0], new_coords.values()))
            new_y = list(map(lambda nc: nc[1], new_coords.values()))
        cost_x.append(frame)
        cost_y.append(curr_cost)
        cost.set_data(cost_x, cost_y)
        cost.axes.axis([0, frame, min(cost_y), max(cost_y)])
        data = np.stack([new_x, new_y]).T
        nodes.set_offsets(data)
        return (nodes, edges, cost)
    ani = animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=interval, repeat=True)
    def save_ani():
        writer = animation.PillowWriter(fps=10,
                                        metadata=dict(artist='Me'),
                                        bitrate=2)
        ani.save('scatter.gif', writer=writer)
    # save_ani()
    plt.show()

if __name__ == '__main__':
    V = {}
    num_points, l, a = parse()
    frames = 300 # Not a cmd line param for now
    for i in range(num_points):
        V[(10 * i, 0)] = (0, 0) # Ai
        V[(10 * i + 4 - float(a[i]) / (4 * l), 0)] = (0, 0) # Bi
        V[(10 * i + 4 + float(a[i]) / (4 * l), 0)] = (0, 0) # Ci
        V[(10 * i + 6, 0)] = (0, 0) # Di
        V[(10 * i + 5, 0)] = (float(-4) / frames, 0) # Ei
    V[(10 * num_points, 0)] = (0, 0) # P
    V[(10 * num_points + float(11) / 10 * num_points, 0)] = (0, 0) # Q
    V[(10 * num_points + float(32) / 10 * num_points, 0)] = (0, 0) # R
    # Note that this is cheating:
    # The extra factor .999 very slightly perturbs coordinates for S
    # so that we can hash to a separate k/v pair
    V[(10 * num_points + float(32) / 10 * num_points * .999, 0)] = \
        ((10 * num_points - (10 * num_points + float(32) / 10 * num_points * .999)) / frames, 0) # S
    assert len(V.keys()) == 5 * num_points + 4, f"{5 * num_points + 4} =/= {len(V.keys())}"
    run_viz(V, find_MBMST, frames, .001)
