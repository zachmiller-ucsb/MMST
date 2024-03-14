import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import numpy.linalg as npl
from MBMST import parse, find_MBMST
from MAMST import find_MAMST

class Visualization:
    def __init__(self, V, frames, interval):
        self.V = V
        self.MBMST = find_MBMST(V)
        # self.MBMST = find_MAMST(V)
        self.frames = frames
        self.interval = interval

    def run(self):
        fig, ax = plt.subplots(2, 1)
        plt.subplots_adjust(hspace=.5)
        nodes = ax[0].scatter(*zip(*map(lambda pv: pv[:2], V)))
        ax[0].set_title("Points in the plane")
        edges = { e : ax[0].plot([e[0][0], e[1][0]], [e[0][1], e[1][1]], color="C0")[0] for e in self.MBMST }
        times = [f for f in range(self.frames + 1)]
        costs = []
        cost_plt = ax[1].plot([], [])[0]
        def calc_lims(c, v):
            cx, cy = c
            vx, vy = v
            cx0, cx1 = cx, cx + vx * self.frames
            cy0, cy1 = cy, cy + vy * self.frames
            min_x, max_x = (cx0, cx1) if cx0 < cx1 else (cx1, cx0)
            min_y, max_y = (cy0, cy1) if cy0 < cy1 else (cy1, cy0)
            return ([min_x, max_x], [min_y, max_y])
        x_lim, y_lim = calc_lims(list(self.V)[0][:2], list(self.V)[0][2:])
        for cv in self.V:
            x_lim_, y_lim_ = calc_lims(cv[:2], cv[2:])
            for lim, lim_ in [(x_lim, x_lim_), (y_lim, y_lim_)]:
                lim[0] = min(lim[0], lim_[0])
                lim[1] = max(lim[1], lim_[1])
        ax[0].set(xlim=x_lim, ylim=y_lim)
        def update(frame):
            new_coords = { cv : np.array(cv[:2]) + np.array(cv[2:]) * frame for cv in self.V }
            curr_cost = .0
            for e, e_plt in edges.items():
                e_plt.set_xdata([new_coords[e[0]][0], new_coords[e[1]][0]])
                e_plt.set_ydata([new_coords[e[0]][1], new_coords[e[1]][1]])
                curr_cost += npl.norm(new_coords[e[1]] - new_coords[e[0]])
            if frame == 0:
                costs.clear()
            costs.append(curr_cost)
            cost_plt.set_data(times[:frame + 1], costs)
            min_cost = min(costs)
            max_cost = max(costs)
            cost_plt.axes.axis([0, frame, 1.2 * (min_cost - max_cost) + max_cost, max_cost])
            ax[1].set_title("MBMST cost = {:.3f}".format(curr_cost))
            new_x = list(map(lambda nc: nc[0], new_coords.values()))
            new_y = list(map(lambda nc: nc[1], new_coords.values()))
            data = np.stack([new_x, new_y]).T
            nodes.set_offsets(data)
            return (nodes, edges, cost_plt)
        ani = animation.FuncAnimation(fig=fig, func=update, frames=self.frames, interval=self.interval, repeat=True)
        def save_ani():
            writer = animation.PillowWriter(fps=10,
                                            metadata=dict(artist='Me'),
                                            bitrate=2)
            ani.save('scatter.gif', writer=writer)
        save_ani()
        # plt.show()

if __name__ == "__main__":
    V = parse()
    viz = Visualization(V, 60, .001)
    viz.run()
