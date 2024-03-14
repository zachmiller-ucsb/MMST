import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
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
        fig, ax = plt.subplots()
        nodes = ax.scatter(*zip(*map(lambda pv: pv[:2], V)))
        edges = { e : ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]], color="C0")[0] for e in self.MBMST }
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
        ax.set(xlim=x_lim, ylim=y_lim)
        def update(frame):
            new_coords = { cv : np.array(cv[:2]) + np.array(cv[2:]) * frame for cv in self.V }
            for e, e_plt in edges.items():
                e_plt.set_xdata([new_coords[e[0]][0], new_coords[e[1]][0]])
                e_plt.set_ydata([new_coords[e[0]][1], new_coords[e[1]][1]])
            new_x = list(map(lambda nc: nc[0], new_coords.values()))
            new_y = list(map(lambda nc: nc[1], new_coords.values()))
            data = np.stack([new_x, new_y]).T
            nodes.set_offsets(data)
            return (nodes, edges)
        ani = animation.FuncAnimation(fig=fig, func=update, frames=self.frames, interval=self.interval, repeat=True)
        def save_ani():
            writer = animation.PillowWriter(fps=10,
                                            metadata=dict(artist='Me'),
                                            bitrate=2)
            ani.save('scatter.gif', writer=writer)
        # save_ani()
        plt.show()

if __name__ == "__main__":
    V = parse()
    viz = Visualization(V, 60, .001)
    viz.run()
