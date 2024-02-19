import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from MBMST import parse, find_MBMST

class Visualization:
    def __init__(self, V, frames, interval):
        self.V = V
        self.MBMST = find_MBMST(V)
        self.frames = frames
        self.interval = interval

    def run(self):
        fig, ax = plt.subplots()
        nodes = ax.scatter(*zip(*self.V.keys()))
        edges = { e : ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]], color="C0")[0] for e in self.MBMST }
        ax.set(xlim=[-10, 10], ylim=[-10, 10])
        mindim = min(abs(ax.get_xlim()[1] - ax.get_xlim()[0]), abs(ax.get_ylim()[1] - ax.get_ylim()[0]))
        div = mindim / self.frames
        def update(frame):
            new_coords = { cv[0] : np.array(cv[0]) + np.array(cv[1]) * frame * div for cv in self.V.items() }
            for e, e_plt in edges.items():
                e_plt.set_xdata([new_coords[e[0]][0], new_coords[e[1]][0]])
                e_plt.set_ydata([new_coords[e[0]][1], new_coords[e[1]][1]])
            new_x = list(map(lambda nc: nc[0], new_coords.values()))
            new_y = list(map(lambda nc: nc[1], new_coords.values()))
            data = np.stack([new_x, new_y]).T
            nodes.set_offsets(data)
            return (nodes, edges)
        ani = animation.FuncAnimation(fig=fig, func=update, frames=self.frames, interval=self.interval, repeat=True)
        plt.show()

if __name__ == "__main__":
    V = parse()
    viz = Visualization(V, 1000, 1)
    viz.run()
