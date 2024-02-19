import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from MBMST import parse

class Visualization:
    def __init__(self, V, frames, interval):
        self.V = V
        self.frames = frames
        self.interval = interval

    def run(self):
        fig, ax = plt.subplots()
        line = ax.scatter(*zip(*self.V.keys()))
        ax.set(xlim=[-3, 3], ylim=[-3, 3])
        mindim = min(abs(ax.get_xlim()[1] - ax.get_xlim()[0]), abs(ax.get_ylim()[1] - ax.get_ylim()[0]))
        div = mindim / self.frames
        def update(frame):
            new_coords = list(map(lambda cv: np.array(cv[0]) + np.array(cv[1]) * frame * div, self.V.items()))
            data = np.stack(new_coords).T
            line.set_offsets(data)
            return line
        ani = animation.FuncAnimation(fig=fig, func=update, frames=self.frames, interval=self.interval, repeat=True)
        plt.show()

if __name__ == "__main__":
    V = parse()
    viz = Visualization(V, 500, 1)
    viz.run()
