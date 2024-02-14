import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from MBMST import parse

DT = .02
TFINAL = 4
FRAMES = int(4 / DT)
INTERVAL = DT * 1000

class Visualization:
    def __init__(self, V, frames, interval):
        # v maps coord tuples to velocity tuples
        self.x = []
        self.y = []
        self.vx = []
        self.vy = []
        for coord, vel in V.items():
            _x, _y = coord
            _vx, _vy = vel
            self.x.append(_x)
            self.y.append(_y)
            self.vx.append(_vx)
            self.vy.append(_vy)
        self.frames = frames
        self.interval = interval

    def run(self):
        fig, ax = plt.subplots()
        ax.set(xlim=[-100, 100], ylim=[-100, 100])
        line = ax.scatter(self.x, self.y)
        def update(frame):
            self.x = [self.x[i] + (self.vx[i] * float(frame) / self.frames) for i in range(len(self.x))]
            self.y = [self.y[i] + (self.vy[i] * float(frame) / self.frames) for i in range(len(self.y))]
            data = np.stack([self.x, self.y]).T
            line.set_offsets(data)
            return (line)
        ani = animation.FuncAnimation(fig=fig, func=update, frames=FRAMES, interval=INTERVAL, repeat=True)
        plt.show()

if __name__ == "__main__":
    V = parse()
    viz = Visualization(V, 10, 3)
    viz.run()
