from random import randint
import sys

if __name__ == '__main__':
    num_points = int(sys.stdin.readline())
    x_lim = tuple([int(x) for x in sys.stdin.readline().split()])
    y_lim = tuple([int(x) for x in sys.stdin.readline().split()])

    print(num_points)
    for i in range(num_points):
        print(randint(*x_lim), randint(*y_lim), randint(*x_lim), randint(*y_lim))