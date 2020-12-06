import numpy as np


def find_neighbor(pos):
    nei_index = {}
    for id, p in enumerate(pos):
        nei_index[id] = []
        for index, nei_p in enumerate(pos):
            if abs(p[0] - nei_p[0]) < 5 and abs(p[1] - nei_p[1]) < 5 and index!=id:
                nei_index[id].append(index)
    return nei_index


if __name__ == '__main__':
    pos = [[1, 1], [5, 5]]
    print(find_neighbor(pos))
