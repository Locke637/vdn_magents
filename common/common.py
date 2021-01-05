import numpy as np
import math


def find_neighbor_id(pos, view_field):
    nei_index = {}
    for id, p in enumerate(pos):
        nei_index[id] = []
        for index, nei_p in enumerate(pos):
            if abs(p[0] - nei_p[0]) < view_field and abs(p[1] - nei_p[1]) < view_field and index != id:
                nei_index[id].append(index)
    return nei_index


def find_neighbor_pos(pos, view_field):
    num_neighbor = 3
    # view_field = 5
    nei_index = {}
    nei_pos = {}
    for id, p in enumerate(pos):
        nei_index[id] = []
        nei_pos[id] = []
        d_p_all = {}
        temp_pos = {}
        for index, nei_p in enumerate(pos):
            d_x = abs(p[0] - nei_p[0])
            d_y = abs(p[1] - nei_p[1])
            if d_x < view_field and d_y < view_field and index != id:
                d_p = d_x + d_y
                d_p_all[index] = d_p
                temp_pos[index] = [nei_p[0] - p[0], nei_p[1] - p[1]]
        if d_p_all:
            d_p_all = sorted(d_p_all.items(), key=lambda item: item[1])
            count = 0
            for idpos in d_p_all:
                if count < num_neighbor:
                    nei_index[id].append(idpos[0])
                    nei_pos[id].append(temp_pos[idpos[0]])
                    count += 1
                else:
                    break

    return nei_index, nei_pos


# def find_pos_index(pos):
#     baseline = 2 * math.pi / 3
#     theta = math.atan2(pos[1], pos[0])
#     if theta < 0:
#         theta += 2 * math.pi
#     index = int(theta / baseline)
#     return index


if __name__ == '__main__':
    pos = [[0, 1], [5, 5], [2, 2], [3, 3], [3, 4], [3, 5], [11, 11], [20, 20]]
    print(find_neighbor_pos(pos))
    # index = find_pos_index(pos[0])
    # print(index)
