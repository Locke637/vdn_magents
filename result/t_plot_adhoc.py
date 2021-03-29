import numpy as np
import pickle
import matplotlib.pyplot as plt


# win_size = 50
# len_limit = 5000


def splot(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    # if len(data) > len_limit:
    #     length = len_limit
    # else:
    #     length = len(data)
    # pdata, std_data = [0 for _ in range(length)], [0 for _ in range(length)]
    ret = np.mean(data)
    return ret


# map_size = 180
# map = 'pursuit'
# num_agent = 8
def plot_one(map_size, map, num_agent, index_l):
    pdata0, pdata1, pdata2 = [], [], []
    coalg = 'qmix'  # maven qmix
    index = index_l[0]
    file_name = coalg + '/' + str(map) + '_' + str(int(num_agent * 0.5)) + '_' + str(map_size) + '_' + str(index)
    pdata0.append(splot(file_name))
    file_name = coalg + '/' + str(map) + '_' + str(num_agent * 2) + '_' + str(map_size) + '_' + str(index)
    pdata0.append(splot(file_name))
    file_name = coalg + '/' + str(map) + '_' + str(num_agent * 2) + '_' + str(map_size) + '_' + str(index)
    pdata0.append(splot(file_name))

    coalg = 'maven'  # maven qmix
    index = index_l[1]
    file_name = coalg + '/' + str(map) + '_' + str(int(num_agent)) + '_' + str(map_size) + '_' + str(
        index) + '_' + str(int(num_agent * 0.5))
    pdata1.append(splot(file_name))
    file_name = coalg + '/' + str(map) + '_' + str(num_agent) + '_' + str(map_size) + '_' + str(index) + '_' + str(
        int(num_agent * 2))
    pdata1.append(splot(file_name))
    file_name = coalg + '/' + str(map) + '_' + str(num_agent) + '_' + str(map_size) + '_' + str(index) + '_' + str(
        int(num_agent * 2))
    pdata1.append(splot(file_name))

    index = index_l[2]
    file_name = 'ours' + '/' + str(map) + '_wdq_' + str(int(num_agent * 0.5)) + '_' + str(map_size) + '_' + str(index)
    pdata2.append(splot(file_name))
    file_name = 'ours' + '/' + str(map) + '_wdq_' + str(num_agent * 2) + '_' + str(map_size) + '_' + str(index)
    pdata2.append(splot(file_name))
    file_name = 'ours' + '/' + str(map) + '_wdq_' + str(num_agent * 2) + '_' + str(map_size) + '_' + str(index)
    pdata2.append(splot(file_name))

    for i in range(3):
        pdata0[i] = pdata0[i]/pdata2[i]
        pdata1[i] = pdata1[i] / pdata2[i]
        pdata2[i] = pdata2[i] / pdata2[i]

    # plt.plot(x0, pdata0, label='qmix')
    # plt.plot(x1, pdata1, label='maven')
    # plt.plot(x2, pdata2, label='ours')
    x = np.arange(3)
    total_width, n = 0.8, 3
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.bar(x, pdata0, width=width, label='a')
    plt.bar(x + width, pdata1, width=width, label='b')
    plt.bar(x + 2 * width, pdata2, width=width, label='c')

    plt.legend(loc='best')


if __name__ == '__main__':
    plt.figure(figsize=(10, 5))
    plt.subplot(131)
    plt.title('Pursuit_Easy')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.grid()
    # plt.xlim(0, len_limit)
    map_size = 180
    map = 'pursuit'
    num_agent = 8
    index_l = ['est', 'est', 'est']
    plot_one(map_size, map, num_agent, index_l)

    plt.subplot(132)
    plt.title('Pursuit_Hard')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.grid()
    # plt.xlim(0, len_limit)
    map_size = 180
    map = 'pursuit'
    num_agent = 8
    index_l = ['est', 'est', 'est']
    plot_one(map_size, map, num_agent, index_l)

    plt.subplot(133)
    plt.title('Battle_Easy')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.grid()
    # plt.xlim(0, len_limit)
    map_size = 180
    map = 'pursuit'
    num_agent = 8
    index_l = ['est', 'est', 'est']
    plot_one(map_size, map, num_agent, index_l)

    plt.show()
