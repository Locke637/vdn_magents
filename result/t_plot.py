import numpy as np
import pickle
import matplotlib.pyplot as plt

win_size = 50
len_limit = 5000


def splot(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    if len(data) > len_limit:
        length = len_limit
    else:
        length = len(data)
    pdata, std_data = [0 for _ in range(length)], [0 for _ in range(length)]
    upper, bound = [], []
    x = range(length)
    for i in range(length):
        if 0 < i <= win_size:
            pdata[i] = np.mean(data[:i])
            std_data[i] = np.std(data[:i])
            upper.append(pdata[i] - std_data[i])
            bound.append(pdata[i] + std_data[i])
        else:
            pdata[i] = np.mean(data[max(i - win_size, 0):i])
            std_data[i] = np.std(data[max(i - win_size, 0):i])
            upper.append(pdata[i] - std_data[i])
            bound.append(pdata[i] + std_data[i])
    return x, pdata, [upper, bound]


# map_size = 180
# map = 'pursuit'
# num_agent = 8
def plot_one(map_size, map, num_agent, index_l):
    coalg = 'vdn'  # maven qmix
    index = index_l[0]
    file_name = coalg + '/' + str(map) + '_' + str(num_agent) + '_' + str(map_size) + '_' + str(index)
    x0, pdata0, std0 = splot(file_name)

    coalg = 'maven'  # maven qmix
    index = index_l[1]
    file_name = coalg + '/' + str(map) + '_' + str(num_agent) + '_' + str(map_size) + '_' + str(index)
    x1, pdata1, std1 = splot(file_name)

    coalg = 'qmix'  # maven qmix
    index = index_l[2]
    file_name = coalg + '/' + str(map) + '_' + str(num_agent) + '_' + str(map_size) + '_' + str(index)
    x2, pdata2, std2 = splot(file_name)

    coalg = 'qtran_base'  # maven qmix
    index = index_l[3]
    file_name = coalg + '/' + str(map) + '_' + str(num_agent) + '_' + str(map_size) + '_' + str(index)
    x3, pdata3, std3 = splot(file_name)

    index = index_l[4]
    file_name = 'ours' + '/' + str(map) + '_wdq_' + str(num_agent) + '_' + str(map_size) + '_' + str(index)
    x7, pdata7, std7 = splot(file_name)

    plt.plot(x0, pdata0, label='vdn')
    plt.fill_between([i * 1 for i in range(len(std0[0]))], std0[0], std0[1], alpha=0.1)
    plt.plot(x1, pdata1, label='maven')
    plt.fill_between([i * 1 for i in range(len(std1[0]))], std1[0], std1[1], alpha=0.1)
    plt.plot(x2, pdata2, label='qmix')
    plt.fill_between([i * 1 for i in range(len(std2[0]))], std2[0], std2[1], alpha=0.1)
    plt.plot(x3, pdata3, label='qtran_base')
    plt.fill_between([i * 1 for i in range(len(std3[0]))], std3[0], std3[1], alpha=0.1)
    plt.plot(x7, pdata7, label='ours')
    plt.fill_between([i * 1 for i in range(len(std7[0]))], std7[0], std7[1], alpha=0.1)
    plt.legend(loc='best')


if __name__ == '__main__':
    plt.figure(figsize=(20, 10))
    plt.subplot(231)
    plt.title('Pursuit_Easy')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.grid()
    plt.xlim(0, len_limit)
    map_size = 180
    map = 'pursuit'
    num_agent = 8
    index_l = ['t', 7, 't', 0, 't']
    plot_one(map_size, map, num_agent, index_l)

    plt.subplot(234)
    plt.title('Pursuit_Hard')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.grid()
    plt.xlim(0, len_limit)
    map_size = 270
    map = 'pursuit'
    num_agent = 8
    index_l = ['t', 7, 0, 0, 7]
    plot_one(map_size, map, num_agent, index_l)

    plt.subplot(232)
    plt.title('Battle_Easy')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.grid()
    plt.xlim(0, len_limit)
    map_size = 80
    map = 'battle'
    num_agent = 5
    index_l = ['0', 'm', 'm', 't', 't']
    plot_one(map_size, map, num_agent, index_l)

    plt.subplot(235)
    plt.title('Battle_Hard')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.grid()
    plt.xlim(0, len_limit)
    map_size = 100
    map = 'battle'
    num_agent = 5
    index_l = ['7', 't', 't', 't', 't']
    plot_one(map_size, map, num_agent, index_l)

    plt.subplot(233)
    plt.title('Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.grid()
    plt.xlim(0, len_limit)
    map_size = 80
    map = 'battle'
    num_agent = 5
    index_l = ['t', 't', 't', 't', 't']
    plot_one(map_size, map, num_agent, index_l)

    plt.subplot(236)
    plt.title('Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.grid()
    plt.xlim(0, len_limit)
    map_size = 100
    map = 'battle'
    num_agent = 5
    index_l = ['t', 't', 't', 't', 't']
    plot_one(map_size, map, num_agent, index_l)
    plt.savefig('rl.png', format='png', bbox_inches='tight')
    plt.show()
