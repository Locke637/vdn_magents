import numpy as np
import pickle
import matplotlib.pyplot as plt

# win_size = 50
#
#
# def splot(file_name):
#     with open(file_name, 'rb') as f:
#         data = pickle.load(f)
#     pdata = [0 for _ in range(len(data))]
#     x = range(len(data))
#     upper, bound = [], []
#     for i in range(len(data)):
#         if 0 < i <= win_size:
#             pdata[i] = np.mean(data[:i])
#         else:
#             pdata[i] = np.mean(data[max(i - win_size, 0):i])
#     return x, pdata

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

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title('Battle')
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.grid()
# plt.xlim(0, len_limit)
map_size = 100
file_name = 'ours/battle_5_' + str(map_size) + '_alt_wo_dq'
x0, pdata0, std0 = splot(file_name)
file_name = 'ours/battle_wdq_5_' + str(map_size) + '_alt_wo_per'
x1, pdata1, std1 = splot(file_name)
file_name = 'ours/battle_wdq_5_' + str(map_size) + '_t'
x2, pdata2, std2 = splot(file_name)

plt.plot(x0, pdata0, label='wo/dq')
plt.fill_between([i * 1 for i in range(len(std0[0]))], std0[0], std0[1], alpha=0.1)
plt.plot(x1, pdata1, label='wo/per')
plt.fill_between([i * 1 for i in range(len(std1[0]))], std1[0], std1[1], alpha=0.1)
plt.plot(x2, pdata2, label='ours')
plt.fill_between([i * 1 for i in range(len(std2[0]))], std2[0], std2[1], alpha=0.1)
plt.legend(loc='best')
plt.xlim(0, 4200)

plt.subplot(122)
plt.title('Pursuit')
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.grid()
plt.xlim(0, len_limit)
map_size = 180
file_name = 'ours/pursuit_8_' + str(map_size) + '_alt_wo_dq'
x0, pdata0, std0 = splot(file_name)
file_name = 'ours/pursuit_wdq_8_' + str(map_size) + '_alt_wo_per'
x1, pdata1, std1 = splot(file_name)
file_name = 'ours/pursuit_wdq_8_' + str(map_size) + '_t'
x2, pdata2, std2 = splot(file_name)

plt.plot(x0, pdata0, label='wo/dq')
plt.fill_between([i * 1 for i in range(len(std0[0]))], std0[0], std0[1], alpha=0.1)
plt.plot(x1, pdata1, label='wo/per')
plt.fill_between([i * 1 for i in range(len(std1[0]))], std1[0], std1[1], alpha=0.1)
plt.plot(x2, pdata2, label='ours')
plt.fill_between([i * 1 for i in range(len(std2[0]))], std2[0], std2[1], alpha=0.1)
plt.legend(loc='best')
plt.savefig('alt.png', format='png', bbox_inches='tight')
plt.show()
