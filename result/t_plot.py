import numpy as np
import pickle
import matplotlib.pyplot as plt

win_size = 50
# file_name = 'vdn/pursuit_8_160_14'
# with open(file_name, 'rb') as f:
#     data1 = pickle.load(f)
# file_name = 'ours/pursuit_8_160_15'
# with open(file_name, 'rb') as f:
#     data2 = pickle.load(f)
# li = np.ones([10, 5, 1])
# idx = [0,1]
# print(li[idx])
# for i in range(20):
#     print(not (np.random.randint(0, 10)))

# file_name = 'ours/pursuit_wdq_8_270_14'
file_name = 'vdn/battle_5_100_t'
with open(file_name, 'rb') as f:
    data1 = pickle.load(f)
# file_name = 'ours/pursuit_wdq_8_270_17'
file_name = 'ours/battle_wdq_5_100_t'
with open(file_name, 'rb') as f:
    data2 = pickle.load(f)
pdata1, pdata2 = [0 for _ in range(len(data1))], [0 for _ in range(len(data2))]
x1 = range(len(data1))
x2 = range(len(data2))
for i in range(len(data1)):
    if 0 < i <= win_size:
        pdata1[i] = np.mean(data1[:i])
    else:
        pdata1[i] = np.mean(data1[max(i - win_size, 0):i])
for i in range(len(data2)):
    if 0 < i <= win_size:
        pdata2[i] = np.mean(data2[:i])
    else:
        pdata2[i] = np.mean(data2[max(i - win_size, 0):i])

plt.plot(x1, pdata1, label='vdn')
plt.plot(x2, pdata2, label='ours')
plt.legend(loc='best')
plt.show()
