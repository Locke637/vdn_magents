import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import interpolate


# file_name = 'meta_3_64'
# file_name = 'goal_b_3_20_100'
def pre_data(file_name):
    # file_name = 'vdn_3_3_32_100_test'
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    win_size = 100
    # print(data)
    r = []
    win = []
    floss = []
    step = []
    std = []
    pr = [0 for _ in range(len(data))]
    pfloss = [0 for _ in range(len(data))]
    pstep = [0 for _ in range(len(data))]
    std_pr = [0 for _ in range(len(data))]
    std_pfloss = [0 for _ in range(len(data))]
    std_pstep = [0 for _ in range(len(data))]
    find_flosses = []
    find_step = []
    for item in data:
        r.append(item[0])
        win.append(item[1])
        floss.append(item[2])
        step.append(item[3])
        if item[1]:
            find_flosses.append(item[2])
            find_step.append(item[3])
    # print(min(find_flosses), np.argmin(find_flosses), find_step[np.argmin(find_flosses)])
    # for i in range(len(data)):
    #     pr[i] = np.mean(r[max(i - win_size, 0):i])
    #     pwin[i] = np.mean(win[max(i - win_size, 0):i])
    #     pfloss[i] = np.mean(floss[max(i - win_size, 0):i])
    #     pstep[i] = np.mean(step[max(i - win_size, 0):i])

    for i in range(len(data)):
        if 0 < i <= win_size:
            pr[i] = np.mean(r[:i])
            # pwin[i] = np.mean(win[max(i - win_size, 0):i])
            pfloss[i] = np.mean(floss[:i])
            pstep[i] = np.mean(step[:i])
            std_pr[i] = np.std(r[:i])
            std_pfloss[i] = np.std(floss[:i])
            std_pstep[i] = np.std(step[:i])
            # std.append([np.std(r[:i]), floss[:i], step[:i]])
        else:
            pr[i] = np.mean(r[max(i - win_size, 0):i])
            # pwin[i] = np.mean(win[max(i - win_size, 0):i])
            pfloss[i] = np.mean(floss[max(i - win_size, 0):i])
            pstep[i] = np.mean(step[max(i - win_size, 0):i])
            std_pr[i] = np.std(r[max(i - win_size, 0):i])
            std_pfloss[i] = np.std(floss[max(i - win_size, 0):i])
            std_pstep[i] = np.std(step[max(i - win_size, 0):i])
            # std.append([np.std(r[max(i - win_size, 0):i]), floss[max(i - win_size, 0):i], step[max(i - win_size, 0):i]])

    x = [i for i in range(len(data))]
    # x = [data[i] for i in range(plotlen)]
    # y = [np.mean(plot_data[i:i + win_size]) for i in x]

    r1_pr = list(map(lambda xl: xl[0] - xl[1], zip(pr, std_pr)))
    r2_pr = list(map(lambda xl: xl[0] + xl[1], zip(pr, std_pr)))
    r1_pfloss = list(map(lambda xl: xl[0] - xl[1], zip(pfloss, std_pfloss)))
    r2_pfloss = list(map(lambda xl: xl[0] + xl[1], zip(pfloss, std_pfloss)))
    r1_pstep = list(map(lambda xl: xl[0] - xl[1], zip(pstep, std_pstep)))
    r2_pstep = list(map(lambda xl: xl[0] + xl[1], zip(pstep, std_pstep)))
    return pr, pfloss, pstep, r1_pr, r2_pr, r1_pfloss, r2_pfloss, r1_pstep, r2_pstep


if __name__ == '__main__':
    file_name_vdn = 'vdn_3_3_32_100'
    file_name_ctce = 'ctce_vdn_3_1_32_1'
    file_name_meta = 'meta_3_32_100'
    pr_vdn, pfloss_vdn, pstep_vdn, r1_pr_vdn, r2_pr_vdn, r1_pfloss_vdn, r2_pfloss_vdn, r1_pstep_vdn, r2_pstep_vdn = pre_data(
        file_name_vdn)
    # pr_vdn, pwin_vdn, pfloss_vdn, pstep_vdn = pre_data(file_name_vdn)
    pr_ctce, pfloss_ctce, pstep_ctce, r1_pr_ctce, r2_pr_ctce, r1_pfloss_ctce, r2_pfloss_ctce, r1_pstep_ctce, r2_pstep_ctce = pre_data(
        file_name_ctce)
    # pr_ctce, pwin_ctce, pfloss_ctce, pstep_ctce = pre_data(file_name_ctce)
    # print(len(pr_ctce))
    pr, pfloss, pstep, r1_pr, r2_pr, r1_pfloss, r2_pfloss, r1_pstep, r2_pstep = pre_data(
        file_name_meta)
    pr_meta, pfloss_meta, pstep_meta, r1_pr_meta, r2_pr_meta, r1_pfloss_meta, \
    r2_pfloss_meta, r1_pstep_meta, r2_pstep_meta = [], [], [], [], [], [], [], [], []
    for i, item in enumerate(pr):
        for _ in range(7):
            pr_meta.append(item)
            pfloss_meta.append(pfloss[i])
            pstep_meta.append(pstep[i])
            r1_pr_meta.append(r1_pr[i])
            r2_pr_meta.append(r2_pr[i])
            r1_pfloss_meta.append(r1_pfloss[i])
            r2_pfloss_meta.append(r2_pfloss[i])
            r1_pstep_meta.append(r1_pstep[i])
            r2_pstep_meta.append(r2_pstep[i])
    x_vdn = np.array(range(len(pr_vdn))) * 10
    x_ctce = np.array(range(len(pr_ctce))) * 10
    x_meta = np.array(range(len(pr_meta))) * 10
    max_len = len(x_vdn)

    font_size = 12
    plt.figure(figsize=(12, 10))
    plt.subplot(221)
    plt.title('Rewards')
    plt.grid()
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    plt.plot(x_vdn, pr_vdn, label='VDN')
    plt.plot(x_ctce[:max_len], pr_ctce[:max_len], label='CTCE')
    plt.plot(x_meta[:max_len], pr_meta[:max_len], label='Ours')
    plt.fill_between([i * 10 for i in range(len(r1_pr_vdn))], r1_pr_vdn, r2_pr_vdn, alpha=0.1)
    plt.fill_between([i * 10 for i in range(len(r1_pr_ctce))], r1_pr_ctce, r2_pr_ctce, alpha=0.1)
    plt.fill_between([i * 10 for i in range(len(r1_pr_meta))], r1_pr_meta, r2_pr_meta, alpha=0.1)
    plt.legend(loc=1, fontsize=font_size)
    plt.xlim(0, 600000)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.subplot(222)
    plt.title('Formation loss')
    plt.grid()
    # plt.xlabel('Episodes')
    plt.plot(x_vdn, pfloss_vdn, label='VDN')
    plt.plot(x_ctce[:max_len], pfloss_ctce[:max_len], label='CTCE')
    plt.plot(x_meta[:max_len], pfloss_meta[:max_len], label='Ours')
    plt.fill_between([i * 10 for i in range(len(r1_pfloss_vdn))], r1_pfloss_vdn, r2_pfloss_vdn, alpha=0.1)
    plt.fill_between([i * 10 for i in range(len(r1_pfloss_ctce))], r1_pfloss_ctce, r2_pfloss_ctce, alpha=0.1)
    plt.fill_between([i * 10 for i in range(len(r1_pfloss_meta))], r1_pfloss_meta, r2_pfloss_meta, alpha=0.1)
    plt.legend(loc=1, fontsize=font_size)
    plt.xlim(0, 600000)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.subplot(223)
    plt.title('Episode steps')
    plt.grid()
    # plt.xlabel('Episodes')
    plt.plot(x_vdn, pstep_vdn, label='VDN')
    plt.plot(x_ctce[:max_len], pstep_ctce[:max_len], label='CTCE')
    plt.plot(x_meta[:max_len], pstep_meta[:max_len], label='Ours')
    plt.fill_between([i * 10 for i in range(len(r1_pstep_vdn))], r1_pstep_vdn, r2_pstep_vdn, alpha=0.1)
    plt.fill_between([i * 10 for i in range(len(r1_pstep_ctce))], r1_pstep_ctce, r2_pstep_ctce, alpha=0.1)
    plt.fill_between([i * 10 for i in range(len(r1_pstep_meta))], r1_pstep_meta, r2_pstep_meta, alpha=0.1)
    plt.legend(loc=1, fontsize=font_size)
    plt.xlim(0, 600000)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.subplot(224)
    plt.title('Pareto fronts')
    x1 = np.array([22, 18.4, 16.4, 15.9])
    y1 = np.array([35, 39, 44, 46.4])
    x1 = x1 / x1[0]
    y1 = y1 / y1[-1]
    plt.grid()
    plt.scatter(x1, y1, marker='x', color='black', zorder=20)
    plt.text(x1[0] + 0.020, y1[0] + 0.0175, "$w=0*w_f$", fontsize=12, style="italic", color="black", weight="light",
             verticalalignment='center',
             horizontalalignment='right')
    plt.text(x1[1] + 0.05, y1[1] + 0.0125, "$w=1*w_f$", fontsize=12, style="italic", color="black", weight="light",
             verticalalignment='center',
             horizontalalignment='right')
    plt.text(x1[2] + 0.06, y1[2] + 0.0125, "$w=2*w_f$", fontsize=12, style="italic", color="black", weight="light",
             verticalalignment='center',
             horizontalalignment='right')
    plt.text(x1[3] + 0.04, y1[3] + 0.01, "$w=3*w_f$", fontsize=12, style="italic", color="black", weight="light",
             verticalalignment='center',
             horizontalalignment='right')
    step = (x1[0] - x1[-1]) / 50
    # y2 = [0.075, 0.067, 0.056]
    x = np.array([x1[0] - i * step for i in range(51)])
    f = interpolate.interp1d(x1, y1, kind='quadratic')
    yvals1 = f(x)
    plt.xlabel('Formation loss', fontsize=font_size)
    plt.ylabel('Makespan', fontsize=font_size)
    plt.plot(x, yvals1)
    plt.savefig('rl.png', format='png', bbox_inches='tight')
    plt.show()
# plt.plot()
