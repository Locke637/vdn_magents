import numpy as np
import torch
from policy.vdn import VDN
from policy.vdn_fixed import VDN_F
from policy.qmix import QMIX
from policy.ours import OURS
from policy.coma import COMA
from policy.reinforce import Reinforce
from policy.central_v import CentralV
from policy.qtran_alt import QtranAlt
from policy.qtran_base import QtranBase
from policy.maven import MAVEN
from torch.distributions import Categorical
from common.arguments import get_common_args, get_coma_args, get_mixer_args
import time


# Agent no communication
class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents*2
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.idact_shape = args.id_dim + args.n_actions
        self.search_actions = np.eye(args.n_actions)
        self.search_ids = np.zeros(self.n_agents)
        if args.alg == 'vdn':
            self.policy = VDN(args)
        elif args.alg == 'qmix':
            self.policy = QMIX(args)
        elif args.alg == 'ours':
            self.policy = OURS(args)
        elif args.alg == 'coma':
            self.policy = COMA(args)
        elif args.alg == 'qtran_alt':
            self.policy = QtranAlt(args)
        elif args.alg == 'qtran_base':
            self.policy = QtranBase(args)
        elif args.alg == 'maven':
            self.policy = MAVEN(args)
        elif args.alg == 'central_v':
            self.policy = CentralV(args)
        elif args.alg == 'reinforce':
            self.policy = Reinforce(args)
        else:
            raise Exception("No such algorithm")
        if args.use_fixed_model:
            args_goal_a = get_common_args()
            args_goal_a.load_model = True
            args_goal_a = get_mixer_args(args_goal_a)
            args_goal_a.learn = False
            args_goal_a.epsilon = 0  # 1
            args_goal_a.min_epsilon = 0
            args_goal_a.map = 'battle'
            args_goal_a.n_actions = args.n_actions
            args_goal_a.episode_limit = args.episode_limit
            args_goal_a.n_agents = args.n_agents
            args_goal_a.state_shape = args.state_shape
            args_goal_a.feature_shape = args.feature_shape
            args_goal_a.view_shape = args.view_shape
            args_goal_a.obs_shape = args.obs_shape
            args_goal_a.real_view_shape = args.real_view_shape
            args_goal_a.load_num = args.load_num
            args_goal_a.use_ja = False
            args_goal_a.mlp_hidden_dim = [512, 512]
            self.fixed_policy = VDN_F(args_goal_a)
        self.args = args
        print('Init Agents')

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, maven_z=None, evaluate=False):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.policy.eval_hidden[:, agent_num, :]

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q value
        if self.args.alg == 'maven':
            maven_z = torch.tensor(maven_z, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                maven_z = maven_z.cuda()
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state, maven_z)
        else:
            if 'qtran' in self.args.alg:
                q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)
            else:
                # print(inputs.shape)
                q_value = self.policy.eval_rnn(inputs)

        # choose action from q value
        if self.args.alg == 'coma' or self.args.alg == 'central_v' or self.args.alg == 'reinforce':
            action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon, evaluate)
        else:
            q_value[avail_actions == 0.0] = - float("inf")
            if np.random.uniform() < epsilon:
                action = np.random.choice(avail_actions_ind)  # action是一个整数
            else:
                action = torch.argmax(q_value)
        return action

    def choose_action_ja(self, obs, neighbor_actions, last_action, agent_num, avail_actions, epsilon, maven_z=None,
                         evaluate=False):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        inputs = np.hstack((inputs, neighbor_actions))
        # print(inputs.shape)
        hidden_state = self.policy.eval_hidden[:, agent_num, :]

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q value
        if self.args.alg == 'maven':
            maven_z = torch.tensor(maven_z, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                maven_z = maven_z.cuda()
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state, maven_z)
        else:
            # q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)
            # st = time.time()
            q_value = self.policy.eval_rnn(inputs)
            # print(time.time()-st)

        # choose action from q value
        if self.args.alg == 'coma' or self.args.alg == 'central_v' or self.args.alg == 'reinforce':
            action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon, evaluate)
        else:
            q_value[avail_actions == 0.0] = - float("inf")
            if np.random.uniform() < epsilon:
                action = np.random.choice(avail_actions_ind)  # action是一个整数
            else:
                action = torch.argmax(q_value)
        return action

    def choose_action_ja_v2(self, obs, neighbor_actions, need_search_agent, last_action, agent_num, avail_actions,
                            epsilon, maven_z=None,
                            evaluate=False):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))

        if need_search_agent:
            q_tot = np.zeros(self.n_actions)
            for search_id in need_search_agent:
                agent_id_one_hot = self.search_ids.copy()
                agent_id_one_hot[search_id] = 1
                # t_neighbor_actions = neighbor_actions.copy()
                for i in range(self.n_actions):
                    t_neighbor_actions = neighbor_actions.copy()
                    search_act = self.search_actions[i]
                    search_idact = np.concatenate([agent_id_one_hot, search_act], axis=0)
                    # print('s', t_neighbor_actions[search_id * self.idact_shape:(search_id + 1) * self.idact_shape])
                    # print(search_idact)
                    t_neighbor_actions[search_id * self.idact_shape:(search_id + 1) * self.idact_shape] = search_idact
                    # print(t_neighbor_actions[search_id * self.idact_shape:(search_id + 1) * self.idact_shape])
                    t_inputs = np.hstack((inputs, t_neighbor_actions))
                    t_inputs = torch.tensor(t_inputs, dtype=torch.float32).unsqueeze(0)
                    inputs_cuda = t_inputs.cuda()
                    q_value = self.policy.eval_rnn(inputs_cuda).squeeze()
                    max_q_index = torch.argmax(q_value)
                    q_tot[max_q_index] += q_value[max_q_index]
        else:
            inputs = np.hstack((inputs, neighbor_actions))
            inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
            inputs = inputs.cuda()
            q_value = self.policy.eval_rnn(inputs)
            q_tot = q_value

        if np.random.uniform() < epsilon:
            action = np.random.choice(avail_actions_ind)  # action是一个整数
        else:
            if isinstance(q_tot, np.ndarray):
                action = np.argmax(q_tot)
            else:
                action = torch.argmax(q_tot)
        return action

    def choose_action_ja_v3(self, obs, neighbor_actions, neighbor_pos, need_search_agent, last_action, agent_num,
                            avail_actions,
                            epsilon, maven_z=None,
                            evaluate=False):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose
        test_q_actions = np.zeros(len(neighbor_actions))
        delta_max_q = 0

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.
        max_q_index_dic = {}

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))

        if need_search_agent:
            compare_neighbor_actions = neighbor_actions.copy()
            # q_tot = np.zeros(self.n_actions)
            q_tot = torch.zeros(self.n_actions).cuda()
            for search_id in need_search_agent:
                agent_pos = neighbor_actions[search_id * self.idact_shape:search_id * self.idact_shape + 2].copy()
                # max_q_temp = -10000
                # print(neighbor_pos)
                # print('ap', agent_pos)
                # t_neighbor_actions = neighbor_actions.copy()
                max_q_temp_one = -100000
                for i in range(self.n_actions):
                    t_neighbor_actions = neighbor_actions.copy()
                    search_act = self.search_actions[i]
                    search_idact = np.concatenate([agent_pos, search_act], axis=0)
                    # print('s', t_neighbor_actions[search_id * self.idact_shape:(search_id + 1) * self.idact_shape])
                    # print(search_idact)
                    t_neighbor_actions[search_id * self.idact_shape:(search_id + 1) * self.idact_shape] = search_idact
                    # print(t_neighbor_actions[search_id * self.idact_shape:(search_id + 1) * self.idact_shape])
                    # print(t_neighbor_actions)
                    t_inputs = np.hstack((inputs, t_neighbor_actions))
                    t_inputs = torch.tensor(t_inputs, dtype=torch.float32).unsqueeze(0)
                    inputs_cuda = t_inputs.cuda()
                    q_value = self.policy.eval_rnn(inputs_cuda).squeeze()

                    max_q_index = torch.argmax(q_value)
                    q_tot[max_q_index] += q_value[max_q_index]
                    # q_tot += q_value.cpu()

                #     max_q_one = torch.max(q_value)
                #     if max_q_one > max_q_temp_one:
                #         max_q_temp_one = max_q_one
                #         add_q = q_value
                #         # search_act_one = np.concatenate([agent_pos, self.search_actions[i]], axis=0)
                #         # compare_neighbor_actions[
                #         # search_id * self.idact_shape:(search_id + 1) * self.idact_shape] = search_act_one
                # q_tot += add_q.cpu()

            # # max_q = torch.max(q_value)
            # compare_inputs = np.hstack((inputs, compare_neighbor_actions))
            # compare_inputs = torch.tensor(compare_inputs, dtype=torch.float32).unsqueeze(0)
            # compare_inputs = compare_inputs.cuda()
            # compare_q_value = self.policy.eval_rnn(compare_inputs).squeeze()
            # compare_max_q = torch.max(compare_q_value)
            #
            # gt_inputs = np.hstack((inputs, test_q_actions))
            # gt_inputs = torch.tensor(gt_inputs, dtype=torch.float32).unsqueeze(0)
            # gt_inputs = gt_inputs.cuda()
            # gt_q_value = self.policy.eval_rnn(gt_inputs).squeeze()
            # gt_max_q = torch.max(gt_q_value)
            #
            # delta_max_q = compare_max_q - gt_max_q
            # q_tot = compare_q_value

            # if torch.argmax(gt_q_value) != torch.argmax(q_value):
            #     delta_act = 1
            # else:
            #     delta_act = 0
            # max_q = torch.max(q_value)
            # if max_q > max_q_temp:
            #     max_q_temp = max_q
            #     max_q_index_dic[search_id] = i
        else:
            inputs = np.hstack((inputs, neighbor_actions))
            inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
            inputs = inputs.cuda()
            q_value = self.policy.eval_rnn(inputs)
            q_tot = q_value

        if np.random.uniform() < epsilon:
            action = np.random.choice(avail_actions_ind)  # action是一个整数
        else:
            if isinstance(q_tot, np.ndarray):
                action = np.argmax(q_tot)
            else:
                action = torch.argmax(q_tot)
        return action, delta_max_q

    def choose_action_ja_vd(self, obs, neighbor_actions, neighbor_pos, need_search_agent, last_action, agent_num,
                            avail_actions,
                            epsilon, maven_z=None,
                            evaluate=False):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose
        delta_max_q = 0

        # # transform agent_num to onehot vector
        # agent_id = np.zeros(self.n_agents)
        # agent_id[agent_num] = 1.
        # max_q_index_dic = {}

        # if self.args.last_action:
        #     inputs = np.hstack((inputs, last_action))
        # if self.args.reuse_network:
        #     inputs = np.hstack((inputs, agent_id))

        q_tot = torch.zeros(self.n_actions).cuda()
        if not neighbor_actions:
            t_inputs = np.hstack((inputs, np.zeros(self.args.idact_dim)))
            t_inputs = torch.tensor(t_inputs, dtype=torch.float32).unsqueeze(0)
            inputs_cuda = t_inputs.cuda()
            q_value = self.policy.eval_rnn(inputs_cuda).squeeze()
            q_tot = q_value
        else:
            # t_inputs = np.hstack((inputs, np.zeros(self.args.idact_dim)))
            # inputs_cuda = torch.tensor(t_inputs, dtype=torch.float32).unsqueeze(0).cuda()
            # q_tot += self.policy.eval_rnn(inputs_cuda).squeeze()
            tmp_q_buffer_list = []
            for index in neighbor_actions.keys():
                if index in need_search_agent:
                    agent_pos = neighbor_actions[index][0:2].copy()
                    # compare_max_q = -100000
                    # tmp_q_buffer = []
                    max_actdim_q_list = torch.ones(self.n_actions) * -100
                    for i in range(self.n_actions):
                        search_act = self.search_actions[i]
                        search_idact = np.concatenate([agent_pos, search_act], axis=0)
                        # print('s', t_neighbor_actions[search_id * self.idact_shape:(search_id + 1) * self.idact_shape])
                        # print(search_idact)
                        # t_neighbor_actions[search_id * self.idact_shape:(search_id + 1) * self.idact_shape] = search_idact
                        # print(t_neighbor_actions[search_id * self.idact_shape:(search_id + 1) * self.idact_shape])
                        # print(t_neighbor_actions)
                        t_inputs = np.hstack((inputs, search_idact))
                        t_inputs = torch.tensor(t_inputs, dtype=torch.float32).unsqueeze(0)
                        inputs_cuda = t_inputs.cuda()
                        q_value = self.policy.eval_rnn(inputs_cuda).squeeze()
                        # # version: add all max q
                        # max_q_index = torch.argmax(q_value)
                        # q_tot[max_q_index] += q_value[max_q_index]

                        # version add up to find true max q
                        # tmp_q_buffer.append(q_value)

                        # version find every dim max q
                        for act_index, act_dim_q in enumerate(q_value):
                            if act_dim_q > max_actdim_q_list[act_index]:
                                max_actdim_q_list[act_index] = act_dim_q

                    # version add up to find true max q
                    # tmp_q_buffer_list.append(tmp_q_buffer)

                    # version find every dim max q
                    q_tot += max_actdim_q_list.cuda()

                    #     # version: add only max q
                    #     t_max_q = torch.max(q_value)
                    #     if compare_max_q < t_max_q:
                    #         add_q_value = q_value
                    #         compare_max_q = t_max_q
                    # q_tot += add_q_value
                else:
                    search_idact = neighbor_actions[index]
                    t_inputs = np.hstack((inputs, search_idact))
                    t_inputs = torch.tensor(t_inputs, dtype=torch.float32).unsqueeze(0)
                    t_inputs = t_inputs.cuda()
                    q_value = self.policy.eval_rnn(t_inputs).squeeze()
                    q_tot += q_value

            # # version find real max q
            # q_tot = self.find_max_q(tmp_q_buffer_list, q_tot)
            # version find every dim max q
            # q_tot = self.find_max_q_edim(tmp_q_buffer_list, q_tot)

        if np.random.uniform() < epsilon:
            action = np.random.choice(avail_actions_ind)  # action是一个整数
        else:
            if isinstance(q_tot, np.ndarray):
                action = np.argmax(q_tot)
            else:
                action = torch.argmax(q_tot)
        return action, delta_max_q

    # def find_max_q_edim(self, tmp_q_buffer_list, q_tot):
    #     return_q_tot = torch.sum(tmp_q_buffer_list, dim=0).cuda() + q_tot
    #     return return_q_tot

    def find_max_q(self, tmp_q_buffer_list, q_tot):
        comapre_q_max = -10000
        for i in range(pow(self.n_actions, len(tmp_q_buffer_list))):
            index = []
            find_index_i = i
            while find_index_i / self.n_actions != 0:
                index.append(find_index_i % self.n_actions)
                find_index_i = int(find_index_i / self.n_actions)
            while len(index) < len(tmp_q_buffer_list):
                index.insert(0, 0)
            tmp_q_tot = torch.zeros(self.n_actions).cuda()
            for list_id, id in enumerate(index):
                tmp_q_tot += tmp_q_buffer_list[list_id][id]
            tmp_q_tot += q_tot
            if max(tmp_q_tot) > comapre_q_max:
                comapre_q_max = max(tmp_q_tot)
                return_q_tot = tmp_q_tot
        return return_q_tot

    def choose_fixed_action(self, obs, last_action, agent_num, avail_actions, epsilon, maven_z=None, evaluate=False):
        epsilon = 0
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.fixed_policy.eval_hidden[:, agent_num, :]

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q value
        # if self.args.alg == 'maven':
        #     maven_z = torch.tensor(maven_z, dtype=torch.float32).unsqueeze(0)
        #     if self.args.cuda:
        #         maven_z = maven_z.cuda()
        #     q_value, self.fixed_policy.eval_hidden[:, agent_num, :] = self.fixed_policy.eval_rnn(inputs, hidden_state,
        #                                                                                          maven_z)
        # else:
            # q_value, self.fixed_policy.eval_hidden[:, agent_num, :] = self.fixed_policy.eval_rnn(inputs, hidden_state)
        q_value = self.fixed_policy.eval_rnn(inputs)

        # choose action from q value
        if self.args.alg == 'coma' or self.args.alg == 'central_v' or self.args.alg == 'reinforce':
            action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon, evaluate)
        else:
            q_value[avail_actions == 0.0] = - float("inf")
            if np.random.uniform() < epsilon:
                action = np.random.choice(avail_actions_ind)  # action是一个整数
            else:
                action = torch.argmax(q_value)
        return action

    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False):
        """
        :param inputs: # q_value of all actions
        """
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[
            -1])  # num of avail_actions
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        # add noise of epsilon
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """

        if epsilon == 0 and evaluate:
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training
        # different episode has different length, so we need to get max length of the batch
        dq = 0
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        if self.args.use_ja:
            dq = self.policy.learn(batch, max_episode_len, train_step, epsilon)
        else:
            self.policy.learn(batch, max_episode_len, train_step, epsilon)
        # print(train_step, self.args.save_cycle)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)
        return dq


# Agent for communication
class CommAgents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        alg = args.alg
        if alg.find('reinforce') > -1:
            self.policy = Reinforce(args)
        elif alg.find('coma') > -1:
            self.policy = COMA(args)
        elif alg.find('central_v') > -1:
            self.policy = CentralV(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        print('Init CommAgents')

    # 根据weights得到概率，然后再根据epsilon选动作
    def choose_action(self, weights, avail_actions, epsilon, evaluate=False):
        weights = weights.unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # 可以选择的动作的个数
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(weights, dim=-1)
        # 在训练的时候给概率分布添加噪音
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """

        if epsilon == 0 and evaluate:
            # 测试时直接选最大的
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action

    def get_action_weights(self, obs, last_action):
        obs = torch.tensor(obs, dtype=torch.float32)
        last_action = torch.tensor(last_action, dtype=torch.float32)
        inputs = list()
        inputs.append(obs)
        # 给obs添加上一个动作、agent编号
        if self.args.last_action:
            inputs.append(last_action)
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents))
        inputs = torch.cat([x for x in inputs], dim=1)
        if self.args.cuda:
            inputs = inputs.cuda()
            self.policy.eval_hidden = self.policy.eval_hidden.cuda()
        weights, self.policy.eval_hidden = self.policy.eval_rnn(inputs, self.policy.eval_hidden)
        weights = weights.reshape(self.args.n_agents, self.args.n_actions)
        return weights.cpu()

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma在训练时也需要epsilon计算动作的执行概率
        # 每次学习时，各个episode的长度不一样，因此取其中最长的episode作为所有episode的长度
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)
