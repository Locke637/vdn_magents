import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time
from common.common import find_neighbor_pos, find_neighbor_id, find_pos_index


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.pos_action_shape = (args.id_dim + args.n_actions) * args.nei_n_agents
        self.args = args
        self.idact_shape = args.id_dim + args.n_actions

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        handles = self.env.get_handles()
        self.env.add_walls(method="random", n=self.n_agents * 2)
        self.env.add_agents(handles[0], method="random", n=self.n_agents)
        self.env.add_agents(handles[1], method="random", n=self.n_agents)
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        fixed_rewards = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)
        if self.args.use_fixed_model:
            self.agents.fixed_policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # sample z for maven
        if self.args.alg == 'maven':
            state = self.env.get_state()
            state = torch.tensor(state, dtype=torch.float32)
            if self.args.cuda:
                state = state.cuda()
            z_prob = self.agents.policy.z_policy(state)
            maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
            maven_z = list(maven_z.cpu())

        while not terminated and step < self.episode_limit:
            num_agents = self.env.get_num(handles[0])
            fixed_num_agents = self.env.get_num(handles[1])
            if num_agents < self.n_agents:
                self.env.add_agents(handles[0], method="random", n=self.n_agents - num_agents)
            if fixed_num_agents < self.n_agents:
                self.env.add_agents(handles[1], method="random", n=self.n_agents - fixed_num_agents)

            obs_all = self.env.get_observation(handles[0])
            fixed_obs_all = self.env.get_observation(handles[1])
            view = obs_all[0]
            feature = obs_all[1]
            fixed_view = fixed_obs_all[0]
            fixed_feature = fixed_obs_all[1]
            obs = []
            fixed_obs = []
            state = self.env.get_global_minimap(3, 3).flatten()

            for j in range(self.n_agents):
                obs.append(np.concatenate([view[j].flatten(), feature[j]]))
                fixed_obs.append(np.concatenate([fixed_view[j].flatten(), fixed_feature[j]]))
                # state = feature[j]
            # obs = self.env.get_obs()
            # state = self.env.get_state()
            actions, avail_actions, actions_onehot, fixed_actions = [], [], [], []
            for agent_id in range(self.n_agents):
                # avail_action = self.env.get_avail_agent_actions(agent_id)
                avail_action = np.ones(self.n_actions)
                if self.args.alg == 'maven':
                    action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon, maven_z, evaluate)
                else:
                    # st = time.time()
                    action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon, evaluate)
                    # print(time.time()-st)
                    if self.args.use_fixed_model:
                        fixed_action = self.agents.choose_fixed_action(fixed_obs[agent_id], last_action[agent_id],
                                                                       agent_id,
                                                                       avail_action, epsilon, evaluate)
                        if isinstance(fixed_action, np.int64):
                            fixed_action = fixed_action.astype(np.int32)
                        else:
                            fixed_action = fixed_action.cpu()
                            fixed_action = fixed_action.numpy().astype(np.int32)
                        fixed_actions.append(fixed_action)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                if isinstance(action, np.int64):
                    action = action.astype(np.int32)
                else:
                    action = action.cpu()
                    action = action.numpy().astype(np.int32)

                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            # reward, terminated, info = self.env.step(actions)
            acts = [[], []]
            acts[0] = np.array(actions)
            # print(actions)
            if self.args.use_fixed_model:
                acts[1] = np.array(fixed_actions)
            else:
                acts[1] = np.array(np.random.randint(0, self.n_actions, size=self.n_agents, dtype='int32'))
            self.env.set_action(handles[0], acts[0])
            self.env.set_action(handles[1], acts[1])
            terminated = self.env.step()
            reward = sum(self.env.get_reward(handles[0]))
            fixed_reward = sum(self.env.get_reward(handles[1]))
            self.env.clear_dead()
            if step == self.episode_limit - 1:
                terminated = 1.

            # win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            fixed_rewards += fixed_reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            # avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_action = np.ones(self.n_actions)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        if self.args.alg == 'maven':
            episode['z'] = np.array([maven_z.copy()])
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return episode, episode_reward, win_tag, fixed_rewards

    def generate_episode_ja(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        handles = self.env.get_handles()
        self.env.add_walls(method="random", n=self.n_agents * 2)
        self.env.add_agents(handles[0], method="random", n=self.n_agents)
        self.env.add_agents(handles[1], method="random", n=self.n_agents)
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        fixed_rewards = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)
        if self.args.use_fixed_model:
            self.agents.fixed_policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # sample z for maven
        if self.args.alg == 'maven':
            state = self.env.get_state()
            state = torch.tensor(state, dtype=torch.float32)
            if self.args.cuda:
                state = state.cuda()
            z_prob = self.agents.policy.z_policy(state)
            maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
            maven_z = list(maven_z.cpu())

        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)

            num_agents = self.env.get_num(handles[0])
            fixed_num_agents = self.env.get_num(handles[1])
            if num_agents < self.n_agents:
                self.env.add_agents(handles[0], method="random", n=self.n_agents - num_agents)
            if fixed_num_agents < self.n_agents:
                self.env.add_agents(handles[1], method="random", n=self.n_agents - fixed_num_agents)
            num_agents = self.env.get_num(handles[0])

            pos = self.env.get_pos(handles[0])
            neighbor_dic = find_neighbor(pos)
            # print(neighbor_dic)
            obs_all = self.env.get_observation(handles[0])
            fixed_obs_all = self.env.get_observation(handles[1])
            view = obs_all[0]
            feature = obs_all[1]
            fixed_view = fixed_obs_all[0]
            fixed_feature = fixed_obs_all[1]
            obs = []
            fixed_obs = []

            for j in range(self.n_agents):
                obs.append(np.concatenate([view[j].flatten(), feature[j]]))
                fixed_obs.append(np.concatenate([fixed_view[j].flatten(), fixed_feature[j]]))
                state = feature[j]
            # obs = self.env.get_obs()
            # state = self.env.get_state()
            actions, avail_actions, actions_onehot, fixed_actions = [], [], [], []
            for agent_id in range(self.n_agents):
                # avail_action = self.env.get_avail_agent_actions(agent_id)
                neighbor_clean_actions = []
                for act_index in range(self.n_agents):
                    if act_index < agent_id:
                        # agent_id_one_hot = np.zeros(self.n_agents)
                        # agent_id_one_hot[act_index] = 1
                        if act_index in neighbor_dic[agent_id]:
                            agent_id_one_hot = [act_index]
                            neighbor_clean_actions = np.concatenate(
                                (neighbor_clean_actions, agent_id_one_hot, actions_onehot[act_index]), axis=0)
                        else:
                            # agent_id_one_hot = np.zeros(self.n_agents)
                            agent_id_one_hot = [-2]
                            neighbor_clean_actions = np.concatenate(
                                (neighbor_clean_actions, agent_id_one_hot, np.zeros(self.n_actions)), axis=0)
                    elif act_index == agent_id:
                        # agent_id_one_hot = np.ones(self.n_agents)
                        agent_id_one_hot = [-1]
                        neighbor_clean_actions = np.concatenate(
                            (neighbor_clean_actions, agent_id_one_hot, np.zeros(self.n_actions)), axis=0)
                    else:
                        if act_index in neighbor_dic[agent_id]:
                            # agent_id_one_hot = np.zeros(self.n_agents)
                            # agent_id_one_hot[act_index] = 1
                            agent_id_one_hot = [act_index]
                            neighbor_clean_actions = np.concatenate(
                                (neighbor_clean_actions, agent_id_one_hot, np.zeros(self.n_actions)), axis=0)
                        else:
                            # agent_id_one_hot = np.zeros(self.n_agents)
                            agent_id_one_hot = [-2]
                            neighbor_clean_actions = np.concatenate(
                                (neighbor_clean_actions, agent_id_one_hot, np.zeros(self.n_actions)), axis=0)
                        # agent_id_one_hot = np.zeros(self.n_agents)
                        # agent_id_one_hot[act_index] = 1
                        # neighbor_clean_actions = np.concatenate(
                        #     (neighbor_clean_actions, agent_id_one_hot, np.zeros(self.n_actions)), axis=0)

                avail_action = np.ones(self.n_actions)
                if self.args.alg == 'maven':
                    action = self.agents.choose_action_ja(obs[agent_id], last_action[agent_id], agent_id,
                                                          avail_action, epsilon, maven_z, evaluate)
                else:
                    # print(neighbor_clean_actions)
                    # print(obs[0].shape)
                    # st = time.time()
                    action = self.agents.choose_action_ja(obs[agent_id], neighbor_clean_actions, last_action[agent_id],
                                                          agent_id,
                                                          avail_action, epsilon, evaluate)
                    # print(time.time() - st)
                    if self.args.use_fixed_model:
                        fixed_action = self.agents.choose_fixed_action(fixed_obs[agent_id], last_action[agent_id],
                                                                       agent_id,
                                                                       avail_action, epsilon, evaluate)
                        if isinstance(fixed_action, np.int64):
                            fixed_action = fixed_action.astype(np.int32)
                        else:
                            fixed_action = fixed_action.cpu()
                            fixed_action = fixed_action.numpy().astype(np.int32)
                        fixed_actions.append(fixed_action)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                if isinstance(action, np.int64):
                    action = action.astype(np.int32)
                else:
                    action = action.cpu()
                    action = action.numpy().astype(np.int32)

                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            # reward, terminated, info = self.env.step(actions)
            acts = [[], []]
            acts[0] = np.array(actions)

            # print(actions)
            if self.args.use_fixed_model:
                acts[1] = np.array(fixed_actions)
            else:
                acts[1] = np.array(np.random.randint(0, self.n_actions, size=self.n_agents, dtype='int32'))
            self.env.set_action(handles[0], acts[0])
            self.env.set_action(handles[1], acts[1])
            terminated = self.env.step()
            reward = sum(self.env.get_reward(handles[0]))
            fixed_reward = sum(self.env.get_reward(handles[1]))
            self.env.clear_dead()
            if step == self.episode_limit - 1:
                terminated = 1.

            # win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            for agent_id in range(self.n_agents):
                real_ja = []
                for act_index in range(self.n_agents):
                    if act_index == agent_id:
                        # agent_id_one_hot = np.ones(self.n_agents)
                        agent_id_one_hot = [-1]
                        real_ja = np.concatenate((real_ja, agent_id_one_hot, np.zeros(self.n_actions)), axis=0)
                    else:
                        # agent_id_one_hot = np.zeros(self.n_agents)
                        # agent_id_one_hot[act_index] = 1
                        if act_index in neighbor_dic[agent_id]:
                            agent_id_one_hot = [act_index]
                            real_ja = np.concatenate((real_ja, agent_id_one_hot, actions_onehot[act_index]), axis=0)
                        else:
                            agent_id_one_hot = [-2]
                            # agent_id_one_hot = np.zeros(self.n_agents)
                            real_ja = np.concatenate((real_ja, agent_id_one_hot, np.zeros(self.n_actions)), axis=0)
                obs[agent_id] = np.concatenate((obs[agent_id], real_ja), axis=0)
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            fixed_rewards += fixed_reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            # avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_action = np.ones(self.n_actions)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        if self.args.alg == 'maven':
            episode['z'] = np.array([maven_z.copy()])
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return episode, episode_reward, win_tag, fixed_rewards

    def generate_episode_ja_v2(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded, n_id = [], [], [], [], [], [], [], [], []
        self.env.reset()
        handles = self.env.get_handles()
        self.env.add_walls(method="random", n=self.n_agents * 2)
        self.env.add_agents(handles[0], method="random", n=self.n_agents)
        self.env.add_agents(handles[1], method="random", n=self.n_agents)
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        fixed_rewards = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)
        if self.args.use_fixed_model:
            self.agents.fixed_policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # sample z for maven
        if self.args.alg == 'maven':
            state = self.env.get_state()
            state = torch.tensor(state, dtype=torch.float32)
            if self.args.cuda:
                state = state.cuda()
            z_prob = self.agents.policy.z_policy(state)
            maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
            maven_z = list(maven_z.cpu())

        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)

            num_agents = self.env.get_num(handles[0])
            fixed_num_agents = self.env.get_num(handles[1])
            if num_agents < self.n_agents:
                self.env.add_agents(handles[0], method="random", n=self.n_agents - num_agents)
            if fixed_num_agents < self.n_agents:
                self.env.add_agents(handles[1], method="random", n=self.n_agents - fixed_num_agents)
            # num_agents = self.env.get_num(handles[0])

            obs_all = self.env.get_observation(handles[0])
            pos = self.env.get_pos(handles[0])
            # print(pos)
            neighbor_dic = find_neighbor_id(pos)
            # print(neighbor_dic)

            fixed_obs_all = self.env.get_observation(handles[1])
            view = obs_all[0]
            feature = obs_all[1]
            fixed_view = fixed_obs_all[0]
            fixed_feature = fixed_obs_all[1]
            obs = []
            fixed_obs = []

            for j in range(self.n_agents):
                obs.append(np.concatenate([view[j].flatten(), feature[j]]))
                fixed_obs.append(np.concatenate([fixed_view[j].flatten(), fixed_feature[j]]))
                state = feature[j]
            # obs = self.env.get_obs()
            # state = self.env.get_state()
            actions, avail_actions, actions_onehot, fixed_actions = [], [], [], []
            for agent_id in range(self.n_agents):
                # avail_action = self.env.get_avail_agent_actions(agent_id)
                neighbor_clean_actions = []
                need_search_neighbor = []
                for act_index in range(self.n_agents):
                    if act_index < agent_id:
                        if act_index in neighbor_dic[agent_id]:
                            # agent_id_one_hot = [act_index]
                            agent_id_one_hot = np.zeros(self.n_agents)
                            agent_id_one_hot[act_index] = 1
                            neighbor_clean_actions = np.concatenate(
                                (neighbor_clean_actions, agent_id_one_hot, actions_onehot[act_index]), axis=0)
                        else:
                            agent_id_one_hot = np.zeros(self.n_agents)
                            # agent_id_one_hot = [-2]
                            neighbor_clean_actions = np.concatenate(
                                (neighbor_clean_actions, agent_id_one_hot, np.zeros(self.n_actions)), axis=0)
                    elif act_index == agent_id:
                        agent_id_one_hot = np.zeros(self.n_agents)
                        # agent_id_one_hot = [-1]
                        neighbor_clean_actions = np.concatenate(
                            (neighbor_clean_actions, agent_id_one_hot, np.zeros(self.n_actions)), axis=0)
                    else:
                        if act_index in neighbor_dic[agent_id]:
                            agent_id_one_hot = np.zeros(self.n_agents)
                            agent_id_one_hot[act_index] = 1
                            # agent_id_one_hot = [act_index]
                            neighbor_clean_actions = np.concatenate(
                                (neighbor_clean_actions, agent_id_one_hot, np.zeros(self.n_actions)), axis=0)
                            need_search_neighbor.append(act_index)
                        else:
                            agent_id_one_hot = np.zeros(self.n_agents)
                            # agent_id_one_hot = [-2]
                            neighbor_clean_actions = np.concatenate(
                                (neighbor_clean_actions, agent_id_one_hot, np.zeros(self.n_actions)), axis=0)
                # need_search_neighbor.append(1)

                avail_action = np.ones(self.n_actions)
                if self.args.alg == 'maven':
                    action = self.agents.choose_action_ja(obs[agent_id], last_action[agent_id], agent_id,
                                                          avail_action, epsilon, maven_z, evaluate)
                else:
                    # print(neighbor_clean_actions)
                    # print(obs[0].shape)
                    # st = time.time()
                    action = self.agents.choose_action_ja_v2(obs[agent_id], neighbor_clean_actions,
                                                             need_search_neighbor, last_action[agent_id], agent_id,
                                                             avail_action, epsilon, evaluate)
                    # print(time.time() - st)
                    if self.args.use_fixed_model:
                        fixed_action = self.agents.choose_fixed_action(fixed_obs[agent_id], last_action[agent_id],
                                                                       agent_id,
                                                                       avail_action, epsilon, evaluate)
                        if isinstance(fixed_action, np.int64):
                            fixed_action = fixed_action.astype(np.int32)
                        else:
                            fixed_action = fixed_action.cpu()
                            fixed_action = fixed_action.numpy().astype(np.int32)
                        fixed_actions.append(fixed_action)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                if isinstance(action, np.int64):
                    action = action.astype(np.int32)
                else:
                    action = action.cpu()
                    action = action.numpy().astype(np.int32)

                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            # reward, terminated, info = self.env.step(actions)
            acts = [[], []]
            acts[0] = np.array(actions)

            # print(actions)
            if self.args.use_fixed_model:
                acts[1] = np.array(fixed_actions)
            else:
                acts[1] = np.array(np.random.randint(0, self.n_actions, size=self.n_agents, dtype='int32'))
            self.env.set_action(handles[0], acts[0])
            self.env.set_action(handles[1], acts[1])
            terminated = self.env.step()
            reward = sum(self.env.get_reward(handles[0]))
            fixed_reward = sum(self.env.get_reward(handles[1]))
            self.env.clear_dead()
            if step == self.episode_limit - 1:
                terminated = 1.

            # win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            for agent_id in range(self.n_agents):
                real_ja = []
                for act_index in range(self.n_agents):
                    if act_index == agent_id:
                        agent_id_one_hot = np.zeros(self.n_agents)
                        # agent_id_one_hot = [-1]
                        real_ja = np.concatenate((real_ja, agent_id_one_hot, np.zeros(self.n_actions)), axis=0)
                    else:
                        if act_index in neighbor_dic[agent_id]:
                            agent_id_one_hot = np.zeros(self.n_agents)
                            agent_id_one_hot[act_index] = 1
                            real_ja = np.concatenate((real_ja, agent_id_one_hot, actions_onehot[act_index]), axis=0)
                        else:
                            agent_id_one_hot = np.zeros(self.n_agents)
                            # agent_id_one_hot = np.zeros(self.n_agents)
                            real_ja = np.concatenate((real_ja, agent_id_one_hot, np.zeros(self.n_actions)), axis=0)
                obs[agent_id] = np.concatenate((obs[agent_id], real_ja), axis=0)

            if self.args.use_dqloss:
                neighbor_ids = np.zeros([self.n_agents, self.n_agents])
                for i in range(self.n_agents):
                    id_list = neighbor_dic[i]
                    id_list_len = len(id_list)
                    if id_list:
                        for id_temp in id_list:
                            neighbor_ids[i][id_temp] = 1
                        neighbor_ids[i][i] = -id_list_len
                n_id.append(neighbor_ids)

            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            fixed_rewards += fixed_reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        o.append(obs)
        s.append(state)
        # n_id.append(neighbor_dic)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            # avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_action = np.ones(self.n_actions)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])
            if self.args.use_dqloss:
                n_id.append(np.zeros((self.n_agents, self.n_agents)))

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       # neighbor_ids=n_id.copy()
                       )
        if self.args.use_dqloss:
            episode['neighbor_ids'] = n_id.copy()
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        if self.args.alg == 'maven':
            episode['z'] = np.array([maven_z.copy()])
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return episode, episode_reward, win_tag, fixed_rewards

    def generate_episode_ja_v3(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        handles = self.env.get_handles()
        self.env.add_walls(method="random", n=self.n_agents * 2)
        self.env.add_agents(handles[0], method="random", n=self.n_agents)
        self.env.add_agents(handles[1], method="random", n=self.n_agents)
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        fixed_rewards = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)
        if self.args.use_fixed_model:
            self.agents.fixed_policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # sample z for maven
        if self.args.alg == 'maven':
            state = self.env.get_state()
            state = torch.tensor(state, dtype=torch.float32)
            if self.args.cuda:
                state = state.cuda()
            z_prob = self.agents.policy.z_policy(state)
            maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
            maven_z = list(maven_z.cpu())

        while not terminated and step < self.episode_limit:
            max_q_infer_actions = {}
            tot_delta_max_q = 0
            need_search_neighbor_dic = {}
            num_agents = self.env.get_num(handles[0])
            fixed_num_agents = self.env.get_num(handles[1])
            if num_agents < self.n_agents:
                self.env.add_agents(handles[0], method="random", n=self.n_agents - num_agents)
            if fixed_num_agents < self.n_agents:
                self.env.add_agents(handles[1], method="random", n=self.n_agents - fixed_num_agents)

            obs_all = self.env.get_observation(handles[0])
            pos = self.env.get_pos(handles[0])
            neighbor_dic, neighbor_pos = find_neighbor_pos(pos)
            fixed_obs_all = self.env.get_observation(handles[1])
            view = obs_all[0]
            feature = obs_all[1]
            fixed_view = fixed_obs_all[0]
            fixed_feature = fixed_obs_all[1]
            obs = []
            fixed_obs = []
            for j in range(self.n_agents):
                obs.append(np.concatenate([view[j].flatten(), feature[j]]))
                fixed_obs.append(np.concatenate([fixed_view[j].flatten(), fixed_feature[j]]))
                state = feature[j]

            actions, avail_actions, actions_onehot, fixed_actions, n_id, real_ja_all = [], [], [], [], [], []
            for agent_id in range(self.n_agents):
                neighbor_clean_actions = {}
                need_search_neighbor = []
                for act_index in range(self.n_agents):
                    if act_index < agent_id:
                        if act_index in neighbor_dic[agent_id]:
                            agent_pos_index = neighbor_dic[agent_id].index(act_index)
                            agent_pos = neighbor_pos[agent_id][agent_pos_index]
                            neighbor_clean_actions[act_index] = np.concatenate((agent_pos, actions_onehot[act_index]),
                                                                               axis=0)
                        # else:
                        #     agent_pos = np.zeros(2)
                        #     neighbor_clean_actions = np.concatenate(
                        #         (neighbor_clean_actions, agent_pos, np.zeros(self.n_actions)), axis=0)
                    # elif act_index == agent_id:
                    #     agent_pos = np.zeros(2)
                    #     neighbor_clean_actions = np.concatenate(
                    #         (neighbor_clean_actions, agent_pos, np.zeros(self.n_actions)), axis=0)
                    else:
                        if act_index in neighbor_dic[agent_id]:
                            agent_pos_index = neighbor_dic[agent_id].index(act_index)
                            agent_pos = neighbor_pos[agent_id][agent_pos_index]
                            neighbor_clean_actions[act_index] = np.concatenate((agent_pos, np.zeros(self.n_actions)),
                                                                               axis=0)
                            need_search_neighbor.append(act_index)
                        # else:
                        #     agent_pos = np.zeros(2)
                        #     neighbor_clean_actions = np.concatenate(
                        #         (neighbor_clean_actions, agent_pos, np.zeros(self.n_actions)), axis=0)

                # print(neighbor_dic)
                # print(neighbor_pos)
                # print(need_search_neighbor, agent_id)
                # print(neighbor_clean_actions)
                avail_action = np.ones(self.n_actions)
                if self.args.alg == 'maven':
                    action = self.agents.choose_action_ja(obs[agent_id], last_action[agent_id], agent_id,
                                                          avail_action, epsilon, maven_z, evaluate)
                else:
                    action, delta_max_q = self.agents.choose_action_ja_vd(obs[agent_id], neighbor_clean_actions,
                                                                          neighbor_pos[agent_id],
                                                                          need_search_neighbor,
                                                                          last_action[agent_id], agent_id,
                                                                          avail_action, epsilon, evaluate)
                    if self.args.use_fixed_model:
                        fixed_action = self.agents.choose_fixed_action(fixed_obs[agent_id], last_action[agent_id],
                                                                       agent_id,
                                                                       avail_action, epsilon, evaluate)
                        if isinstance(fixed_action, np.int64):
                            fixed_action = fixed_action.astype(np.int32)
                        else:
                            fixed_action = fixed_action.cpu()
                            fixed_action = fixed_action.numpy().astype(np.int32)
                        fixed_actions.append(fixed_action)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                if isinstance(action, np.int64):
                    action = action.astype(np.int32)
                else:
                    action = action.cpu()
                    action = action.numpy().astype(np.int32)

                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot
                # max_q_infer_actions[agent_id] = max_q_index_dic
                tot_delta_max_q += delta_max_q
                need_search_neighbor_dic[agent_id] = need_search_neighbor

            # count = 0
            # tot_len = sum([len(need_search_neighbor_dic[k]) for k in range(self.n_agents)])
            # if tot_len:
            #     for test_i in range(self.n_agents):
            #         if max_q_infer_actions[test_i]:
            #             for test_id in need_search_neighbor_dic[test_i]:
            #                 infer_act = max_q_infer_actions[test_i][test_id]
            #                 true_act = actions[test_id]
            #                 if infer_act == true_act:
            #                     count += 1
            #     infer_rate += count
            #     infer_step += 1

            acts = [[], []]
            acts[0] = np.array(actions)

            if self.args.use_fixed_model:
                acts[1] = np.array(fixed_actions)
            else:
                acts[1] = np.array(np.random.randint(0, self.n_actions, size=self.n_agents, dtype='int32'))
            self.env.set_action(handles[0], acts[0])
            self.env.set_action(handles[1], acts[1])
            terminated = self.env.step()
            reward = sum(self.env.get_reward(handles[0]))
            fixed_reward = sum(self.env.get_reward(handles[1]))
            self.env.clear_dead()
            if step == self.episode_limit - 1:
                terminated = 1.

            for agent_id in range(self.n_agents):
                real_ja = []
                for act_index in range(self.n_agents):
                    if act_index in neighbor_dic[agent_id]:
                        agent_pos_index = neighbor_dic[agent_id].index(act_index)
                        agent_pos = neighbor_pos[agent_id][agent_pos_index]
                        real_ja = np.concatenate((real_ja, agent_pos, actions_onehot[act_index]), axis=0)
                    else:
                        agent_pos = np.zeros(2)
                        real_ja = np.concatenate((real_ja, agent_pos, np.zeros(self.n_actions)), axis=0)
                real_ja_all.append(real_ja)
                # obs[agent_id] = np.concatenate((obs[agent_id], real_ja), axis=0)

            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            # if self.args.use_dqloss:
            #     n_id.append(np.zeros((self.n_agents, self.n_agents)))
            if self.args.use_ja:
                neighbor_ids = np.zeros([self.n_agents, self.n_agents])
                for i in range(self.n_agents):
                    id_list = neighbor_dic[i]
                    id_list_len = len(id_list)
                    if id_list:
                        for id_temp in id_list:
                            neighbor_ids[i][id_temp] = 1
                        neighbor_ids[i][i] = -id_list_len
                n_id.append(neighbor_ids)
            episode_reward += reward
            fixed_rewards += fixed_reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            # avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_action = np.ones(self.n_actions)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])
            if self.args.use_ja:
                n_id.append(np.zeros((self.n_agents, self.n_agents)))
                real_ja_all.append(np.zeros((self.n_agents, self.n_agents * (self.args.id_dim + self.args.n_actions))))

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       # neighbor_ids=n_id.copy()
                       )
        if self.args.use_ja:
            episode['neighbor_ids'] = n_id.copy()
            episode['neighbor_idacts'] = real_ja_all.copy()
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        if self.args.alg == 'maven':
            episode['z'] = np.array([maven_z.copy()])
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        # if infer_step:
        #     win_tag = infer_rate / infer_step
        # else:
        #     win_tag = 0
        win_tag = tot_delta_max_q
        return episode, episode_reward, win_tag, fixed_rewards


# RolloutWorker for communication
class CommRolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init CommRolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []

            # get the weights of all actions for all agents
            weights = self.agents.get_action_weights(np.array(obs), last_action)

            # choose action for each agent
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.agents.choose_action(weights[agent_id], avail_action, epsilon, evaluate)

                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            # if terminated:
            #     time.sleep(1)
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
            # print('Epsilon is ', self.epsilon)
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return episode, episode_reward, win_tag
