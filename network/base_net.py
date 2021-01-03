import torch.nn as nn
import torch.nn.functional as f
import torch
import time


class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, input_shape_view, input_shape_feature, args):
        super(RNN, self).__init__()
        self.args = args
        self.input_shape_view = input_shape_view
        self.input_shape_feature = input_shape_feature

        self.fc1 = nn.Linear(input_shape_view, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim + input_shape_feature, args.rnn_hidden_dim + input_shape_feature)
        self.fc2 = nn.Linear(args.rnn_hidden_dim + input_shape_feature, args.n_actions)

    def forward(self, obs, hidden_state):
        view = obs[:, :self.input_shape_view]
        feature = obs[:, -self.input_shape_feature:]

        # x = f.relu(self.fc1(obs))
        x = f.relu(self.fc1(view))
        h = torch.cat((x, feature), dim=1)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim + self.input_shape_feature)
        h = self.rnn(h, h_in)
        q = self.fc2(h)
        return q, h


class MLP(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape_view, input_shape_feature, args):
        super(MLP, self).__init__()
        self.args = args
        self.input_shape_view = input_shape_view
        self.input_shape_feature = input_shape_feature

        self.fc1 = nn.Linear(input_shape_view, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim + input_shape_feature, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs):
        # print(obs)
        view = obs[:self.input_shape_view]
        feature = obs[self.input_shape_feature:]
        x = f.relu(self.fc1(view))
        h = torch.cat((x, feature), dim=0)
        # print(self.fc1(obs))
        h = f.relu(self.fc2(h))
        q = self.fc3(h)
        # print(q)
        return q


class ConvNet_RNN(nn.Module):

    def __init__(self, input_shape, input_shape_view, input_shape_feature, args):
        super(ConvNet_RNN, self).__init__()
        self.args = args
        self.input_shape_view = input_shape_view
        self.input_shape_feature = input_shape_feature

        self.layer1 = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(4032, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim + input_shape_feature, args.rnn_hidden_dim + input_shape_feature)
        self.fc2 = nn.Linear(args.rnn_hidden_dim + input_shape_feature, args.n_actions)

    def forward(self, obs, hidden_state):
        view = obs[:, :self.input_shape_view]
        # print(view.shape)
        view = view.view(-1, 10, 10, 5)
        feature = obs[:, -self.input_shape_feature:]

        # x = f.relu(self.fc1(obs))
        out = self.layer1(view)
        out = self.layer2(out)
        out = out.reshape(-1, 4032)
        # print(out.size())
        x = f.relu(self.fc1(out))
        h = torch.cat((x, feature), dim=1)
        # print(h.size())
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim + self.input_shape_feature)
        h = self.rnn(h, h_in)
        q = self.fc2(h)
        return q, h


class ConvNet_MLP(nn.Module):

    def __init__(self, real_shape_view, input_shape_view, input_shape_feature, args):
        super(ConvNet_MLP, self).__init__()
        self.args = args
        self.input_shape_view = input_shape_view
        self.input_shape_feature = input_shape_feature
        self.real_shape_view = real_shape_view

        self.layer1 = nn.Sequential(
            nn.Conv2d(real_shape_view[0], 32, kernel_size=3, stride=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(512, args.mlp_hidden_dim[0])
        self.fc2 = nn.Linear(args.mlp_hidden_dim[0] + input_shape_feature, args.mlp_hidden_dim[0])
        self.fc3 = nn.Linear(args.mlp_hidden_dim[0], args.n_actions)

    def forward(self, obs):
        # st = time.time()
        # print(self.input_shape_view)
        view = obs[:, :self.input_shape_view]
        # print(view.shape)
        view = view.view(-1, self.real_shape_view[0], self.real_shape_view[1], self.real_shape_view[2])
        feature = obs[:, -self.input_shape_feature:]

        # x = f.relu(self.fc1(obs))
        out = self.layer1(view)
        out = self.layer2(out)
        out = out.reshape(-1, 512)
        # print(out.size())
        x = f.relu(self.fc1(out))
        h = torch.cat((x, feature), dim=1)
        # print(h.size())
        # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim + self.input_shape_feature)
        h = f.relu(self.fc2(h))
        q = self.fc3(h)
        # print(time.time() - st)

        # h = torch.cat((x, feature), dim=1)
        # # print(h.size())
        # # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim + self.input_shape_feature)
        # h = f.relu(self.fc2(h))
        # q = self.fc3(h)
        return q
    #
    # def __init__(self,, input_shape_view, input_shape_feature, args):
    #     super(ConvNet, self).__init__()
    #     self.layer1 = nn.Sequential(
    #         nn.Conv2d(input_shape_view, 32, kernel_size=3, stride=1, padding=2),
    #         nn.BatchNorm2d(16),
    #         nn.ReLU(),
    #         )
    #
    #     self.layer2 = nn.Sequential(
    #         nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),
    #         nn.BatchNorm2d(32),
    #         nn.ReLU(),
    #         )
    #
    #     self.fc1 = nn.Linear(7 * 7 * 32, num_classes)
    #     # 前馈网络过程
    #
    # def forward(self, x):
    #     out = self.layer1(x)
    #     out = self.layer2(out)
    #     out = out.reshape(out.size(0), -1)
    #     out = self.fc(out)
    #     return out


class ConvNet_MLP_Ja(nn.Module):

    def __init__(self, real_shape_view, input_shape_view, input_shape_feature, args):
        super(ConvNet_MLP_Ja, self).__init__()
        self.args = args
        self.input_shape_view = input_shape_view
        self.input_shape_feature = input_shape_feature
        self.real_shape_view = real_shape_view
        self.len_idact = args.id_dim + args.act_dim
        self.input_len_idact = 1 + args.act_dim
        # print(self.len_idact, args.id_dim, args.act_dim)
        self.len_id = args.id_dim
        self.len_act = args.act_dim
        self.neighbor_actions_view = self.input_len_idact * args.n_agents
        self.agents_num = args.n_agents

        self.layer1 = nn.Sequential(
            nn.Conv2d(real_shape_view[0], 32, kernel_size=3, stride=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(192, args.rnn_hidden_dim)
        # print(args.rnn_hidden_dim + input_shape_feature + self.len_idact)
        self.fc2 = nn.Linear(args.rnn_hidden_dim + input_shape_feature + self.len_idact, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs):
        # st = time.time()
        # print(obs[0].shape)
        # q = torch.zeros(self.len_act)
        tot_local_q = []
        view = obs[:, :self.input_shape_view]
        # print(view.shape)
        view = view.view(-1, self.real_shape_view[0], self.real_shape_view[1], self.real_shape_view[2])
        feature = obs[:, self.input_shape_view:self.input_shape_view + self.input_shape_feature]
        neighbor_action = obs[:, -self.neighbor_actions_view:]
        # feature_ja = []

        # x = f.relu(self.fc1(obs))
        out = self.layer1(view)
        out = self.layer2(out)
        out = out.reshape(-1, 192)
        # print(out.size())
        x = f.relu(self.fc1(out))
        self_id = torch.zeros(self.len_id).cuda()
        search_act = torch.eye(self.len_act).cuda()
        all_id = torch.eye(self.len_id).cuda()
        x_ja_all = []
        tot_local_q = []
        # st = time.time()
        # t_t = time.time()
        # print('tt', t_t - st)

        for index in range(len(obs)):
            # print(self.neighbor_actions_view)
            # x_ja = torch.unsqueeze(x_ja, 1)
            # print(act_index)
            x_ja = []
            for act_index in range(self.agents_num):
                temp_idact = neighbor_action[index,
                             act_index * self.input_len_idact:(act_index + 1) * self.input_len_idact]
                temp_id = temp_idact[:1]
                temp_act = temp_idact[-self.len_act:]
                # print(temp_idact)
                # print(temp_id)
                # st1 = time.time()
                # print(temp_id)
                # print(temp_id, temp_id[0][0], temp_id[0][0] >= -1)
                if temp_id[0] >= -1:
                    # print('j', time.time() - st1)
                    if temp_id[0] == -1:
                        # st2 = time.time()
                        # t_idact = torch.cat([t_id.cuda(), temp_act[index]], dim=0)
                        # print(feature[index], t_idact)
                        # feature_ja.append(torch.cat((feature[index], t_idact), dim=0))

                        t_x = torch.cat((x[index], feature[index], self_id, temp_act), dim=0)
                        t_x = torch.unsqueeze(t_x, 0)
                        if isinstance(x_ja, list):
                            x_ja = t_x
                        else:
                            x_ja = torch.cat([x_ja, t_x], dim=0)
                        # print('p1', time.time() - st2)

                        # print(feature_ja)
                    else:
                        # st3 = time.time()
                        if not any(temp_act):
                            # label = torch.tensor(range(self.len_act))
                            # search_act = torch.zeros(self.len_act, self.len_act).scatter_(1, label, 1)
                            # print(search_act)
                            # t_feature = []
                            for s_act in search_act:
                                t_idact = torch.cat((all_id[temp_id[0].cpu().numpy()], s_act))
                                # t_feature = torch.cat((feature[index], t_idact))
                                t_x = torch.cat((x[index], feature[index], t_idact), dim=0)
                                t_x = torch.unsqueeze(t_x, 0)
                                if isinstance(x_ja, list):
                                    x_ja = t_x
                                else:
                                    x_ja = torch.cat([x_ja, t_x], dim=0)
                            # self.find_max_feature(x[index], t_feature)
                        else:
                            # t_idact = temp_idact[index]
                            # print(all_id[temp_id], temp_id)
                            t_idact = torch.cat((all_id[temp_id[0].cpu().numpy()], temp_act))
                            # feature_ja.append(torch.cat((feature[index], t_idact), dim=0))
                            t_x = torch.cat((x[index], feature[index], t_idact), dim=0)
                            t_x = torch.unsqueeze(t_x, 0)
                            if isinstance(x_ja, list):
                                x_ja = t_x
                            else:
                                x_ja = torch.cat([x_ja, t_x], dim=0)
                        # print('p2', time.time() - st3)
                # else:
                #     # gt = time.time()
                #     # print('gt', gt - st1)
                #     pass

                # print('now', time.time() - st1)
                # print(x_ja_all, x_ja)
            # if isinstance(x_ja_all, list):
            #     x_ja_all = x_ja
            # else:
            #     if not isinstance(x_ja, list):
            #         x_ja_all = torch.cat([x_ja_all, x_ja], dim=0)

            h = f.relu(self.fc2(x_ja))
            q = self.fc3(h)
            q = torch.sum(q, dim=0).unsqueeze(0)
            # print(q.size())
            if isinstance(tot_local_q, list):
                tot_local_q = q
            else:
                tot_local_q = torch.cat([tot_local_q, q], dim=0)

            # h = torch.cat((x, feature), dim=1)
            # print(x_ja.size())
            # x_ja_tensor = torch.tensor(x_ja)
            # x_ja = torch.tensor(x_ja.cuda())
        # nc_t = time.time()
        # print('nc:', nc_t - st)
        # print(x_ja_all.size())

        # print(q.size())
        # if isinstance(tot_local_q, list):
        #     tot_local_q = q
        # else:
        #     tot_local_q = torch.cat([tot_local_q, q], dim=0)

        # ft = time.time()
        # print('ft', ft - t_t)
        # print(q)
        # print(tot_local_q)
        # print(tot_local_q)

        # print(q)
        # print(tot_local_q)
        return tot_local_q

    # def find_max_feature(self, x, t_feature):
    #     q = []
    #     for f in t_feature:
    #         h = torch.cat((x, f), dim=1)
    #         h = f.relu(self.fc2(h))
    #         q.append(self.fc3(h))
    #     return torch.argmax(q)


class ConvNet_MLP_Ja_v2(nn.Module):

    def __init__(self, real_shape_view, input_shape_view, input_shape_feature, args):
        super(ConvNet_MLP_Ja_v2, self).__init__()
        self.args = args
        self.input_shape_view = input_shape_view
        self.input_shape_feature = input_shape_feature
        self.real_shape_view = real_shape_view
        # self.len_idact = args.id_dim + args.act_dim
        self.input_len_idact = args.id_dim + args.act_dim
        # print(self.len_idact, args.id_dim, args.act_dim)
        self.len_id = args.id_dim
        self.len_act = args.act_dim
        self.neighbor_actions_view = self.input_len_idact
        self.agents_num = args.n_agents

        self.layer1 = nn.Sequential(
            nn.Conv2d(real_shape_view[0], 32, kernel_size=3, stride=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # self.fc1 = nn.Linear(512, args.mlp_hidden_dim[0])
        # # print(args.rnn_hidden_dim + input_shape_feature + self.neighbor_actions_view)
        # self.fc2 = nn.Linear(args.mlp_hidden_dim[0] + input_shape_feature, args.mlp_hidden_dim[0])
        # self.fc3 = nn.Linear(args.mlp_hidden_dim[0] + self.neighbor_actions_view, args.mlp_hidden_dim[1])
        # self.fc4 = nn.Linear(args.mlp_hidden_dim[1], args.n_actions)

        self.fc1 = nn.Linear(512, args.mlp_hidden_dim[0])
        # print(args.rnn_hidden_dim + input_shape_feature + self.neighbor_actions_view)
        self.fc2 = nn.Linear(args.mlp_hidden_dim[0] + input_shape_feature + self.neighbor_actions_view,
                             args.mlp_hidden_dim[1])
        self.fc3 = nn.Linear(args.mlp_hidden_dim[1], args.n_actions)

    def forward(self, obs):
        # print(self.input_shape_view)
        # print(obs)
        view = obs[:, :self.input_shape_view]
        view = view.view(-1, self.real_shape_view[0], self.real_shape_view[1], self.real_shape_view[2])
        feature_w_neighbor_action = obs[:, self.input_shape_view:]
        # feature = obs[:, self.input_shape_view:self.input_shape_view + self.input_shape_feature]
        # neighbor_action = obs[:, self.input_shape_view + self.input_shape_feature:]
        # print(feature_w_neighbor_action.size())
        # feature = obs[:, self.input_shape_view:self.input_shape_view + self.input_shape_feature]
        # neighbor_action = obs[:, -self.neighbor_actions_view:]

        out = self.layer1(view)
        out = self.layer2(out)
        out = out.reshape(-1, 512)
        x = f.relu(self.fc1(out))
        h = torch.cat((x, feature_w_neighbor_action), dim=1)
        h = f.relu(self.fc2(h))
        q = self.fc3(h)
        # print(h.size())
        # h = torch.cat((x, feature), dim=1)
        # h = f.relu(self.fc2(h))
        # h = torch.cat((h, neighbor_action), dim=1)
        # h = f.relu(self.fc3(h))
        # q = self.fc4(h)

        return q


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Critic of Central-V
class Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, 1)

    def forward(self, inputs):
        x = f.relu(self.fc1(inputs))
        x = f.relu(self.fc2(x))
        q = self.fc3(x)
        return q
