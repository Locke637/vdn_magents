import torch.nn as nn
import torch.nn.functional as f
import torch


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

    def __init__(self, input_shape, input_shape_view, input_shape_feature, args):
        super(ConvNet_MLP, self).__init__()
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
        self.fc2 = nn.Linear(args.rnn_hidden_dim+input_shape_feature, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs):
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
        # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim + self.input_shape_feature)
        h = f.relu(self.fc2(h))
        q = self.fc3(h)
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
