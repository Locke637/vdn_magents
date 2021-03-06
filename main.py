from runner import Runner
# from smac.env import StarCraft2Env
import magent
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, \
    get_commnet_args, get_g2anet_args


def get_config_pursuit(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})

    predator = cfg.register_agent_type(
        "predator",
        {
            'width': 2, 'length': 2, 'hp': 1, 'speed': 1,
            'view_range': gw.CircleRange(5), 'attack_range': gw.CircleRange(2),
            'attack_penalty': -0.2
        })

    prey = cfg.register_agent_type(
        "prey",
        {
            'width': 1, 'length': 1, 'hp': 1, 'speed': 1.5,
            'view_range': gw.CircleRange(4), 'attack_range': gw.CircleRange(0)
        })

    predator_group = cfg.add_group(predator)
    prey_group = cfg.add_group(prey)

    a = gw.AgentSymbol(predator_group, index='any')
    b = gw.AgentSymbol(prey_group, index='any')

    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=[a, b], value=[1, -1])

    return cfg


def get_config_double_attack(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"embedding_size": 10})

    deer = cfg.register_agent_type(
        "deer",
        {'width': 1, 'length': 1, 'hp': 5, 'speed': 1,
         'view_range': gw.CircleRange(1), 'attack_range': gw.CircleRange(0),
         'step_recover': 0.2,
         'kill_supply': 8,
         })

    tiger = cfg.register_agent_type(
        "tiger",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 1,
         'view_range': gw.CircleRange(4), 'attack_range': gw.CircleRange(1),
         'damage': 1, 'step_recover': -0.2,
         })

    tiger_group = cfg.add_group(tiger)
    deer_group = cfg.add_group(deer)

    a = gw.AgentSymbol(tiger_group, index='any')
    b = gw.AgentSymbol(tiger_group, index='any')
    c = gw.AgentSymbol(deer_group, index='any')

    # tigers get reward when they attack a deer simultaneously
    e1 = gw.Event(a, 'attack', c)
    e2 = gw.Event(b, 'attack', c)
    cfg.add_reward_rule(e1 & e2, receiver=[a, b], value=[1, 1])

    return cfg


if __name__ == '__main__':
    view_dic = {'pursuit': 5, 'battle': 6, 'double_attack': 4}
    num_neighbor_dic = {'pursuit': 3, 'battle': 4, 'double_attack': 1}
    reward_event_dic = {'pursuit': 0.7, 'battle': 0, 'double_attack': 0}
    for i in range(1):
        args = get_common_args()
        args.alg = 'ours'
        if args.alg.find('coma') > -1:
            args = get_coma_args(args)
        elif args.alg.find('central_v') > -1:
            args = get_centralv_args(args)
        elif args.alg.find('reinforce') > -1:
            args = get_reinforce_args(args)
        else:
            args = get_mixer_args(args)
        if args.alg.find('commnet') > -1:
            args = get_commnet_args(args)
        if args.alg.find('g2anet') > -1:
            args = get_g2anet_args(args)
        # env = StarCraft2Env(map_name=args.map,
        #                     step_mul=args.step_mul,
        #                     difficulty=args.difficulty,
        #                     game_version=args.game_version,
        #                     replay_dir=args.replay_dir)
        # env = magent.GridWorld("battle", map_size=30)
        args.map_size = 80  # pursuit:180 270;battle:80 100
        args.env_name = 'battle'
        args.map = args.alg
        args.name_time = 'est'
        # alt_wo_per alt_wo_dq
        env = magent.GridWorld(args.env_name, map_size=args.map_size)
        # env = magent.GridWorld(get_config_double_attack(args.map_size))  # pursuit 180 270 330
        handles = env.get_handles()
        eval_obs = None
        feature_dim = env.get_feature_space(handles[0])
        view_dim = env.get_view_space(handles[0])
        real_view_shape = view_dim
        v_dim_total = view_dim[0] * view_dim[1] * view_dim[2]
        obs_shape = (v_dim_total + feature_dim[0],)

        # env_info = env.get_env_info()
        # print(env.action_space[0][0])
        args.n_actions = env.action_space[0][0]
        args.fixed_n_actions = env.action_space[1][0]
        args.n_agents = 5  # pursuit:8;battle:5
        args.more_walls = 1  # pursuit:1;battle:1 10
        if args.alg == 'maven' and args.load_model:
            args.est_mod_num = 1
        else:
            args.est_mod_num = 1
        args.use_v1 = False
        if args.use_v1:
            args.nei_n_agents = args.n_agents
            args.id_dim = args.n_agents
        else:
            args.nei_n_agents = args.n_agents
            args.id_dim = 2
        args.state_shape = feature_dim[0]
        # args.obs_shape = obs_shape[0]
        args.view_shape = v_dim_total
        args.act_dim = env.action_space[0][0]
        args.idact_dim = args.id_dim + args.act_dim
        args.view_field = view_dic[args.env_name]
        args.num_neighbor = num_neighbor_dic[args.env_name]
        args.reward_event = reward_event_dic[args.env_name]
        # args.id_dim = 2
        # print(args.view_shape)
        # print(obs_shape[0])
        args.feature_shape = feature_dim[0]
        args.real_view_shape = real_view_shape
        args.episode_limit = 350
        if args.env_name == 'battle':
            args.use_fixed_model = True
        else:
            args.use_fixed_model = False
        args.load_num = 4  # 4
        args.use_ja = True
        args.use_dqloss = True
        args.use_per = True
        if args.use_per:
            args.sample_times = 5
        if args.use_dqloss:
            args.alpha_dq_loss = 0.005  # 0.005
            args.env_name += '_wdq'
        if args.use_ja:
            # args.obs_shape = obs_shape[0] + args.nei_n_agents * (args.id_dim + args.act_dim)
            args.obs_shape = obs_shape[0]
        else:
            args.obs_shape = obs_shape[0]
        runner = Runner(env, args)
        if args.learn:
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        # env.close()
