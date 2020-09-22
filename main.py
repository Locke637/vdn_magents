from runner import Runner
# from smac.env import StarCraft2Env
import magent
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, \
    get_commnet_args, get_g2anet_args

if __name__ == '__main__':
    for i in range(1):
        args = get_common_args()
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
        env = magent.GridWorld("pursuit", map_size=40)
        handles = env.get_handles()
        eval_obs = None
        feature_dim = env.get_feature_space(handles[0])
        view_dim = env.get_view_space(handles[0])
        v_dim_total = view_dim[0] * view_dim[1] * view_dim[2]
        obs_shape = (v_dim_total + feature_dim[0],)
        act_dim = env.action_space

        # env_info = env.get_env_info()
        # print(env.action_space[0][0])
        args.n_actions = env.action_space[0][0]
        args.n_agents = 2
        args.state_shape = feature_dim[0]
        args.obs_shape = obs_shape[0]
        args.view_shape = v_dim_total
        # print(obs_shape[0])
        args.feature_shape = feature_dim[0]
        args.episode_limit = 550
        runner = Runner(env, args)
        if args.learn:
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        # env.close()
