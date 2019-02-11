import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=16,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=20,
                        help='number of forward steps in A2C (default: 20)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='ppo batch size (default: 64)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--tau', type=float, default=0.1,
                        help='ppo tau parameter -- NOT SET (default: 0.1)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='Whether to use gae for a2c -> ppo')
    parser.add_argument('--num-stack', type=int, default=1,
                        help='number of frames to stack (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='log interval, one log per n updates (default: 100)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num-frames', type=int, default=10e6,
                        help='number of frames to train (default: 10e6)')
    parser.add_argument('--env-name', default='VizDoom',
                        help='environment to train on (default: VizDoom)')
    parser.add_argument('--config-path', default='./scenarios/basic.cfg',
                        help='vizdoom configuration file path (default: ./scenarios/basic.cfg)')
    parser.add_argument('--source-models-path', default='./models',
                        help='directory from where to load source task models [A2T only] (default: ./models)')
    parser.add_argument('--log-dir', default='./',
                        help='directory to save agent logs (default: /tmp/vizdoom)')
    parser.add_argument('--save-dir', default='./models',
                        help='directory to save agent logs (default: ./models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debugging')
    parser.add_argument('--shaped', action='store_true', default=False,
                        help='Trains using shaped intrinsic reward')
    parser.add_argument('--bots', action='store_true', default=False,
                        help='Is the scenario with bots? (default: False)')
    parser.add_argument('--roe', action='store_true', default=False,
                        help='Trains using Rairty of Events (default: False)')
    parser.add_argument('--visual', action='store_true', default=False,
                        help='Trains with visuals (default: False)')
    parser.add_argument('--num-events', type=int, default=26,
                        help='number of events to record (default: 26)')
    parser.add_argument('--capacity', type=int, default=100,
                        help='Size of the event buffer (default: 100)')
    parser.add_argument('--num-vars', type=int, default=17,
                        help='number of vars to record (default: 17)')
    parser.add_argument('--qd', action='store_true', default=False,
                        help='RoE QD (default: False)')
    parser.add_argument('--exp-id', type=int, required=True,
                        help='Experiment ID')
    parser.add_argument('--agent-id', type=int, required=True,
                        help='Experiment ID')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
