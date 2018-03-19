import argparse
import os
import pickle

import torch
from torch.autograd import Variable

from envs import make_env
from vec_env import VecEnv
from time import sleep
import matplotlib.animation as animation
import numpy as np
import scipy.misc
from pylab import *

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--algo', default='a2c',
                    help='algorithm to use: a2c | acktr')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-stack', type=int, default=1,
                    help='number of frames to stack (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='VizDoom',
                    help='environment to train on (default: VizDoom)')
parser.add_argument('--config-path', default='./scenarios/deady_corridor.cfg',
                    help='vizdoom configuration file path (default: ./scenarios/basic.cfg)')
parser.add_argument('--load-dir', default='./models/',
                    help='directory with models')
parser.add_argument('--log-dir', default='/tmp/doom/',
                    help='directory to save agent logs (default: /tmp/doom)')
parser.add_argument('--roe', action='store_true', default=False,
                    help='Loads the RoE model (default: False)')
parser.add_argument('--demo', action='store_true', default=True,
                    help='Play in real-time with visuals (default: False)')
parser.add_argument('--record', action='store_true', default=False,
                    help='Record game (default: False)')
parser.add_argument('--heatmap', action='store_true', default=False,
                    help='Saves data for heatmaps (default: False)')
args = parser.parse_args()

try:
    os.makedirs(args.log_dir)
except OSError:
    pass

envs = VecEnv([make_env(0, config_file_path=args.config_path, visual=args.demo)], record=args.record)

scenario = args.config_path.split("/")[1].split(".")[0]
exp_name = scenario + ("_event" if args.roe else "")

print("Scenario: " + scenario)
print("Experiment: " + exp_name)

if args.roe:
    model_name = args.algo + "/vizdoom_" + scenario.split("-")[0] + "_event"
else:
    model_name = args.algo + "/vizdoom_" + scenario.split("-")[0]

print("Model: " + model_name)
actor_critic = torch.load(os.path.join(args.load_dir, model_name + ".pt"))

actor_critic.eval()

obs_shape = envs.observation_space_shape
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
current_obs = torch.zeros(1, *obs_shape)

if args.record:
    try:
        os.remove("recording_" + scenario + ".lmp")
    except Exception as e:
        pass

def update_current_obs(obs):
    shape_dim0 = envs.observation_space_shape[0]
    obs = torch.from_numpy(obs).float()
    if args.num_stack > 1:
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
    current_obs[:, -shape_dim0:] = obs

obs = envs.reset()
update_current_obs(obs)
#vars = envs.get_all_game_variables()
#vars = torch.from_numpy(to_input_vars(vars)).float()

num_episodes = 10 if not args.record else 1
total_rewards = []
episode_cnt = 0
episode_reward = 0.0
total_kills = []

frame = 0

deterministic = True
num_of_events = 26
episode_events = np.zeros(num_of_events)

positions = []
positions_episode = []
position = envs.get_position()[0]
positions_episode.append(position)

while episode_cnt < num_episodes:
    if args.demo:
        sleep(1/24)

    # Save frames
    #scipy.misc.imsave('./frames/' + scenario + '_' + str(frame) + '.jpg', current_obs.numpy()[0][0])
    frame += 1

    #actor_critic.vars = Variable(vars)
    value, action = actor_critic.act(Variable(current_obs, volatile=True),
                                     deterministic=deterministic)
    if deterministic:
        cpu_actions = action.data.cpu().numpy()  # Enable for deterministic play
    else:
        cpu_actions = action.data.squeeze(1).cpu().numpy()

    # Obser reward and next obs
    obs, reward, done, _, events = envs.step([cpu_actions[0]])

    # Fix reward
    if scenario in ["deathmatch", "my_way_home"]:
        reward[0] *= 100
    if scenario == "deadly_corridor":
        reward[0] = 1 if events[0][2] >= 1 else 0

    #print('Frame', frame)
    #print ('Reward:', reward[0] * 100)

    position = envs.get_position()[0]
    positions_episode.append(position)

    if events[0][15] > 0:
        print("kill: " + str(events[0][15]))

    #vars = torch.from_numpy(np.array(to_input_vars(vars))).float()
    episode_reward += reward[0] * 100
    episode_events = episode_events + np.array(events[0])

    if done:
        #print("Reward: " + str(episode_reward))
        positions.append(np.copy(positions_episode))
        positions_episode = []

        total_rewards.append(episode_reward)
        episode_cnt += 1
        episode_reward = 0.0
        episode_game_variables = envs.get_all_game_variables()[0]
        total_kills.append(episode_events[15])
        episode_events = np.zeros(num_of_events)

        obs = envs.reset()

        position = envs.get_position()[0]
        positions_episode.append(position)

        #actor_critic = torch.load(os.path.join(args.load_dir, log_file_name.split(".log")[0] + ".pt"))
        #actor_critic.eval()

    update_current_obs(obs)

print ('Avg reward:', np.mean(total_rewards))
print ('Std. dev reward:', np.std(total_rewards))
print ('Avg kills:', np.mean(total_kills))
print ('Std. dev. kills:', np.std(total_kills))

heat_name = scenario + "_" + model_name + ".p"

if args.heatmap:
    pickle.dump( positions, open( "./heat_data/" + exp_name, "wb" ) )

envs.close()
