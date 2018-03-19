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
from vizdoom import *

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--env-name', default='VizDoom',
                    help='environment to train on (default: VizDoom)')
parser.add_argument('--config-path', default='./scenarios/deathmatch.cfg',
                    help='vizdoom configuration file path (default: ./scenarios/deathmatch.cfg)')
parser.add_argument('--recording', default='./recordings/deathmatch_event.lmp',
                    help='vizdoom recording file path (default: ./recording/deathmatch_event.lmp)')
args = parser.parse_args()

scenario = args.config_path.split("/")[1].split(".")[0]

print("Scenario: " + scenario)

save_movie = False

game = DoomGame()

# Use other config file if you wish.
game.load_config(args.config_path)
#game.set_episode_timeout(100)
game.close()

# New render settings for replay
game.set_screen_resolution(ScreenResolution.RES_1280X720)
game.set_screen_format(ScreenFormat.CRCGCB)
game.set_render_hud(True)

# Replay can be played in any mode.
game.set_mode(Mode.SPECTATOR)
game.set_window_visible(True)
game.set_sound_enabled(True)

game.init()

print("\nREPLAY OF EPISODE")
print("************************\n")

# Replays episodes stored in given file. Sending game command will interrupt playback.
#game.replay_episode("recording_health_gathering_412.lmp")
game.replay_episode(args.recording)
print("FPS="+str(game.get_ticrate()))

frame = 0
while not game.is_episode_finished():
    s = game.get_state()
    #sleep(0.1 / 24)

    if save_movie:
        scipy.misc.imsave('./frames/' + scenario + '_' + str(frame) + '.jpg', np.swapaxes(np.swapaxes(s.screen_buffer,0,2),0,1))

    frame += 1

    # Use advance_action instead of make_action.
    game.advance_action()

    r = game.get_last_reward()
    # game.get_last_action is not supported and don't work for replay at the moment.

    print("State #" + str(s.number))
    print("Game variables:", s.game_variables[0])
    print("Reward:", r)
    print("=====================")

print("Episode finished.")
print("total reward:", game.get_total_reward())
print("************************")

game.close()
