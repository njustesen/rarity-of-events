import numpy as np
from multiprocessing import Process, Pipe
import scipy.misc
import os
from vizdoom import *
import math
from arguments import get_args

args = get_args()


# Processes Doom screen image to produce cropped and resized image. 
def process_frame(frame):
    s = frame[10:-10,30:-30]
    print(s.shape)
    s = scipy.misc.imresize(s,[84,84])
    #s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    s = [s / 255.0]
    return s


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x
    # prev_agent_health = 0
    # prev_agent_ammo = 0
    log_file = None
    get_bin = lambda x, n: format(x, 'b').zfill(n)
    total_reward = 0.0
    episode_reward = 0.0
    episode_cnt = 0.0
    total_episode_cnt = 0
    total_kills = 0.0
    episode_kills = 0.0
    step_cnt = 0
    position_history = []
    vars = []
    episode_events = np.zeros(args.num_events)

    def meters_walked(position_history):
        meters = 0
        last_position = None
        for p in position_history:
            if last_position is None:
                last_position = p
            else:
                distance = math.sqrt((p[0] - last_position[0]) ** 2 + (p[1] - last_position[1]) ** 2) / 100
                if distance > 1:
                    meters += 1
                    last_position = p

        return meters

    def get_vizdoom_vars(vizdoom, position_history):
        vars = [meters_walked(position_history),  # 0
                vizdoom.get_game_variable(GameVariable.HEALTH),  # 1
                vizdoom.get_game_variable(GameVariable.ARMOR),  # 2
                vizdoom.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO),  # 3
                vizdoom.get_game_variable(GameVariable.WEAPON0),  # 4
                vizdoom.get_game_variable(GameVariable.WEAPON1),  # 5
                vizdoom.get_game_variable(GameVariable.WEAPON2),  # 6
                vizdoom.get_game_variable(GameVariable.WEAPON3),  # 7
                vizdoom.get_game_variable(GameVariable.WEAPON4),  # 8
                vizdoom.get_game_variable(GameVariable.WEAPON5),  # 9
                vizdoom.get_game_variable(GameVariable.WEAPON6),  # 10
                vizdoom.get_game_variable(GameVariable.WEAPON7),  # 11
                vizdoom.get_game_variable(GameVariable.WEAPON8),  # 12
                vizdoom.get_game_variable(GameVariable.WEAPON9),  # 13
                vizdoom.get_game_variable(GameVariable.KILLCOUNT) + vizdoom.get_game_variable(GameVariable.FRAGCOUNT),
                # 14
                vizdoom.get_game_variable(GameVariable.DEATHCOUNT),  # 15
                vizdoom.get_game_variable(GameVariable.SELECTED_WEAPON)]  # 16
        return np.array(vars)

    def get_events(vars, last_vars):

        events = np.zeros(args.num_events)

        if np.count_nonzero(last_vars) == 0:
            return events

        # If died -> no event
        if vars[15] > last_vars[15]:
            return events

        # 0. Movement
        if vars[0] > last_vars[0]:
            events[0] = 1

        # 1. Health increase
        if vars[1] > last_vars[1]:
            events[1] = 1

        # 2. Armor increase
        if vars[2] > last_vars[2]:
            events[2] = 1

        # 3. Ammo decrease
        if vars[3] < last_vars[3]:
            events[3] = 1

        # 4. Ammo increase
        if vars[3] > last_vars[3]:
            events[4] = 1

        # 5-14. Weapon pickup 0-9
        for i in range(4, 14):
            if vars[i] > last_vars[i]:
                events[i + 1] = 1

        # 15-24 Kill increase - for each weapon
        if vars[14] > last_vars[14]:
            events[15] = 1
            for i in range(0, 9):
                if vars[16] == i:  # If selected weapon
                    events[16 + i] = 1

        return events

    while True:
        cmd, data = remote.recv()
        if data is None:
            import random
            data = random.randint(0, 2**env.get_available_buttons_size() - 1)
        action = [True if i == '1' else False for i in get_bin(data, env.get_available_buttons_size())]
        
        if cmd == 'step':
            if len(vars) == 0:
                vars = get_vizdoom_vars(env, position_history)
            reward = env.make_action(action)
            last_vars = vars
            pos = [env.get_game_variable(GameVariable.POSITION_X),
                   env.get_game_variable(GameVariable.POSITION_Y)]
            position_history.append(pos)
            vars = get_vizdoom_vars(env, position_history)
            events = get_events(vars, last_vars)
            if not env.is_episode_finished():
                ob = process_frame(env.get_state().screen_buffer)
                episode_kills = vars[14]
            reward = reward / 100.0                                 # normalizing the reward
            episode_reward += reward
            step_cnt += 1
            done = env.is_episode_finished()
            if done:
                total_kills += episode_kills
                env.new_episode()
                position_history = []
                vars = get_vizdoom_vars(env, position_history)
                ob = process_frame(env.get_state().screen_buffer)
                total_reward += episode_reward
                episode_cnt += 1
                total_episode_cnt += 1
                episode_reward = 0.0
            remote.send((ob, reward, done, 0.0, events))
        elif cmd == 'pos':
            pos = [env.get_game_variable(GameVariable.POSITION_X),
                   env.get_game_variable(GameVariable.POSITION_Y)]
            remote.send(pos)
        elif cmd == 'gv':
            remote.send(vars)
        elif cmd == 'log':
            if log_file is None:
                continue
            if episode_cnt == 0.0:
                continue
            avg_reward = round(total_reward / episode_cnt, 5)
            log_file.write(str(step_cnt) + ', ' + str(avg_reward) + '\n')
            log_file.flush()
            total_reward = 0.0
            episode_cnt = 0.0
            total_kills = 0.0
        elif cmd == 'reset':
            env.new_episode()
            ob = process_frame(env.get_state().screen_buffer)
            remote.send(ob)
        elif cmd == 'reset_task':
            print ('reset_task: Not implemented')
            raise NotImplementedError
        elif cmd == 'close':
            print ('Terminating doom environment')
            if log_file is not None:
                log_file.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((2**env.get_available_buttons_size(), (1, 84, 84)))
        else:
            raise NotImplementedError


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x, log_file=""):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class VecEnv():
    def __init__(self, env_fns):
        """
        envs: list of vizdoom game environments to run in subprocesses
        """
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]

        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.action_space_shape, self.observation_space_shape = self.remotes[0].recv()

    def step(self, actions):
        cumul_rewards = None
        cumul_dones = None
        cumul_events = None
        for _ in range(4):  # Frame skip
            for remote, action in zip(self.remotes, actions):
                remote.send(('step', action))
            results = [remote.recv() for remote in self.remotes]
            obs, rews, dones, infos, events = zip(*results)
            if cumul_rewards is None:
                cumul_rewards = np.stack(rews)
            else:
                cumul_rewards += np.stack(rews)
            if cumul_dones is None:
                cumul_dones = np.stack(dones)
            else:
                cumul_dones |= np.stack(dones)
            if cumul_events is None:
                cumul_events = events
            else:
                cumul_events = np.add(cumul_events, events)
        return np.stack(obs), cumul_rewards, cumul_dones, infos, cumul_events

    def get_game_variables(self, id):
        self.remotes[id].send(['gv', None])
        return self.remotes[id].recv()

    def get_all_game_variables(self):
        for remote in self.remotes:
            remote.send(('gv', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_position(self):
        for remote in self.remotes:
            remote.send(('pos', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def log(self):
        for remote in self.remotes:
            remote.send(('log', None))
        return

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)
