import copy
import glob
import os
import time
import sys
import signal
import pickle
import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from event_buffer import EventBuffer, EventBufferSQLProxy
from arguments import get_args
from envs import make_env
from vec_env import VecEnv
from model import CNNPolicy
from storage import RolloutStorage
# from gif import make_gif

envs = None

args = get_args()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def main():
    print("###############################################################")
    print("#################### VIZDOOM LEARNER START ####################")
    print("###############################################################")

    save_path = os.path.join(args.save_dir, str(args.exp_id))
    num_updates = int(args.num_frames) // args.num_steps // args.num_processes
    reward_name = ""
    if args.roe:
        reward_name = "_event"
    scenario_name = args.config_path.split("/")[1].split(".")[0]
    print("############### " + scenario_name + " ###############")
    log_file_name = "vizdoom_" + scenario_name + reward_name + "_" + str(args.exp_id) + "_" + str(args.agent_id) + ".log"
    log_event_file_name = "vizdoom_" + scenario_name + reward_name + "_" + str(args.exp_id) + "_" + str(args.agent_id) + ".eventlog"
    log_event_reward_file_name = "vizdoom_" + scenario_name + reward_name + "_" + str(args.exp_id) + "_" + str(args.agent_id) + ".eventrewardlog"
    start_updates = 0
    start_step = 0
    best_final_rewards = -1000000.0

    os.environ['OMP_NUM_THREADS'] = '1'

    cig = "cig" in args.config_path
    global envs
    es = [make_env(i, args.config_path, visual=args.visual, cig=cig) for i in range(args.num_processes)]
    envs = VecEnv([
        es[i] for i in range(args.num_processes)
    ])

    obs_shape = envs.observation_space_shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    if args.resume:
        actor_critic = torch.load(os.path.join(save_path, log_file_name + ".pt"))
        filename = glob.glob(os.path.join(args.log_dir, log_file_name))[0]
        #if args.roe:
            # TODO: Load event buffer
        with open(filename) as file:
            lines = file.readlines()
            start_updates = (int)(lines[-1].strip().split(",")[0])
            start_steps = (int)(lines[-1].strip().split(",")[1])
            num_updates += start_updates
    else:
        if not args.debug:
            try:
                os.makedirs(args.log_dir)
            except OSError:
                files = glob.glob(os.path.join(args.log_dir, log_file_name))
                for f in files:
                    os.remove(f)
                with open(log_file_name, "w") as myfile:
                    myfile.write("")
                files = glob.glob(os.path.join(args.log_dir, log_event_file_name))
                for f in files:
                    os.remove(f)
                with open(log_event_file_name, "w") as myfile:
                    myfile.write("")
                files = glob.glob(os.path.join(args.log_dir, log_event_reward_file_name))
                for f in files:
                    os.remove(f)
                with open(log_event_reward_file_name, "w") as myfile:
                    myfile.write("")
        actor_critic = CNNPolicy(obs_shape[0], envs.action_space_shape)
        
    action_shape = 1

    if args.cuda:
        actor_critic.cuda()

    optimizer = optim.RMSprop(actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
    
    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space_shape)
    current_obs = torch.zeros(args.num_processes, *obs_shape)

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space_shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)
    last_game_vars = []
    for i in range(args.num_processes):
        last_game_vars.append(np.zeros(args.num_events))

    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])
    episode_intrinsic_rewards = torch.zeros([args.num_processes, 1])
    final_intrinsic_rewards = torch.zeros([args.num_processes, 1])
    episode_events = torch.zeros([args.num_processes, args.num_events])
    final_events = torch.zeros([args.num_processes, args.num_events])

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    def mean_distance_to_nearest_neighbor(elite_events):
        d = []
        nearest = None
        for a in range(len(elite_events)):
            for b in range(len(elite_events)):
                if a != b:
                    elite_a = elite_events[a]
                    elite_b = elite_events[b]
                    dist = np.linalg.norm(elite_a - elite_b)
                    if nearest is None or dist < nearest:
                        nearest = dist
            if nearest is not None:
                d.append(nearest)
            nearest = None
        return np.mean(d)

    def distance_to_nearest_neighbor(elite_events, events):
        nearest = None
        for elite_a in elite_events:
            dist = np.linalg.norm(elite_a - events)
            if nearest is None or dist < nearest:
                nearest = dist
        return nearest

    def add_to_archive(frame):
        try:
            os.makedirs(save_path)
        except OSError:
            pass
        fitness = final_rewards.mean()
        behavior = event_buffer.get_last_own_events_mean(len(final_rewards))  # From event buffer
        #print("Fitness:", fitness)
        #print("Behavior:", behavior)
        neighbors = event_buffer.get_neighbors(behavior)
        if len(neighbors) == 0:
            add = True
            #print("- Cell empty")
        else:
            add = True
            for neighbor in neighbors:
                if fitness < neighbor.fitness:
                    add = False
                    #print("- Competition lost")
                    break
        if add:
            if len(neighbors) > 0:
                event_buffer.remove_elites(neighbors)
                #print(f"- Removing elites {[neighbor.elite_id for neighbor in neighbors]}")
            for neighbor in neighbors:
                try:
                    #print(f"- Deleting model {neighbor.elite_id}")
                    os.remove(os.path.join(save_path, f"{neighbor.elite_id}.pt"))
                except:
                    print("Error while deleting model with id : ", neighbor.elite_id)
            name = str(uuid.uuid1())
            event_buffer.add_elite(name, behavior, fitness, frame)
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()
            torch.save(save_model, os.path.join(save_path, f"{name}.pt"))

    # Create event buffer
    event_buffer = EventBufferSQLProxy(args.num_events, args.capacity, args.exp_id, args.agent_id, qd=args.qd)

    event_episode_rewards = []

    start = time.time()
    for j in np.arange(start_updates, num_updates):
        for step in range(args.num_steps):

            value, action = actor_critic.act(Variable(rollouts.observations[step], volatile=True))
            cpu_actions = action.data.squeeze(1).cpu().numpy()
            obs, reward, done, info, events = envs.step(cpu_actions)
            intrinsic_reward = []

            # Fix broken rewards - upscale
            for i in range(len(reward)):
                if scenario_name in ["deathmatch", "my_way_home"]:
                    reward[i] *= 100
                if scenario_name == "deadly_corridor":
                    reward[i] = 1 if events[i][2] >= 1 else 0

            for e in events:
                if args.roe:
                    intrinsic_reward.append(event_buffer.intrinsic_reward(e))
                else:
                    r = reward[len(intrinsic_reward)]
                    intrinsic_reward.append(r)

            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            intrinsic_reward = torch.from_numpy(np.expand_dims(np.stack(intrinsic_reward), 1)).float()
            #events = torch.from_numpy(np.expand_dims(np.stack(events), args.num_events)).float()
            events = torch.from_numpy(events).float()
            episode_rewards += reward
            episode_intrinsic_rewards += intrinsic_reward
            episode_events += events

            # Event stats
            event_rewards = []
            for ei in range(0,args.num_events):
                ev = np.zeros(args.num_events)
                ev[ei] = 1
                er = event_buffer.intrinsic_reward(ev)
                event_rewards.append(er)

            event_episode_rewards.append(event_rewards)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_intrinsic_rewards *= masks
            final_events *= masks
            final_rewards += (1 - masks) * episode_rewards
            final_intrinsic_rewards += (1 - masks) * episode_intrinsic_rewards
            final_events += (1 - masks) * episode_events

            for i in range(args.num_processes):
                if done[i]:
                    event_buffer.record_events(np.copy(final_events[i].numpy()), frame=j*args.num_steps*args.num_processes)
                    add_to_archive(step)

            episode_rewards *= masks
            episode_intrinsic_rewards *= masks
            episode_events *= masks

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs)

            rollouts.insert(step, current_obs, action.data, value.data, intrinsic_reward, masks)

        final_episode_reward = np.mean(event_episode_rewards, axis=0)
        event_episode_rewards = []

        next_value = actor_critic(Variable(rollouts.observations[-1], volatile=True))[0].data

        if hasattr(actor_critic, 'obs_filter'):
            actor_critic.obs_filter.update(rollouts.observations[:-1].view(-1, *obs_shape))

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        values, action_log_probs, dist_entropy = actor_critic.evaluate_actions(Variable(rollouts.observations[:-1].view(-1, *obs_shape)), Variable(rollouts.actions.view(-1, action_shape)))

        values = values.view(args.num_steps, args.num_processes, 1)
        action_log_probs = action_log_probs.view(args.num_steps, args.num_processes, 1)
        advantages = Variable(rollouts.returns[:-1]) - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(Variable(advantages.data) * action_log_probs).mean()

        optimizer.zero_grad()
        (value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef).backward()

        nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)

        optimizer.step()

        rollouts.observations[0].copy_(rollouts.observations[-1])

        if j % args.log_interval == 0:

            envs.log()
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            log = "Updates {}, num timesteps {}, FPS {}, mean/max reward {:.5f}/{:.5f}, mean/max intrinsic reward {:.5f}/{:.5f}"\
                .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            final_rewards.mean(),
                            final_rewards.max(),
                            final_intrinsic_rewards.mean(),
                            final_intrinsic_rewards.max()
                        )
            print(log)

    envs.close()
    time.sleep(5)

if __name__ == "__main__":
   main()

