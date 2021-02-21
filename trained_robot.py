import os
import sys
import gym
import time
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
from agent import TQC
import cv2
from PIL import Image
from gym_grasping.envs.robot_sim_env import RobotSimEnv
from replay_buffer_depth import ReplayBufferDepth
from helper import FrameStack


def evaluate_policy(policy, args, env, episode=25):
    """
    Args:
       param1(): policy
       param2(): writer
       param3(): episode default 1 number for path to save the video
    """
    size = args.size
    different_seeds = False
    seeds = [x for x in range(episode)]
    obs_shape = (args.history_length, size, size)
    action_shape = (args.action_dim,)
    size = 256
    memory = ReplayBufferDepth((size, size), (size, size, 3), (size, size, 3), 15000, "cuda")
    path = "eval"
    goals  = 0
    avg_reward = 0.
    total_steps = 0
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    episode = 0
    use_random = False
    print("continue ", memory.idx)
    while True:
        print("Episode {} goals {} buffer size {}".format(episode, goals, memory.idx))
        if different_seeds:
            torch.manual_seed(s)
            np.random.seed(s)
            env.seed(s)
        obs = env.reset()
        # print("for agent", obs.shape)
        # obs = obs.transpose(1,2,0)
        done = False
        # use_random= True
        for step in range(args.max_episode_steps):
            if use_random:
                action = env.action_space.sample()
            else:
                action = policy.select_action(np.array(obs))
            new_obs, RGB_image, reward, done, depth = env.step(action)
            
            # print("rgb", RGB_image.shape)
            # cv2.imshow("RGB_image", RGB_image)
            # cv2.waitKey(0)
            
            # s = new_obs.transpose(1,2,0)
            # cv2.imshow("RGB_image", s)
            #frame = cv2.imwrite("new_obs{}.png".format(step), np.array(s))
            #cv2.waitKey(0)
            # print("depth size", depth_array.shape)
            #frame = cv2.imwrite("{}/{}wi{}.png".format(path, s, step), np.array(new_obs)[0])
            #frame = cv2.imwrite("{}/{}wi{}.png".format(path, s, step), np.array(info["depth"]))
            # array_RGB = np.swapaxes(new_obs, 0, 2)
            #array_RGB = np.stack([array_RGB[2], array_RGB[0], array_RGB[1]], axis=0)
            #array_RGB = array_RGB.transpose(1,2,0)
            # array_RGB = new_obs
            
            # cv2.imwrite("test.png", array_RGB)
            depth_array =  np.array(depth)
            memory.add(depth_array, RGB_image)
            if memory.idx % 5000 == 0:    
                memory.save_memory("depth_memory-{}".format(memory.idx))
            elif memory.idx >= 10000:
                memory.save_memory("depth_memory-{}".format(memory.idx))
                return
            done_bool = 0 if step + 1 == args.max_episode_steps else float(done)
            if done:
                episode += 1
                if step < 49:
                    total_steps += step 
                    goals +=1
                break
            obs = new_obs
            avg_reward += reward
    
    
    avg_reward /= len(seeds)
    print("reached goal {} of {}".format(goals, episode))
    if goals != 0:
        print("Average step if reached {} ".format(float(total_steps) / goals))
    print ("---------------------------------------")
    print ("Average Reward over the Evaluation Step: {} of {} Episode".format(avg_reward, len(seeds)))
    print ("---------------------------------------")
    return avg_reward




def main(args):
    """ Starts different tests

    Args:
        param1(args): args

    """
    env= gym.make(args.env_name, renderer='egl')
    env = FrameStack(env, args)
    state = env.reset()
    state_dim = 200
    action_dim = env.action_space.shape[0]
    args.action_dim = action_dim
    max_action = float(1)
    min_action = float(-1)
    args.target_entropy=-np.prod(action_dim)
    
    policy = TQC(state_dim, action_dim, max_action, args) 
    directory = "pretrained/"
    if args.agent is None:
        filename = "kuka_block_grasping-v0-97133reward_-1.05-agentTCQ" # 93 %
        filename = "kuka_block_grasping-v0-7953reward_-1.13" # 93 %
    else:
        filename = args.agent

    filename = directory + filename
    print("Load " , filename)
    policy.load(filename)
    policy.actor.training = False
    if args.eval:
        evaluate_policy(policy, args,  env, args.epi)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="kuka_block_grasping-v0", type=str, help='Name of a environment (set it to any Continous environment you want')
    parser.add_argument('--seed', default=True, type=bool, help='use different seed for each episode')
    parser.add_argument('--epi', default=25, type=int)
    parser.add_argument('--max_episode_steps', default=50, type=int)    
    parser.add_argument('--lr-critic', default= 0.0005, type=int)               # Total number of iterations/timesteps
    parser.add_argument('--lr-actor', default= 0.0005, type=int)               # Total number of iterations/timesteps
    parser.add_argument('--lr_alpha', default=3e-4, type=float)
    parser.add_argument('--lr_encoder', default=1e-4, type=float)      # Divide by 5
    parser.add_argument('--save_model', default=True, type=bool)     # Boolean checker whether or not to save the pre-trained model
    parser.add_argument('--batch_size', default= 256, type=int)      # Size of the batch
    parser.add_argument('--discount', default=0.99, type=float)      # Discount factor gamma, used in the calculation of the total discounted reward
    parser.add_argument('--tau', default=0.005, type= float)        # Target network update rate
    parser.add_argument('--size', default=84, type=int)
    parser.add_argument('--num_q_target', default=4, type=int)    # amount of qtarget nets
    parser.add_argument('--tensorboard_freq', default=5000, type=int)    # every nth episode write in to tensorboard
    parser.add_argument('--device', default='cuda', type=str)    # amount of qtarget nets
    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--top_quantiles_to_drop_per_net", default=2, type=int)
    parser.add_argument("--n_nets", default=5, type=int)
    parser.add_argument('--history_length', default=3, type=int)
    parser.add_argument('--locexp', type=str)     # Maximum value
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--eval', type=bool, default= True)
    parser.add_argument('--buffer_size', default=3.5e5, type=int)
    parser.add_argument('--agent', default=None, type=str)
    parser.add_argument('--save_buffer', default=False, type=bool)
    arg = parser.parse_args()
    main(arg)
