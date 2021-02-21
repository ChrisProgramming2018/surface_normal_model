from collections import deque
import numpy as np
import cv2
import torch
from gym import Wrapper



class FrameStack(Wrapper):
    r"""Observation wrapper that stacks the observations in a rolling manner.
    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v0', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].
    .. note::
        To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
    .. note::
        The observation space must be `Box` type. If one uses `Dict`
        as observation space, it should apply `FlattenDictWrapper` at first.
    Example::
        >>> import gym
        >>> env = gym.make('PongNoFrameskip-v0')
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(4, 210, 160, 3)
    Args:
        env (Env): environment object
        num_stack (int): number of stacks
        lz4_compress (bool): use lz4 to compress the frames internally
    """
    def __init__(self, env, args):
        super(FrameStack, self).__init__(env)
        self.state_buffer = deque([], maxlen=args.history_length)
        self.env = env
        self.size = args.size
        self.device = args.device
        self.history_length = args.history_length

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        state = self._create_next_obs(observation)
        observation = cv2.resize(observation,(256, 256))
        depth = cv2.resize(info["depth"],(256, 256))
        return state, observation, reward, done, depth

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        state = self._stacked_frames(observation)
        return state

    def _create_next_obs(self, state):
        state =  cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state,(self.size, self.size))
        state = torch.tensor(state, dtype=torch.uint8, device=self.device)
        self.state_buffer.append(state)
        state = torch.stack(list(self.state_buffer), 0)
        state = state.cpu()
        obs = np.array(state)
        return obs


    def _stacked_frames(self, state):
        state =  cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state,(self.size, self.size))
        state = torch.tensor(state, dtype=torch.uint8, device=self.device)
        zeros = torch.zeros_like(state)
        for idx in range(self.history_length - 1):
            self.state_buffer.append(zeros)
        self.state_buffer.append(state)

        state = torch.stack(list(self.state_buffer), 0)
        state = state.cpu()
        obs = np.array(state)
        return obs


def time_format(sec):
    """     
    Args:
    param1():
    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, round(secs,2)





def write_into_file(pathname, text):
    """
    """
    with open(pathname+".txt", "a") as myfile:
        myfile.write(text)
        myfile.write('\n')

def evaluate_policy(policy, writer, total_timesteps, args, env, episode=5):
    """
    
    
    Args:
       param1(): policy
       param2(): writer
       param3(): episode default 1 number for path to save the video
    """

    path = mkdir("","eval/" + str(total_timesteps) + "/")
    size = args.size
    print(path)
    avg_reward = 0.
    seeds = [x for x in range(episode)]
    goal= 0
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        env.seed(s)
        obs = env.reset()
        done = False
        for step in range(args.max_episode_steps):
            action = policy.select_action(np.array(obs))
            
            obs, reward, done, _ = env.step(action)
            #cv2.imshow("wi", cv2.resize(obs[:,:,::-1], (300,300)))
            # frame = cv2.imwrite("{}/wi{}.png".format(path, step), np.array(obs))
            if done:
                avg_reward += reward * args.reward_scalling
                goal +=1
                break
            #cv2.waitKey(10)
            avg_reward += reward * args.reward_scalling

    avg_reward /= len(seeds)
    writer.add_scalar('Evaluation reward', avg_reward, total_timesteps)
    print ("---------------------------------------")
    print ("Average Reward over the Evaluation Step: {}  realed {} ".format(avg_reward, goal))
    print ("---------------------------------------")
    return avg_reward

