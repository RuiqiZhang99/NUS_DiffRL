# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import random
import torch


class CriticDataset:
    def __init__(self, batch_size, obs, target_values, shuffle = False, drop_last = False):
        self.obs = obs.view(-1, obs.shape[-1])
        self.target_values = target_values.view(-1)
        self.batch_size = batch_size

        if shuffle:
            self.shuffle()
        
        if drop_last:
            self.length = self.obs.shape[0] // self.batch_size
        else:
            self.length = ((self.obs.shape[0] - 1) // self.batch_size) + 1
    
    def shuffle(self):
        index = np.random.permutation(self.obs.shape[0])
        self.obs = self.obs[index, :]
        self.target_values = self.target_values[index]

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.obs.shape[0])
        return {'obs': self.obs[start_idx:end_idx, :], 'target_values': self.target_values[start_idx:end_idx]}

class ActorDataset:
    def __init__(self, horizon, buffer_info):
        
        self.obs = buffer_info['obs']
        self.next_obs = buffer_info['next_obs']
        self.raw_actions = buffer_info['raw_actions']
        self.actions = buffer_info['actions']
        self.rewards = buffer_info['rewards']
        self.l2a_grads = buffer_info['l2a_grads']
        self.log_probs = buffer_info['log_probs']
        self.done_mask = buffer_info['done_mask']
        self.deltas = buffer_info['deltas']
        self.target_values = buffer_info['target_values']
        self.next_values = buffer_info['next_values']
        self.values = buffer_info['values']
        self.gamma = buffer_info['gamma']
        self.horizon = horizon

    def __len__(self):
        return self.length
    
    def sample_horizon(self, times):
        seed_index = []
        times_counter = 0
        
        if self.horizon == self.rewards.shape[0]:
            obs = self.obs
            raw_actions = self.raw_actions
            actions = self.actions
            l2a_grads = self.l2a_grads
            log_probs = self.log_probs
            done_mask = self.done_mask
            next_values = self.next_values
            deltas = self.deltas
            gamma = self.gamma
            rewards = self.rewards
            values = self.values

        else:
            while True:
                seed = random.randint(0, self.obs.shape[0]-self.horizon-1)
                if seed in seed_index:
                    continue
                start_idx = min(seed*self.horizon, self.obs.shape[0]-self.horizon)
                end_idx = min((seed+1)*self.horizon, self.obs.shape[0])
                if times_counter == 0:
                    obs = self.obs[start_idx:end_idx, :, :]
                    raw_actions = self.raw_actions[start_idx:end_idx, :, :]
                    actions = self.actions[start_idx:end_idx, :, :]
                    l2a_grads = self.l2a_grads[start_idx:end_idx, :, :]
                    log_probs = self.log_probs[start_idx:end_idx, :, :]
                    done_mask = self.done_mask[start_idx:end_idx, :]
                    gamma = self.gamma[start_idx:end_idx, :]
                    rewards = self.rewards[start_idx:end_idx, :]
                    deltas = self.deltas[start_idx:end_idx, :]
                    values = self.values[start_idx:end_idx, :]
                    next_values = self.next_values[start_idx:end_idx, :]
                else:
                    obs = torch.cat((obs, self.obs[start_idx:end_idx, :, :]), dim=1)
                    raw_actions = torch.cat((raw_actions, self.raw_actions[start_idx:end_idx, :, :]), dim=1)
                    actions = torch.cat((actions, self.actions[start_idx:end_idx, :, :]), dim=1)
                    l2a_grads = torch.cat((l2a_grads, self.l2a_grads[start_idx:end_idx, :, :]), dim=1)
                    log_probs = torch.cat((log_probs, self.log_probs[start_idx:end_idx, :, :]), dim=1)
                    done_mask = torch.cat((done_mask, self.done_mask[start_idx:end_idx, :]), dim=1)
                    gamma = torch.cat((gamma, self.gamma[start_idx:end_idx, :]), dim=1)
                    rewards = torch.cat((rewards, self.rewards[start_idx:end_idx, :]), dim=1)
                    deltas = torch.cat((deltas, self.deltas[start_idx:end_idx, :]), dim=1)
                    values = torch.cat((values, self.values[start_idx:end_idx, :]), dim=1)
                    next_values = torch.cat((next_values, self.next_values[start_idx:end_idx, :]), dim=1)
                times_counter += 1
                if times_counter >= times:
                    break
        
        return {'obs': obs, 
                'raw_actions': raw_actions,
                'actions': actions, 
                'l2a_grads': l2a_grads, 
                'log_probs': log_probs, 
                'done_mask': done_mask,
                'gamma': gamma,
                'deltas': deltas,
                'rewards': rewards,
                'values': values,
                'next_values': next_values,
                }


'''
class ActorDataset:
    def __init__(self, horizon, buffer_info):
        
        self.obs = buffer_info['obs']
        self.next_obs = buffer_info['next_obs']
        self.raw_actions = buffer_info['raw_actions']
        self.actions = buffer_info['actions']
        self.rewards = buffer_info['rewards']
        self.l2a_grads = buffer_info['l2a_grads']
        self.log_probs = buffer_info['log_probs']
        self.done_mask = buffer_info['done_mask']
        self.target_values = buffer_info['target_values']
        self.next_values = buffer_info['next_values']
        self.gamma = buffer_info['gamma']

        self.horizon = horizon

    def __len__(self):
        return self.length
    
    def sample_horizon(self, index):
        start_idx = min(index*self.horizon, self.obs.shape[0]-self.horizon)
        end_idx = min((index + 1)*self.horizon, self.obs.shape[0])
        return {'obs': self.obs[start_idx:end_idx, :, :], 
                'next_obs': self.next_obs[start_idx:end_idx, :, :],
                'raw_actions': self.raw_actions[start_idx:end_idx, :, :],
                'actions': self.actions[start_idx:end_idx, :, :], 
                'rewards': self.rewards[start_idx:end_idx, :],  
                'l2a_grads': self.l2a_grads[start_idx:end_idx, :, :], 
                'log_probs': self.log_probs[start_idx:end_idx, :, :], 
                'done_mask': self.done_mask[start_idx:end_idx, :],
                'target_values': self.target_values[start_idx:end_idx, :],
                'next_values': self.next_values[start_idx:end_idx, :],
                'gamma': self.gamma[start_idx:end_idx, :],
                }


class ActorDataset:
    def __init__(self, horizon, buffer_info):
        self.obs = buffer_info['obs']
        self.next_obs = buffer_info['next_obs']
        self.raw_actions = buffer_info['raw_actions']
        self.actions = buffer_info['actions']
        self.rewards = buffer_info['rewards']
        self.r2a_grads = buffer_info['r2a_grads']
        self.nv2a_grads = buffer_info['nv2a_grads']
        self.log_probs = buffer_info['log_probs']
        self.done_mask = buffer_info['done_mask']
        self.target_values = buffer_info['target_values']
        self.next_values = buffer_info['next_values']
        self.values = buffer_info['values']

        self.horizon = horizon

    def __len__(self):
        return self.length
    
    def sample_horizon(self, index):
        start_idx = min(index*self.horizon, self.obs.shape[0]-self.horizon)
        end_idx = min((index + 1)*self.horizon, self.obs.shape[0])
        return {'obs': self.obs[start_idx:end_idx, :, :], 
                'next_obs': self.next_obs[start_idx:end_idx, :, :],
                'raw_actions': self.raw_actions[start_idx:end_idx, :, :],
                'actions': self.actions[start_idx:end_idx, :, :], 
                'rewards': self.rewards[start_idx:end_idx, :],  
                'r2a_grads': self.r2a_grads[start_idx:end_idx, start_idx:end_idx, :, :],
                'nv2a_grads': self.nv2a_grads[start_idx:end_idx, start_idx:end_idx, :, :], 
                'log_probs': self.log_probs[start_idx:end_idx, :, :], 
                'done_mask': self.done_mask[start_idx:end_idx, :],
                'target_values': self.target_values[start_idx:end_idx, :],
                'next_values': self.next_values[start_idx:end_idx, :],
                'values': self.values[start_idx:end_idx, :],
                }
'''    