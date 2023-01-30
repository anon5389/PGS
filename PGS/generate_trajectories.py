import numpy as np
import torch
import pickle
from tqdm import tqdm

def get_grads(x, f, device="gpu"):
    x_tensor = torch.from_numpy(x).to(device)
    x_tensor.requires_grad_()
    y = f(x_tensor)
    grad_f_x = torch.autograd.grad(y.sum(), x_tensor)[0].cpu().detach().numpy() 
    
    return grad_f_x

def link(x1, grads):
    ## link observations through gradients
    l =  x1[1:] - x1[:-1]
    action = l / grads
    linked = True
    if  np.isnan(action).any():
        linked = False    
    
    return action , linked


def get_percentiles(y, length):
    percentiles = [ (i+1) * 100/length for i in range(length-1)]
    first_perc = np.percentile(y, 0)
    indices_per_percentile = []
    for j in range(length-1):
        second_perc = np.percentile(y, percentiles[j])
        indices = np.where((y>=first_perc) & (y<second_perc))[0]
        indices_per_percentile.append(indices)
        first_perc = second_perc
    
    indices = np.where(y>=second_perc)[0]
    indices_per_percentile.append(indices)
    return indices_per_percentile

def generate_trajectory(x, y, f, pools, length=50):
    # generate one trajectory with specified length
    bool_ = False
    traj = {'observations':[],
            "next_observations":[],
            'actions':[],
            'rewards':[],
            'dones':[],
            'prediction':[],
            'gradients':[]}
    
    idx_0 = np.zeros(length, dtype=np.int32)
    linked = True
    while not(linked):
        for i in range(length):
            idx_0[i] = np.random.choice(pools[i])

        x_0 =  x[idx_0] 
        grads = get_grads(x_0[:-1], f)
        reward = y[idx_0[1:]] - y[idx_0[:-1]]
        action, linked = link(x_0, x_0, grads)

    #save data
    traj['observations'] = x_0[:-1]
    traj['next_observations'] = x_0[1:]
    traj['actions'] = action
    traj['rewards'] = reward
    traj['gradients'] = grads
    traj['dones'] = np.array([0]*(length-1) + [1])
    
   
    return traj, bool_

def generate_offline_dataset(x, y, surrogate, dataset_name, size=20000, length=50):
    # create a dataset of observation for offline RL
    dataset = {'observations':[],
            'next_observations':[],
            'actions':[],
            'rewards':[],
            'dones':[],}

    pools = get_percentiles(y, length)
    for j in tqdm(range(size)):

        bool_ = True
        while bool_:
            traj, bool_ = generate_trajectory(x, y, surrogate, pools, length)

        dataset['observations'].append(traj['observations'])
        dataset['next_observations'].append(traj['next_observations'])
        dataset['actions'].append(traj['actions'])
        dataset['rewards'].append(traj['rewards'])
        dataset['dones'].append(traj['dones'])

        if (j%5000==0 and j>1) or (j==size-1):
            print("saving dataset ....")
            traj_cleaned = {'observations':[],
                'next_observations':[],
                'actions':[],
                'rewards':[],
                'dones':[],}
            observations = np.array(dataset["observations"])
            next_observations = np.array(dataset["next_observations"])
            actions = np.array(dataset["actions"])
            rewards = np.array(dataset["rewards"])
            dones = np.array(dataset["dones"])

            observations = observations.reshape((-1, observations.shape[-1]))
            next_observations = next_observations.reshape((-1, next_observations.shape[-1]))
            actions = actions.reshape((-1, actions.shape[-1]))
            rewards = rewards.reshape((-1, rewards.shape[-1]))
            dones = dones.reshape((-1, 1))

            
            traj_cleaned['observations'] = observations
            traj_cleaned['next_observations'] = next_observations
            traj_cleaned['actions'] = actions
            traj_cleaned['rewards'] = rewards
            traj_cleaned['dones'] = dones

            with open('traj_'+dataset_name+'.pickle', 'wb') as handle:
                pickle.dump(traj_cleaned, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset


