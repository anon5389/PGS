import numpy as np
import torch
import tensorflow as tf
from generate_trajectories import get_grads

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 
def load_top_observations(x, y, top_k):
    top_indices = tf.math.top_k(y[:, 0], k=top_k)[1]
    top_indices = top_indices.numpy()
    top_vals = y[top_indices]
    top_observations = x[top_indices, :]

    return top_observations, top_vals

def step(observation, action, surrogate, scale=10, device=device):
    grads = get_grads(observation, surrogate, device=device)
    next_observation = observation + action * scale * grads 
    
    return next_observation

def generate_designs(policy, surrogate, start_observations, scale=10,
            max_traj_length=50, deterministic=True, device=device):
    
    ## sample trajectories following policy
    for _ in range(max_traj_length):
        obs_tensor = np.array(start_observations)
        obs_tensor = torch.from_numpy(obs_tensor).to(device)

        action = policy(obs_tensor, deterministic=deterministic)#[0, :]
        action = action[0].cpu().detach().numpy()

        next_observation,  = step(start_observations, action, surrogate, scale=scale, vec=True)
        start_observations = next_observation

    return next_observation

def load_y(task_name):
    dic2y = np.load("npy/dic2y.npy", allow_pickle=True).item()
    y_min, y_max = dic2y[task_name]

    return y_min, y_max

def oracle_evaluate_designs(task, task_name, x, discrete=False):
    if discrete:
        x = x.astype(np.int64)
    else:
        x = task.denormalize_x(x)
    y_min, y_max = load_y(task_name)
 
    y = task.predict(x)
    max_y = (np.max(y)-y_min)/(y_max-y_min)
    
    return max_y


        