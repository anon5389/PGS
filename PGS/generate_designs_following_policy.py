import pickle
import numpy as np
import torch
import tensorflow as tf
from train_surrogate import NeuralNetwork
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_surrogate(path, input_shape = 60, hidden_size = 2048):
    ## load surrogate model
    model = NeuralNetwork(input_shape, hidden_size).to(device)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model.float()


def get_grads(x, f):
    x_tensor = torch.from_numpy(x).to(device)
    x_tensor.requires_grad_()
    y = f(x_tensor)

    grad_f_x = torch.autograd.grad(y.sum(), x_tensor)[0].cpu().detach().numpy() 
    
    return grad_f_x

def step(observation, action, surrogate, scale=10, device=device):

    grads = get_grads(observation, surrogate)
    next_observation = observation + action * scale * grads 
    next_observation_tensor = torch.from_numpy(next_observation).to(device).float()
    surrogate_reward = surrogate(next_observation_tensor)
    rwrd = surrogate_reward.cpu().detach().numpy()
    
    return next_observation, rwrd


def vec_sample(policy, surrogate, start_observations, scale=10,
            max_traj_length=50, deterministic=False, device=device, random=False):
    
    ## sample trajectories following policy
    trajs = []
    observations = []
    actions = []
    next_observations = []

    for _ in range(max_traj_length):
        obs_tensor = np.array(start_observations)
        obs_tensor = torch.from_numpy(obs_tensor).to(device)

        action = policy(obs_tensor, deterministic=deterministic)#[0, :]
        action = action[0].cpu().detach().numpy()

        next_observation, surrogate_reward = step(start_observations, action, surrogate, scale=scale, vec=True)
        observations.append(start_observations)
        actions.append(action)
        next_observations.append(next_observation)
        start_observations = next_observation

    trajs.append(dict(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        next_observations=np.array(next_observations, dtype=np.float32),
        ))

    return trajs

def get_agent_trajectories(policy, surrogate, start_observations, path, max_traj_length=50, scales=[10,], random=False):
    metrics = {}
    trajs = {str(key):[] for key in scales}
    for sc_ in tqdm(scales):
        trajs[str(sc_)] = vec_sample(policy, surrogate, start_observations, scale=sc_, max_traj_length=max_traj_length, deterministic=True, random=random)

    with open(path, 'wb') as handle:
        pickle.dump(trajs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return metrics
 
def load_top_observations(x, y, top_k):
    top_indices = tf.math.top_k(y[:, 0], k=top_k)[1]
    top_indices = top_indices.numpy()
    top_vals = y[top_indices]
    top_observations = x[top_indices, :]

    return top_observations, top_vals

def get_data(dataset_name, discrete=False):
    # 
    if discrete:
        with open('discrete_data/'+ dataset_name +'.pickle', 'rb') as f:
            dataset = pickle.load(f)
    else:
        with open('continuous_data/'+ dataset_name +'.pickle', 'rb') as f:
            dataset = pickle.load(f)    
        
    
    return dataset["x"], dataset["y"]



        