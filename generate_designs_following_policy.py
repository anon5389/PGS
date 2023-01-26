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

def normalize(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_std = np.where(x_std == 0.0, 1.0, x_std)

    return (x - x_mean) / x_std


if __name__ == '__main__':
    discrete = True
    random = False
    max_traj_length = 50
    ## read task data
    if discrete:
        dataset_names = ['TFBind8', 'GFP', 'UTR']
    else:
        dataset_names = ["AntMorphology", "DKittyMorphology", "Superconductor"]
   
    dataset_ = dataset_names[1]
    top_k = 128 
    if discrete:
        x, y = get_data(dataset_ + "_32", discrete=discrete)
        surrogate_path = dataset_ + "_32_std_dnn.pt"
    else:
        x, y = get_data(dataset_, discrete=discrete)
        surrogate_path = dataset_ + "_std_dnn.pt"

    if discrete :
        x_normalized = x 
    else:
        x_normalized = normalize(x)
    
    ## get top designs in data
    start_observations, _ = load_top_observations(x_normalized, y, top_k=top_k)
    
    ## load surrogate model
    print(start_observations.shape[1])
    surrogate = get_surrogate(surrogate_path, input_shape=start_observations.shape[1])
    if discrete:
        scale = [2.0 * np.sqrt(x.shape[1])]
    else:
        scale = [0.05 * np.sqrt(x.shape[1])]
        

    for no in range(5):
        ## Load trained agent model_utr_rnd_adv_20k_no_2_100.pkl
        agent_path = "agents/" + dataset_ + "/agent_no_"+str(no+1)+".pkl"
        save_path = "evaluation/" + dataset_ + "/final_trajs_" + str(top_k) + ".pickle"
        
        with open(agent_path, 'rb') as handle:
            agent = pickle.load(handle)
        
        policy = agent["sac"].policy
        
        get_agent_trajectories(policy, surrogate, start_observations, save_path, 
                                        max_traj_length=max_traj_length, scale=scale, random=random)
        