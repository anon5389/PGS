import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import torch
import pickle
from tqdm import tqdm
# seed = 1234
# np.random.seed(seed)

def get_data(dataset_name, discrete=False):
    #read datasset
    if discrete:
        with open('discrete_data/'+ dataset_name +'.pickle', 'rb') as f:
            dataset = pickle.load(f)
    else:
        with open('continuous_data/'+ dataset_name +'.pickle', 'rb') as f:
            dataset = pickle.load(f)
    
    return dataset["x"], dataset["y"]

def normalize(x):
    #normalize data
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_std = np.where(x_std == 0.0, 1.0, x_std)

    return (x - x_mean) / x_std


# def get_grads(x, f, vec=False):
#     #gradients of f with respect to x
#     x_tensor = torch.from_numpy(x).to(device)
#     x_tensor.requires_grad_()
#     y = f(x_tensor)
#     if vec :
#         grad_f_x = torch.autograd.grad(y.sum(), x_tensor)[0].cpu().detach().numpy()
#     else:
#         grad_f_x = torch.autograd.grad(y, x_tensor)[0].cpu().detach().numpy()
    
#     #set low gradients to zero
#     grad_f_x[np.where(grad_f_x<0.05)] = 0
    
#     return grad_f_x

def get_grads(x, model):
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        y = model(x_tensor)
        y = tf.reduce_sum(y)
    dy_dx = tape.gradient(y, x_tensor).numpy()
    dy_dx[np.where(dy_dx<0.05)] = 0

    return dy_dx


def link(x1, x2, grads, vec=False):
    ## link observations through gradients
    if vec:
        l =  x1[1:] - x1[:-1]
        action = l / grads
        action = np.nan_to_num(action)
        action[np.abs(action) >10] = 0
    else:
        l = x2 - x1
        action = l / grads
        # action = np.where(np.isnan(action) or np.isinf(action), 1.0, action)
        action = np.nan_to_num(action)
        # reward = f.predict(x1)
        # from numpy import inf
        # action[action == inf] = 1
        # action[action == -inf] = 1

        action[np.abs(action) >10] = 0
    
    return action 


def get_percentiles(y, length):
    percentiles = [ (i+1) * 100/length for i in range(length-1)]
    first_perc = np.percentile(y, 0)
    indices_per_percentile = []
    for j in range(length-1):
        second_perc = np.percentile(y, percentiles[j])
        # if j == length-2:
        #     indices = np.where(y>=second_perc)[0]
        # else:
        indices = np.where((y>=first_perc) & (y<second_perc))[0]
        indices_per_percentile.append(indices)
        first_perc = second_perc
    
    indices = np.where(y>=second_perc)[0]
    indices_per_percentile.append(indices)
    return indices_per_percentile

def vectorized_generate_trajectory(x, y, f, pools, length=50, policy="random", advantage=False, plus_grads=False):
    # generate one trajectory with specified length
    bool_ = False
    traj = {'observations':[],
            "next_observations":[],
            'actions':[],
            'rewards':[],
            'dones':[],
            'prediction':[],
            'gradients':[]}

    if policy=="random":
        #sample random observations
        idx_0 = np.random.choice(np.arange(x.shape[0]), length, replace=False)
        x_0 =  x[idx_0] 
        vec = True
        grads = get_grads(x_0[:-1], f)
        grads_next = get_grads(x_0[1:], f)
        if advantage:
            # print("here")
            reward = y[idx_0[1:]] - y[idx_0[:-1]]
        else:
            reward = y[idx_0[1:]]
        action = link(x_0, x_0, grads, vec=vec)
        #save data
        if plus_grads:
            traj['observations'] = np.hstack((x_0[:-1], grads))
            traj['next_observations'] = np.hstack((x_0[1:], grads_next))
        else:
            traj['observations'] = x_0[:-1]
            traj['next_observations'] = x_0[1:]
        traj['actions'] = action
        traj['rewards'] = reward
        # traj['prediction'].append(pred)
        traj['gradients'] = grads
        #episode length is T=50, like in COMS paper: the number of gadient ascent steps 
        traj['dones'] = np.array([0]*48 + [1])

    elif policy=="percentile":
        idx_0 = np.zeros(length, dtype=np.int32)
        for i in range(length):
            idx_0[i] = np.random.choice(pools[i])
        # idx_0 = all_indices #np.random.choice(np.arange(x.shape[0]), length, replace=False)
        x_0 =  x[idx_0] 
        vec = True
        grads = get_grads(x_0[:-1], f)
        if advantage:
            # print("here")
            reward = y[idx_0[1:]] - y[idx_0[:-1]]
        else:
            # print("here")
            reward = y[idx_0[1:]]
        action = link(x_0, x_0, grads, vec=vec)

        #save data
        traj['observations'] = x_0[:-1]
        traj['next_observations'] = x_0[1:]
        traj['actions'] = action
        traj['rewards'] = reward
        # traj['prediction'].append(pred)
        traj['gradients'] = grads
        #episode length is T=50, like in COMS paper: the number of gadient ascent steps 
        traj['dones'] = np.array([0]*48 + [1])
    
    elif policy=="cluster":
        matrix = np.load(dataset_+"_negihbors.npy")
        idx_0 = np.zeros(length, dtype=np.int32)
        sample = np.random.choice(np.arange(x.shape[0]))
        for i in range(length):
            idx_0[i] = np.random.choice(matrix[sample][1:])
            sample = idx_0[i]
        
        x_0 =  x[idx_0] 
        vec = True
        grads = get_grads(x_0[:-1], f)
        if advantage:
            # print("here")
            reward = y[idx_0[1:]] - y[idx_0[:-1]]
        else:
            # print("here")
            reward = y[idx_0[1:]]
        action = link(x_0, x_0, grads, vec=vec)

        #save data
        traj['observations'] = x_0[:-1]
        traj['next_observations'] = x_0[1:]
        traj['actions'] = action
        traj['rewards'] = reward
        # traj['prediction'].append(pred)
        traj['gradients'] = grads
        #episode length is T=50, like in COMS paper: the number of gadient ascent steps 
        traj['dones'] = np.array([0]*48 + [1])
    
    elif policy=="up_neigh":
        matrix = np.load(dataset_+"_100_negihbors.npy")
        idx_0 = np.zeros(length, dtype=np.int32)
        sample = np.random.choice(np.arange(x.shape[0]))
        for i in range(length):
            value = y[sample]
            y_neighbors = y[matrix[sample][1:]]
            # print(y_neighbors)
            y_indices = np.where(y_neighbors>=value)[0]
            print(y_indices)
            # print(value)
            # print(y_indices)
            # print(y[matrix[sample][1:]][y_indices])
            # print("#######################################")
            if len(y_indices)<20 and i==0:
                sample = np.random.choice(np.arange(x.shape[0]))
                value = y[sample]
                y_neighbors = y[matrix[sample][1:]]
                y_indices = np.where(y_neighbors>=value)[0]

            if len(y_indices)<1:
                sample = idx_0[i-1]
                value = y[sample]
                y_neighbors = y[matrix[sample][1:]]
                y_indices = np.where(y_neighbors>=value)[0]
                print(y_indices)
                
            idx_0[i] = np.random.choice(matrix[sample][1:][y_indices])
            sample = idx_0[i]

    elif policy=="up":
        top_k = 50
        idx_0 = np.zeros(length, dtype=np.int32)
        sample = np.random.choice(np.arange(x.shape[0]))
        # top_indices = np.argpartition(y[:, 0], -top_k)[-top_k:]
        for i in range(length):
            value = y[sample]
            y_indices = np.where(y>value)[0]
            # top_indices = sorted(range(len(y[y_indices, 0])), 
            #                     key = lambda sub: y[y_indices, 0][sub])[:top_k]
            if len(y_indices)<51:
                bool_ = True
                break
            top_indices = np.argpartition(y[y_indices, 0], top_k)[:top_k]
            group = y_indices[top_indices]
            distances = np.linalg.norm(x[group] - x[sample], axis=1)
            nearest_neighbor_ids = distances.argsort()[:10]

            idx_0[i] = np.random.choice(group[nearest_neighbor_ids])
            sample = idx_0[i]
            
        x_0 =  x[idx_0] 
        vec = True
        grads = get_grads(x_0[:-1], f)
        if advantage:
            # print("here")
            reward = y[idx_0[1:]] - y[idx_0[:-1]]
        else:
            # print("here")
            reward = y[idx_0[1:]]
        action = link(x_0, x_0, grads, vec=vec)

        #save data
        traj['observations'] = x_0[:-1]
        traj['next_observations'] = x_0[1:]
        traj['actions'] = action
        traj['rewards'] = reward
        # traj['prediction'].append(pred)
        traj['gradients'] = grads
        #episode length is T=50, like in COMS paper: the number of gadient ascent steps 
        traj['dones'] = np.array([0]*48 + [1])        
    
    return traj, bool_

def generate_offline_dataset(x, y, f, dataset_name, size=20000, length=50, policy="random", advantage=False, plus_grads=False):
    # create a dataset of observation for offline RL
    dataset = {'observations':[],
            'next_observations':[],
            'actions':[],
            'rewards':[],
            'dones':[],
            # 'prediction':[],
            'gradients':[]}
    pools = get_percentiles(y, length)
    for j in tqdm(range(size)):
        # traj = generate_trajectory(x, y, f, length, policy)
        bool_ = True
        while bool_:
            traj, bool_ = vectorized_generate_trajectory(x, y, f, pools, length, policy, advantage=advantage, plus_grads=plus_grads)
        # dataset.append(traj)
        dataset['observations'].append(traj['observations'])
        dataset['next_observations'].append(traj['next_observations'])
        dataset['actions'].append(traj['actions'])
        dataset['rewards'].append(traj['rewards'])
        dataset['dones'].append(traj['dones'])
        # dataset['prediction'].append(traj['prediction'])
        dataset['gradients'].append(traj['gradients'])

        if (j==size-1):
            print("saving dataset ....")
            traj_cleaned = {'observations':[],
                'next_observations':[],
                'actions':[],
                'rewards':[],
                'dones':[],
                # 'prediction':[],
                'gradients':[]}
            observations = np.array(dataset["observations"])
            next_observations = np.array(dataset["next_observations"])
            actions = np.array(dataset["actions"])
            rewards = np.array(dataset["rewards"])
            dones = np.array(dataset["dones"])
            # prediction = np.array(dataset["prediction"])
            gradients = np.array(dataset["gradients"])

            observations = observations.reshape((-1, observations.shape[-1]))
            next_observations = next_observations.reshape((-1, next_observations.shape[-1]))
            actions = actions.reshape((-1, actions.shape[-1]))
            rewards = rewards.reshape((-1, rewards.shape[-1]))
            dones = dones.reshape((-1, 1))
            # prediction = prediction.reshape((-1, prediction.shape[-1]))
            gradients = gradients.reshape((-1, gradients.shape[-1]))

            
            traj_cleaned['observations'] = observations
            traj_cleaned['next_observations'] = next_observations
            traj_cleaned['actions'] = actions
            traj_cleaned['rewards'] = rewards
            traj_cleaned['dones'] = dones
            # traj_cleaned['prediction'] = prediction
            traj_cleaned['gradients'] = gradients
            with open('traj_'+dataset_name+'_rob_'+policy+'_adv_'+str(advantage)+'_seed_'+str(seed)+'_size_'+str(size)+'.pickle', 'wb') as handle:
                pickle.dump(traj_cleaned, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset


if __name__ == '__main__':
    ## read data
    discrete = True
    seed = "no_5"
    size = 20000 #// 2
    length = 50
    policy = "random"
    advantage = True
    plus_grads = False
    if discrete:
        dataset_names = ['TFBind8', 'GFP', 'UTR']
    else:
        dataset_names = ["AntMorphology", "DKittyMorphology", "HopperController", "Superconductor"]
    # 
    dataset_ = dataset_names[2] + "_32"
    model = keras.models.load_model('utr_model')

    x, y = get_data(dataset_, discrete=discrete)
    input_shape = x.shape[1:]
    # print(input_shape)
    # print(2 * np.sqrt(np.prod(input_shape)))
    # sys.exit("Error message")
    if discrete:
        # print(x.max(), x.min())
        # print(y.max(), y.min())
        # sys.exit("Error message")
        x_normalized = x #.numpy()
        y_normalized = y
    else:
        x_normalized = normalize(x)
        y_normalized = normalize(y)
    # x_normalized = x #.numpy()
    # y_normalized = y
    ## load surrogate
    # print(np.unique(y, return_counts=True))
    # sys.exit()

    
    ## Generate dataset
    print(dataset_, seed)
    dataset = generate_offline_dataset(x_normalized, y_normalized, model, dataset_, 
                                        size=size, length=length, policy=policy, 
                                        advantage=advantage, plus_grads=plus_grads)
    # print(np.array(dataset["rewards"]).reshape(10, 49)[:2])
    # with open('traj_'+dataset_+'_grads_aware.pickle', 'wb') as handle:
    #     pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
