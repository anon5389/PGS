from copy import deepcopy
import pickle
import absl.app
import absl.flags
from .conservative_sac import ConservativeSAC
from .replay_buffer import batch_to_torch, subsample_batch
from .model import TanhGaussianPolicy, FullyConnectedQFunction
from .utils import Timer, define_flags_with_default, set_random_seed, get_user_flags, prefix_metrics
from .utils import WandBLogger
from viskit.logging import logger, setup_logger
from vae import train_vae
from surrogate import prepare_dataloaders, get_surrogate, train_surrogate, load_surrogate
from generate_trajectories import generate_offline_dataset
from evaluate_policy import load_top_observations, generate_designs, oracle_evaluate_designs
import design_bench
import tensorflow as tf
import numpy as np


FLAGS_DEF = define_flags_with_default(
    task="TFBind8-Exact-v0",
    normalize_ys=True,
    normalize_xs=True,

    train_surrogate = False,
    surrogate_hidden_size=2048, 
    surrogate_batch_size=128,
    surrgoate_epochs=50,
    surrogate_lr=0.0003,

    trajectory_length=50,
    n_trajectories=20000,
    top_k=128,
    eval_length=50,

    seed=42,
    device='cuda',
    save_model=True,
    batch_size=256,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,

    policy_arch='256-256',
    qf_arch='256-256',
    
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    top_p=20,
    n_epochs=401,
    save_epoch=50,
    bc_epochs=0,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,

    cql=ConservativeSAC.get_default_config(),
    logging=WandBLogger.get_default_config(),
)

def get_simulated_dataset(path):
    with open(path, 'rb') as handle: 
        traj = pickle.load(handle)
    traj["actions"] = traj["actions"]

    return traj

def main(argv):
    FLAGS = absl.flags.FLAGS

    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=FLAGS.seed,
        base_log_dir=FLAGS.logging.output_dir,
        include_exp_prefix_sub_dir=False
    )

    set_random_seed(FLAGS.seed)
    task = FLAGS.task
    task_name = task.split("-")[0]
    task = design_bench.make(task)

    if FLAGS.normalize_ys:
        task.map_normalize_y()
    if FLAGS.normalize_xs:
        task.map_normalize_x()
    if task.is_discrete:
        vae_model = train_vae(task, task_name=task_name)
        x = vae_model.encoder_cnn.predict(task.x)[0]
    else:
        x = task.x
    
    y = task.y

    x = np.array(x)
    y = np.array(y)

    train_loader, val_loader = prepare_dataloaders(x, y , batch_size=FLAGS.surrogate_batch_size)
    if FLAGS.train_surrogate:
        surrogate = get_surrogate(x.shape[1], FLAGS.surrogate_hidden_size)
        surrogate = train_surrogate(surrogate, train_loader, val_loader, task_name, 
                                    n_epochs=FLAGS.surrogate_epochs, lr=FLAGS.surrogate_lr, device=FLAGS.device)
    else:
        surrogate_path = "surrogate_models/"+task_name+"_surrogate.pt"
        surrogate = load_surrogate(surrogate_path, x.shape[1], FLAGS.surrogate_hidden_size)
    perc = np.percentile(y, 100 - FLAGS.top_p)
    indices = np.where(y[:,0]>=perc)[0]
    x = x[indices]
    y = y[indices]
    start_observations = load_top_observations(x, y, FLAGS.top_k)
    if task.is_discrete:
        scale = 2.0 * np.sqrt(x.shape[1])
    else:
        scale = 0.05 * np.sqrt(x.shape[1])
    dataset = generate_offline_dataset(x, y, surrogate, task_name, 
                                size=FLAGS.trajectory_length, length=FLAGS.n_trajectories)

    dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
    dataset['actions'] = np.clip(dataset['actions'], -FLAGS.clip_action, FLAGS.clip_action)
    
    observation_space_dim = dataset['observations'].shape[1]
    action_space_dim = dataset['actions'].shape[1]

    policy = TanhGaussianPolicy(
        observation_space_dim,
        action_space_dim,
        arch=FLAGS.policy_arch,
        log_std_multiplier=FLAGS.policy_log_std_multiplier,
        log_std_offset=FLAGS.policy_log_std_offset,
        orthogonal_init=FLAGS.orthogonal_init,
    )

    qf1 = FullyConnectedQFunction(
        observation_space_dim,
        action_space_dim,
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf1 = deepcopy(qf1)

    qf2 = FullyConnectedQFunction(
        observation_space_dim,
        action_space_dim,
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf2 = deepcopy(qf2)

    if FLAGS.cql.target_entropy >= 0.0:
        FLAGS.cql.target_entropy = -np.prod((observation_space_dim, 1)).item()

    sac = ConservativeSAC(FLAGS.cql, policy, qf1, qf2, target_qf1, target_qf2)
    sac.torch_to_device(FLAGS.device)

    viskit_metrics = {}
    all_designs = {}
    oracle_scores = []
    for epoch in range(FLAGS.n_epochs):
        metrics = {'epoch': epoch}

        with Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch):
                batch = subsample_batch(dataset, FLAGS.batch_size)
                batch = batch_to_torch(batch, FLAGS.device)
                metrics.update(prefix_metrics(sac.train(batch, bc=epoch < FLAGS.bc_epochs), 'sac'))
                
        if epoch%FLAGS.save_epoch==0 and epoch>0:
            save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
            wandb_logger.save_pickle(save_data, 'model_'+task_name+'_'+str(epoch)+'.pkl')
            designs = generate_designs(sac.policy, surrogate, start_observations, scale=scale,
                                        max_traj_length=FLAGS.eval_length, deterministic=True, device=FLAGS.device)
            if task.is_discrete:
                designs = vae_model.decoder_cnn.predict(designs)
                designs = tf.argmax(designs, axis=2, output_type=tf.int32)

            score = oracle_evaluate_designs(task, task_name, designs, discrete=task.is_discrete)
            oracle_scores.append(score)
            all_designs[epoch] = designs

        metrics['train_time'] = train_timer()
        metrics['epoch_time'] = train_timer() 
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    with open(task_name+"_designs_top_"+str(100 - FLAGS.top_p), 'wb') as handle:
        pickle.dump(all_designs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(task_name+"_oracle_scores_top_"+str(100 - FLAGS.top_p), "wb") as fp:
        pickle.dump(oracle_scores, fp)
    

if __name__ == '__main__':
    absl.app.run(main)
