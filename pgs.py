from copy import deepcopy
import pickle

import numpy as np


import absl.app
import absl.flags

from .conservative_sac import ConservativeSAC
from .replay_buffer import batch_to_torch, get_d4rl_dataset, subsample_batch
from .model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from .sampler import StepSampler, TrajSampler
from .utils import Timer, define_flags_with_default, set_random_seed, print_flags, get_user_flags, prefix_metrics
from .utils import WandBLogger
from viskit.logging import logger, setup_logger

FLAGS_DEF = define_flags_with_default(
    max_traj_length=60,
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

    n_epochs=300,
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

    ## path to offline data
    path="SimpleSAC/traj_UTR_32_seed_no_5_size_20000.pickle"

    task_name = "utr_20k_no_5"
    dataset = get_simulated_dataset(path)
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
    for epoch in range(FLAGS.n_epochs):
        metrics = {'epoch': epoch}

        with Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch):
                batch = subsample_batch(dataset, FLAGS.batch_size)
                batch = batch_to_torch(batch, FLAGS.device)
                metrics.update(prefix_metrics(sac.train(batch, bc=epoch < FLAGS.bc_epochs), 'sac'))
                
        if FLAGS.save_model:
            if epoch%100==0 and epoch>0:
                save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
                wandb_logger.save_pickle(save_data, 'model_'+task_name+'_'+str(epoch)+'.pkl')
            else:
                save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
                wandb_logger.save_pickle(save_data, 'model_'+task_name+'.pkl')

        metrics['train_time'] = train_timer()
        metrics['epoch_time'] = train_timer() 
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)



if __name__ == '__main__':
    absl.app.run(main)
