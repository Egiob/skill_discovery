from torch.optim import optimizer
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

import numpy as np

from stable_baselines3 import DIAYN, SAC, PPO


from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.environment import UnityEnvironment

from gym_unity.envs import UnityToGymWrapper
import os
from stable_baselines3.common.exp_utils import linear_schedule, multi_step_schedule, export_to_onnx


@hydra.main(config_path="./", config_name="config")
def run_exp(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    cwd = os.getcwd()
    print("Working directory : {}".format(os.getcwd()))
    DIAYN_cfg = cfg['DIAYN']
    device = DIAYN_cfg['device']
    env_id = DIAYN_cfg['env_id']
    prior_type = DIAYN_cfg['prior_type']
    n_skills= DIAYN_cfg['n_skills']
    buffer_size = int(DIAYN_cfg['buffer_size'])
    ent_coef = DIAYN_cfg['ent_coef']
    seed = DIAYN_cfg['seed']
    steps = DIAYN_cfg['steps']
    train_freq = tuple(DIAYN_cfg['train_freq'])
    gradient_steps = DIAYN_cfg['gradient_steps']
    learning_starts = DIAYN_cfg['learning_starts']
    batch_size = DIAYN_cfg['batch_size']
    tau = DIAYN_cfg['tau']
    gamma = DIAYN_cfg['gamma']
    
    
    disc_on = DIAYN_cfg['discriminator']['input']
    disc_arch = DIAYN_cfg['discriminator']['net_arch']
    combined_rewards = DIAYN_cfg['combined_rewards']['enable']
    smerl = DIAYN_cfg['combined_rewards']['smerl']
    eps = DIAYN_cfg['combined_rewards']['eps']
    beta = DIAYN_cfg['combined_rewards']['beta']
    policy_class = DIAYN_cfg['policy']['class']
    policy_arch = DIAYN_cfg['policy']['net_arch']
    policy_optimizer_name = DIAYN_cfg['policy']['optimizer']
    lr_schedule_type = DIAYN_cfg['policy']['lr_schedule']['type']
    
    assert policy_optimizer_name in ['Adam', 'SGD']
    if policy_optimizer_name == 'Adam':
        policy_optim_class = torch.optim.Adam
    elif policy_optimizer_name == 'SGD':
        policy_optim_class = torch.optim.SGD
        momentum = DIAYN_cfg['policy']['momentum']
    optimizer_kwargs = {'momentum': momentum}
    policy_kwargs = {'optimizer_kwargs':optimizer_kwargs,
                     'optimizer_class' : policy_optim_class}
    assert lr_schedule_type in ['constant', 'linear', 'multi-step']
    if lr_schedule_type=='constant':
        lr = DIAYN_cfg['policy']['lr_schedule']['init_value']
    elif lr_schedule_type=='linear':
        init_value = DIAYN_cfg['policy']['lr_schedule']['init_value']
        fin_value = DIAYN_cfg['policy']['lr_schedule']['fin_value']
        end = DIAYN_cfg['policy']['lr_schedule']['end']
        lr = linear_schedule(init_value, fin_value, end)
    elif lr_schedule_type=='multi-step':
        values = DIAYN_cfg['policy']['lr_schedule']['values']
        milestones = DIAYN_cfg['policy']['lr_schedule']['milestones']
        lr = multi_step_schedule(values, milestones)


        
    build_path = DIAYN_cfg['unity']['build_path']
    time_scale = DIAYN_cfg['unity']['time_scale']
    worker_id = DIAYN_cfg['unity']['worker_id']
    
    
    assert prior_type == 'categorical-uniform'
    if prior_type == 'categorical-uniform':
        prior = torch.distributions.OneHotCategorical(probs = 1/n_skills * torch.ones(n_skills))
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale=time_scale)

    if policy_arch is not None:
        policy_kwargs['net_arch'] = list(policy_arch)

    discriminator_kwargs = {}
    if disc_arch is not None:
        discriminator_kwargs['net_arch'] = list(disc_arch)
        
    env = UnityEnvironment(build_path,no_graphics=True,worker_id=worker_id,side_channels=[channel])
    gym_env = UnityToGymWrapper(env, allow_multiple_obs=False)
    print(type(ent_coef))
    model = DIAYN(policy=policy_class,
              env=gym_env,
              prior=prior,
              verbose=0,
              device=device,
              tensorboard_log=os.path.join(cwd,'logs'),
              train_freq=train_freq,
              gradient_steps=gradient_steps,
              buffer_size=buffer_size,
              tau=tau,
              gamma=gamma,
              learning_starts=learning_starts,
              learning_rate=lr,
              batch_size=batch_size,
              ent_coef=ent_coef,
              disc_on = disc_on,
              discriminator_kwargs=discriminator_kwargs,
              seed = seed,
              combined_rewards = combined_rewards,
              beta=beta,
              smerl=smerl,
              eps=eps,
              policy_kwargs=policy_kwargs,
         )
    model.learn(total_timesteps=steps, log_interval=5)
    model.save(os.path.join(cwd,'model'))
    export_to_onnx(model, os.path.join(cwd,'model'))
    
if __name__ == "__main__":
    run_exp()