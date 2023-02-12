import sys
import torch
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from arguments import get_args
from ppo import PPO
from network import FeedForwardNN
from eval_policy import eval_policy


def train(env, hyperparameters, actor_model, critic_model):
    """
    Trains the model.
    """

    model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)

    # Tries to load in an existing actor/critic model to continue training on
    if actor_model != '' and critic_model != '':
        print("Loading in {} and {}...".format(actor_model, critic_model), flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print("Successfully loaded.", flush=True)

    # Don't train from scratch if user accidentally forgets actor/critic model
    elif actor_model != '' or critic_model != '':
        print("Error: Either specify both actor/critic model or none at all", flush=True)
        sys.exit(0)
    else:
        print("\n*** Training from scratch ***", flush=True)

    model.learn(total_timesteps=15_000_000)


def test(env, actor_model):
    """
    Tests the model
    """

    print("Testing {}".format(actor_model), flush=True)

    # If the actor model is not specified, then exit
    if actor_model == '':
        print("Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    # Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Build policy and load in the saved actor model
    policy = FeedForwardNN(obs_dim, act_dim)
    policy.load_state_dict(torch.load(actor_model))

    # Evaluate our policy
    eval_policy(policy=policy, env=env, render=True)


def main(args):
    """
    The main function to run
    """

    hyperparameters = {
        'buffer_size': 2048,
        'batch_size': 512,
        'max_episode_length': 20_000,
        'gamma': 0.99,
        'epochs': 10,
        'render': True,
        'actor_lr': 3e-4,
        'critic_lr': 3e-3,
        'epsilon_clip': 0.2,
        'gae_lambda': 0.95,
        'entropy_strength': 0.001,
        'icm': True,
        'seed': 1234
    }

    engineConfigChannel = EngineConfigurationChannel()
    unity_env = UnityEnvironment(file_name='RSTAR_Simulation', seed=hyperparameters['seed'],
                                 side_channels=[engineConfigChannel])
    env = UnityToGymWrapper(unity_env=unity_env, action_space_seed=hyperparameters['seed'])

    if args.mode == 'train':
        engineConfigChannel.set_configuration_parameters(width=1028, height=768, quality_level=1, time_scale=20)
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
    else:
        engineConfigChannel.set_configuration_parameters(width=1028, height=768, quality_level=1, time_scale=1)
        test(env=env, actor_model=args.actor_model)


if __name__ == '__main__':
    args = get_args()  # Parse arguments from command line
    main(args)
