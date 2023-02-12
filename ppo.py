"""
The file contains the PPO class to train with.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter

from ICM import ICM


class PPO:
    def __init__(self, policy_class, env, **hyperparameters):
        """
        Initializes the PPO model, including hyper-parameters.
        """

        # Create writer for tensorboard
        self.writer = SummaryWriter()

        # Initialize env and hyper-parameters
        self.env = env
        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Initialize networks
        self.actor = policy_class(self.obs_dim, self.act_dim).to(self.device)
        self.critic = policy_class(self.obs_dim, 1).to(self.device)

        # Initialize optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # Initialize curiosity module
        if self.icm:
            self.icm = ICM(self.obs_dim, self.act_dim)
            self.icm_optim = torch.optim.Adam(self.icm.parameters(), lr=self.actor_lr)

        # Create the covariance matrix for get_action
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5, device=self.device)
        self.cov_mat = torch.diag(self.cov_var)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,  # timesteps so far
            'i_so_far': 0,  # iterations so far
            'batch_lengths': [],  # episodic lengths in batch
            'batch_rewards': [],  # episodic returns in batch
            'actor_losses': [],  # losses of actor network in current iteration
            'critic_losses': [],  # losses of critic network in current iteration
            'total_time': 0  # time from start of the learning
        }

    def learn(self, total_timesteps):
        """
        Train the actor and critic networks. Here is where the main PPO algorithm resides
        """

        while self.logger['t_so_far'] < total_timesteps:
            states, actions, old_log_pi, returns, rewards, episodes_lengths = self.rollout()

            # Increment the number of iterations and time steps so far
            self.logger['t_so_far'] += np.sum(episodes_lengths)
            self.logger['i_so_far'] += 1

            values = self.critic(states).squeeze()
            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

            for _ in range(self.epochs):
                for batch in range(0, self.buffer_size, self.batch_size):
                    values, curr_log_pi = self.evaluate(states[batch:batch + self.batch_size],
                                                        actions[batch:batch + self.batch_size])

                    ratios = torch.exp(curr_log_pi - old_log_pi[batch:batch + self.batch_size])

                    # Calculate surrogate losses
                    surr1 = ratios * advantages[batch:batch + self.batch_size]
                    surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages[
                                                                                                batch:batch + self.batch_size]

                    entropy = (-curr_log_pi * torch.exp(curr_log_pi)).sum()
                    actor_loss = -(torch.min(surr1, surr2)).mean() - self.entropy_strength * entropy
                    critic_loss = nn.MSELoss()(values, returns[batch:batch + self.batch_size])

                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    self.actor_optim.step()

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()

                    self.logger['actor_losses'].append(actor_loss.detach().item())
                    self.logger['critic_losses'].append(critic_loss.detach().item())

                if self.icm:
                    self.icm.zero_grad()
                    _, icm_loss = self.icm.calc_loss(states=states[:-1], next_states=states[1:],
                                                     actions=actions[:-1])
                    icm_loss.backward()
                    self.icm_optim.step()

    def rollout(self):
        """
        This is where we collect the batch of data from simulation. Since this is an on-policy algorithm,
        we will need to collect a fresh batch of data each time we iterate the actor/critic networks.
        """

        # Batch data
        states = []
        actions = []
        log_probs = []
        episodes_lengths = []
        rewards_per_ep = []
        dones = []

        t = 0  # Keeps track of how many timesteps we've run so far this batch

        while t < self.buffer_size:
            ep_rewards = []  # rewards collected per episode
            ep_t = 0
            state = self.env.reset()
            done = False
            while ep_t < self.max_episode_length and not done:
                self.save_render_log(batch_step=t, batch_ep=len(episodes_lengths))
                t += 1
                ep_t += 1

                action, log_prob = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)

                states.append(torch.tensor(state, device=self.device, dtype=torch.float32))
                actions.append(torch.tensor(action, device=self.device, dtype=torch.float32))
                log_probs.append(log_prob)
                ep_rewards.append(reward)
                dones.append(done)

                state = next_state

            # Track episodic length and rewards
            episodes_lengths.append(ep_t)
            rewards_per_ep.append(ep_rewards)

        states = torch.stack(states)
        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)

        all_rewards = sum(rewards_per_ep, [])

        if self.icm:
            intrinsic_rewad, _ = self.icm.calc_loss(states=states[:-1], next_states=states[1:], actions=actions[:-1])
            intrinsic_rewad = np.concatenate([intrinsic_rewad, np.array([0])])
            all_rewards = np.array(all_rewards) + np.clip(intrinsic_rewad, 0, 1)

        all_returns = self.get_returns(all_rewards, dones)
        # advantages = self.gae(states, all_rewards, dones)

        self.logger['batch_rewards'] = rewards_per_ep
        self.logger['batch_lengths'] = episodes_lengths

        return states, actions, log_probs, all_returns, all_rewards, episodes_lengths  # , advantages

    def gae(self, states, rewards, dones):
        values = self.critic(states).squeeze().detach().cpu().numpy()
        advantages = np.zeros(len(rewards), dtype=np.float32)

        for t in range(len(rewards) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards) - 1):
                a_t += discount * (rewards[k] + self.gamma * values[k + 1] * (1 - int(dones[k])) - values[k])
                discount *= self.gamma * self.gae_lambda
            advantages[t] = a_t
        advantages = torch.tensor(advantages).to(self.device)
        return advantages

    def get_returns(self, batch_rewards, batch_dones):
        """
        Compute the Reward-To-Go of each timestep in a batch given the rewards.
        """

        batch_returns = []
        discounted_return = 0
        for time_step, (reward, done) in enumerate(zip(reversed(batch_rewards), reversed(batch_dones))):
            if done or time_step % self.max_episode_length == 0:
                discounted_return = 0
            discounted_return = reward + discounted_return * self.gamma
            batch_returns.insert(0, discounted_return)

        batch_returns = torch.tensor(batch_returns, device=self.device, dtype=torch.float32)
        return batch_returns

    def get_action(self, obs):

        mean = self.actor(obs)  # Same thing as calling self.actor.forward(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().cpu().numpy(), log_prob.detach().cpu()

    def evaluate(self, batch_states, batch_actions):
        """
        Estimate the values of each observation, and the log probs of each action in the most recent batch
        with the most recent iteration of the actor network. Should be called from learn.
        """

        V = self.critic(batch_states).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        mean = self.actor(batch_states)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_actions)

        return V, log_probs

    def save_render_log(self, batch_step, batch_ep):
        if self.logger['i_so_far'] == 0:
            return

        if batch_step == 0:
            self._log_summary()  # print
            if self.logger['i_so_far'] % self.save_freq == 0:  # save
                itr = self.logger['i_so_far']
                torch.save(self.actor.state_dict(), f"./SavedNets/ClimbingStep_actor_{itr}_iter.pth")
                torch.save(self.critic.state_dict(), f"./SavedNets/ClimbingStep_critic_{itr}_iter.pth")
                print("The model has been saved", flush=True)

        if self.render and self.logger['i_so_far'] % self.render_every_i == 0 and batch_ep == 0:  # render
            self.env.render()

        elif self.render and batch_ep == 1:  # stop render
            self.env.close()

    def _init_hyperparameters(self, hyperparameters):

        # Algorithm hyper-parameters
        self.buffer_size = 2048
        self.batch_size = 512
        self.max_episode_length = 1600
        self.epochs = 5
        self.actor_lr = 0.0003
        self.critic_lr = 0.002
        self.gamma = 0.95
        self.epsilon_clip = 0.2
        self.entropy_strength = 0.001
        self.gae_lambda = 0.95
        self.seed = None
        self.icm = None
        self.device = "cpu"

        # Technical parameters
        self.render = True,
        self.render_every_i = 10  # Only render every n iterations
        self.save_freq = 30  # How often we save in number of iterations

        print("\nHyper-Parameters:\n" + "-" * 17 + "\n", flush=True)
        for param, val in hyperparameters.items():  # print hyper-parameters and save them variables
            exec('self.' + param + ' = ' + str(val))
            print(f"{param}: {val}", flush=True)
        print(f"device: {self.device}", flush=True)

        # Sets the seed if specified
        if self.seed is not None:
            assert type(self.seed) == int

            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            # self.env.seed(self.seed)
            # self.env.action_space.seed(self.seed)

    def _log_summary(self):
        """
        Print to stdout what we have logged so far in the most recent batch.
        """

        # Calculate logging values.
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        self.logger['total_time'] += delta_t

        total_hours = self.logger['total_time'] // 3600
        total_minutes = self.logger['total_time'] // 60 - total_hours * 60

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']

        avg_ep_lens = np.mean(self.logger['batch_lengths'])
        avg_ep_returns = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rewards']])
        avg_actor_loss = np.mean(self.logger['actor_losses'])
        avg_critic_loss = np.mean(self.logger['critic_losses'])

        # Write to tensorboard
        self.writer.add_scalar('Per Iteration/Average Episode Return', avg_ep_returns, i_so_far)
        self.writer.add_scalar('Per Iteration/Average Episode Length', avg_ep_lens, i_so_far)
        self.writer.add_scalar('Per Iteration/Average Actor Loss', avg_actor_loss, i_so_far)
        self.writer.add_scalar('Per Iteration/Average Critic Loss', avg_critic_loss, i_so_far)

        self.writer.add_scalar('Per Steps/Average Episode Return', avg_ep_returns, t_so_far)
        self.writer.add_scalar('Per Steps/Average Episode Length', avg_ep_lens, t_so_far)
        self.writer.add_scalar('Per Steps/Average Actor Loss', avg_actor_loss, t_so_far)
        self.writer.add_scalar('Per Steps/Average Critic Loss', avg_critic_loss, t_so_far)

        # Print logging statements
        print(flush=True)
        print("-------------------- Iteration #{} --------------------".format(self.logger['i_so_far']), flush=True)
        print("Average Episodic Length: {:.2f}".format(avg_ep_lens), flush=True)
        print("Average Episodic Return: {:.3f}".format(avg_ep_returns), flush=True)
        print("Average Actor Loss: {:.5f}".format(avg_actor_loss), flush=True)
        print("Timesteps So Far: {}".format(self.logger['t_so_far']), flush=True)
        print("Iteration took: {:.2f} secs".format(delta_t), flush=True)
        print("Total learning time: Hours: {:.0f} | Minutes: {:.0f}".format(total_hours, total_minutes), flush=True)
        print("------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lengths'] = []
        self.logger['batch_rewards'] = []
        self.logger['actor_losses'] = []
        self.logger['critic_losses'] = []
