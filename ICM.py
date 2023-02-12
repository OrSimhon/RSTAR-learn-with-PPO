import torch as T
import torch.nn as nn
import torch.nn.functional as F


class ICM(nn.Module):
    def __init__(self, obs_dim, act_dim, alpha=0.3, beta=0.2):
        super().__init__()
        self.beta = beta
        self.alpha = alpha

        # Forward model
        self.frw_layer1 = nn.Linear(obs_dim + act_dim, 256)
        self.frw_layer2 = nn.Linear(256, obs_dim)

    def forward(self, state, action):
        # forward model
        # action = action.reshape((action.size()[0], 1))
        forward_input = T.cat([state, action], dim=1)
        frw_activation_1 = F.elu(self.frw_layer1(forward_input))
        eval_next_state = self.frw_layer2(frw_activation_1)

        return eval_next_state

    def calc_loss(self, states, next_states, actions):
        eval_next_state = self.forward(states, actions)
        L_F = self.beta * nn.MSELoss()(eval_next_state, next_states)
        intrinsic_reward = self.alpha * ((eval_next_state - next_states).pow(2)).mean(dim=1)

        return intrinsic_reward.detach().cpu().numpy(), L_F
