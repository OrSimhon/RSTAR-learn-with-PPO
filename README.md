# RSTAR Learn to climb over a step with RL

This repository presents an implementation of the Proximal Policy Optimization [1] (PPO-Clip) reinforcement learning algorithm. The environment used in this project was built using the Unity3D ML-Agents toolkit.

### Algorithms and Techniques
* Curiosity term was added by the Intrinsic Curiosity Module (ICM) algorithm [2].
* The advantage function is calculated using the Generalized Advantage Estimator (GAE) algorithm [3].
* An entropy term is added to the policy loss to encourage exploration.

### Requirements
* PyTorch
* Numpy

### Environment
The environment in this project features the RSTAR [4], a crawling robot capable of changing its shape and center of mass location through its sprawl and four-bar extension mechanisms. This allows the robot to adjust its height and width dimensions and pitch upwards, enabling it to overcome obstacles such as steps and navigate through narrow passages.

<!-- ![](https://github.com/OrSimhon/RSTAR-learn2climb-with-PPO/blob/master/Ep_230_AdobeExpress.gif) -->
<p align="center">
  <img src="https://github.com/OrSimhon/RSTAR-learn2climb-with-PPO/blob/master/Ep_230_AdobeExpress.gif">
</p>

The goal of the RSTAR in this environment is to climb over a step in an energy-efficient and time-efficient manner. The robot receives the height of the step as a feature when observing the environment and must take effective action based on this information.

[1] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1705.05363)\
[2] [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363)\
[3] [Generalized Advantage Estimator](https://arxiv.org/abs/1506.02438)\
[4] [Rising STAR, a highly reconfigurable sprawl tuned robot](https://ieeexplore.ieee.org/document/8289322)
