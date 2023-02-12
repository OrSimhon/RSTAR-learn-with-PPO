"""
This file is used only to evaluate our trained policy
"""


def log_summary(ep_len, ep_ret, ep_num):
    """
    Print to stdout what we have logged so far in the most recent episode.
    """

    # Round decimal places for more aesthetic logging messages
    ep_len = str(round(ep_len, 2))
    ep_ret = str(round(ep_ret, 2))

    # Print logging statements
    print(flush=True)
    print("------------------------- Episode #{} -------------------------".format(ep_num), flush=True)
    print("Episode Length: {}".format(ep_len), flush=True)
    print("Episodic Return: {}".format(ep_ret), flush=True)
    print("---------------------------------------------------------------")
    print(flush=True)


def rollout(policy, env, render):
    """
    Returns a generator to roll out each episode given a trained policy and environment to test on
    :return: A generator object rollout
    """

    while True:
        obs = env.reset()
        done = False
        t = 0
        ep_return = 0  # Episodic return

        while not done:
            t += 1

            if render:
                env.render()

            # Query deterministic action from policy and perform it
            action = policy(obs).detach().numpy()
            obs, rew, done, _ = env.step(action)

            # Sum all episodic rewards as we go along
            ep_return += rew

        yield t, ep_return


def eval_policy(policy, env, render=False):
    """
    The main function to evaluate our policy with. It will iterate a generator object "rollout",
    which will simulate each episode and return the most recent episode's length and return.
    """
    for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, env, render)):
        log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)