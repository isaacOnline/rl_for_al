class scorer():
    def __init__(self):
        pass

    def score(self, fout, env, trials = 10000):
        observation = env.reset()
        total_dist = this_round_dist = 0
        total_num_samples = this_round_num_samples = 0
        total_reward = this_round_reward = 0
        num_runs = 0
        N = len(fout) - 1
        while num_runs < trials:
            true_mvmt = fout[int(observation)]
            pct_into_hyp_space = int(true_mvmt / (observation / N))
            observation, reward, done, _ = env.step(pct_into_hyp_space)
            this_round_dist += true_mvmt
            this_round_num_samples += 1
            this_round_reward += reward
            if done:
                num_runs += 1
                total_dist += this_round_dist
                total_num_samples += this_round_num_samples
                total_reward += this_round_reward
                observation = env.reset()

                # reset things we're keeping track of
                this_round_dist = 0
                this_round_num_samples = 0
                this_round_reward = 0

        avg_reward = total_reward / num_runs
        avg_ns = total_num_samples / num_runs
        avg_dist = total_dist / num_runs
        print(f'num runs: {num_runs}')
        print(f'avg reward: {avg_reward}')
        print(f'avg ns: {avg_ns}')
        print(f'avg dist: {avg_dist}')
        return avg_reward, avg_ns, avg_dist