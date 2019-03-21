import numpy as np

class Reward:
	pass

class StaticReward(Reward):
	def __init__(self, value):
		self.value = value

	def get(self):
		return value

class NormalReward(Reward):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def get(self):
		return np.random.normal(self.mean, self.std)

class Bandit:
	def __init__(self, arms):
		self.no_of_arms = arms
		self.arms = [np.random.normal(0, 1) for _ in range(arms)]

	def step(self, arm):
		return np.random.normal(self.arms[arm], 1)

def episilon_greedy(bandit, no_of_steps, epsilon, seed = None):
	np.random.seed(seed)

	Q = [0] * bandit.no_of_arms
	N = [0] * bandit.no_of_arms

	sum_of_rewards = 0

	for _ in range(no_of_steps):
		if np.random.random() > epsilon:
			# greedy
			action = np.argmax(Q)
		else:
			# random
			action = np.random.choice(bandit.no_of_arms)

		reward = bandit.step(action)

		sum_of_rewards += reward

		N[action] += 1
		Q[action] += (1 / N[action]) * (reward - Q[action])

	mean_reward = sum_of_rewards / no_of_steps

	return mean_reward, Q, N

def main():
	no_of_arms = 10
	no_of_steps = 1000
	epsilon = 0.1

	bandit = Bandit(no_of_arms)
	
	print(episilon_greedy(bandit, no_of_steps, epsilon))


if __name__ == '__main__':
	main()