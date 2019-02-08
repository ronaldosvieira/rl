import numpy as np

class Bandit:
	def __init__(self, arms):
		self.arms = [np.random.normal(0, 1) for _ in range(arms)]

	def step(self, arm):
		return np.random.normal(self.arms[arm], 1)

def main():
	no_of_arms = 10
	max_steps = 1000
	epsilon = 0.1

	bandit = Bandit(no_of_arms)
	Q = [0] * no_of_arms
	N = [0] * no_of_arms

	sum_of_rewards = 0

	for _ in range(max_steps):
		if np.random.random() > epsilon:
			# greedy
			action = np.argmax(Q)
		else:
			# random
			action = np.random.choice(no_of_arms)

		reward = bandit.step(action)

		sum_of_rewards += reward

		N[action] += 1
		Q[action] += (1 / N[action]) * (reward - Q[action])

	print(sum_of_rewards / max_steps)
	print(bandit.arms)
	print(Q)
	print(N)


if __name__ == '__main__':
	main()