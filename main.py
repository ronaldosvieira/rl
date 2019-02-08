import numpy as np

class Bandit:
	def __init__(self, arms):
		self.arms = arms

	def step(self, arm):
		return np.random.normal(self.arms[arm], 1)

def generate_bandit(arms):
	return Bandit([np.random.normal(0, 1) for _ in range(arms)])

def main():
	bandit = generate_bandit(10)

if __name__ == '__main__':
	main()