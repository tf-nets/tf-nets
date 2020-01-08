from tensorflow.keras import initializers
import numpy as np
import math

class PriorProbability(initializers.Initializer):
	def __init__(self, probability = 0.01):
		self.probability = probability

	def get_config(self):
		return {
			'probability': self.probability
		}

	def __call__(self, shape, dtype = np.float32):
		return np.ones(shape) * -math.log((1 - self.probability) / self.probability)