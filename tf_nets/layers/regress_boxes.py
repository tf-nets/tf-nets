import tensorflow as tf
import numpy as np
import tf_nets

class RegressBoxes(tf.keras.layers.Layer):
	def __init__(self, mean = None, std = None, *args, **kwargs):
		if mean is None:
			mean = np.array([0, 0, 0, 0])
		if std is None:
			std = np.array([0.2, 0.2, 0.2, 0.2])

		if isinstance(mean, (list, tuple)):
			mean = np.array(mean)
		if isinstance(std, (list, tuple)):
			std = np.array(std)

		self.mean = mean
		self.std = std
		super(RegressBoxes, self).__init__(*args, **kwargs)

	def call(self, inputs, **kwargs):
		anchors, localization = inputs
		return tf_nets.bbox_transform(anchors, localization, mean = self.mean, std = self.std)

	def compute_output_shape(self, input_shape):
		return input_shape[0]

	def get_config(self):
		config = super(RegressBoxes, self).get_config()
		config.update({
			'mean': self.mean,
			'std': self.std
		})
		return config