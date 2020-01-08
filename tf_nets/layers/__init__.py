import tensorflow as tf
import numpy as np
from tensorflow import keras
import tf_nets

class UpsampleLike(keras.layers.Layer):
	def call(self, inputs, **kwargs):
		source, target = inputs
		target_shape = tf.shape(target)
		return tf.compat.v1.image.resize_images(source, (target_shape[1], target_shape[2]), tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners = False)

	def compute_output_shape(self, input_shape):
		return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1])

class RegressBoxes(keras.layers.Layer):
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

class ClipBoxes(keras.layers.Layer):
	def call(self, inputs, **kwargs):
		image, boxes = inputs
		shape = keras.backend.cast(tf.shape(image), keras.backend.floatx())
		_, height, width, _ = tf.unstack(shape, axis = 0)
		x1, y1, x2, y2 = tf.unstack(boxes, axis = -1)
		x1, y1, x2, y2 = (
			tf.clip_by_value(x1, 0, width - 1),
			tf.clip_by_value(y1, 0, height - 1),
			tf.clip_by_value(x2, 0, width - 1),
			tf.clip_by_value(y2, 0, height - 1)
		)
		return tf.stack([x1, y1, x2, y2], axis = 2)

	def compute_output_shape(self, input_shape):
		return input_shape[1]