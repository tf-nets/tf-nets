import tensorflow as tf
import tf_nets
import numpy as np

class Anchors(tf.keras.layers.Layer):
	def __init__(self, size, stride, ratios = None, scales = None, *args, **kwargs):
		self.size = size
		self.stride = stride
		self.ratios = ratios
		self.scales = scales
		self.num_anchors = len(ratios) * len(scales)
		self.anchors = tf.keras.backend.variable(tf_nets.generate_anchors(
			base_size = self.size,
			ratios = self.ratios,
			scales = self.scales
		))
		super(Anchors, self).__init__(*args, **kwargs)

	def call(self, inputs, **kwargs):
		features = inputs
		feature_shape = tf.shape(features)
		anchors = tf_nets.shift(feature_shape[1:3], self.stride, self.anchors)
		anchors = tf.tile(tf.expand_dims(anchors, axis = 0), (feature_shape[0], 1, 1))
		return anchors

	def compute_output_shape(self, input_shape):
		if None not in input_shape:
			total = np.prod(input_shape[1:3]) * self.num_anchors
			return (input_shape[0], total, 4)
		else:
			return (input_shape[0], None, 4)

	def get_config(self):
		config = super(Anchors, self).get_config()
		config.update({
			'size': self.size,
			'stride': self.stride,
			'ratios': self.ratios,
			'scales': self.scales
		})
		return config

def num_anchors(config):
	return len(config.ratio) * len(config.scale)

def build(config, features):
	return tf.keras.layers.Concatenate(axis = 1, name = 'anchors')([
		Anchors(
			size = config.size[i],
			stride = config.stride[i],
			ratios = config.ratio,
			scales = config.scale,
			name='anchors_{}'.format(i))(feature)
		for i, feature in enumerate(features)
	])