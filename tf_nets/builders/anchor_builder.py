import tensorflow as tf
import tf_nets
import numpy as np

class BaseAnchors(tf.keras.layers.Layer):
	def __init__(self, config, *args, **kwargs):
		super(BaseAnchors, self).__init__(*args, **kwargs)
		self.sizes = config.size
		self.strides = config.stride
		self.ratios = config.ratio
		self.scales = config.scale

	def num_anchors(self):
		return len(self.ratios) * len(self.scales)

class Anchors(BaseAnchors):
	def __init__(self, config, *args, **kwargs):
		super(Anchors, self).__init__(config, *args, **kwargs)
		self.anchors = tf.keras.backend.variable(tf_nets.generate_anchors(
			base_size = self.sizes,
			ratios = self.ratios,
			scales = self.scales
		))

	def call(self, inputs, **kwargs):
		features = inputs
		feature_shape = tf.shape(features)
		anchors = tf.shift(features[1:3], self.strides, self.anchors)
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
				'size': self.sizes,
				'stride': self.strides,
				'ratios': self.ratios,
				'scales': self.scales
			})
			return config

def build(config):
	return Anchors(config)