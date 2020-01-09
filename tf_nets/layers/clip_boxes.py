import tensorflow as tf
from tensorflow.keras import backend as K

class ClipBoxes(tf.keras.layers.Layer):
	def call(self, inputs, **kwargs):
		image, boxes = inputs
		shape = K.cast(tf.shape(image), K.floatx())
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