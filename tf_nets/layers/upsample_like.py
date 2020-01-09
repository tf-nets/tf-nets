import tensorflow as tf

class UpsampleLike(tf.keras.layers.Layer):
	def call(self, inputs, **kwargs):
		source, target = inputs
		target_shape = tf.shape(target)
		return tf.compat.v1.image.resize_images(source, (target_shape[1], target_shape[2]), tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners = False)

	def compute_output_shape(self, input_shape):
		return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1])