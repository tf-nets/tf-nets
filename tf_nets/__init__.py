import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

def bbox_transform(boxes, deltas, mean = None, std = None):
	if mean is None:
		mean = [0, 0, 0, 0]
	if std is None:
		std = [0.2, 0.2, 0.2, 0.2]
	width = boxes[:, :, 2] + boxes[:, :, 0]
	height = boxes[:, :, 3] + boxes[:, :, 1]
	x1 = boxes[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
	y1 = boxes[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
	x2 = boxes[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
	y2 = boxes[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height
	return tf.stack([x1, x2, y1, y2], axis = 2)

def generate_anchors(base_size, ratios, scales):
	num_anchors = len(ratios) * len(scales)
	anchors = np.zeros((num_anchors, 4))
	anchors[:, 2:] = base_size * np.transpose(np.tile(scales, (2, len(ratios))))
	areas = anchors[:, 2] * anchors[:, 3]
	anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
	anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))
	anchors[:, 0::2] -= np.transpose(np.tile(anchors[:, 2] * 0.5, (2, 1)))
	anchors[:, 1::2] -= np.transpose(np.tile(anchors[:, 3] * 0.5, (2, 1)))
	return anchors

def shift(shape, stride, anchors):
	x = (K.arange(0, shape[1], dtype=K.floatx()) + K.constant(0.5, dtype=K.floatx())) * stride
	y = (K.arange(0, shape[0], dtype=K.floatx()) + K.constant(0.5, dtype=K.floatx())) * stride
	x, y = tf.meshgrid(x, y)
	x = K.reshape(x, [-1])
	y = K.reshape(y, [-1])
	shifts = K.stack([x, y, x, y], axis = 0)
	shifts = K.transpose(shifts)
	number_of_anchors = K.shape(anchors)[0]
	k = K.shape(shifts)[0]
	shifted_anchors = K.reshape(anchors, [1, number_of_anchors, 4]) + K.cast(K.reshape(shifts, [k, 1, 4]), K.floatx())
	shifted_anchors = K.reshape(shifted_anchors, [k * number_of_anchors, 4])
	return shifted_anchors