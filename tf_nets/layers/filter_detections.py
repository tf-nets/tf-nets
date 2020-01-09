import tensorflow as tf
from tensorflow.keras import backend as K

def filter_detections(boxes, classification, other = [], class_specific_filter = True, nms = True,
		score_threshold = 0.5, max_detections = 300, nms_threshold = 0.5):
	def _filter_detections(scores, labels):
		indices = tf.where(K.greater(scores, score_threshold))
		if nms:
			filtered_boxes = tf.gather_nd(boxes, indices)
			filtered_scores = K.gather(scores, indices)[:, 0]
			nms_indices = tf.image.non_max_suppression(
				filtered_boxes, filtered_scores, max_output_size = max_detections,
				iou_threshold = nms_threshold
			)
			indices = K.gather(indices, nms_indices)
		labels = tf.gather_nd(labels, indices)
		indices = K.stack([indices[:, 0], labels], axis = 1)
		return indices
	if class_specific_filter:
		all_indices = []
		for c in range(int(classification.shape[1])):
			scores = classification[:, c]
			labels = c * tf.ones((K.shape(scores)[0], ), dtype = 'int64')
			all_indices.append(_filter_detections(scores, labels))
		indices = K.concatenate(all_indices, axis=0)
	else:
		scores = K.max(classification, axis = 1)
		labels = K.argmax(classification, axis = 1)
		indices = _filter_detections(scores, labels)
	scores = tf.gather_nd(classification, indices)
	labels = indices[:, 1]
	scores, top_indices = tf.nn.top_k(scores, k = K.minimum(
		max_detections, K.shape(scores)[0]
	))
	indices = K.gather(indices[:, 0], top_indices)
	boxes = K.gather(boxes, indices)
	labels = K.gather(labels, top_indices)
	other_ = [K.gather(o, indices) for o in other]
	pad_size = K.minimum(0, max_detections - K.shape(scores)[0])
	boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values = -1)
	scores = tf.pad(scores, [[0, pad_size]], constant_values = -1)
	labels = tf.pad(labels, [[0, pad_size]], constant_values = -1)
	other_ = [
		tf.pad(o, [[0, pad_size]] + [[0, 0]])
		for _ in range(1, len(o.shape))
		for o in other_
	]
	boxes.set_shape([max_detections, 4])
	scores.set_shape([max_detections])
	labels.set_shape([max_detections])
	for o, s in zip(other_, [list(K.int_shape(o)) for o in other]):
		o.set_shape([max_detections] + s[1:])
	return [boxes, scores, labels] + other_

class FilterDetections(tf.keras.layers.Layer):
	def __init__(self, nms = True, class_specific_filter = True, nms_threshold = 0.5, score_threshold = 0.05,
			max_detections = 300, **kwargs):
		self.nms = nms
		self.class_specific_filter = class_specific_filter
		self.nms_threshold = nms_threshold
		self.score_threshold = score_threshold
		self.max_detections = max_detections
		super(FilterDetections, self).__init__(**kwargs)
	def call(self, inputs, **kwargs):
		boxes = inputs[0]
		classification = inputs[1]
		other = inputs[2:]
		def _filter_detections(args):
			boxes, classification, other = args
			return filter_detections(
				boxes, classification, other,
				nms = self.nms, class_specific_filter = self.class_specific_filter,
				score_threshold = self.score_threshold, max_detections = self.max_detections,
				nms_threshold = self.nms_threshold
			)
		return K.map_fn(
			_filter_detections, elems = [boxes, classification, other],
			dtype = [K.floatx(), K.floatx(), 'int64'] + [o.dtype for o in other]
		)
	def compute_output_shape(self, input_shape):
		return [
			(input_shape[0][0], self.max_detections, 4),
			(input_shape[1][0], self.max_detections),
			(input_shape[1][0], self.max_detections)
		] + [
			tuple([input_shape[i][0], self.max_detections] + list(input_shape[i][2:])) for i in range(2, len(input_shape))
		]
	def compute_mask(self, inputs, mask = None):
		return (len(inputs) + 1) * [None]
	def get_config(self):
		config = super(FilterDetections, self).get_config()
		config.update({
			'nms': self.nms,
			'class_specific_filter': self.class_specific_filter,
			'nms_threshold': self.nms_threshold,
			'score_threshold': self.score_threshold,
			'max_detections' : self.max_detections
		})
		return config