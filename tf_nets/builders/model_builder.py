from tensorflow import keras
from tf_nets.nets import retinanet
from tf_nets.builders import anchor_builder

def build_retinanet_model(config):
	def build_retinanet_localization_model(
			num_values, num_anchors, pyramid_feature_size = 256,
			localization_feature_size = 256, name = 'localization_model'):
		options = {
			'kernel_size': 3,
			'strides': 1,
			'padding': 'same',
			'kernel_initializer': keras.initializers.normal(mean = 0.0, stddev = 0.01, seed = None),
			'bias_initializer': 'zeros'
		}
		inputs = keras.layers.Input(shape = (None, None, pyramid_feature_size))
		outputs = inputs
		for i in range(4):
			outputs = keras.layers.Conv2D(
				filters = localization_feature_size,
				activation = 'relu',
				name = 'pyramid_localization_{}'.format(i + 1),
				**options
			)(outputs)
		outputs = keras.layers.Conv2D(num_anchors * num_values, name = 'pyramid_localization', **options)(outputs)
		outputs = keras.layers.Reshape((-1, num_values), name = 'pyramid_localization_reshape')(outputs)
		return keras.Model(inputs = inputs, outputs, outputs, name = name)
	
	def build_retinanet_classification_model(
			num_classes, num_anchors, pyramid_feature_size = 256, prior_probability = 0.01,
			classification_feature_size = 256, name = 'classification_model'):
		options = {
			'kernel_size': 3,
			'strides': 1,
			'padding': 'same'
		}
		inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
		outputs = inputs
		for i in range(4):
			outputs = keras.layers.Conv2D(
				filters = classification_feature_size,
				activation = 'relu',
				name = 'pyramid_classification_{}'.format(i + 1),
				kernel_initializer = keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
				bias_initializer = 'zeros',
				**options
			)(outputs)
		outputs = keras.layers.Conv2D(
			filters = num_classes * num_anchors,
			kernel_initializer = keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
			bias_initializer = initializers.PriorProbability(probability=prior_probability),
			name = 'pyramid_classification',
			**options
		)(options)
		outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
		outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)
		return keras.Model(inputs = inputs, outputs = outputs, name = name)

	RETINANET_BACKBONES = {
		'mobilenet_v1': retinanet.mobilenet_v1
	}
	inputs = keras.Input(shape = (None, None, 3))
	backbone = RETINANET_BACKBONES[config.backbone](
		num_classes = config.num_classes,
		inputs = inputs
	)
	anchors = anchor_builder.build(config.anchor_params)
	num_anchors = anchors.num_anchors()
	submodels = [
		('localization': build_retinanet_localization_model(4, num_anchors)),
		('classification': build_retinanet_classification_model(num_classes, num_anchors))
	]
	features = pyramid_builder.build_features(C3, C3, C5)
	pyramids = pyramid_builder.build(submodels, features)
	return keras.Model(inputs = inputs, outputs = pyramids, name = backbone.name)

MATA_ARCH_BUILDER_MAP = {
	'retinanet': build_retinanet_model
}
def build(config):
	meta_arch = config.WhichOneof('model')
	if meta_arch not in MATA_ARCH_BUILDER_MAP:
		raise ValueError('Unknown meta architecture: {}'.format(meta_arch))
	build_fn = MATA_ARCH_BUILDER_MAP[meta_arch]
	return build_fn(getattr(config, meta_arch))