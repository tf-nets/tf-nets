from tensorflow import keras
from tf_net.builders import pyramid_features_builder

def build(
		num_classes,
		inputs,
		backbone_layers,
		num_anchors = None,
		create_pyramid_features = None,
		submodels = None,
		name = 'tf_net'):
	if num_anchors is None:
		num_anchors = AnchorParameters.defautl.num_anchors()
	if submodels is None:
		submodels = default_submodels(num_classes, num_anchors)
	C3, C4, C5 = backbone_layers
	if create_pyramid_features is None:
		create_pyramid_features = pyramid_features_builder.build()
	features = create_pyramid_features(C3, C4, C5)
	pyramids = __build_pyramids(submodels, features)
	return keras.Model(inputs = inputs, outputs = pyramids, name = name)