from tensorflow import keras
from tf_nets.nets import retinanet

def build_retinanet_model(config):
	RETINANET_BACKBONES = {
		'mobilenet_v1': retinanet.mobilenet_v1
	}
	inputs = keras.Input(shape = (None, None, 3))
	backbone = RETINANET_BACKBONES[config.backbone](
		num_classes = config.num_classes,
		inputs = inputs
	)
	
	print(config)

MATA_ARCH_BUILDER_MAP = {
  'retinanet': build_retinanet_model
}
def build(config):
	meta_arch = config.WhichOneof('model')
	if meta_arch not in MATA_ARCH_BUILDER_MAP:
		raise ValueError('Unknown meta architecture: {}'.format(meta_arch))
	build_fn = MATA_ARCH_BUILDER_MAP[meta_arch]
	return build_fn(getattr(config, meta_arch))