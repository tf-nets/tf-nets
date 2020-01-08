import tensorflow as tf
from google.protobuf import text_format
from tf_nets.protos import pipeline_pb2
from tf_nets.builders import model_builder

flags = tf.compat.v1.app.flags

__flags__ = [
	('pipeline_config_path', None, 'Path to pipline configuration file'),
	('output_directory', None, 'Path to write outputs')
]

__flags_required__ = [
	'pipeline_config_path',
	'output_directory'
]

for flag in __flags__:
  flags.DEFINE_string(*flag)

for required_flag in __flags_required__:
  flags.mark_flag_as_required(required_flag)

__FLAGS__ = flags.FLAGS

if __name__ == '__main__':
	pipeline_config = pipeline_pb2.PipelineConfig()
	with tf.compat.v1.gfile.GFile(__FLAGS__.pipeline_config_path) as __file:
		text_format.Merge(__file.read(), pipeline_config)
	detection_model, baseline_model = model_builder.build(pipeline_config.model)
