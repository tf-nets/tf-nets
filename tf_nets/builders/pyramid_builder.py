from tensorflow import keras
from tf_nets import layers

def build_features(C3, C4, C5, feature_size = 256):
	P5 = keras.layers.Conv2D(feature_size, kernel_size = 1, strides = 1, padding = 'same', name = 'C5_reduced')(C5)
	P5_upsampled = layers.UpsampleLike(name = 'P5_upsampled')([P5, C4])
	P5 = keras.layers.Conv2D(feature_size, kernel_size = 3, strides = 1, padding = 'same', name = 'P5')(P5)
	P4 = keras.layers.Conv2D(feature_size, kernel_size = 1, strides = 1, padding = 'same', name = 'C4_reduced')(C4)
	P4_upsampled = layers.UpsampleLike(name = 'P4_upsampled')([P5_upsampled, C3])
	P4 = keras.layers.Conv2D(feature_size, kernel_size = 3, strides = 1, padding = 'same', name = 'P4')(P4)
	P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
	P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
	P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)
	P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)
	P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
	P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)
	return [P3, P4, P5, P6, P7]

def build(models, features):
	def build_model(name, model, features):
		return keras.layers.Concatenate(axis = 1, name = name)([
			model(feature) for feature in features
		])
	return [
		build_model(name, model, features) for name, model in models
	]