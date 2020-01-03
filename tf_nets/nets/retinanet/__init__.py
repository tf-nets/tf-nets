from tensorflow import keras

def mobilenet_v1(num_classes, inputs, modifier = None):
	from tensorflow.keras.applications import MobileNet
	backbone = MobileNet(input_tensor = inputs, include_top = False, pooling = None)
	layer_names = ['conv_pw_5_relu', 'conv_pw_11_relu', 'conv_pw_13_relu']
	layer_outputs = [backbone.get_layer(name).output for name in layer_names]
	backbone = keras.models.Model(inputs=inputs, outputs=layer_outputs, name=backbone.name)
	if modifier:
		backbone = modifier(backbone)
	return backbone