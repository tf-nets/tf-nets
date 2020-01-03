
class BaseAnchors:
	def __init__(self, config):
		self.sizes = config.size
		self.strides = config.stride
		self.ratios = config.ratio
		self.scales = config.scale

	def num_anchors(self):
		return len(self.ratios) * len(self.scales)

class Anchors(BaseAnchors):
	def __init__(self, config):
		super(Anchors, self).__init__(config)

def build(config):
	return Anchors(config)