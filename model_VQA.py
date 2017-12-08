class VQAModel:

	def __init__(self):
		pass

	def build_model(self):
		# build a model to be trained to answer questions (use for training)
		pass

	def build_generator(self):
		# build a model to answer questions to images (use for testing)
		pass

	def _embed_question(self):
		# embed question somehow
		pass

	def _embed_image(self):
		# embed image using bidirectional lstm
		pass

	def _fuse(self):
		# fuse embedded question and image to 1 vector
		pass
