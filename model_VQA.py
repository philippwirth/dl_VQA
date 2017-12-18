import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.contrib.rnn import BidirectionalGridLSTMCell

class Mode(Enum):
	TRAIN = 0
	GENERATE = 1

class VQAModel:

	def __init__(self, rnn_size, rnn_layer, bi_lstm_size,
				 batch_size, input_embedding_size, vocabulary_size,
				 drop_out_rate,
				 max_words_q, n_sub_images,
				 dim_hidden, dim_output):
		
		self.rnn_size = rnn_size 								# size of lstm for question embedding
		self.rnn_layer = rnn_layer 								# number of layers of lstm for question embedding
		self.bi_lstm_size = bi_lstm_size						# size of biLSTM for image embedding
		self.batch_size = batch_size 							# batch size (for question embedding?)
		self.input_embedding_size = input_embedding_size 		# word embedding size
		self.vocabulary_size = vocabulary_size 					# size of vocabulary in questions (and answers?)
		self.drop_out_rate = drop_out_rate 						# dropout rate for lstm layers (0 = no dropout)
		self.max_words_q = max_words_q 							# maximal number of words
		self.n_sub_images = n_sub_images						# number of images one image gets split into (wow much english, very talk)
		self.dim_hidden = dim_hidden							# dimension of the question/word embedding
		self.dim_output = dim_output 							# output dimension

		# question embedding
		self.embed_ques_W = tf.Variable(
			tf.random_uniform([self.vocabulary_size, self.input_embedding_size], -0.08, 0.08), name='embed_ques_W')

		# question embedding: encode words as one vector representing the question
		self.lstm_1 = rnn_cell.LSTMCell(rnn_size, input_embedding_size, use_peepholes=True)
		self.lstm_dropout_1 = rnn_cell.DropoutWrapper(self.lstm_1, output_keep_prob=1 - self.drop_out_rate)
		self.lstm_2 = rnn_cell.LSTMCell(rnn_size, rnn_size, use_peepholes=True)
		self.lstm_dropout_2 = rnn_cell.DropoutWrapper(self.lstm_2, output_keep_prob=1 - self.drop_out_rate)
		self.stacked_lstm = rnn_cell.MultiRNNCell([self.lstm_dropout_1, self.lstm_dropout_2])

		# image embedding

		# image embedding: biLSTM (only 1 layer atm)
		self.bi_lstm = BidirectionalGridLSTMCell(bi_lstm_size, use_peepholes=True)	

		# question/image fusing
		self.embed_q_state_W = tf.Variable(tf.random_uniform([2*rnn_size*rnn_layer, dim_hidden], -0.08, 0.08), name='embed_q_state_W')
		self.embed_q_state_b = tf.Variable(tf.random_uniform([dim_hidden], -0.08, 0.08), name='embed_q_state_b')

		bilstm_state_size = self.bi_lstm.state_size
		self.embed_i_state_W = tf.Variable(tf.random_uniform([bilstm_state_size, dim_hidden], -0.08, 0.08), name='ebed_i_state_W')
		self.embed_i_state_b = tf.Variable(tf.random_uniform([dim_hidden], -0.08, 0.08), name='embed_i_state_b')

		self.embed_scor_W = tf.Variable(tf.random_uniform([dim_hidden, dim_output], -0.08, 0.08), name='embed_scor_W')
		self.embed_scor_b = tf.Variable(tf.random_uniform([dim_output], -0.08, 0.08), name='embed_scor_b')


	def build_model(self, mode=Mode.TRAIN):
		'''
			build a model to be trained to answer questions
		'''

		# embed question and image
		question, q_output, q_state = self._embed_question()
		image, i_output, i_state = self._embed_image()

		# fuse question and image to 1 vector
		scores = self._fuse(q_state, i_state)

		# predict answer / train model
		if (mode == Mode.TRAIN):
			label = tf.placeholder(tf.int64, [self.batch_size])
			cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(scores, label)
			loss = tf.reduce_mean(cross_entropy)
			return loss, None, image, question, label
		elif (mode == Mode.GENERATE):
			generated_answer = scores
			return None, generated_answer, image, question, None
		else:
			raise ValueError("Bad mode!")


	def _embed_question(self):
		'''
			embed a question using a 2-layer lstm, 
			returns question (placeholder), output (hidden state / output of last lstm), state (state after last lstm)
		'''

		# question placeholder, get array of question words
		question = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q])

		# lstm
		state = tf.zeros([self.batch_size, self.stacked_lstm.state_size])
		for i in range(self.max_words_q):
			if i == 0:
				ques_emb_linear = tf.zeros([self.batch_size, self.input_embedding_size])
			else:
				tf.get_variable_scope().reuse_variables() # reuse same weights as previous lstm (it's the same)
				ques_emb_linear = tf.nn.embedding_lookup(self.embed_ques_W, question[:, i - 1])

			ques_emb_drop = tf.nn.dropout(ques_emb_linear, 1 - self.drop_out_rate)
			ques_emb = tf.tanh(ques_emb_drop)

			output, state = self.stacked_lstm(ques_emb, state)

		return question, output, state


	def _embed_image(self):
		'''
			embed an image using resnet & bidirectional lstm
			returns image (placeholder), output (output of the biLSTM), state (of the biLSTM)
		'''

		# TODO: RESNET STUFF HERE
		resnet_out = None # output from resnet, should be a tensor of dim: self.batch_size x n_sub_images

		# embed image with a biLSTM
		state = tf.zeros([self.batch_size, self.bi_lstm.state_size])
		for i in range(self.n_sub_images+1):
			if i == 0:
				img_emb_linear = tf.zeros([self.batch_size, self.input_embedding_size])
			else:
				tf.get_variable_scope().reuse_variables()
				img_emb_linear = resnet_out[:, i-1]

			img_emb_drop = tf.nn.dropout(img_emb_linear, 1 - self.drop_out_rate)
			img_emb = tf.tanh(img_emb_drop)

			output, state = self.bi_lstm(img_emb, state)

		return resnet_out, output, state


	def _fuse(self, q_state, i_state):
		'''
			fuse embedded question and image to 1 vector
		'''

		# non-linear activation of question state
		q_state_drop = tf.nn.dropout(q_state, 1-self.drop_out_rate)
		q_state_linear = tf.nn.xw_plus_b(q_state_drop, self.embed_q_state_W, self.embed_q_state_b)
		q_state_emb = tf.tanh(q_state_linear)

		# non-linear activation of image state
		i_state_drop = tf.nn.dropout(i_state, 1-self.drop_out_rate)
		i_state_linear = tf.nn.xw_plus_b(i_state_drop, self.embed_i_state_W, self.embed_i_state_b)
		i_state_emb = tf.tanh(i_state_linear)

		# fuse w/ pointwise multiplication
		scores = tf.mul(q_state_emb, i_state_emb)
		scores_drop = tf.nn.dropout(scores, 1-self.drop_out_rate)
		scores_emb = tf.nn.xw_plus_b(scores_drop, self.embed_scor_W, self.embed_scor_b)
		return scores_emb
