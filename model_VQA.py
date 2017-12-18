import tensorflow as tf
from tensorflow.models.rnn import rnn_cell

class VQAModel:

	def __init__(self, rnn_size, rnn_layer, batch_size, input_embedding_size, vocabulary_size, drop_out_rate, max_words_q):
		self.rnn_size = rnn_size # size of lstm for question embedding
		self.rnn_layer = rnn_layer # number of layers of lstm for question embedding
		self.batch_size = batch_size # batch size (for question embedding?)
		self.input_embedding_size = input_embedding_size # word embedding size
		self.vocabulary_size = vocabulary_size # size of vocabulary in questions (and answers?)
		self.drop_out_rate = drop_out_rate # dropout rate for lstm layers (0 = no dropout)
		self.max_words_q = max_words_q # maximal number of words

		# question-embedding
		self.embed_ques_W = tf.Variable(
			tf.random_uniform([self.vocabulary_size, self.input_embedding_size], -0.08, 0.08), name='embed_ques_W')

		# question embedding: encode words as one vector representing the question
		self.lstm_1 = rnn_cell.LSTMCell(rnn_size, input_embedding_size, use_peepholes=True)
		self.lstm_dropout_1 = rnn_cell.DropoutWrapper(self.lstm_1, output_keep_prob=1 - self.drop_out_rate)
		self.lstm_2 = rnn_cell.LSTMCell(rnn_size, rnn_size, use_peepholes=True)
		self.lstm_dropout_2 = rnn_cell.DropoutWrapper(self.lstm_2, output_keep_prob=1 - self.drop_out_rate)
		self.stacked_lstm = rnn_cell.MultiRNNCell([self.lstm_dropout_1, self.lstm_dropout_2])


	def build_model(self):
		# build a model to be trained to answer questions (use for training)
		question, output, state = self._embed_question()
		pass

	def build_generator(self):
		# build a model to answer questions to images (use for testing)
		pass

	def _embed_question(self):
		# embed question, returns question (placeholder), output (hidden state / output of last lstm), state (state after last lstm)
		question = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q]) # question placeholder, get array of question words

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
		# embed image using bidirectional lstm
		pass

	def _fuse(self):
		# fuse embedded question and image to 1 vector
		pass
