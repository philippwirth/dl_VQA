# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn # stacked_... is just a mulitlayer version of the non-stack model
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn


class VQAModel:

    def __init__(self, rnn_size, bi_lstm_size,
                 batch_size,
                 input_embedding_size, image_embedding_size,
                 vocabulary_size,
                 drop_out_rate,
                 max_words_q, n_sub_images,
                 dim_hidden, dim_output):

        self.rnn_size = rnn_size 								# size of lstm for question embedding
        self.bi_lstm_size = bi_lstm_size						# size of biLSTM for image embedding
        self.batch_size = batch_size 							# batch size (for question embedding?)
        self.input_embedding_size = input_embedding_size 		# word embedding size
        self.image_embedding_size = image_embedding_size		# image embedding size
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
        self.lstm_1 = rnn.LSTMCell(rnn_size, use_peepholes=True, state_is_tuple=False, reuse=tf.get_variable_scope().reuse)
        self.lstm_dropout_1 = rnn.DropoutWrapper(self.lstm_1, output_keep_prob=1 - self.drop_out_rate)
        self.lstm_2 = rnn.LSTMCell(rnn_size, use_peepholes=True, state_is_tuple=False, reuse=tf.get_variable_scope().reuse)
        self.lstm_dropout_2 = rnn.DropoutWrapper(self.lstm_2, output_keep_prob=1 - self.drop_out_rate) 
        self.stacked_lstm = rnn.MultiRNNCell([self.lstm_dropout_1, self.lstm_dropout_2], state_is_tuple=False)

        # image embedding

        # image embedding: biLSTM (only 1 layer atm)
        self.lstm_3 = rnn.LSTMCell(bi_lstm_size, image_embedding_size, use_peepholes=True, num_proj=1, reuse=tf.get_variable_scope().reuse)
        self.lstm_dropout_3 = rnn.DropoutWrapper(self.lstm_3, output_keep_prob=1 - self.drop_out_rate)
        self.lstm_4 = rnn.LSTMCell(bi_lstm_size, image_embedding_size, use_peepholes=True, num_proj=1, reuse=tf.get_variable_scope().reuse)
        self.lstm_dropout_4 = rnn.DropoutWrapper(self.lstm_4, output_keep_prob=1 - self.drop_out_rate)

        # question/image fusing
        self.embed_q_state_W = tf.Variable(tf.random_uniform([2*rnn_size*2, dim_hidden], -0.08, 0.08), name='embed_q_state_W')
        self.embed_q_state_b = tf.Variable(tf.random_uniform([dim_hidden], -0.08, 0.08), name='embed_q_state_b')

        bilstm_state_size = self.bi_lstm_size # is this correct @philip? was 'self.bi_lstm.state_size' (syntax error)
        self.embed_image_W = tf.Variable(tf.random_uniform([image_embedding_size, dim_hidden], -0.08, 0.08), name='ebed_i_state_W')
        self.embed_image_b = tf.Variable(tf.random_uniform([dim_hidden], -0.08, 0.08), name='embed_i_state_b')

        self.embed_scor_W = tf.Variable(tf.random_uniform([dim_hidden, dim_output], -0.08, 0.08), name='embed_scor_W')
        self.embed_scor_b = tf.Variable(tf.random_uniform([dim_output], -0.08, 0.08), name='embed_scor_b')


    def build_model(self, mode="train"):
        '''
            build a model to be trained to answer questions
        '''

        # embed question and image
        question, q_output, q_state = self._embed_question(mode=mode)
        resnet_out, i_output, images = self._embed_image(mode=mode)

        # fuse question and image to 1 vector
        scores = self._fuse(q_state, images)

        # predict answer / train model
        if (mode == "train"):
            label = tf.placeholder(tf.int64, [self.batch_size], name='tf_label')
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=label)
            loss = tf.reduce_mean(cross_entropy)
            return loss, None, resnet_out, question, label
        elif (mode == "generate"):
            generated_answer = scores
            return None, generated_answer, resnet_out, question, None
        else:
            raise ValueError("Bad mode!")


    def _embed_question(self, mode):
        '''
            embed a question using a 2-layer lstm,
            returns question (placeholder), output (hidden state / output of last lstm), state (state after last lstm)
        '''

        print("embedding question..")

        # question placeholder, get array of question words
        question = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q], name='tf_question')

        # lstm
        with tf.variable_scope("question"):
            state = tf.zeros([self.batch_size, 2*2*self.rnn_size])
            for i in range(self.max_words_q):
                if i == 0:
                	ques_emb_linear = tf.zeros([self.batch_size, self.input_embedding_size])
                	if mode == "generate":
                		print("MODE: generate")
                		#tf.get_variable_scope().reuse_variables()
                else:
                    tf.get_variable_scope().reuse_variables() # reuse same weights as previous lstm (it's the same)
                    ques_emb_linear = tf.nn.embedding_lookup(self.embed_ques_W, question[:, i - 1])

                ques_emb_drop = tf.nn.dropout(ques_emb_linear, 1 - self.drop_out_rate)
                ques_emb = tf.tanh(ques_emb_drop)

                output, state = self.stacked_lstm(ques_emb, state) #, scope="question")

        return question, output, state


    def _embed_image(self, mode):
        '''
            embed an image using resnet & bidirectional lstm
            returns weighted sub-images, output (output of the biLSTM), state (of the biLSTM)
        '''

        print("embedding image..")

        # TODO: RESNET STUFF HERE
        #resnet_out = None # output from resnet, should be a tensor of dim: self.batch_size x n_sub_images x dim_resout
        resnet_out = tf.placeholder(tf.float32, [self.batch_size, self.n_sub_images, self.image_embedding_size], name='tf_images')

        # weight sub-images with biLSTM
        with tf.variable_scope("image"):

        	if mode == "generate":
        		print("MODE: generate")
        		#tf.get_variable_scope().reuse_variables()

        	outputs, output_states = bidirectional_dynamic_rnn(self.lstm_dropout_3, self.lstm_dropout_4, resnet_out, dtype=tf.float32)

        	fwd_out, bwd_out = outputs
        	weights = tf.nn.softmax(tf.add(fwd_out, bwd_out))

        # multiply resnet output with weights
        	weighted_out = tf.zeros([self.batch_size, self.n_sub_images, self.image_embedding_size])
        	weighted_out = tf.multiply(resnet_out, weights)

        return resnet_out, outputs, weighted_out


    def _fuse(self, q_state, weighted_images):
        '''
            fuse embedded question and images to 1 vector
        '''

        print("fusing..")

        # non-linear activation of question state
        q_state_drop = tf.nn.dropout(q_state, 1-self.drop_out_rate)
        q_state_linear = tf.nn.xw_plus_b(q_state_drop, self.embed_q_state_W, self.embed_q_state_b)
        q_state_emb = tf.tanh(q_state_linear)

        for i in range(self.n_sub_images):

            # non-linear activation of weighted image
            image_drop = tf.nn.dropout(weighted_images[:, i], 1-self.drop_out_rate)
            image_linear = tf.nn.xw_plus_b(image_drop, self.embed_image_W, self.embed_image_b)
            image_emb = tf.tanh(image_linear)

            # fuse w/ pointwise multiplication
            scores = tf.multiply(q_state_emb, image_emb)
            scores_drop = tf.nn.dropout(scores, 1-self.drop_out_rate)
            scores_emb = tf.nn.xw_plus_b(scores_drop, self.embed_scor_W, self.embed_scor_b)

            if (i == 0):
                result = scores_emb
            else:
                result = tf.concat([result, scores_emb], 1)

        return result