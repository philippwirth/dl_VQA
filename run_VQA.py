# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import json
import os
import time

from model_VQA import VQAModel
from data_VQA import VQAData

class VQAMain:

    '''
        INIT: all settings/variables are specified here
    '''
    def __init__(self, split=1):

        # global parameters
        print("initializing parameters..")

        # data settings
        self.data_settings = {
            'input_json': './data/data_prepro_s'+ str(split) +'.json',
            'input_img_h5': './data/data_img_s'+ str(split) +'.h5',
            'input_ques_h5': './data/data_prepro_s'+ str(split) +'.h5',
            'img_norm': 0,
            'image_feature_size': 2048}

        self.split = 1
        self.fetch_data = VQAData(**self.data_settings)

        # model settings
        self.model_settings = {
            'rnn_size': 512,
            'bi_lstm_size': 512,			# temporary
            'batch_size': 500,
            'input_embedding_size': 200,
            'image_embedding_size': self.data_settings['image_feature_size'],
            'max_words_q': 26,
            'n_sub_images': 9,				# temporary
            'dim_hidden': 1024,
            'dim_output': 1000}

        # train parameters
        self.learning_rate = 0.0003
        self.decay_factor = 0.99997592083	# dafuq
        self.batch_size = self.model_settings['batch_size']

        # checkpoint_path
        self.checkpoint_path = 'model_save/'
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        # misc
        self.gpu_id = 0
        self.max_itr = 1 # for testing reasons put back to 150000 !!!!!!!!!!
        self.n_epochs = 300
        self.verbose = True

    '''
        TRAIN: call to train a VQAModel
    '''
    def train(self):

        print("Op: TRAIN")
        t_start_total = time.time()

        print("loading dataset..")
        dataset, img_feature, train_data = self.fetch_data.get_data_train(split=self.split)
        num_train = train_data['question'].shape[0]

        # update model options
        self.model_settings['vocabulary_size'] = len(dataset['ix_to_word'].keys())
        self.model_settings['drop_out_rate'] = 0.5

        print("constructing model..")
        model = VQAModel(**self.model_settings)
        tf_loss, _, tf_image, tf_question, tf_label = model.build_model(mode="train")

        print("initializing session..")
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
        saver = tf.train.Saver(max_to_keep=100)

        tvars = tf.trainable_variables()
        lr = tf.Variable(self.learning_rate)
        opt = tf.train.AdamOptimizer(learning_rate=lr)

        gvs = opt.compute_gradients(tf_loss, tvars)
        clipped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
        train_op = opt.apply_gradients(clipped_gvs)

        tf.initialize_all_variables().run()

        print("start training..")
        for itr in range(self.max_itr):

            if self.verbose:
                print("starting epoch " + str(itr))

            t_start = time.time()

            # shuffle training data
            index = np.random.random_integers(0, num_train-1, self.batch_size)

            current_question = train_data['question'][index, :]
            current_length_q = train_data['length_q'][index]
            current_answers = train_data['answers'][index]
            current_img_list = train_data['img_list'][index]
            current_img = img_feature[current_img_list, :]

            # training process
            _, loss = sess.run(
                        [train_op, tf_loss],
                        feed_dict={
                            tf_image: current_img,
                            tf_question: current_question,
                            tf_label: current_answers
                        })

            current_learning_rate = lr * self.decay_factor
            lr.assign(current_learning_rate).eval()

            t_stop = time.time()
            if self.verbose:
                print("Iteration: " + str(itr) + " Loss: " + str(loss))
                print("Time Cost: " + str(t_stop - t_start) + "s")
            if itr > 0 and np.mod(itr, 100) == 0:
                print("Iteration: " + str(itr) + " Loss: " + str(loss) + " Learning Rate:" + str(lr.eval()))
                print("Time Cost: " + str(t_stop - t_start) + "s")
            if itr > 0 and np.mod(itr, self.max_itr) == 0:
                print("Iteration " + str(itr) + " is done - saving the model..")
                saver.save(sess, os.path.join(self.checkpoint_path, 'model'), global_step=itr)

        print("done..")
        saver.save(sess, os.path.join(self.checkpoint_path, 'model'), global_step=self.n_epochs)
        t_stop_total = time.time()
        print("Overall Time: " + str(t_stop_total - t_start_total) + "s")

    '''
        TEST: call to test a VQAModel
    '''
    def test(self, model_path=None):

        print("Op: TEST")
        t_start_total = time.time()

        if model_path is None:
            model_path = os.path.join(self.checkpoint_path, ('model-' + str(self.n_epochs)))

        print("loading dataset..")
        dataset, img_feature, test_data = self.fetch_data.get_data_test()
        num_test = test_data['question'].shape[0]

        # update model settings
        self.model_settings['vocabulary_size'] = len(dataset['ix_to_word'].keys())
        self.model_settings['drop_out_rate'] = 0.	# zero for testing!

        print("constructing model..")
        model = VQAModel(**self.model_settings)
        _, tf_answer, tf_image, tf_question, _ = model.build_model(mode="generate")

        print("initializing session..")
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
        #saver = tf.train.Saver()
        print("Model at: " + model_path)
        saver = tf.train.import_meta_graph(model_path + '.meta')
        saver.restore(sess, model_path)

        print("generating answers..")
        result = []
        for current_batch_start_idx in range(0, num_test-1, self.batch_size):

            t_start = time.time()

            # fetch questions & images
            if current_batch_start_idx + self.batch_size < num_test:
                current_batch_file_idx = range(current_batch_start_idx, current_batch_start_idx + self.batch_size)
            else:
                current_batch_file_idx = rangE(current_batch_start_idx, num_test)

            current_question = test_data['question'][current_batch_file_idx,:]
            current_length_q = test_data['length_q'][current_batch_file_idx]
            current_img_list = test_data['img_list'][current_batch_file_idx]
            current_ques_id = test_data['ques_id'][current_batch_file_idx]
            current_img = img_feature[current_img_list, :]

            # deal with the last batch (this shit ugly af)
            if len(current_img) < 500:
                pad_img = np.zeros((500-len(current_img),dim_image),dtype=np.int)
                pad_q = np.zeros((500-len(current_img),max_words_q),dtype=np.int)
                pad_q_len = np.zeros(500-len(current_length_q),dtype=np.int)
                pad_q_id = np.zeros(500-len(current_length_q),dtype=np.int)
                pad_ques_id = np.zeros(500-len(current_length_q),dtype=np.int)
                pad_img_list = np.zeros(500-len(current_length_q),dtype=np.int)
                current_img = np.concatenate((current_img, pad_img))
                current_question = np.concatenate((current_question, pad_q))
                current_length_q = np.concatenate((current_length_q, pad_q_len))
                current_ques_id = np.concatenate((current_ques_id, pad_q_id))
                current_img_list = np.concatenate((current_img_list, pad_img_list))

            # get answer
            generated_ans = sess.run(
                tf_answer,
                feed_dict={
                    tf_image: current_img,
                    tf_question: current_question
                })

            top_ans = np.argmax(generated_ans, axis=1)

            # initialize json list
            for i in range(500):
                ans = dataset['ix_to_ans'][str(top_ans[i]+1)]
                if (current_ques_id[i] == 0):
                    continue
                result.append({u'answer': ans, u'question_id': str(current_ques_id[i])})

            t_stop = time.time()
            print("Test batch: ", current_barch_file_idx[0])
            print("Time Cost:", str(t_stop - t_start) + "s")

        print("done..")
        print("Overall Time:" + str(t_stop_total - t_start_total) + "s")
        print("saving result..")
        my_list = list(result)
        dd = json.dump(my_list, open('data.json', 'w'))


'''
    RUN EVERYTHING!!! 
'''
vqa_main = VQAMain()	# this
vqa_main.train()		# should
tf.reset_default_graph()# ! 
vqa_main.test()			# work














