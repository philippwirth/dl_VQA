import h5py, json
import numpy as np


class VQAData:

    def __init__(self, input_json, input_img_h5, input_ques_h5, img_norm, image_feature_size):
        self.input_json = input_json
        self.input_img_h5 = input_img_h5
        self.input_ques_h5 = input_ques_h5
        self.img_norm = img_norm
        self.image_feature_size = image_feature_size # size of image feature
        pass

    def right_align(self, seq, lengths):
        v = np.zeros(np.shape(seq))
        N = np.shape(seq)[1]
        for i in range(np.shape(seq)[0]):
            v[i][N - lengths[i]:N - 1] = seq[i][0:lengths[i] - 1]
        return v

    def get_fake_img_feature(self, img_list):
        N = len(np.unique(img_list)) # number of unique images
        assert N == np.max(img_list) + 1 # the images should be numbered from 0 to max(img_list)
        feature = np.random.rand(N, 9, self.image_feature_size)
        return feature

    def get_data_train(self, split=1):
        '''
            get the training data
        '''

        dataset = {}
        training_data = {}
        # load json file
        print('loading json file...')
        with open(self.input_json) as data_file:
            data = json.load(data_file)
        for key in data.keys():
            dataset[key] = data[key]

        # load question h5 file
        print('loading question h5 file...')
        with h5py.File(self.input_ques_h5, 'r') as hf:
            # total number of training data is 215375
            # question is (26, )
            tem = hf.get('ques_train')
            training_data['question'] = np.array(tem) - 1 # -1 because +1 added in preprocessing (to work with torch originally)
            # max length is 23
            tem = hf.get('ques_length_train')
            training_data['length_q'] = np.array(tem)
            # total 82460 img
            tem = hf.get('img_pos_train')
            # convert into 0~82459
            training_data['img_list'] = np.array(tem) - 1
            # answer is 1~1000
            tem = hf.get('answers')
            training_data['answers'] = np.array(tem) - 1

        # load image feature
        print('loading image feature...')
        # TODO: true image features
        # with h5py.File(self.input_img_h5, 'r') as hf:
        #     tem = hf.get('images_train')
        #     img_feature = np.array(tem)
        img_feature = self.get_fake_img_feature(training_data['img_list'])

        print('question aligning')
        training_data['question'] = self.right_align(training_data['question'], training_data['length_q'])

        print('Normalizing image feature')
        if self.img_norm:
            tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
            img_feature = np.divide(img_feature, np.transpose(np.tile(tem, (self.image_feature_size, 1))))

        return dataset, img_feature, training_data

    def get_data_test(self, split=1):
        '''
            get the training data
        '''
        dataset = {}
        test_data = {}
        # load json file
        print('loading json file...')
        with open(self.input_json) as data_file:
            data = json.load(data_file)
        for key in data.keys():
            dataset[key] = data[key]


        # load h5 file
        print('loading h5 file...')
        with h5py.File(self.input_ques_h5, 'r') as hf:
            # total number of training data is 215375
            # question is (26, )
            tem = hf.get('ques_test')
            test_data['question'] = np.array(tem) - 1
            # max length is 23
            tem = hf.get('ques_length_test')
            test_data['length_q'] = np.array(tem)
            # total 82460 img
            tem = hf.get('img_pos_test')
            # convert into 0~82459
            test_data['img_list'] = np.array(tem) - 1
            # quiestion id
            tem = hf.get('question_id_test')
            test_data['ques_id'] = np.array(tem)
            # MC_answer_test
            tem = hf.get('MC_ans_test')
            test_data['MC_ans_test'] = np.array(tem)

        # load image feature
        print('loading image feature...')
        # TODO: true image features
        # with h5py.File(self.input_img_h5, 'r') as hf:
        #     tem = hf.get('images_test')
        #     img_feature = np.array(tem)
        img_feature = self.get_fake_img_feature(test_data['img_list'])

        print('question aligning')
        test_data['question'] = self.right_align(test_data['question'], test_data['length_q'])

        if self.img_norm:
            print('Normalizing image feature')
            # TODO: fix normalization, e.g. using https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy ?
            tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
            img_feature = np.divide(img_feature, np.transpose(np.tile(tem, (self.image_feature_size, 1))))

        return dataset, img_feature, test_data




if __name__ == '__main__':
    dataSet = VQAData('./data/data_prepro_s1.json', '', './data/data_prepro_s1.h5', 0, 2048)
    dataset, img_feature, train_data = dataSet.get_data_train()
    print("len(dataset): " + str(len(dataset)))
    print("img_feature.shape: " + str(img_feature.shape))
    print("len(train_data): " + str(len(train_data)))
