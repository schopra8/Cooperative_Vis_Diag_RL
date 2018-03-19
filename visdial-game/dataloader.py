import h5py
import json
import numpy as np

class DataLoader(object):
    def __init__(self, visdial_params_json, visdial_data_h5, img_data_h5, dsplits,
                                     img_norm=False, verbose=False):
        self.data = {}
        self.verbose = verbose
        if self.verbose:
            print 'DataLoader loading json file:', visdial_params_json
        visdial_params = json.load(open('../data/{}'.format(visdial_params_json)))

        self.word2ind = visdial_params['word2ind']
        self.vocabSize = len(self.word2ind)

        # add <START> and <END> to vocabulary
        self.word2ind['<START>'] = self.vocabSize + 1
        self.word2ind['<END>'] = self.vocabSize + 2
        self.vocabSize += 2
        if self.verbose:
            print 'Vocabulary size (with <START>,<END>):{}'.format(self.vocabSize)

        self.ind2word = visdial_params['ind2word']
        self.ind2word[str(self.word2ind['<START>'])] = '<START>'
        self.ind2word[str(self.word2ind['<END>'])] = '<END>'
        self.ind2word_int = {}
        for key, value in self.ind2word.iteritems():
            self.ind2word_int[int(key)] = value
        self.ind2word = self.ind2word_int

        if self.verbose:
            print 'DataLoader loading h5 file:', visdial_data_h5
        visdial_data = h5py.File('../data/{}'.format(visdial_data_h5), 'r')

        if self.verbose:
            print 'DataLoader loading h5 file:', img_data_h5
        img_data = h5py.File('../data/{}'.format(img_data_h5), 'r')

        self.num_dialogs = {}

        for dsplit in dsplits:
            # read image list, if image features are needed
            img_features = np.array(img_data['images_'+dsplit])

            # Normalize the image features (if needed)
            # TODO: normalize image by default
            if self.verbose:
                print 'Normalizing image features..'
            norm = np.sqrt(np.sum(img_features * img_features, axis=1))
            img_features = img_features / np.expand_dims(norm, 1)

            # note: img_pos == np.arange(num_images)
            self.data[dsplit+'_img_fv'] = img_features
            # self.data[dsplit+'_img_pos'] = np.array(visdial_data['img_pos_'+dsplit])
            # self.data[dsplit+'_img_pos'] += 1

            # read captions
            self.data[dsplit+'_cap'] = np.array(visdial_data['cap_'+dsplit])
            self.data[dsplit+'_cap_len'] = np.array(visdial_data['cap_length_'+dsplit])

            # read question related information
            self.data[dsplit+'_ques'] = np.array(visdial_data['ques_'+dsplit])
            self.data[dsplit+'_ques_len'] = np.array(visdial_data['ques_length_'+dsplit])

            # read answer related information
            self.data[dsplit+'_ans'] = np.array(visdial_data['ans_'+dsplit])
            self.data[dsplit+'_ans_len'] = np.array(visdial_data['ans_length_'+dsplit])

            # print information for data type
            if self.verbose:
                print '{}:\n\tNo. of dialogs: {}\n\tNo. of rounds: {}\n\tMax ques len: {}\n\tMax ans len: {}\n'.format(
                        dsplit,
                        self.data[dsplit+'_ques'].shape[0],
                        self.data[dsplit+'_ques'].shape[1],
                        self.data[dsplit+'_ques'].shape[2],
                        self.data[dsplit+'_ans'].shape[2])

            # record some stats
            if dsplit == 'train':
                self.num_dialogs['train'] = self.data['train_ques'].shape[0]
            if dsplit == 'val':
                self.num_dialogs['val'] = self.data['val_ques'].shape[0]

            # assume similar stats across multiple data subsets
            # maximum number of questions per image, ideally 10
            # maximum length of question
            # self.maxQuesCount, self.maxQuesLen, _ = self.data[dsplit+'_ques'].shape
            # maximum length of answer
            # self.maxAnsLen = self.data[dsplit+'_ans'].shape[2]

        # prepareDataset for training
        for dsplit in dsplits:
            self.wrapDelimiterTokens('ques', dsplit)
            self.wrapDelimiterTokens('ans', dsplit)

    def wrapDelimiterTokens(self, sentence_type, dsplit):
        # prefix sentence_type with <START>, <END>, and increment lengths by 2
        sentences = self.data[dsplit+'_'+sentence_type]
        sentenceLengths = self.data[dsplit+'_'+sentence_type+'_len']

        numDialogs, numRounds, maxLength = sentences.shape

        new_sentences = np.zeros((numDialogs, numRounds, maxLength+2), dtype=np.int32)
        new_sentences[:,:,0] = self.word2ind['<START>']

        # go over each answer and modify
        endTokenId = self.word2ind['<END>']
        for dialog_id in xrange(numDialogs):
            for round_id in xrange(numRounds):
                length = sentenceLengths[dialog_id][round_id]
                # only if nonzero
                if length > 0:
                    new_sentences[dialog_id][round_id][1:length+1] = sentences[dialog_id][round_id][0:length]
                    new_sentences[dialog_id][round_id][length+1] = endTokenId
                else:
                    print 'Warning: empty answer at ({} {} {})'.format(dialog_id, round_id, length)

        self.data[dsplit+'_'+sentence_type] = new_sentences
        self.data[dsplit+'_'+sentence_type+'_len'] += 2

    def getTrainBatch(self, batch_size):
        # shuffle all the indices
        size = self.num_dialogs['train']
        ordering = np.random.permutation(size)

        for i in xrange(0, size, batch_size):
            inds = ordering[i:i+batch_size]
            yield self.getIndexData(inds, 'train')

    def getEvalBatch(self, batch_size):
        size = self.num_dialogs['val']
        print size
        size = size - 1 # Crashed with val size

        for i in xrange(0, size, batch_size):
            inds = range(i,min(size,i+batch_size))
            yield self.getIndexData(inds, 'val')

    def getIndexData(self, inds, dsplit):
        # get the question lengths
        images = self.data[dsplit+'_img_fv'][inds]

        captions = self.data[dsplit+'_cap'][inds]
        caption_lengths = self.data[dsplit+'_cap_len'][inds]

        # get questions
        true_questions = self.data[dsplit+'_ques'][inds]
        true_question_lengths = self.data[dsplit+'_ques_len'][inds]

        # get the answer lengths
        true_answers = self.data[dsplit+'_ans'][inds]
        true_answer_lengths = self.data[dsplit+'_ans_len'][inds]

        return images, captions, caption_lengths, true_questions, true_question_lengths, true_answers, true_answer_lengths, inds

    def getEvalImageEmbeddings(self):
        return self.data['val_img_fv']

if __name__ == '__main__':
    # dataloader = DataLoader('visdial_params.json', 'visdial_data.h5', 'data_img.h5', ['train'], verbose=True)
    # batch_generator = dataloader.getTrainBatch(20)
    # count = 0
    # for batch in batch_generator:
    #     if count % 100 == 0:
    #         print count
    #     count += 1

    eval_dataloader = DataLoader('visdial_params.json', 'visdial_data.h5', 'data_img.h5', ['val'], verbose=True)
    imgs = eval_dataloader.getEvalImageEmbeddings()
    print imgs.shape
    # batch_generator = eval_dataloader.getEvalBatch(20)
    # count = 0
    # for batch in batch_generator:
    #     if count % 100 == 0:
    #         print count
    #     count += 1