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

    self.ind2word = visdial_params['word2ind']
    self.ind2word[str(self.word2ind['<START>'])] = '<START>'
    self.ind2word[str(self.word2ind['<END>'])] = '<END>'

    if self.verbose:
      print 'DataLoader loading h5 file:', visdial_data_h5
    visdial_data = h5py.File('../data/{}'.format(visdial_data_h5), 'r')

    if self.verbose:
      print 'DataLoader loading h5 file:', img_data_h5
    img_data = h5py.File('../data/{}'.format(img_data_h5), 'r')

    self.num_dialogs = {}

    for dsplit in dsplits:
      # read image list, if image features are needed
      self.img_features = np.array(img_data['images_'+dsplit])

      # Normalize the image features (if needed)
      # TODO: normalize image by default
      # if img_norm:
      #   print('Normalizing image features..')
      #   local nm = torch.sqrt(torch.sum(torch.cmul(imgFeats, imgFeats), 2));
      #   imgFeats = torch.cdiv(imgFeats, nm:expandAs(imgFeats)):float();

      # note: img_pos == np.arange(num_images)
      self.data[dsplit+'_img_fv'] = self.img_features
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

    new_sentences = np.zeros((numDialogs, numRounds, maxLength+2), dtype=np.uint32)
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
      inds = ordering[i%size:(i%size+batch_size)]
      yield self.getIndexData(inds, 'train')

  def getEvalBatch(self, start_id, batch_size):
    # get the next start id and fill up current indices till then
    next_start_id = math.min(self.num_dialogs['val']+1, start_id + batch_size)

    # dumb way to get range (complains if cudatensor is default)
    # TODO: replace this
    # inds = torch.LongTensor(next_start_id - start_id)
    # for ii in xrange(start_id, next_start_id - 1):
    #   inds[ii - start_id + 1] = ii

     # Index question, answers, image features for batch
    batch_output = self.getIndexData(inds, 'val')

    return batch_output, next_start_id

  def getIndexData(self, inds, dsplit):
    # get the question lengths
    images = self.data[dsplit+'_img_fv'][img_inds]

    captions = self.data[dsplit+'_cap'][inds]
    caption_lengths = self.data[dsplit+'_cap_len'][inds]

    # get questions
    true_questions = self.data[dsplit+'_ques'][inds]
    true_question_lengths = self.data[dsplit+'_ques_len'][inds]

    # get the answer lengths
    true_answers = self.data[dsplit+'_ans'][inds]
    true_answer_lengths = self.data[dsplit+'_ans_len'][inds]

    return images, captions, caption_lengths, true_questions, true_question_lengths, true_answers, true_answer_lengths, inds

if __name__ == '__main__':
  dataloader = DataLoader('visdial_params.json', 'visdial_data.h5',
                            'data_img.h5', ['train'], verbose=True)
  batch_generator = dataloader.getTrainBatch(5)
  for batch in batch_generator:
    print batch
