import json

'''
unique_img_train 82783
unique_img_val 40504
ind2word 8845
word2ind 8845
'''

if __name__ == '__main__':
  filename = 'visdial_params.json'
  data = json.load(open('../data/{}'.format(filename)))
  for key, val in data.iteritems():
    print key, type(val), len(val)

  # print data['unique_img_train'][0:3]

  # indices = data['ind2word'].keys()[0:3]
  # for ind in indices:
  #   print ind, data['ind2word'][ind]

  # words = data['word2ind'].keys()[0:3]
  # for word in words:
  #   print word, data['word2ind'][word]

  # word = '<START>'
  # print word
  # print data['word2ind'][word]
  # print data['ind2word'][str(data['word2ind'][word])]

  # print data['ind2word']['0']
  print data['ind2word']['8845']
