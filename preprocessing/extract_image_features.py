import urllib, cStringIO
import pandas as pd
import numpy as np

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from imagenet_utils import decode_predictions

def get_image_urls(filepath):
  return pd.read_csv(filepath)['image_ids'].tolist(), pd.read_csv(filepath)['image_urls'].tolist()

def extract_image_features(model, img_url):
  print 'URL', img_url
  file = cStringIO.StringIO(urllib.urlopen(img_url).read())

  img = image.load_img(file, target_size=(224, 224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  return model.predict(x)

if __name__ == '__main__':
  filename = 'mini.csv'
  img_ids, img_urls = get_image_urls('../image_urls/{}'.format(filename))

  # model = VGG16(weights='imagenet')
  # model.layers.pop()

  for img_id, img_url in zip(img_ids, img_urls):
    # img_id = int(img_url[-15:-4])
    print image_id
    # features = extract_image_features(model, img_url)
    # print decode_predictions(features)
    # np.save('resized_2.npy', features)
