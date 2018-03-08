import pandas as pd
from pycocotools.coco import COCO

def get_image_urls(annFile, image_ids):
  # initialize COCO api for instance annotations
  coco = COCO(annFile)

  images = coco.loadImgs(image_ids)
  return [image['coco_url'] for image in images]

if __name__ == '__main__':
  # data = ('coco_images_train_stripped.csv', 'train2014', 'train')
  # data = ('coco_images_val_stripped.csv', 'train2014', 'val')
  data = ('coco_images_test_stripped.csv', 'val2014', 'test')

  annFile = '../annotations/instances_{}.json'.format(data[1])

  image_ids = pd.read_csv('../../image_ids/{}'.format(data[0]))['image_ids'].tolist()
  print len(image_ids)
  image_urls = get_image_urls(annFile, image_ids)
  pd.DataFrame({'image_ids': image_ids, 'image_urls': image_urls}).to_csv('../../image_urls/{}_urls.csv'.format(data[2]), index=False)
