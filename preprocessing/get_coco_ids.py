import ijson
import pandas as pd

DATA_FOLDER = '../data/'

def get_ids(filename):
  image_ids = []
  splits = []
  parser = ijson.parse(open(DATA_FOLDER + filename))
  count = 0
  printed = False
  for prefix, event, value in parser:
    if (prefix, event) == ('item.image_id', 'number'):
      image_ids.append(value)
      count += 0.5
      printed = False
    elif (prefix, event) == ('item.split', 'string'):
      splits.append(value)
      count += 0.5
      printed = False
    if count % 1000 == 0 and not printed:
      print int(count / 1000)
      printed = True
  return image_ids, splits

if __name__ == '__main__':
  # split = 'train_stripped'
  # split = 'val_stripped'
  # split = 'test_stripped'
  filename = 'visdial_0.5_' + split + '.json'

  image_ids, splits = get_ids(filename)
  print 'len(image_ids)', len(image_ids)
  print 'len(splits)', len(splits)
  print image_ids[:3], splits[:3]
  data = {'image_ids':image_ids, 'splits':splits}
  pd.DataFrame(data).to_csv('coco_images_' + split + '.csv', index=False)

