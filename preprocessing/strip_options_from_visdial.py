DATA_FOLDER = '../data/'

def strip_options(filename):
  with open(DATA_FOLDER + filename + '_stripped.json', 'w') as outfile:
    with open(DATA_FOLDER + filename + '.json') as infile:
      window = list(infile.read(12))
      count = 0
      outfile.write(''.join(window))
      filebuffer = []
      strip = False
      while True:
        c = infile.read(1)
        if c == '': break  # EOF
        window = window[1:] + [c]
        if ''.join(window) == ', "options":':
          strip = True
        if not strip:
          filebuffer.append(c)
        if strip and c == ']':
          strip = False
          outfile.write(''.join(filebuffer[:-11]))
          filebuffer = []
        if count % 1e6 == 0:
          print int(count / 1e6)
        count += 1
      outfile.write(''.join(filebuffer))

if __name__ == '__main__':
  split = 'train'
  # split = 'val'
  # split = 'test'
  filename = 'visdial_0.5_' + split

  strip_options(filename)