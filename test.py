def main():
  x = []
  def helper():
    x = 5
    return x
  f = helper()
  x.append(f)
  print x

if __name__ == '__main__':
  main()