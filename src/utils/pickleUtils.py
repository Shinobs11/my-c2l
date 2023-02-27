import blosc, pickle

def pload(filename: str):
    fname = filename + ".pickle.blosc"
    res = None
    with open(fname, mode="rb") as f:
      res = pickle.loads(blosc.decompress(f.read()))
    return res


def pdump(obj, filename:str):
    fname = filename + ".pickle.blosc"
    with open(fname, mode="wb") as f:
      f.write(blosc.compress(pickle.dumps(obj)))

def pjoin(pathname):
  return pathname + ".pickle.blosc"

