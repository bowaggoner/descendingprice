from collections import Sequence
from bisect import bisect_left

# got this approach from stackoverflow
class MySeq(Sequence):
  def __init__(self, array, f):
    self.array = array
    self.f = f

  def __len__(self):
    return len(self.array)

  def __getitem__(self, i):
    return self.f(self.array[i])

def lazy_bisect(x, array, f, lo=0, hi=None):
  if hi is None:
    hi = len(array)
  seq = MySeq(array, f)
  return bisect_left(seq, x, lo, hi)

