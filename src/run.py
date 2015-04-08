#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function, division
from os import environ, getcwd
import sys

# Update PYTHONPATH
cwd = getcwd()  # Current Directory
axe = cwd + '/axe/'  # AXE
pystat = cwd + '/pystats/'  # PySTAT
sys.path.extend([axe, pystat, cwd])

from Prediction import *
from _imports import *
from abcd import _Abcd
from cliffsDelta import cliffs
from methods1 import *
import numpy as np
import pandas as pd
from sk import rdivDemo
from pdb import set_trace
from dEvol import tuner
from os import walk
from demos import cmd
from latex import latex
from weights import weights as W


class run():

  def __init__(
          self,
          pred=CART,
          _smoteit=True,
          _n=-1,
          _tuneit=False,
          dataName=None,
          reps=24):

    self.dataName = dataName
    self.pred = pred
    self.out_pred = []
    self._smoteit = _smoteit
    self._tuneit = _tuneit
    self.train, self.test = self.categorize()
    self.reps = reps
    self._n = _n
    self.tunedParams = None if not _tuneit else tuner(
        self.pred, self.train[_n])

  def categorize(self):
    dir = '../Data'
    self.projects = [Name for _, Name, __ in walk(dir)][0]
    self.numData = len(self.projects)  # Number of data
    one, two = explore(dir)
    data = [one[i] + two[i] for i in xrange(len(one))]

    def withinClass(data):
      N = len(data)
      return [(data[:n], [data[n]]) for n in range(1, N)]

    def whereis():
      for indx, name in enumerate(self.projects):
        if name == self.dataName:
          return indx

    return [
        dat[0] for dat in withinClass(
            data[
                whereis()])], [
        dat[1] for dat in withinClass(
            data[
                whereis()])]  # Train, Test

  def go(self):
    for _ in xrange(self.reps):
      predRows = []
      train_DF = createTbl(self.train[self._n], isBin=True, bugThres=1)
      test_df = createTbl(self.test[self._n], isBin=True, bugThres=1)
      actual = Bugs(test_df)
      before = self.pred(train_DF, test_df,
                         tunings=self.tunedParams,
                         smoteit=self._smoteit)

      self.out_pred.append(_Abcd(before=actual, after=before)[-1])

    if self._smoteit:
      if self._tuneit:
        suffix = "(SMOTE, Tune)"
      else:
        suffix = "(SMOTE)"
    else:
      if self._tuneit:
        suffix = "(Tune)"
      else:
        suffix = ""

    self.out_pred.insert(0, str(self.pred.__doc__) + suffix)
    return self.out_pred

  def go1(self):
    import csv
    for _ in xrange(self.reps):
      predRows = []
      train_DF = createTbl(self.train[self._n], isBin=True, bugThres=1)
      test_df = createTbl(self.test[self._n], isBin=True, bugThres=1)
      actual = Bugs(test_df)
      before = []
      header = [h.name for h in train_DF.headers]
      for smote in [True, False]:
        for p in [CART, rforest]:
          header += [p.__doc__] if not smote else [p.__doc__ + ' (SMOTE)']
          before.append(p(train_DF, test_df,
                          tunings=self.tunedParams,
                          smoteit=smote))
      body = []
      for b, one, two, three, four in zip(test_df._rows, before[0], before[1], before[2], before[3]):
        newCell = b.cells[:-1] + [one, two, three, four]
        body.append(newCell)
      with open('./raw/' + self.dataName + '.csv', 'w+') as fwrite:
        writer = csv.writer(fwrite, delimiter=',')
        writer.writerow(header)
        for b in body:
          writer.writerow(b)

  def histplot(self, lst):
    from numpy import median
    from numpy import array
    from numpy import std
    from numpy import arange
    import matplotlib.pyplot as plt
    N = len(lst[0])
    ind = arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(
        ind,
        [l * 100 for l in lst[0]],
        width,
        color=[
            0.9,
            0.9,
            0.9])
    ax.set_ylabel('WEIGHTS in %')
    ax.set_xlabel('Features')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(tuple(str(n) for n in xrange(N)))
    plt.savefig('_figs/%s.jpg' % (self.dataName))

  def fWeight(self, criterion='Variance'):
    train_DF = createTbl(self.train[self._n], isBin=True, bugThres=1)
    lbs = W(use=criterion).weights(train_DF)
    sortedLbs = sorted([l / max(lbs[0]) for l in lbs[0]], reverse=True)
    L = [l / max(lbs[0]) for l in lbs[0]]
    self.histplot([L, lbs[1]])


def _weights():
  for file in ['ant', 'camel', 'ivy', 'forrest',
               'jedit', 'poi', 'log4j',
               'lucene', 'velocity',
               'xalan', 'xerces']:
    print("## %s" % (file))
    R = run(dataName=file).fWeight()


def _test(isLatex=False):
  # print("All but one")
  # tune = [False]  # , True]
  #   smote = [True, False]
  #   if isLatex:
  #     latex().preamble()
  for file in ['ant', 'camel', 'ivy', 'forrest',
               'jedit', 'poi', 'log4j',
               'lucene', 'velocity',
               'xalan', 'xerces']:
    print("## %s" % (file))
    R = run(dataName=file).go1()

#     E = []
#     if isLatex:
#       latex().subsection(file)
#     else:
# print("## %s\n```" % (file))
#     for pred in [CART, rforest]:
#       for t in tune:
#         for s in smote:
#           R = run(
#               pred=pred,
#               dataName=file,
#               _tuneit=t,
#               _smoteit=s).go1()
#           E.append(R)
#
#     rdivDemo(E, isLatex=isLatex)
#     if isLatex:
#       latex().postamble()
#     else:
#       print('```')
#   if isLatex:
#     print("\\end{table*}\n\\end{document}")


if __name__ == '__main__':
  _weights()
#   eval(cmd())
