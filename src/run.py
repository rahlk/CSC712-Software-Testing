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


def write(str):
  sys.stdout.write(str)
  sys.stdout.flush()


class ABCD():

  "Statistics Stuff, confusion matrix, all that jazz..."

  def __init__(self, before, after):
    self.actual = before
    self.predicted = after
    self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0
    self.abcd()

  def abcd(self):
    for a, b in zip(self.actual, self.predicted):
      if a == 1 and b == 1:
        self.TP += 1
      if a == 0 and b == 0:
        self.TN += 1
      if a == 0 and b == 1:
        self.FP += 1
      if a == 1 and b == 0:
        self.FN += 1

  def all(self):
    Sen = self.TP / (self.TP + self.FN)
    Spec = self.TN / (self.TN + self.FP)
    Prec = self.TP / (self.TP + self.FP)
    Acc = (self.TP + self.TN) / (self.TP + self.FN + self.TN + self.FP)
    F1 = 2 * self.TP / (2 * self.TP + self.FP + self.FN)
    G = 2 * Sen * Spec / (Sen + Spec)
    G1 = Sen * Spec / (Sen + Spec)
    return Sen, Spec, Prec, Acc, F1, G


class run():

  def __init__(
          self,
          pred=CART,
          _smoteit=True,
          _n=-1,
          _tuneit=False,
          dataName=None,
          reps=8):

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

    out_pred = [str(self.pred.__doc__) + suffix]
    for _ in xrange(self.reps):
      predRows = []
      train_DF = createTbl(self.train[self._n], isBin=True, bugThres=2)
      test_df = createTbl(self.test[self._n], isBin=True, bugThres=2)
      actual = Bugs(test_df)
      before = self.pred(train_DF, test_df,
                         tunings=self.tunedParams,
                         smoteit=self._smoteit)

      out_pred.append(_Abcd(before=actual, after=before)[-1])

    return out_pred

  def goRaw(self):

    def iqr(lst):
      lst = sorted(lst)
      return lst[int(len(lst) * 0.75)] - lst[int(len(lst) * 0.25)]

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

    out = []

    for _ in xrange(self.reps):
      predRows = []
      train_DF = createTbl(self.train[self._n], isBin=True, bugThres=1)
      test_df = createTbl(self.test[self._n], isBin=True, bugThres=1)
      actual = Bugs(test_df)
      before = self.pred(train_DF, test_df,
                         tunings=self.tunedParams,
                         smoteit=self._smoteit)

      out.append(ABCD(before=actual, after=before).all())

    med = [np.median([o[i] for o in out]) for i in xrange(len(out[0]))]
    quart = [iqr([o[i] for o in out]) for i in xrange(len(out[0]))]
    return self.pred.__doc__ + suffix, med, quart

  def go1(self):
    predRows = []
    train_DF = createTbl(self.train[self._n], isBin=True, bugThres=1)
    actual = Bugs(train_DF)
    print(self.dataName,
          len(actual),
          sum(actual),
          sum(actual) / len(actual) * 100)
#       with open('./raw/'+self.dataname, 'w+') as fwrite:


def _test(isLatex=True):
  tune = [True, False]
  smote = [True]
  for file in ['ant', 'camel', 'ivy',
               'jedit', 'log4j', 'lucene',
               'poi', 'synapse', 'velocity',
               'pbeans', 'xalan', 'xerces']:
    E = []
    for pred in [rforest]:
      for t in tune:
        for s in smote:
          R = run(
              reps=10,
              pred=pred,
              dataName=file,
              _tuneit=t,
              _smoteit=s).go()

          E.append(R)
    rdivDemo(E, isLatex=isLatex)
#     if isLatex:
#       latex().postamble()
#     else:
#       print('```')
#   if isLatex:
#     print("\\end{table*}\n\\end{document}")


def say(a, b, c):
  write(a)

  for med, iqr in zip(b, c):
    write(',%0.2f' % (med))  # + ', ' + str(iqr))

  print('')


def _testRaw(file):
  tune = [True, False]
  smote = [True]
  print("##%s\n" % (file))
  print('Dataset,', 'Pd,', '1-Pf,', 'Precision,', 'Accuracy,', 'F,', 'G')
#   print(',', 8 * 'med, ', 'med')
  for pred in [rforest]:
    for t in tune:
      for s in smote:
        name, med, iqr = run(
            pred=pred,
            dataName=file,
            _tuneit=t,
            _smoteit=s).goRaw()
        say(name, med, iqr)


def _test2(isLatex=True):
  for file in ['ant', 'camel', 'ivy',
               'jedit', 'log4j',
               'lucene', 'poi', 'synapse', 'velocity',
               'xalan']:
    run(
        pred=CART,
        dataName=file,
        _tuneit=False,
        _smoteit=False).go1()


if __name__ == '__main__':
  for file in ['ant', 'camel', 'ivy',
               'jedit', 'log4j']:
    #           , 'pbeans',
    #           'lucene',
    #           'poi', 'synapse',
    #           'velocity', 'xalan', 'xerces']:
    _testRaw(file)
#   _test()
#   eval(cmd())
