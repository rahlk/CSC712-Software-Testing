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

class run():

  def __init__(
          self,
          pred=CART,
          _smoteit=True,
          _n=-1,
          _tuneit=False,
          dataName=None,
          reps=10):

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
      train_DF = createTbl(self.train[self._n], isBin=True)
      test_df = createTbl(self.test[self._n], isBin=True)
      actual = Bugs(test_df)
      before = self.pred(train_DF, test_df,
                         tunings=self.tunedParams,
                         smoteit=self._smoteit)

      self.out_pred.append(_Abcd(before=actual, after=before)[-1])

    if self._smoteit:
      if self._tuneit:
        suffix = "_s+tune_"
      else:
        suffix = "_s_"
    else:
      if self._tuneit:
        suffix = "_tune_"
      else:
        suffix = "_"

    self.out_pred.insert(0, self.dataName + suffix + str(self.pred.__doc__))
    return self.out_pred


def _test(isLatex = False):
  tune = [False]#, True]
  smote = [True, False]
  if isLatex: latex().preamble()
  for file in ['ant', 'camel', 'ivy',
               'jedit', 'poi', 'log4j',
               'lucene', 'pbeans', 'velocity',
               'xalan', 'xerces']:
    E = []
    if isLatex: latex().subsection(file)
    for pred in [CART, rforest]:
      for t in tune:
        for s in smote:
          R = run(
               pred=pred,
               dataName=file,
               _tuneit=t,
               _smoteit=s).go()
          E.append(R)
    
    rdivDemo(E, isLatex=isLatex)
    

  
if __name__ == '__main__':
  eval(cmd())
