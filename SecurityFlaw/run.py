#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function
from __future__ import division
from os import environ
from os import getcwd
from os import walk
from os import system
from pdb import set_trace
from random import uniform as rand
from random import randint as randi
from random import sample
from subprocess import call
from subprocess import PIPE
import pandas
import sys
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.constants.codata import precision
# Update PYTHONPATH
HOME = environ['HOME']
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystat/'  # PySTAT
cwd = getcwd()  # Current Directory
WHAT = '../SOURCE/'
sys.path.extend([axe, pystat, cwd, WHAT])
from table import clone
from abcd import _Abcd
from sk import rdivDemo
from sk import scottknott
from smote import SMOTE
from methods1 import *
from Prediction import formatData
from cliffsDelta import cliffs
from demos import cmd
from numpy import median


class predictor():

  def __init__(
          self,
          train=None,
          test=None,
          tuning=None,
          smoteit=False,
          duplicate=True):
    self.train = train
    self.test = test
    self.tuning = tuning
    self.smoteit = smoteit
    self.duplicate = duplicate

  def CART(self):
    "  CART"
    # Apply random forest Classifier to predict the number of bugs.
    if self.smoteit:
      self.train = SMOTE(
          self.train,
          atleast=1,
          atmost=300,
          resample=self.duplicate)

    if not self.tuning:
      clf = DecisionTreeClassifier(random_state=1)
    else:
      clf = DecisionTreeRegressor(max_depth=int(self.tunings[0]),
                                  min_samples_split=int(self.tunings[1]),
                                  min_samples_leaf=int(self.tunings[2]),
                                  max_features=float(self.tunings[3] / 100),
                                  max_leaf_nodes=int(self.tunings[4]),
                                  criterion='entropy', random_state=1)
    train_df = formatData(self.train)
    test_df = formatData(self.test)
    features = train_df.columns[:-2]
    klass = train_df[train_df.columns[-2]]
    # set_trace()
    clf.fit(train_df[features].astype('float32'), klass.astype('float32'))
    preds = clf.predict(
        test_df[test_df.columns[:-2]].astype('float32')).tolist()
    return preds

  def rforest(self):
    "  RF"
    # Apply random forest Classifier to predict the number of bugs.
    if self.smoteit:
      self.train = SMOTE(
          self.train,
          atleast=1,
          atmost=301,
          resample=self.duplicate)

    if not self.tuning:
      clf = RandomForestClassifier(random_state=1)
    else:
      clf = RandomForestClassifier(n_estimators=int(tunings[0]),
                                   max_features=tunings[1] / 100,
                                   min_samples_leaf=int(tunings[2]),
                                   min_samples_split=int(tunings[3]),
                                   random_state=1)
    train_df = formatData(self.train)
    test_df = formatData(self.test)
    features = train_df.columns[:-2]
    klass = train_df[train_df.columns[-2]]
    # set_trace()
    clf.fit(train_df[features].astype('float32'), klass.astype('float32'))
    preds = clf.predict(
        test_df[test_df.columns[:-2]].astype('float32')).tolist()
    return preds


class fileHandler():

  def __init__(self, dir='./Data/'):
    self.dir = dir

  def file2pandas(self, file):
    fread = open(file, 'r')
    rows = [line for line in fread]
    head = rows[0].strip().split(',')  # Get the headers
    body = [[1 if r == 'Y' else 0 if r == 'N' else r for r in row.strip().split(',')]
            for row in rows[1:]]
    return pandas.DataFrame(body, columns=head)

  def explorer2(self):
    files = [filenames for (
        dirpath,
        dirnames,
        filenames) in walk(self.dir)][0]
    for f in files:
      return [self.dir + f]

  def flatten(self, x):
    """
    Takes an N times nested list of list like [[a,b],[c, [d, e]],[f]]
    and returns a single list [a,b,c,d,e,f]
    """
    result = []
    for el in x:
      if hasattr(el, "__iter__") and not isinstance(el, basestring):
        result.extend(self.flatten(el))
      else:
        result.append(el)
    return result

  def kFoldCrossVal(self, data, k=5):
    acc = []
    sen = []
    spec = []
    prec = []
    chunks = lambda l, n: [l[i:i + n] for i in range(0, len(l), int(n))]
    from random import shuffle, sample
    rows = data._rows
    shuffle(rows)
    sqe = chunks(rows, int(len(rows) / k))
    if len(sqe) > k:
      sqe = sqe[:-2] + [sqe[-2] + sqe[-1]]
    for indx in xrange(k):
      testRows = sqe.pop(indx)
      trainRows = self.flatten([s for s in sqe if not s in testRows])
      train, test = clone(data, rows=[
          i.cells for i in trainRows]), clone(data, rows=[
              i.cells for i in testRows])
      train_df = formatData(train)
      test_df = formatData(test)
      actual = test_df[
          test_df.columns[-2]].astype('float32').tolist()
      before = predictor(train=train, test=test).rforest()
      acc.append(_Abcd(before=actual, after=before)[0])
      sen.append(_Abcd(before=actual, after=before)[1])
      spec.append(_Abcd(before=actual, after=before)[2])
      prec.append(_Abcd(before=actual, after=before)[-3])
      sqe.insert(k, testRows)
    return acc, sen, spec, prec

  def crossval(self, k=2):
    cv_acc = ['          Accuracy  ']
    cv_prec = ['         Precision  ']
    cv_sen = ['Sensitivity (Recall)']
    cv_spec = ['         Specificity']
    for _ in xrange(k):
      proj = self.explorer2()
      data = createTbl(proj, isBin=False, _smote=True)
      a, b, c, d = self.kFoldCrossVal(data, k=k)
      cv_acc.extend(a)
      cv_sen.extend(b)
      cv_spec.extend(c)
      cv_prec.extend(d)
    return cv_acc, cv_sen, cv_spec, cv_prec


def _doCrossVal():
  cv_acc = []
  cv_sen = []
  cv_spec = []
  cv_prec = []
  a, b, c, d = fileHandler().crossval(k=5)
  cv_acc.append(a)
  cv_sen.append(b)
  cv_spec.append(c)
  cv_prec.append(d)
# print(r'### Precision')
  rdivDemo(cv_acc, isLatex=False)
# print(r'### Sensitivity')
  rdivDemo(cv_sen, isLatex=False)
# print(r'### Sensitivity')
  rdivDemo(cv_spec, isLatex=False)
  rdivDemo(cv_prec, isLatex=False)

if __name__ == '__main__':
  _doCrossVal()
