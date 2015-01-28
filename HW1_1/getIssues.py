""" Code used to extract data from GITHUB"""
from __future__ import print_function
from pygithub3 import Github
import time
import datetime

def d2s(x):
  temp = time.mktime(time.strptime(str(x), "%Y-%m-%d %H:%M:%S"))
  return (time.time() - temp) / (3600 * 24)

auth = dict(login = 'rahlk', password = 'nb20152dq')
gh = Github(**auth)

octocat_issues = gh.issues.list_by_repo('scikit-learn', 'scikit-learn', state = "all")
indx = 0
print('Index, State, Days, Rate')
for page in (octocat_issues):
  for resource in (page):
    indx += 1
    now = d2s(resource.created_at)
    if indx % 50 == 0:
      print('{0:1d},{1:6s},{2:}'.format(indx , resource.state, now))  # ," closed at: ",d2s(resource.closed_at)
