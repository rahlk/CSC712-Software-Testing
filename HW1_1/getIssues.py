""" Code used to extract data from GITHUB"""
from __future__ import print_function, division
from pygithub3 import Github
import time
import datetime
indx = 0

def d2s(x):
  temp =time.mktime(time.strptime(str(x), "%Y-%m-%d %H:%M:%S"))
  return (time.time() - temp)/(3600*24)

auth = dict(login='rahlk', password='nb20152dq')
gh = Github(**auth)

octocat_issues = gh.issues.list_by_repo('scikit-learn','scikit-learn',state="all")

print('Issues, State, Cumulative Days, Rate(Bugs/Day)')
for _, page in enumerate(octocat_issues):
  for _, resource in enumerate(page):
  	indx += 1
  	if indx%50 == 0:
	    print(('{0},{1},{2:.0f},{3:.2f}').format(indx,resource.state,
	    			d2s(resource.created_at),indx/d2s(resource.created_at)))