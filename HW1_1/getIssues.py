""" Code used to extract data from GITHUB"""

from pygithub3 import Github
import time
import datetime
def d2s(x):
  temp =time.mktime(time.strptime(str(x), "%Y-%m-%d %H:%M:%S"))
  return (time.time() - temp)/(3600*24)

auth = dict(login='rahlk', password='nb20152dq')
gh = Github(**auth)

octocat_issues = gh.issues.list_by_repo('scikit-learn','scikit-learn',state="all")

for page in enumerate(octocat_issues):
  for resource in page:
    print "State: ",resource.state," Days: ",d2s(resource.created_at)#," closed at: ",d2s(resource.closed_at)