from __future__ import print_function

class latex():
  def __init__(self):
    pass

  def preamble(self):
        print(r"""\documentclass{article}
              \usepackage{colortbl}
              \usepackage{fullpage}
              \usepackage{booktabs}                        
              \usepackage{bigstrut}
              \usepackage[table]{xcolor}
              \usepackage{picture}
              \newcommand{\quart}[4]{\begin{picture}(100,6)
              {\color{black}\put(#3,3){\circle*{4}}\put(#1,3){\line(1,0){#2}}}\end{picture}}
              \begin{document}
              """)

  def subsection(self, str):
    print("\\subsection*{%s}"%(str))

  def postamble(self):
    print("\end{document}")
