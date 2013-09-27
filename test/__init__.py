import os

from YSOVAR import plot

outroot = os.path.join(os.path.dirname(__file__),'testout')

plot.filetype = ['.pdf']
    
if not os.path.exists(outroot):
    os.makedirs(outroot)
