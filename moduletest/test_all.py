import os
import sys

outdir='tmp'

if os.path.exists(os.path.join(outdir, 'test.csv')):
    os.remove(os.path.join(outdir, 'test.csv'))

os.system('python {} --outdir tmp'.format(os.path.join(sys.path[0], 'test_relu.py')))
os.system('python {} --outdir tmp'.format(os.path.join(sys.path[0], 'test_sigmoid.py')))
os.system('python {} --outdir tmp'.format(os.path.join(sys.path[0], 'test_flatten.py')))
os.system('python {} --outdir tmp'.format(os.path.join(sys.path[0], 'test_silu.py')))
