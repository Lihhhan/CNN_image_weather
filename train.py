#coding=utf-8
import caffe

from caffe import layers as L
from caffe import params as P

#设置GPU模式
#caffe.set_device(0)
#caffe.set_mode_gpu()

solver = caffe.SGDSolver('models/weather/solver.prototxt')
solver.solve()



