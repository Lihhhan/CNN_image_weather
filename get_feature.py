#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import caffe
import sys
import pickle
import struct
import sys,cv2
caffe_root = '/home/han/workspace/caffe/'  
# 运行模型的prototxt
deployPrototxt =  'models/weather/deploy.prototxt'
# 相应载入的modelfile
modelFile = 'caffemodel/_iter_79000.caffemodel'
# meanfile 也可以用自己生成的
meanFile = 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
# 需要提取的图像列表
imageListFile = 'snow.txt'
imageBasePath = '/home/han/weather_database/'

# 初始化函数的相关操作
def initilize():
    print 'initilize ... '

    sys.path.insert(0, caffe_root + 'python')
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(deployPrototxt, modelFile,caffe.TEST)
    return net  
# 提取特征并保存为相应地文件
def extractFeature(imageList, net):
    # 对输入数据做相应地调整如通道、尺寸等等
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(caffe_root + meanFile).mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  
    transformer.set_channel_swap('data', (2,1,0))  
    # set net to batch size of 1 如果图片较多就设置合适的batchsize 
    net.blobs['data'].reshape(1,3,227,227)      #这里根据需要设定，如果网络中不一致，需要调整
    num=0
    for imagefile in imageList:
        imagefile_abs = os.path.join(imageBasePath, imagefile)
        print imagefile_abs
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(imagefile_abs))
        out = net.forward()
        fea_file = 'feature.txt' 
        num +=1
        print 'Num ',num,' extract feature ',fea_file
        with  open(fea_file,'w') as f:
            for x in xrange(0, net.blobs['fc8_weather'].data.shape[0]):
                for y in xrange(0, net.blobs['fc8_weather'].data.shape[1]):
                    f.write(struct.pack('f', net.blobs['fc8_weather'].data[x,y]))

# 读取文件列表
def readImageList(imageListFile):
    imageList = []
    with open(imageListFile,'r') as fi:
        while(True):
            line = fi.readline().strip().split()# every line is a image file name
            if not line:
                break
            imageList.append(line[0]) 
    print 'read imageList done image num ', len(imageList)
    return imageList

if __name__ == "__main__":
    net = initilize()
    imageList = readImageList(imageListFile) 
    extractFeature(imageList, net)

