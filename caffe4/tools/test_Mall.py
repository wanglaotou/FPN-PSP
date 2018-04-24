#!/usr/bin/env python
import numpy as np
import skimage 
import os,sys,cv2
import pickle
from pyheatmap.heatmap import HeatMap

#from colour import Color
import requests

from matplotlib import pyplot as plt
from PIL import Image
from pylab import *
from skimage import transform,data

caffe_root = '/home/luohongling/crowd_count/MCNN2.3/caffe/'
sys.path.insert(0,caffe_root+'python')

os.chdir(caffe_root)
import caffe




net_file = '/ssd1/crowd_count/luohongling/MCNN2.3/pretrain1/deploy.prototxt'
caffe_model = '/ssd1/crowd_count/luohongling/MCNN2.3/pretrain1/models/roi_iter_2200000.caffemodel'


caffe.set_mode_gpu()
net = caffe.Net(net_file,caffe_model,caffe.TEST)



#print estcount

def fileMap(filepath):
    mapDir = os.path.join(filepath,'map.txt')
    roiDir = os.path.join(filepath,'roi/')
    mapList = np.loadtxt(mapDir,str,delimiter=' ')
    test_num = mapList[:,0].size
    total_MAE = 0.0
    total_MSE = 0.0
    i=0
    total_num = 0.0
    saveFile = '/ssd1/crowd_count/luohongling/MCNN2.3/pretrain1/test/test_mall_roi_220w.txt'
    save = file(saveFile,"a")
    save.write("net_file: " + str(net_file)+"\n")
    save.write("caffe_model: "+str(caffe_model)+"\n")
    save.write("#image  #gtcount  #estcount  #abs(gtcount-estcount)  #MAE  #MSE" + "\n")
    # for mall
    roi_name = 'mall_roi.txt'
    roi_file = os.path.join(roiDir,roi_name)
    roi_info = np.loadtxt(roi_file)


    #print test_num
    for m in mapList:
        #print m
        
        jpgDir = os.path.join(filepath,m[0])
        #print jpgDir  
        txtDir = os.path.join(filepath,m[1])
        if(os.path.isfile(txtDir)):
            with open(txtDir,'r') as f:
                gtcount = float(f.readline())
                total_num = total_num + gtcount
                print("The gtcount is %s"%(gtcount))
                save.write(str(m[0])+": ")
                save.write(str(gtcount)+"  ")
        if(os.path.isfile(jpgDir)):
            im = caffe.io.load_image(jpgDir)
        i=i+1

        net.blobs['data'].reshape(1,im.shape[2],im.shape[0],im.shape[1])
        transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
        transformer.set_transpose('data',(2,0,1))
        transformer.set_raw_scale('data',255)
        transformer.set_channel_swap('data',(2,1,0))

        net.blobs['data'].data[...] = transformer.preprocess('data',im)    
        out = net.forward()        
        feat = net.blobs['estdmap'].data[0]
        feat = np.reshape(feat, (feat.shape[1], feat.shape[2], -1))
        
        dst = np.reshape(feat,(feat.shape[0], feat.shape[1]))
        #color_feat = density_heatmap(feat)
        
        roi_scale=transform.rescale(roi_info,0.25)
        est_roi = np.multiply(dst,roi_scale)

        #color_map = density_heatmap(est_roi)
            #cv2.imshow("img",color_map);

            #cv2.waitKey(0);
            #np.savetxt("est_165050.txt",est_roi)
 
        estcount = np.sum(est_roi)
        print("The roi estcount is %.2f"%(estcount))
        save.write(str(estcount)+"  ")
        error = estcount-gtcount
        save.write(str(error)+"\n")

        total_MAE = total_MAE + np.abs(gtcount - estcount)
        total_MSE = total_MSE + np.square(gtcount - estcount)
    print("The scene has %d test img"%(i))
    MAE = total_MAE / i
    MSE = np.sqrt(total_MSE / i)
    print("The total_MAE is %.2f"%(total_MAE))
    print("The total_MSE is %.2f"%(total_MSE))
    print("The MAE is %.2f"%(MAE))
    print("The MSE is %.2f"%(MSE))
    print("Total num is %.2f"%(total_num))
    save.write("Total_num = "+str(total_num)+"\n")
    save.write("Total_MAE = "+str(total_MAE)+"\n")
    save.write("MAE = "+str(MAE)+"\n")
    save.write("MSE = "+str(MSE)+"\n")
    save.close()


fileMap('/ssd1/crowd_count/luohongling/data/mall_dataset/test_data')
#fileMap('/ssd1/crowd_count/luohongling/data/merge_data/test_data')

#fileMap('/ssd1/crowd_count/luohongling/data/ShanghaiExpo/test_data')
#fileMap('/ssd1/crowd_count/luohongling/data/ShanghaiTech/Part_A/test_data_remove')
#fileMap('/ssd1/crowd_count/luohongling/data/ShanghaiTech/Part_B/test_data')
#fileMap('/ssd1/crowd_count/luohongling/data/ShanghaiTech/Part_A/test_data')




#def eachFile(filepath):
    #pathDir = os.listdir(filepath) 
    #for s in pathDir:
        #newDir = os.path.join(filepath,s)
        #if(os.path.splitext(newDir)[1]=='.txt'):
            #f = open(newDir)
            #gtcount = f.readline()
            #print gtcount'''
        #if(os.path.splitext(newDir)[1]=='.jpg'):
            #im = caffe.io.load_image(newDir)
            #net.blobs['data'].data[...] = transformer.preprocess('data',im)
            #out = net.forward()
            #estcount = net.blobs['estcount'].data[0].flatten()
            #print estcount




#eachFile('/ssd1/crowd_count/luohongling/data/ShanghaiTech/part_A_final/test_data')
