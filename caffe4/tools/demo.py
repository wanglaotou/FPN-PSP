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


caffe_root = '/home/luohongling/crowd_count/MCNN3.1/caffe/'
sys.path.insert(0,caffe_root+'python')

os.chdir(caffe_root)
import caffe

net_file = '/home/luohongling/crowd_count/model/MCNN3.1/2017-08-24/pretrain1_loss/deploy.prototxt'
caffe_model = '/home/luohongling/crowd_count/model/MCNN3.1/2017-08-24/pretrain1_loss/models/mydata_iter_950000.caffemodel'


caffe.set_mode_gpu()
net = caffe.Net(net_file,caffe_model,caffe.TEST)
src = "saveFile = '/home/luohongling/crowd_count/model/MCNN3.1/2017-08-24/pretrain1_loss/heatMap/"


def demoDMap(filepath):
    mapDir = os.path.join(filepath,'map_test.txt')
    mapList = np.loadtxt(mapDir,str,delimiter=' ')
    test_num = mapList[:,0].size
    
    #print testList
    #print type(testList)

    #total_MAE = 0.0
    #total_MSE = 0.0
    num =0;
    for m in mapList:
        num=num+1
        jpgDir = os.path.join(filepath,m[0])
        a = m[0]
        #print a

        b=a.replace('Image','Label')
        c=b.replace('jpg','txt')
        #print c      
        txtDir = os.path.join(filepath,c)
        n1 = c.index('frames')
        dir1 = c[n1:n1+23] 
        save_dir = os.path.join(src,dir1)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        n2 = c.index('frame_')
        name = c[n2:n2+15]
        if(os.path.isfile(txtDir)):
            with open(txtDir,'r') as f:
                gtcount = float(f.readline())
                print("The gtcount is %s"%(gtcount))
        if(os.path.isfile(jpgDir)):
            im = caffe.io.load_image(jpgDir)
        net.blobs['data'].reshape(1,im.shape[2],im.shape[0],im.shape[1])
            #print net.blobs['data'].data.shape
        transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
        transformer.set_transpose('data',(2,0,1))
            #print net.blobs['data'].data.shape
        transformer.set_raw_scale('data',255)
        transformer.set_channel_swap('data',(2,1,0))

        net.blobs['data'].data[...] = transformer.preprocess('data',im)
        print im.shape
        out = net.forward()
        #print 'blobs:'
        #print [(k, v.data.shape) for k, v in net.blobs.items()]

        estcount = net.blobs['estcount'].data[0]
        print("The estcount is %.2f"%(estcount))

        feat = net.blobs['estdmap'].data[0]


        #feat.reshape(feat.shape[1],feat.shape[2],feat.shape[0])
        feat = np.reshape(feat, (feat.shape[1], feat.shape[2], -1))
        dst = np.reshape(feat,(feat.shape[0], feat.shape[1]))
        save1 = save_dir+'/'
        np.savetxt(save1+str(name),dst)

        #color_map = density_heatmap(feat)
        #cv2.imwrite(src+str(num)+'.jpg',color_map)
        #cv2.imshow("img",color_map)

        #cv2.waitKey(0)


     



        
        #print feat.shape,feat.dtype
        #width = feat.shape[1]
        #height = feat.shape[0]




        #feat -= feat.min()
        #feat /= feat.max()
        #dst = np.reshape(feat,(feat.shape[0], feat.shape[1]))

        #np.savetxt("test1.txt",dst)
        
        #print dst.shape
        #print dst


def density_heatmap(density_map):
    import matplotlib.pyplot as plt
    from colour import Color
    density_range = 100

    #max_range = 255
    #min_range = 0
    #gap = max_range - min_range
    gradient = np.linspace(0, 1, density_range)
    img_width = density_map.shape[1]
    img_height = density_map.shape[0]
    color_map = np.empty([img_height, img_width, 3], dtype='uint8')
    # get gradient color using rainbow
    cmap = plt.get_cmap("rainbow") 
    blue = Color("blue") 
    hex_colors = list(blue.range_to(Color("red"), density_range))
    rgb_colors = [[rgb * 255 for rgb in color.rgb] for color in hex_colors][::-1]
   
    for i in range(img_height):
        for j in range(img_width):

            val_density = max(0.0, min(1.0, density_map[i][j]))
            #ratio = min_range + int(round(gap*val_density))
            ratio = int(round((density_range-1)*val_density))
            for k in range(3):
                #color_map[i][j][k] = int(cmap(gradient[ratio])[:3][k]*255)
                color_map[i][j][k] = rgb_colors[ratio][k]
    return color_map











        #dimg = Image.fromarray(np.uint8(dst))
        #dimg.show()
        


        #dst = dst*255
        #print dst.shape
        
        #cv2.imshow("img", dimg)
        #cv2.imwrite("/ssd1/crowd_count/luohonglincg/MCNN1.3/crowd_net/Part_A/pretrain1/test.jpg",dimg)
        #cv2.waitKey(0)



       # a = np.array(feat,np.int32)c

        #feat.save('/ssd1/crowd_count/luohongling/MCNN1.3/crowd_net/Part_A/pretrain1/test.jpg')
        #dimg = Image.fromarray(a)
        #dimg = dimg.covert('L')
        #dimg.show()
        #dst = dimg.resize((im.shape[0], feat.shape[1]))
        #print feat.shape
        #feat -= feat.min()
        #feat /= feat.max()
        #dst = transform.resize(feat,(im.shape[0], feat.shape[1]))
        #print dst.shapec

     
        #print feat
        #print feat.shape

        #cv2.imshow("img", dst)
        #cv2.waitKey(0)



       # with open('LastLayerOutput.pickle','wb') as f:
            #pickle.dump(feat,f)
        #vis_square(feat,padval=1)

        


def vis_square(data,padsize=1,padval=0):
    data-=data.min()
    data/=data.max()

    n = int(np.ceil(np.sqrt(data.shape[0])))
    print("The n is %.2f"%(n))
    print("The n is %.2f"%(data.ndim))
    padding = ((0,n**2-data.shape[0]),(0,padsize),(0,padsize)) + ((0,0),)*(data.ndim - 3)
    data = np.pad(data,padding,mode='constant',constant_values=(padval,padval))
    data = data.reshape((n,n) + data.shape[1:]).transpose((0,2,1,3)+ tuple(range(4,data.ndim+1)))
    data = data.reshape((n*data.shape[1],n*data.shape[3]) + data.shape[4:])
    print data.shape
    print data
    cv2.imshow("img", data)
    cv2.waitKey(0)


    #plt.show(data)

   #datamap[]
    #for i in data:


    #hm = HeatMap(data)
    #hm.heatmap(save_as="heat.png")

    
fileMap('/home/luohongling/crowd_count/data/MyData')

#demoDMap('/ssd1/crowd_count/luohongling/data/ShanghaiTech/part_A_final/test_data/')







'''def eachFile(filepath):
    pathDir = os.listdir(filepath) 
    for s in pathDir:
        newDir = os.path.join(filepath,s)
        if(os.path.splitext(newDir)[1]=='.txt'):
            f = open(newDir)
            gtcount = f.readline()
            print gtcount'''
        #if(os.path.splitext(newDir)[1]=='.jpg'):
            #im = caffe.io.load_image(newDir)
            #net.blobs['data'].data[...] = transformer.preprocess('data',im)
            #out = net.forward()
            #estcount = net.blobs['estcount'].data[0].flatten()
            #print estcount




#eachFile('/ssd1/crowd_count/luohongling/data/ShanghaiTech/part_A_final/test_data')
