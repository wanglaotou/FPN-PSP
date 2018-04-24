#!/usr/bin/env python
import numpy as np
import skimage
import os
import sys
import cv2


from matplotlib import pyplot

caffe_root = '/home/luohongling/crowd_count/caffe/MCNN3.1/caffe/'
sys.path.insert(0,caffe_root+'python')

os.chdir(caffe_root)
import caffe


#net_file = '/home/luohongling/crowd_count/model/MCNN3.1/2017-08-24/pretrain1_adddata/deploy.prototxt'
#caffe_model = '/home/luohongling/crowd_count/model/MCNN3.1/2017-08-24/pretrain1_adddata/models/mydata_iter_360000.caffemodel'

net_file = '/home/luohongling/crowd_count/model/MCNN3.1/2017-08-24/pretrain1_loss/deploy.prototxt'
caffe_model = '/home/luohongling/crowd_count/model/MCNN3.1/2017-08-24/pretrain1_loss/models/mydata_iter_1000000.caffemodel'



caffe.set_mode_gpu()       
net = caffe.Net(net_file,caffe_model,caffe.TEST)
# print 
#transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#transformer.set_transpose('data',(2,0,1))
#transformer.set_raw_scale('data',255)
#transformer.set_channel_swap('data',(2,0,1))
#print net.blobs['data'].data.shape


# im = caffe.io.load_image(img)
#net.blobs['data'].data[...] = transformer.preprocess('data',im)

#out = net.forward()

#f = open(labels_filename)
#gtcount = f.readline();

#gtcount = np.loadtxt(labels_filename);

#print gtcount

#estcount = net.blobs['estcount'].data[0].flatten()


#print estcount

def fileMap(filepath):
    mapDir = os.path.join(filepath,'map_test.txt')
    mapList = np.loadtxt(mapDir,str,delimiter=' ')
    test_num = mapList[:,0].size
    total_MAE = 0.0
    total_MSE = 0.0
    total_num = 0
    total_estcount = 0.0
    saveFile = '/home/luohongling/crowd_count/model/MCNN3.1/2017-08-24/pretrain1_loss/test/test_100w.txt'
    save = file(saveFile,"a")
    save.write("net_file: " + str(net_file)+"\n")
    save.write("caffe_model: "+str(caffe_model)+"\n")
    save.write("#image  #gtcount  #estcount  #abs(gtcount-estcount)  #MAE  #MSE" + "\n")


    #print test_num
    for m in mapList:
        #print m
        
        jpgDir = os.path.join(filepath,m[0])
        #print jpgDir
        a = m[0]
        #print a

        b=a.replace('Image','Label')
        c=b.replace('jpg','txt')
        #print c
        txtDir = os.path.join(filepath,c)
        if(os.path.isfile(txtDir)):
            with open(txtDir,'r') as f:
                gtcount = float(f.readline())
                total_num = total_num + gtcount
                print("The gtcount is %s"%(gtcount))
                save.write(str(m[0])+": ")
                save.write(str(gtcount)+"  ")
        if(os.path.isfile(jpgDir)):
            im = caffe.io.load_image(jpgDir)
            #channels = im.channels()
            #print im.ndim
            #int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
                            #CV_LOAD_IMAGE_GRAYSCALE);
            #Mat cv_img = imread(filename, cv_read_flag);

            #print im.shape[1]
            #print im.shape[0]
            #print im.shape[2]
            #print im.shapec
            net.blobs['data'].reshape(1,im.shape[2],im.shape[0],im.shape[1])
            #print net.blobs['data'].data.shape

            transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})


            transformer.set_transpose('data',(2,0,1))
            #print net.blobs['data'].data.shape
            transformer.set_raw_scale('data',255)
            transformer.set_channel_swap('data',(2,1,0))

            net.blobs['data'].data[...] = transformer.preprocess('data',im)

            #cv2.imshow("img", im)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            print net.blobs['data'].data.shape
            out = net.forward()

            estcount = net.blobs['estcount'].data[0]
            #estdmap = net.blobs['estdmap'].data[0]
            #ecount = out['estcount']
            print("The estcount is %.2f"%(estcount))
            save.write(str(estcount)+"  ")
            error = estcount-gtcount
            save.write(str(error)+"\n")
            #print ecount
        total_estcount = total_estcount + estcount
        total_MAE = total_MAE + np.abs(gtcount - estcount)
        total_MSE = total_MSE + np.square(gtcount - estcount)
    MAE = total_MAE / test_num
    MSE = np.sqrt(total_MSE / test_num)
    print("The total_MAE is %.2f"%(total_MAE))
    print("The total_MSE is %.2f"%(total_MSE))
    print("The MAE is %.2f"%(MAE))
    print("The MSE is %.2f"%(MSE))
    print("The Total num is %.2f"%(total_num))
    print("The P is %.2f"%((total_num-total_MAE)/total_num))
    save.write("total_num = "+str(total_num)+"\n")
    save.write("total_MAE = "+str(total_MSE)+"\n")
    save.write("MAE = "+str(MAE)+"\n")
    save.write("MSE = "+str(MSE)+"\n")
    save.write("P = "+str((total_num-total_MAE)/total_num)+"\n")
    #save.write("P2 = "+str(np.abs(total_num-total_estcount)/total_num)+"\n")
    save.close()


fileMap('/home/luohongling/crowd_count/data/MyData')
#fileMap('/ssd1/crowd_count/luohongling/data/Mall/test_data')
#fileMap('/ssd1/crowd_count/luohongling/data/merge_data/test_data')

#fileMap('/ssd1/crowd_count/luohongling/data/ShanghaiExpo/test_data')
#fileMap('/ssd1/crowd_count/luohongling/data/ShanghaiTech/Part_A/test_data_remove')
#fileMap('/ssd1/crowd_count/luohongling/data/ShanghaiTech/Part_B/test_data')
#fileMap('/ssd1/crowd_count/luohongling/data/ShanghaiTech/Part_A/test_data')

#fileMap('/ssd1/crowd_count/luohongling/data/mall_dataset/test_data')



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
