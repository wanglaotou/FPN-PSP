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

#print estcount
def fileMap(filepath,netFile,caffeModel,saveFile):
    mapDir = os.path.join(filepath,'map_test.txt')
    mapList = np.loadtxt(mapDir,str,delimiter=' ')
    test_num = mapList[:,0].size
    total_MAE = 0.0
    total_MSE = 0.0
    total_gtcount = 0
    total_estcount = 0.0
    
    save = file(saveFile,"a")
    save.write("net_file: " + str(netFile)+"\n")
    save.write("caffe_model: "+str(caffeModel)+"\n")
    save.write("#image  #gtcount  #estcount  #abs(gtcount-estcount)  #MAE  #MSE" + "\n")


    ## for each img test
    for m in mapList:
        jpgDir = os.path.join(filepath,m[0])
        #print jpgDir

        ## read label file
        a = m[0]
        b=a.replace('Image','Label')
        c=b.replace('jpg','txt')
        txtDir = os.path.join(filepath,c)
        #print txtDir
        if(os.path.isfile(txtDir)):
            with open(txtDir,'r') as f:
                gtcount = float(f.readline())
                total_gtcount = total_gtcount + gtcount
                print("The gtcount is %s"%(gtcount))
                save.write(str(m[0])+": ")
                save.write(str(gtcount)+"  ")

        ## predict
        if(os.path.isfile(jpgDir)):
            ## 1> read img
            im = caffe.io.load_image(jpgDir)
            print im.shape[0]
            print im.shape[1]
            print im.shape[2]
            print im.shapec

            ## 2> set net img size
            net.blobs['data'].reshape(1,im.shape[2],im.shape[0],im.shape[1])
            print net.blobs['data'].data.shape

            ## 3> Transformer
            transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
            transformer.set_transpose('data',(2,0,1))
            transformer.set_raw_scale('data',255)
            transformer.set_channel_swap('data',(2,1,0))

            ## 4> preprocess and Load transformer img to net memory
            net.blobs['data'].data[...] = transformer.preprocess('data',im)
            print net.blobs['data'].data.shape

            ## 5> infer
            out = net.forward()
            %timeit net.forward()

            estcount = net.blobs['estcount'].data[0]
            #estdmap = net.blobs['estdmap'].data[0]

            print("The estcount is %.2f"%(estcount))
            save.write(str(estcount)+"  ")
            error = estcount-gtcount
            save.write(str(error)+"\n")

            cv2.imshow("img", im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
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
    print("The total_gtcount is %.2f"%(total_gtcount))
    print("The P is %.2f"%((total_gtcount-total_MAE)/total_gtcount))
    save.write("total_gtcount = "+str(total_gtcount)+"\n")
    save.write("total_MAE = "+str(total_MSE)+"\n")
    save.write("MAE = "+str(MAE)+"\n")
    save.write("MSE = "+str(MSE)+"\n")
    save.write("P = "+str((total_gtcount-total_MAE)/total_gtcount)+"\n")
    #save.write("P2 = "+str(np.abs(total_num-total_estcount)/total_num)+"\n")
    save.close()

if __name__=='__main__':
    netFile = '/home/luohongling/crowd_count/model/MCNN3.1/2017-08-24/pretrain1_loss/deploy.prototxt'
    caffeModel = '/home/luohongling/crowd_count/model/MCNN3.1/2017-08-24/pretrain1_loss/models/mydata_iter_1000000.caffemodel'
    caffe.set_device(0)
    caffe.set_mode_gpu()       
    net = caffe.Net(netFile,caffeModel,caffe.TEST)

    saveFile = '/home/luohongling/crowd_count/model/MCNN3.1/2017-08-24/pretrain1_loss/test/test_100w.txt'
    filePath = '/home/luohongling/crowd_count/data/MyData'
    fileMap(filePath,netFile,caffeModel,saveFile)


    #fileMap('/ssd1/crowd_count/luohongling/data/Mall/test_data')
    #fileMap('/ssd1/crowd_count/luohongling/data/merge_data/test_data')

    #fileMap('/ssd1/crowd_count/luohongling/data/ShanghaiExpo/test_data')
    #fileMap('/ssd1/crowd_count/luohongling/data/ShanghaiTech/Part_A/test_data_remove')
    #fileMap('/ssd1/crowd_count/luohongling/data/ShanghaiTech/Part_B/test_data')
    #fileMap('/ssd1/crowd_count/luohongling/data/ShanghaiTech/Part_A/test_data')