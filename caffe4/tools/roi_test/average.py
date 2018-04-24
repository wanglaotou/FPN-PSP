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
caffe_model = '/ssd1/crowd_count/luohongling/MCNN2.3/pretrain1/models/roi_iter_2600000.caffemodel'


caffe.set_mode_gpu()
net = caffe.Net(net_file,caffe_model,caffe.TEST)
#src = "/ssd1/crowd_count/luohongling/data/ShanghaiExpo/test_roi/"
#src = "/ssd1/crowd_count/luohongling/MCNN1.5/crowd_net/Part_A/2017-07-21/pretrain1_sigma0.12/dmap/"


def demoDMap(filepath):
    mapDir = os.path.join(filepath,'map.txt')
    #for roiList in os.listdir(filepath)
    roiDir = os.path.join(filepath,'roi/')
    #print(roiDir)
    
    mapList = np.loadtxt(mapDir,str,delimiter=' ')

    saveFile = '/ssd1/crowd_count/luohongling/MCNN2.3/pretrain1/test/Eaverage_260w.txt'
    save = file(saveFile,"a")
    save.write("net_file: " + str(net_file)+"\n")
    save.write("caffe_model: "+str(caffe_model)+"\n")
    save.write("#image  #gtcount  #estcount  #abs(gtcount-estcount)  #MAE  #MSE" + "\n")


    test_num = mapList[:,0].size
    print test_num
 
    total_MAE = 0.0
    total_MSE = 0.0
    i=0
    total_num = 0
 
    for m in mapList:
        jpgDir = os.path.join(filepath,m[0])
        scene_id = m[0][0:6]
        roi_name = scene_id+'.txt'
        roi_file = os.path.join(roiDir,roi_name)
        roi_info = np.loadtxt(roi_file)
      
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
        color_feat = density_heatmap(feat)
        
        roi_scale=transform.rescale(roi_info,0.25)
        est_roi = np.multiply(dst,roi_scale)

        color_map = density_heatmap(est_roi)
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
    print("The Total num is %.2f"%(total_num))

    save.write("total_num = "+str(total_num)+"\n")
    save.write("total_MAE = "+str(total_MSE)+"\n")
    save.write("MAE = "+str(MAE)+"\n")
    save.write("MSE = "+str(MSE)+"\n")
    save.close()




        #cv2.imshow("img",roi_scale);

        #cv2.waitKey(0);
        
        #print(roi_info.shape)
        #print(transform.rescale(roi_info,0.25).shape)

        #roi_info.reshape(roi_info.shape[0]/4,roi_info.shape[1]/4)















        #mask=zeros((srcImg.shape[0],srcImg.shape[1]))
        #vector<vector<point>> contour
        #vector<Point> pts

        #contour = cvCreateSeq(CV_SEQ_ELTYPE_POINT|CV_SEQ_FLAG_CLOSED,sizeof())
        #pts.push_back(Point(9,186))
        #pts.push_back(Point(310,101))
        #pts.push_back(Point(610,117))
        #pts.push_back(Point(714,122))
        #pts.push_back(Point(717,573))
        #pts.push_back(Point(7,570))

        #contour.push_back(pts)
        #cv2.drawContours(mask,contour,0,all(255),-1)
        #srcImg.copyTo(dstImg,mask)



        #color_map = density_heatmap(srcImg)
        #cv2.imwrite(src+'heat_'+str(m[0]),color_map)
        #cv2.imshow("img",dstImg);

        #cv2.waitKey(0);


     



        
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
        #print dst.shape

     
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
    #print data.shape
    #print datac
    #cv2.imshow("img", data)
    #cv2.waitKey(0)


    #plt.show(data)

   #datamap[]
    #for i in data:


    #hm = HeatMap(data)
    #hm.heatmap(save_as="heat.png")

    


#demoDMap('/ssd1/crowd_count/luohongling/data/ShanghaiTech/part_A_final/test_data/')
demoDMap('/ssd1/crowd_count/luohongling/data/ShanghaiExpo/test_data')







#'''def eachFile(filepath):
 #   pathDir = os.listdir(filepath) 
  #  for s in pathDir:
   #     newDir = os.path.join(filepath,s)
    #    if(os.path.splitext(newDir)[1]=='.txt'):
     #       f = open(newDir)
      #      gtcount = f.readline()
       #     print gtcount'''
        #if(os.path.splitext(newDir)[1]=='.jpg'):
            #im = caffe.io.load_image(newDir)
            #net.blobs['data'].data[...] = transformer.preprocess('data',im)
            #out = net.forward()
            #estcount = net.blobs['estcount'].data[0].flatten()
            #print estcount




#eachFile('/ssd1/crowd_count/luohongling/data/ShanghaiTech/part_A_final/test_data')
