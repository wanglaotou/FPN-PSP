#include "caffe/util/data_augment.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

namespace caffe {

/*********************  data augment*******************************************/
void GaussianBlur(Mat OriImage,Mat& DstImage) {
  int ksize=15;
  int randnum = rand()%20;
  double sigma = randnum/10.0;  
  GaussianBlur(OriImage, DstImage, Size(ksize,ksize), sigma);
}

void SmoothBlur(Mat OriImage,Mat& DstImage) {
  int smooth_param = 1 + 2*(rand()%2);
  cv::blur(OriImage, DstImage, cv::Size(smooth_param, smooth_param));
}

void MedianBlur(Mat OriImage,Mat& DstImage) {
  int smooth_param = 3 + 2*(rand()%2);
  cv::medianBlur(OriImage, DstImage, smooth_param);
}

void BoxFilter(Mat OriImage,Mat& DstImage) {
  int smooth_param = 3 + 2*(rand()%2);
  cv::boxFilter(OriImage, DstImage, -1, cv::Size(smooth_param*2,smooth_param*2));
}

static double generateGaussianNoise() {
  const double pix2 = 6.2831853071795864769252866;
  static bool hasSpare = false;  
  static double rand1, rand2;  

  if(hasSpare)  
  {  
    hasSpare = false;  
    return sqrt(rand1) * sin(rand2);  
  }  

  hasSpare = true;  

  rand1 = rand() / ((double) RAND_MAX);  
  if(rand1 < 1e-100) rand1 = 1e-100;  
  rand1 = -2 * log(rand1);  
  rand2 = (rand() / ((double) RAND_MAX)) * pix2;  

  return sqrt(rand1) * cos(rand2);  
}

void GaussianNoise(Mat OriImage,Mat& DstImage) {
  int degree = rand()%24;
  DstImage = OriImage.clone();
  int channels = OriImage.channels();  

  int nRows = DstImage.rows;  
  int nCols = DstImage.cols * channels;  

  if(DstImage.isContinuous()){  
    nCols *= nRows;  
    nRows = 1;  
  }  

  int i,j;  
  uchar* p;  
  for(i = 0; i < nRows; ++i){  
    p = DstImage.ptr<uchar>(i);  
    for(j = 0; j < nCols; ++j){  
      double val = p[j] + generateGaussianNoise() * degree; 
      if(val < 0)  
        val = 0;  
      if(val > 255)  
        val = 255;  

      p[j] = (uchar)val;  

    }  
  }  
}

void salt(Mat OriImage, Mat& DstImage, int n){
    for(int k=0;k<n;k++)
    {
        int i=rand()%OriImage.cols;
        int j=rand()%OriImage.rows;
        if(OriImage.channels()==1)
        {
            DstImage.at<uchar>(j,i)=255;
        }
        else if(OriImage.channels()==3)
        {
            DstImage.at<Vec3b>(j,i)[0]=255;
            DstImage.at<Vec3b>(j,i)[1]=255;
            DstImage.at<Vec3b>(j,i)[2]=255;
        }
    }
}

void sharpen2D(Mat OriImage, Mat& DstImage){
    //¹¹ÔìºË
    Mat kernel(3,3,CV_32F,Scalar(0));
    //¶ÔºËÔªËØ½øÐÐ¸³Öµ
    kernel.at<float>(1,1)=5.0;
    kernel.at<float>(0,1)=-1.0;
    kernel.at<float>(2,1)=-1.0;
    kernel.at<float>(1,0)=-1.0;
    kernel.at<float>(1,2)=-1.0;
    filter2D(OriImage,DstImage,OriImage.depth(),kernel);
}

void colorReduce(Mat OriImage, Mat& DstImage, int div = 8){
    int nl=OriImage.rows;
    int nc=OriImage.cols*OriImage.channels();
    for(int j=0;j<nl;j++)
    {
        uchar *data=DstImage.ptr<uchar>(j);
        for(int i=0;i<nc;i++)
        {
            data[i]=data[i]/div*div+div/2;
        }
    }
}

void GetNoiseImg(Mat OriImage, Mat& DstImage) {
  int rnd = rand()%5;
  // LOG(INFO) << "noise:  " << rnd;
  if (rnd == 0) {
    GaussianBlur(OriImage, DstImage);
  }else if(rnd == 1) {
    SmoothBlur(OriImage, DstImage);
  }else if(rnd == 2) {
    GaussianNoise(OriImage, DstImage);
  }else if(rnd == 3) {
    int noise_num = 2000;
    salt(OriImage, DstImage, noise_num);
  }else if(rnd == 4) {
    colorReduce(OriImage, DstImage);
  }else {
    DstImage = OriImage;
  }

}

    

}  // namespace caffe
