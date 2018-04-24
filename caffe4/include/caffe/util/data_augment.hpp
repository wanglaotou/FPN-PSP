#ifndef _CAFFE_DATA_AUGMENT_HPP_
#define _CAFFE_DATA_AUGMENT_HPP_
#include "caffe/blob.hpp"
#include <opencv2/core/core.hpp>
using namespace cv;

namespace caffe {
    
void GetNoiseImg(Mat OriImage, Mat& DstImage);

}  // namespace caffe

#endif  // CAFFE_UTIL_CROWD_HPP_