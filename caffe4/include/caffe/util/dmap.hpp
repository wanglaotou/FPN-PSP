#ifndef _CAFFE_UTIL_DMAP_HPP_
#define _CAFFE_UTIL_DMAP_HPP_
#include <opencv2/core/core.hpp>
#include "caffe/util/util_crowd.hpp"
#include <iostream>
#include <algorithm>

using namespace cv;

namespace caffe {

void fspecial(const int size, const float sigma, Mat &gaussian_kernel);
void GetDensityMap(int hei_img, int wid_img, int ds_times,
                   const vector<HeadLocation<int> >& head_location,
                   const float sigma,
                   int neighbor_num, 
                   const float ksize_param, 
                   const float sigma_param,
                   Mat& cv_dmap);

void ConvertDensityMapForShow(const Mat &density, cv::Size sz, Mat &dmap_sh);

}  // namespace caffe

#endif  // CAFFE_UTIL_DMAP_HPP_
