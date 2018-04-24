#ifndef _CAFFE_UTIL_CROWD_HPP_
#define _CAFFE_UTIL_CROWD_HPP_
#include "caffe/blob.hpp"
#include <opencv2/core/core.hpp>
using namespace cv;

namespace caffe {
    
template <typename Dtype>
struct HeadLocation { // from label file 
    Dtype x;
    Dtype y;
    Dtype w;
    Dtype h;
};


void GetImageBlobShape(const bool is_color, const int height, const int width, vector<int> &shape);
void GetDmapBlobShape(const int ds_times, int hei_img, int wid_img, vector<int> &shape);
void ReadLocationFromTextFile(const std::string filename, int hei_img, int wid_img, vector<HeadLocation<int> > &gt_loc);
void ReadRoiPointFromTextFile(const std::string filename, Mat& roi_mat);
void ReadDmapFromTextFile(const std::string filename, int ds_times, int hei_img, int wid_img, Mat& cv_dmap);

template <typename Dtype>
void TransformDmap(const Mat &cv_img, Dtype* data_label);

void MakeDirectoryPath(string &path);
bool IsPointInRect(const HeadLocation<int> &pnt, const Rect &roi);
void CropDataByRoi(const Mat &cv_img, const Mat &dmap_original, const vector<HeadLocation<int> > &gt_loc, 
                   const Rect &roi_rect, bool has_dmap, int ds_times,
                   Mat &roi_img, Mat &roi_dmap, vector<HeadLocation<int> > &gt_loc_roi);
void GetRoiFromPnt(const Mat &roi_mat, int hei_img, int wid_img, int ds_times, Rect &roi_rect);
void RandomGenerateRoi(int hei_img, int wid_img, int min_sz_crop, int max_sz_crop, int ds_times, Rect &crop_rect);
void GetSplitRoiByIdx(int times_split, int idx_roi_split, int hei_img, int wid_img, int ds_times, Rect &split_rect);
void ModifyRect(int hei_img, int wid_img, int ds_times, Rect &rect);
int GetDownTimesNum(const int ds_times);
void GetScaleGt(const vector<HeadLocation<int> > &gt_loc, int scale, vector<HeadLocation<int> > &gt_loc_roi);
void GetDmapGt(vector<HeadLocation<int> > &gt_loc, int ds_times);


}  // namespace caffe

#endif  // CAFFE_UTIL_CROWD_HPP_
