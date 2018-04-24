
#include "caffe/util/util_crowd.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

namespace caffe {
// Compute initial shape for crowd image
void GetImageBlobShape(const bool is_color, const int height, const int width, vector<int> &shape) {
    const int channels = is_color ? 3 : 1;
    CHECK_GT(height, 0);
    CHECK_GT(width, 0);

    // Build BlobShape.
    shape.resize(4);
    shape[0] = 1;
    shape[1] = channels;
    shape[2] = height;
    shape[3] = width;
    return;
}


void GetSizeDmap(int hei_img, int wid_img, int ds_times, int &hei_dmap, int &wid_dmap) {
    CHECK_GT(ds_times, 0);
    hei_dmap = hei_img;
    wid_dmap = wid_img;
    for (int i = 0; i < ds_times; ++i) {
        hei_dmap = hei_dmap / 2 + (hei_dmap % 2);
        wid_dmap = wid_dmap / 2 + (wid_dmap % 2);
    }
}



// Compute initial shape for density map( by pooling times)
void GetDmapBlobShape(const int ds_times, int hei_img, int wid_img, vector<int> &shape) {
    int chn_dmap = 1;
    int hei_dmap, wid_dmap;
    GetSizeDmap(hei_img, wid_img, ds_times, hei_dmap, wid_dmap);
    // Check dimensions.
    CHECK_GT(hei_dmap, 0);
    CHECK_GT(wid_dmap, 0);
    
    // Build BlobShape.
    shape.resize(4);
    shape[0] = 1;
    shape[1] = chn_dmap;
    shape[2] = hei_dmap;
    shape[3] = wid_dmap;
    return;
}

//Read ground truth head location of crowd images
void ReadLocationFromTextFile(const std::string filename, int hei_img, int wid_img, vector<HeadLocation<int> > &gt_loc) {
    std::ifstream infile(filename.c_str());
    CHECK(infile.good()) << "Failed to open file " << filename;
    int num_crowd;
    CHECK(infile >> num_crowd);
    CHECK_GE(num_crowd, 0) << "Number of crowd must be positive!";
    gt_loc.clear();
    gt_loc.resize(num_crowd);
    
    for (int i = 0; i < num_crowd; i++) {   
        HeadLocation<int> location;
        CHECK(infile >> location.x >> location.y); 
        CHECK(0 <= location.x && 0 <= location.y && wid_img>location.x && hei_img>location.y);   
        gt_loc[i] = location;
    }
    infile.close();
}

//Read ROI of crowd images
void ReadRoiPointFromTextFile(const std::string filename, Mat& roi_mat) {
    DLOG(INFO) << "Opening file " << filename;
    std::ifstream infile(filename.c_str());
    CHECK(infile.good()) << "Failed to open file " << filename;
    
    int num_point;    
    CHECK(infile >> num_point);
    CHECK_GE(num_point, 0) << "Number of crowd must be positive!";
    Mat roi_M(num_point, 2, CV_32FC1,Scalar(0));

    for (int i = 0; i < num_point; i++) { 
        for(int j = 0; j < 2; j++){
            infile >> roi_M.at<int>(i,j);
        }
    }
    roi_mat = roi_M.clone();
    infile.close();   
}

void ReadDmapFromTextFile(const std::string filename, int ds_times, int hei_img, int wid_img, Mat& cv_dmap) {
    DLOG(INFO) << "Opening file " << filename;   
    std::ifstream infile(filename.c_str()); 
    CHECK(infile.good()) << "Failed to open file " << filename;
    int dmap_h ,dmap_w;
    infile >> dmap_w >> dmap_h;
    cv_dmap.create(dmap_h, dmap_w, CV_32F);
    
    for(int i=0; i<cv_dmap.rows; i++){
        for(int j=0; j<cv_dmap.cols; j++){
            infile >> cv_dmap.ptr<float>(i)[j];
        }
    }

    infile.close();    
}

//Convert cvMat to data blob
template <typename Dtype>
void TransformDmap(const Mat &dmap, Dtype* data_label) {
    int hei_dmap = dmap.rows;
    int wid_dmap = dmap.cols;
    CHECK_EQ(dmap.channels(), 1);
    CHECK_EQ(dmap.step, dmap.cols*dmap.elemSize());
    memcpy(data_label, dmap.data, dmap.step*dmap.rows);    
}

template void TransformDmap(const Mat &dmap, double* data_label);
template void TransformDmap(const Mat &dmap, float* data_label);


void MakeDirectoryPath(string &path) {
    string str_end = path.substr(path.length()-1);
    if (str_end != "/") {
        path = path.append("/");
    }
}

bool IsPointInRect(const HeadLocation<int> &pnt, const Rect &roi) {
    if (pnt.x>roi.x && pnt.y>roi.y && pnt.x<(roi.x+roi.width-1) && pnt.y<(roi.y+roi.height-1))
        return true;
    else
        return false;
}

int GetSuitableLen(int len_in, int len_img, int factor) {
    CHECK_GE(len_img, len_in);
    int len_out = len_in;
    if (len_in % factor == 0 ) {
        len_out = len_in;
    } else {
        int len_floor = len_in/factor * factor;
        int len_ceil  = (len_in/factor+1) * factor;
        if ((len_floor-len_in)<(len_ceil-len_in) || len_ceil>=len_img) {
            len_out = len_floor;
        } else {
            len_out = len_ceil;
        }
    }
    return len_out;
}

int GetSuitableX(int x_in, int wid_old, int wid_new, int wid_img) {
    CHECK_GE(wid_img, wid_new);
    int x_out = x_in;
    float mid = float(x_in) + float(wid_old)*0.5;
    float lef_new = mid - float(wid_new)*0.5;
    float rig_new = mid + float(wid_new)*0.5;
    int rig = min(wid_img-1, cvRound(rig_new));
    x_out = max(0, rig-wid_new+1);
    return x_out;
}

//ds_times is used to specify the rect_out
void ModifyRect(int hei_img, int wid_img, int ds_times, Rect &rect) {
    int factor = pow(2, ds_times+1);
    int wid_s = GetSuitableLen(rect.width, wid_img, factor);
    int hei_s = GetSuitableLen(rect.height, hei_img, factor);
    int x_s = GetSuitableX(rect.x, rect.width, wid_s, wid_img);
    int y_s = GetSuitableX(rect.y, rect.height, hei_s, hei_img);
    rect = Rect(x_s, y_s, wid_s, hei_s);
}

void GetRoiFromPnt(const Mat &roi_mat, int hei_img, int wid_img, int ds_times, Rect &roi_rect) {
    Mat sortPoint(roi_mat.rows,roi_mat.cols,CV_32F,Scalar(0));
    sort(roi_mat,sortPoint,CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
    int minWRoi = max(sortPoint.ptr<int>(0)[0], 1);
    int maxWRoi = min(sortPoint.ptr<int>(roi_mat.rows-1)[0], wid_img);
    int minHRoi = max(sortPoint.ptr<int>(0)[1], 1);
    int maxHRoi = min(sortPoint.ptr<int>(roi_mat.rows-1)[1], hei_img);
    
    int roi_height = maxHRoi - minHRoi;
    int roi_width  = maxWRoi - minWRoi;
    roi_rect = Rect(minWRoi, minHRoi, roi_width, roi_height);
    ModifyRect(hei_img, wid_img, ds_times, roi_rect); 
}

void CropDataByRoi(const Mat &cv_img, const Mat &dmap_original, const vector<HeadLocation<int> > &gt_loc, 
                   const Rect &roi_rect, bool has_dmap, int ds_times,
                   Mat &roi_img, Mat &roi_dmap, vector<HeadLocation<int> > &gt_loc_roi) {
    roi_img = cv_img(roi_rect);

    if (has_dmap) {
        Rect dmap_rect;
        int factor = pow(2, ds_times);
        dmap_rect.x = int(float(roi_rect.x)/float(factor));
        dmap_rect.y = int(float(roi_rect.y)/float(factor));
        GetSizeDmap(roi_img.rows, roi_img.cols, ds_times, dmap_rect.height, dmap_rect.width);
        roi_dmap = dmap_original(dmap_rect); 
        
        // Mat den_sh;
        // ConvertDensityMapForShow(roi_dmap, roi_dmap.size(), den_sh);
        // imshow("roi_dmap", roi_dmap);
        // waitKey(0);
    } else {
        //process points
        for (int i=0; i<gt_loc.size(); i++) {
            if (IsPointInRect(gt_loc[i], roi_rect)) {
                HeadLocation<int> loc;
                loc.x = gt_loc[i].x-roi_rect.x;
                loc.y = gt_loc[i].y-roi_rect.y;
                loc.w = gt_loc[i].w;
                loc.h = gt_loc[i].h;
                gt_loc_roi.push_back(loc);
            }
        }
    }
}

void RandomGenerateRoi(int hei_img, int wid_img, int min_sz_crop, int max_sz_crop, int ds_times, Rect &crop_rect) {
    CHECK_GT(wid_img, min_sz_crop);
    int min_wid = min_sz_crop;
    int max_wid = min(max_sz_crop, wid_img-1);
    int wid = cvRound(double(rand())/double(RAND_MAX)*(max_wid-min_wid)) + min_wid;
    int hei = cvRound(float(wid)*float(hei_img)/float(wid_img));
    int x = cvRound(double(rand())/double(RAND_MAX)*(wid_img-wid));
    int y = cvRound(double(rand())/double(RAND_MAX)*(hei_img-hei));
    crop_rect = Rect(x, y, wid, hei);
    ModifyRect(hei_img, wid_img, ds_times, crop_rect);
}

void GetSplitRoiByIdx(int times_split, int idx_roi_split, int hei_img, int wid_img, int ds_times, Rect &split_rect) {
    CHECK_GT(times_split, 0);
    int wid = cvRound(float(wid_img)/times_split);
    int hei = cvRound(float(hei_img)/times_split);
    int idx_y = idx_roi_split / times_split;
    int idx_x = idx_roi_split % times_split;
    int x = idx_x * wid;
    int y = idx_y * hei;
    int rig = x+wid-1;
    if (rig>=wid_img)
        wid = wid_img-x;
    int bot = y+hei-1;
    if (bot>=hei_img)
        hei = hei_img-y;
    split_rect = Rect(x, y, wid, hei);
    ModifyRect(hei_img, wid_img, ds_times, split_rect);
}

int GetDownTimesNum(const int ds_times) {
    CHECK_GT(ds_times, 0);
    int exact_ds_num = 1;
    for(int i = 0; i<ds_times; i++) {
        exact_ds_num *= 2;
    }
    return exact_ds_num;
}

void GetScaleGt(const vector<HeadLocation<int> > &gt_loc, int scale, vector<HeadLocation<int> > &gt_loc_roi) {
    gt_loc_roi.resize(gt_loc.size());
    for(int i = 0; i< gt_loc.size(); i++) {
        gt_loc_roi[i].x = gt_loc[i].x / scale;
        gt_loc_roi[i].y = gt_loc[i].y / scale;
        gt_loc_roi[i].w = gt_loc[i].w / scale;
        gt_loc_roi[i].h = gt_loc[i].h / scale;
    }

}

void GetDmapGt(vector<HeadLocation<int> > &gt_loc, int ds_times) {
    CHECK_GT(ds_times, 0);
    int factor = pow(2, ds_times);
    for(int i = 0; i< gt_loc.size(); i++) {
        gt_loc[i].x = gt_loc[i].x / factor;
        gt_loc[i].y = gt_loc[i].y / factor;
        gt_loc[i].w = gt_loc[i].w / factor;
        gt_loc[i].h = gt_loc[i].h / factor;
    }

}
    

}  // namespace caffe
