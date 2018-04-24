#include "caffe/util/dmap.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <algorithm>

namespace caffe {
const float DIST_DEFAULT = 25.0;

//2D Gaussain kernel, similar to function fspecial in Matlab
//Input size must be odd number
void fspecial(const int size, const float sigma, Mat &gaussian_kernel) {
    int r_size = (size-1) / 2;
    Mat kernel(size, size, CV_32F);
    float simga_2 = float(2) * sigma * sigma;
    for(int i = -r_size; i <= r_size; ++i) {
        int h = i + r_size;
        for (int j = (-r_size); j <= r_size; ++j) {
            int w = j + r_size;
            float v = exp(-(static_cast<float>(i*i) + static_cast<float>(j*j)) / simga_2);
            kernel.ptr<float>(h)[w] = v;
        }
    }
    Scalar sum_value = sum(kernel);
    kernel.convertTo(gaussian_kernel, CV_32F, (1/sum_value[0]));
    return;
}


// Generate gaussian kernel based density map
void GetDensityMap(int hei_img, int wid_img, int ds_times,
                   const vector<HeadLocation<int> >& head_location,
                   const float sigma,
                   int neighbor_num, 
                   const float ksize_param, 
                   const float sigma_param,
                   Mat& cv_dmap) {
    vector<int> dmap_shape;
    GetDmapBlobShape(ds_times, hei_img, wid_img, dmap_shape);   
    cv_dmap.create(dmap_shape[2], dmap_shape[3], CV_32F);


    const int num_head = head_location.size();
    vector<float> mean_dist(num_head, DIST_DEFAULT);

    // LOG(INFO)<<"head_num:"<<num_head;
    float max_dist = 0.0;
    for(int i = 0; i < num_head; i++) {
        // float valuew = min(head_location[i].w, hei_img - head_location[i].y);
        // valuew = min(valuew, wid_img - head_location[i].x);
        // float valueh = min(valueh, hei_img - head_location[i].y);
        // valueh = min(valueh, wid_img - head_location[i].x);
        // mean_dist[i] = max(min(valuew, valueh), 1);

        // mean_dist[i] = min(hei_img - head_location[i].y, wid_img - head_location[i].x);
        mean_dist[i] = max(min(head_location[i].w, head_location[i].h), 1);
        if(mean_dist[i] > max_dist){
            max_dist = mean_dist[i];
        }
    }

    // max_dist = 25;
    // max_dist = std::max((float)28, max_dist);
    int b_size = 2 * ceil(max_dist) +1;
    int r_size = (b_size - 1) / 2;
    Mat extend_dmap(cv_dmap.rows +  b_size - 1, cv_dmap.cols + b_size - 1, CV_32F, Scalar(0));
    
    for(int i=0; i<num_head;i++){
        int k_size = ceil(ksize_param * mean_dist[i]);
        // int k_size = 15;

        if(k_size % 2 == 0){
            k_size = k_size+1;
        }
        
        // k_size = std::max(7, k_size);
        // k_size = 40;
        // if(k_size < 7){
        //     k_size = 7;
        // }
        
        float sigma = sigma_param * k_size;
        // float sigma = 1.5;
        Mat gaussian_kernel;
        fspecial(k_size, sigma, gaussian_kernel);      
        // LOG(INFO) << "k_size scale fspecial:" << k_size;
 
        int loc_x = min(max(head_location[i].x, 0), cv_dmap.cols);
        int loc_y = min(max(head_location[i].y, 0), cv_dmap.rows);

        CHECK_GE(loc_x, 0) << "invalid head location: x!";
        CHECK_GE(loc_y, 0) << "invalid head location: y!";
        CHECK_LE(loc_x, cv_dmap.cols) << "invalid head location: x!";
        CHECK_LE(loc_y, cv_dmap.rows) << "invalid head location: y!";           
        
        Rect location_box;
        location_box.x = max(0, loc_x + r_size - (k_size - 1) / 2);
        location_box.y = max(0, loc_y + r_size - (k_size - 1) / 2);
        location_box.width = min(k_size, extend_dmap.cols - location_box.x );
        location_box.height = min(k_size, extend_dmap.rows - location_box.y );

        // location_box.width = min(location_box.width, location_box.height);
        // location_box.height = location_box.width;
        // LOG(INFO)<<"x:"<<location_box.x;
        // LOG(INFO)<<"y:"<<location_box.y;
        // LOG(INFO)<<"cols:"<<extend_dmap.cols;
        // LOG(INFO)<<"rows:"<<extend_dmap.rows;
        // LOG(INFO)<<"w:"<<location_box.width;
        // LOG(INFO)<<"h:"<<location_box.height;
        // LOG(INFO)<<"k_size:"<<k_size;
        
        Mat kernel_location(extend_dmap, location_box);
        kernel_location = kernel_location + gaussian_kernel;
        // LOG(INFO)<<"k_size2:"<<k_size;
    }
    // LOG(INFO)<<"map.rows:"<<extend_dmap.rows;
    // LOG(INFO)<<"map.cols:"<<extend_dmap.cols;

    Rect dmap_loc;
    dmap_loc.x = r_size;
    dmap_loc.y = r_size;
    dmap_loc.width = cv_dmap.cols;
    dmap_loc.height = cv_dmap.rows;

    Mat dstroi = extend_dmap(dmap_loc);
    // LOG(INFO)<<"r_size:"<<r_size;
    // LOG(INFO)<<"dstroi_sum: "<<cv::sum(dstroi);
    // LOG(INFO)<<"dstroi.rows:"<<dstroi.rows;
    // LOG(INFO)<<"dstroi.cols:"<<dstroi.cols;
    dstroi.convertTo(cv_dmap, CV_32F, 1, 0);
}


void ConvertDensityMapForShow(const Mat &density, cv::Size sz, Mat &dmap_sh) {
    const double MAX_PIXEL = 255.0;
    const double MIN_PIXEL = 0.0;
    double gap = MAX_PIXEL - MIN_PIXEL;
    double min_den, max_den;
    minMaxLoc(density, &min_den, &max_den);
    //LOG(INFO) << "MIN: " << min_den << ", MAX: " <<max_den;
    double alpha=1.0, beta=0.0;
    if (max_den-min_den>0.00001) {
        alpha = gap/(max_den-min_den);
        beta  = MIN_PIXEL-alpha*min_den;
    } else {
        alpha = 1.0;
        beta  = 0.0;
    }
    density.convertTo(dmap_sh, CV_8UC1, alpha, beta);
    resize(dmap_sh, dmap_sh, sz);
    // imshow("density_map", dmap_sh);
    // waitKey(0);
}
    

}  // namespace caffe
