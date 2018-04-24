//  Author:    Hongling Luo

#ifdef USE_OPENCV
#include <fstream>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "caffe/layers/crowd_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/util/dmap.hpp"
#include "caffe/util/util_crowd.hpp"
#include "caffe/util/data_augment.hpp"


namespace caffe {
    
using namespace cv;

template <typename Dtype>
CrowdDataLayer<Dtype>::~CrowdDataLayer<Dtype>() {
    this->StopInternalThread();
}

//Shuffle crowd image and ground truth location.
template <typename Dtype>
void CrowdDataLayer<Dtype>::ShuffleData() {
    caffe::rng_t* prefetch_rng =
    static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

//Return an unsigned integrate number
template <typename Dtype>
unsigned int CrowdDataLayer<Dtype>::PrefetchRand() {
    CHECK(prefetch_rng_);
    caffe::rng_t* prefetch_rng =
    static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    return (*prefetch_rng)();
}

template <typename Dtype>
void CrowdDataLayer<Dtype>::GetParamTransform() {
    const bool has_mean_values = this->transform_param_.mean_value_size() > 0;
    int channel = is_color_ ? 3 : 1;
    if (has_mean_values) {
        const int mean_channel = this->transform_param_.mean_value_size();
        CHECK(mean_channel == 1 || mean_channel == channel) <<
        "Specify either 1 mean_value or as many as channels: " << channel;
        if (channel > 1 && mean_channel == 1) {
            // Replicate the mean_value for simplicity
            for (int c = 0; c < channel; ++c) {
                mean_values_.push_back(this->transform_param_.mean_value(0));
            }
        } else {
            for (int c = 0; c < channel; ++c) {
                mean_values_.push_back(this->transform_param_.mean_value(c));
            }
        }
    } else {
        // If not specify mean value, set to zero
        for (int c = 0; c < channel; ++c) {
            mean_values_.push_back(Dtype(0));
        }
    }
    transform_scale_ = this->transform_param_.scale();
}

template <typename Dtype>
void CrowdDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
    downsamp_times_ = this->layer_param_.image_data_param().downsamp_times();
    base_sigma_ = this->layer_param_.image_data_param().base_sigma();
    is_color_  = this->layer_param_.image_data_param().is_color();
    dmap_online_ = this->layer_param_.image_data_param().dmap_online();
    string root_folder = this->layer_param_.image_data_param().root_folder();
    MakeDirectoryPath(root_folder);

    times_split_ = this->layer_param_.image_data_param().times_split();
    max_crop_ = this->layer_param_.image_data_param().max_crop();
    min_crop_ = this->layer_param_.image_data_param().min_crop();
    neighbor_num_ = this->layer_param_.image_data_param().neighbor_num();
    ksize_param_ = this->layer_param_.image_data_param().ksize_param();
    sigma_param_ = this->layer_param_.image_data_param().sigma_param();
    scale_param_ = this->layer_param_.image_data_param().scale_param();
    is_augment_ = this->layer_param_.image_data_param().is_augment();
    LOG(INFO) << "times_split_ set up: " << times_split_;

    // 1.  Configure mean values and transform scale
    GetParamTransform();

    // 2. Read the image and location file with filenames.
    const string& source = this->layer_param_.image_data_param().source();
    std::ifstream infile(source.c_str());    
    CHECK(infile.good()) << "Failed load source file: "<<source;

    string img_filename;
    string dmap_filename;
    int head_num;
    Dtype flag;
    HeadLocation<int> head_loc;
    vector<HeadLocation<int> > loc_info_tmp;
    string loc_filename;
    string roi_filename;
    while (infile >> img_filename >> head_num) {
        LabelInfo<string> label_info;
        loc_info_tmp.clear();
       
        // LOG(INFO) << "load file :"<<img_filename <<", head:"<<head_num;
        for(int i = 0 ; i< head_num;i++) {
            infile >> flag >> head_loc.x >> head_loc.y >> head_loc.w >> head_loc.h;
            // LOG(INFO) << "load lbl :"<<head_loc.x << ","<<head_loc.y<< ","<<head_loc.w<< ","<<head_loc.h;
            loc_info_tmp.push_back(head_loc);
        }

        if(!dmap_online_) {
            string rep_name = img_filename;
            string fnd1 = "RoiImg";
            string dmap_name;            
            if(downsamp_times_ == 2) {
                dmap_name = rep_name.replace(rep_name.find(fnd1),fnd1.length(),"Dmap/Dmap4");
            }else if(downsamp_times_ == 3) {
                dmap_name = rep_name.replace(rep_name.find(fnd1),fnd1.length(),"Dmap/Dmap8");
            }else {
                // do nothing
            }
            int npos = dmap_name.rfind('.');
            dmap_filename = root_folder + dmap_name.substr(0,npos) + ".txt";
        }else {
            // do nothing
        }

        label_info.img_name = root_folder + img_filename;
        label_info.loc_info = loc_info_tmp;
        label_info.dmap_name = dmap_filename;
        lines_.push_back(label_info); 
    }
    infile.close();

    // 3. Randomly shuffle data.
    if (this->layer_param_.image_data_param().shuffle()) {
        const unsigned int prefetch_rng_seed = caffe_rng_rand();
        prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
        ShuffleData();
    }
    LOG(INFO) << "A total of " << lines_.size() << " crowd images.";

    // 4. Read an crowd image to initialize the top blob.
    // 4.1. crowd image.
    lines_id_ = 0;
    Mat cv_img = imread(lines_[lines_id_].img_name, is_color_ ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
    int orgCols = cv_img.cols;
    int orgRows = cv_img.rows;
    int restCols = orgCols % 64;
    int restRows = orgRows % 64;
    int extCols = (64 - restCols)%64;
    int extRows = (64 - restRows)%64;
    Mat ext_img;
   // resize(cv_img,ext_img,Size(cv_img.cols,cv_img.cols),0,0,INTER_LINEAR);
    // copyMakeBorder(cv_img,ext_img,0,extRows,0,extCols,BORDER_CONSTANT);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].img_name;
    vector<int> data_shape;
    GetImageBlobShape(is_color_, 128, 128, data_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->data_.Reshape(data_shape);
    }
    top[0]->Reshape(data_shape);

    // 4.2. density map.
    vector<int> dmap_shape;
    GetDmapBlobShape(downsamp_times_, 128, 128, dmap_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->label_.Reshape(dmap_shape);
    }
    top[1]->Reshape(dmap_shape);  
    idx_get_split_ = 2;
}

// This function is called on prefetch thread.
template <typename Dtype>
void CrowdDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {   
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    double dmap_time = 0;
    CPUTimer timer;

    ImageDataParameter image_data_param = this->layer_param_.image_data_param();
    const bool mirror = this->transform_param_.mirror();
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
   
    // 1.1. Reshape crowd image (data) and density map (label) of batch.
    const int lines_size = lines_.size();
    int batch_size  = 1;
    for (int item_id = 0; item_id < batch_size; ++item_id) {
        CHECK_GT(lines_size, lines_id_);
        /////////read inputs/////////
        timer.Start();
        Mat cv_img = imread(lines_[lines_id_].img_name, is_color_ ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
        CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].img_name;
        const int img_height = cv_img.rows;
        const int img_width  = cv_img.cols;
        int restCols = img_width % 64;
        int restRows = img_height % 64;
        int extCols = (64 - restCols)%64;
        int extRows = (64 - restRows)%64;
        Mat ext_img;
       // resize(cv_img,ext_img,Size(cv_img.cols,cv_img.cols),0,0,INTER_LINEAR);
        // copyMakeBorder(cv_img,ext_img,0,extRows,0,extCols,BORDER_CONSTANT);
        bool has_dmap = false;
        Mat dmap_original;
        if((lines_[lines_id_].dmap_name.size()>0) && (!dmap_online_)) {
            ReadDmapFromTextFile(lines_[lines_id_].dmap_name, downsamp_times_, img_height, img_width, dmap_original); 
            has_dmap = true;
        }
        vector<HeadLocation<int> > gt_loc = lines_[lines_id_].loc_info;
        read_time += timer.MicroSeconds();
        
       // LOG(INFO) << "img name " <<lines_[lines_id_].img_name;
       // LOG(INFO) << "cv_img.cols " <<cv_img.cols;
       // LOG(INFO) << "cv_img.rows " <<cv_img.rows;
       // LOG(INFO) << "ext_img.cols " <<ext_img.cols;
       // LOG(INFO) << "ext_img.rows " <<ext_img.rows;
        /////////split and crop///////// 
        Mat crop_img;
        Mat crop_dmap;
        vector<HeadLocation<int> > gt_loc_crop;
        if (times_split_ < 0) {
            Rect crop_rect;
            RandomGenerateRoi(cv_img.rows, cv_img.cols, min_crop_, max_crop_, downsamp_times_, crop_rect); 
            CropDataByRoi(cv_img, dmap_original, gt_loc, crop_rect, has_dmap, downsamp_times_, crop_img, crop_dmap, gt_loc_crop);
        } else if (times_split_ > 0) {  //split the image
            Rect crop_rect;
            GetSplitRoiByIdx(times_split_, idx_get_split_, cv_img.rows, cv_img.cols, downsamp_times_, crop_rect); 
            CropDataByRoi(cv_img, dmap_original, gt_loc, crop_rect, has_dmap, downsamp_times_, crop_img, crop_dmap, gt_loc_crop);
        } else {  //no crop
            crop_img = cv_img;      //crop_img = ext_img (fpn)
            if (has_dmap) {
                crop_dmap = dmap_original;
            } else {
                gt_loc_crop = gt_loc;
            }
        }

        /* show
        // Mat img_show = crop_img.clone();
        // LOG(INFO) << "times_split_: " << times_split_;
        // LOG(INFO) << "head size :" << gt_loc_crop.size();
        // for(int i = 0; i< gt_loc_crop.size(); i++) {
        //     circle(img_show, cv::Point(gt_loc_crop[i].x + gt_loc_crop[i].w/2 , gt_loc_crop[i].y + gt_loc_crop[i].h/2), gt_loc_crop[i].h, Scalar(0,0,255)); 
        // }
        // imshow("imgsplit", img_show);
        */

        /////////data augment  1> do mirror///////// 
        bool do_mirror = false;
        if (mirror) {
            if (PrefetchRand()%2==0)
                do_mirror = true;
        }
        if (do_mirror) {
            cv::flip(crop_img, crop_img, 1);
            if (has_dmap) {
                cv::flip(crop_dmap, crop_dmap, 1);
            } else {
                for (int c=0; c<gt_loc_crop.size(); c++) {
                    gt_loc_crop[c].x = crop_img.cols-1 - gt_loc_crop[c].x;   //horizontal mirror
                }
            }  
        }

        
        //scale augment
        Mat img_out, dmap_out;
        img_out = crop_img;
        vector<HeadLocation<int> > gt_loc_scale;
        int scale_param = scale_param_;
        // int scale_param = rand() % scale_param_ + 1;
        if(scale_param > 1 && dmap_online_) {
            resize(img_out, img_out, cv::Size(img_out.cols /scale_param, img_out.rows /scale_param ));
            GetScaleGt(gt_loc_crop, scale_param, gt_loc_scale);
        }else {
            gt_loc_scale = gt_loc_crop;
        }

        /////////prepare output. if pnt, generate dmap///////// 
        if(!dmap_online_) {
            dmap_out = crop_dmap.clone();
        } else {
            timer.Start();
            GetDmapGt(gt_loc_scale, downsamp_times_);
            // LOG(INFO) << "img size " << img_out.rows << " , "<< img_out.cols << " , ";
            // LOG(INFO) << "img name: " << lines_[lines_id_].img_name;
            GetDensityMap(img_out.rows, img_out.cols, downsamp_times_, gt_loc_scale, base_sigma_,
                          neighbor_num_, ksize_param_, sigma_param_, dmap_out);
            dmap_time += timer.MicroSeconds();

            // LOG(INFO) << "img size " << img_out.rows << " , "<< img_out.cols << " , ";
            // LOG(INFO) << "dmp size after:" << dmap_out.rows << " , "<< dmap_out.cols << " , ";
            // LOG(INFO) << "img scale " << scale_param;
            // LOG(INFO) << "img name " <<lines_[lines_id_].img_name;
            // imshow("img", img_out);
            // Mat dmap_sh;
            // ConvertDensityMapForShow(dmap_out, dmap_out.size(), dmap_sh);
            // imshow("density", dmap_sh);
            // waitKey(0);
        }

    
        //imshow("img bef", img_out);
        ///////// data augment //////////////
        if(is_augment_ == true) {
            GetNoiseImg(img_out, img_out);
        }  
        // imshow("noise", img_out);
        // waitKey(0);



        ///////////////////// output  //////////////////////////////////////////////////
        if(downsamp_times_ == 2) {
            // resize(img_out, img_out, cv::Size(img_out.cols  /4 * 4, img_out.rows /4 *4 ));
            // resize(dmap_out, dmap_out, cv::Size(img_out.cols /4, img_out.rows / 4));
        }else if(downsamp_times_ == 3) {
            resize(img_out, img_out, cv::Size(dmap_out.cols * 8 /16 * 16, dmap_out.rows * 8/16 *16 ));
            resize(dmap_out, dmap_out, cv::Size(img_out.cols /8, img_out.rows / 8));
        }else {
            // do nothing
        }  


        ///////// Transform /////////////////////////////////////////
        timer.Start();
        vector<int> top_shape = this->data_transformer_->InferBlobShape(img_out);
        // LOG(INFO) << "img shape " << top_shape[0] << " , "<< top_shape[1]<< " , "<< top_shape[2]<< " , "<< top_shape[3]<< " , ";
        this->transformed_data_.Reshape(top_shape);
        CHECK_EQ(batch_size, 1);
        top_shape[0] = batch_size;
        batch->data_.Reshape(top_shape);
        CHECK_EQ(item_id, 0);
        int offset = batch->data_.offset(item_id);
        this->transformed_data_.set_cpu_data(batch->data_.mutable_cpu_data() + offset);
        this->data_transformer_->Transform(img_out, &(this->transformed_data_));

        vector<int> shape_dmap;
        shape_dmap.push_back(batch_size);
        shape_dmap.push_back(1);
        shape_dmap.push_back(dmap_out.rows);
        shape_dmap.push_back(dmap_out.cols);
        // LOG(INFO) << "dmp shape " << shape_dmap[0] << " , "<< shape_dmap[1]<< " , "<< shape_dmap[2]<< " , "<< shape_dmap[3]<< " , ";
        batch->label_.Reshape(shape_dmap);
        CHECK_EQ(item_id, 0);
        offset = batch->label_.offset(item_id);
        TransformDmap(dmap_out, batch->label_.mutable_cpu_data() + offset);

        trans_time += timer.MicroSeconds();

        lines_id_++;
        if (lines_id_ >= lines_size) {  // We have reached the end. Restart from the first.
            DLOG(INFO) << "Restarting data prefetching from start.";
            lines_id_ = 0;
            if (this->layer_param_.image_data_param().shuffle()) {
                ShuffleData();
            }
            if (times_split_>0) {
                idx_get_split_ += 1;
                if (idx_get_split_ >= times_split_*times_split_) {
                    idx_get_split_ = 0;
                }
            }
        }

    }
          
    batch_timer.Stop();
    // LOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    // LOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    // LOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
    // LOG(INFO) << "Generate Density map: " << dmap_time / 1000 << "ms.";
}

INSTANTIATE_CLASS(CrowdDataLayer);
REGISTER_LAYER_CLASS(CrowdData);
    
}  // namespace caffe
#endif  // USE_OPENCV

//------luohongling add end------
