//  Create on: 2017/07/05 YCKJ
//  Author:    Hongling Luo

#ifndef CAFFE_CROWD_DATA_LAYER_HPP_
#define CAFFE_CROWD_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/util_crowd.hpp"
using namespace cv;


namespace caffe {
    
/**
 * @brief Provides data and density map to the Net.
 *  top[0]: crowd image
 *  top[1]: density map
 */    

template <typename Dtype>
struct LabelInfo{ // the label_info each img for train needed
    Dtype img_name;
    vector<HeadLocation<int> > loc_info;
    Dtype dmap_name;
    Dtype roi_name;
};

template <typename Dtype>
class CrowdDataLayer : public BasePrefetchingDataLayer<Dtype> {
public:
    explicit CrowdDataLayer(const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype>(param) {}
    virtual ~CrowdDataLayer();
    virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
    
    virtual inline const char* type() const { return "CrowdData"; }
    virtual inline int ExactNumBottomBlobs() const { return 0; }
    virtual inline int ExactNumTopBlobs() const { return 2; }
    
protected:
    shared_ptr<Caffe::RNG> prefetch_rng_;
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    void GetParamTransform();
    
    // data operation
    virtual void ShuffleData();
    virtual void load_batch(Batch<Dtype>* batch);
    virtual unsigned int PrefetchRand();        
    
    int lines_id_;
    int downsamp_times_;
    bool is_color_;
    int times_split_;
    int idx_get_split_;
    int max_crop_;
    int min_crop_;
    int neighbor_num_;
    int scale_param_;
    float ksize_param_;
    float sigma_param_;
    Dtype transform_scale_;
    Dtype base_sigma_;
    bool dmap_online_;
    bool is_augment_;
    vector<LabelInfo<string> > lines_;    
    
    vector<Dtype> mean_values_;
};
    
    
}  // namespace caffe

#endif  // CAFFE_CROWD_DATA_LAYER_HPP_
