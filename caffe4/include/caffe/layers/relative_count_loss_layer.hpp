#ifndef CAFFE_RELATIVE_COUNT_LOSS_LAYER_HPP_  
#define CAFFE_RELATIVE_COUNT_LOSS_LAYER_HPP_  
  
  
#include <vector>  
  
#include "caffe/blob.hpp"  
#include "caffe/layer.hpp"  
#include "caffe/proto/caffe.pb.h"  
#include "caffe/layers/loss_layer.hpp"  
  
  
namespace caffe {  
  
template <typename Dtype>  
class RelativeCountLossLayer : public LossLayer<Dtype> {  
 public:  
  explicit RelativeCountLossLayer(const LayerParameter& param)  
      : LossLayer<Dtype>(param){}  
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top);  
  
  
  virtual inline const char* type() const { return "RelativeCountLoss"; }  
  
  
  virtual inline bool AllowForceBackward(const int bottom_index) const {  
    return true;  
  }  
  
  
 protected:  
  /// @copydoc AbsoluteLossLayer  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top);  
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top);  
  
  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,  
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);  
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,  
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);  
  
  
  Blob<Dtype> dis0_;
  Blob<Dtype> dis1_;
  Blob<Dtype> tmp_;  
};  
  
  
}  // namespace caffe  
  
  
#endif    // __CAFFE_RELATIVE_COUNT_LOSS_LAYER_HPP_  
