#include <vector>

#include "caffe/layers/euclidean_hardmining_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
bool EuclideanHardminingLossLayer<Dtype>::SortByLoss(const Dtype &loss_1  ,const Dtype &loss_2) {
  return loss_1 > loss_2; 
}

template <typename Dtype>
void EuclideanHardminingLossLayer<Dtype>::PushLoss(std::vector<Dtype> &loss_vec, const Dtype* diff_data, int count) {
  for(int i = 0; i < count ;i++) {
    if(diff_data[i] < 0) {
      loss_vec.push_back(-diff_data[i]);
    }else {
      loss_vec.push_back(diff_data[i]);
    }
    
  }
}

template <typename Dtype>
void EuclideanHardminingLossLayer<Dtype>::PrintLoss(std::vector<Dtype> &loss_vec) {
  for(size_t i = 0;i < loss_vec.size(); i++) {
    LOG(INFO)<< "loss_" << i << ": "<< loss_vec[i];
  }
}


template <typename Dtype>
void EuclideanHardminingLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanHardminingLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;

  //////zt add :for hard sample///////sort diff_cpu_data()
  hard_sample_loss_vec_.clear();
  hard_sample_loss_vec_.resize(count);
  const Dtype* diff_data = diff_.cpu_data();
  Dtype* diff_mutable_data = diff_.mutable_cpu_data();
  PushLoss(hard_sample_loss_vec_, diff_data, count);

  LOG(INFO)<<"count :" << count;  
  // LOG(INFO)<<"lossSort Before:";  
  // PrintLoss(hard_sample_loss_vec_);
  std::sort(hard_sample_loss_vec_.begin(), hard_sample_loss_vec_.end(), SortByLoss);
  // LOG(INFO)<<"forward sortLoss:"; 
  // PrintLoss(hard_sample_loss_vec_);

  hard_sample_ratio_ = this->layer_param_.euclidean_hardmining_loss_param().hardmining_ratio();
  int hard_sample_num = count * hard_sample_ratio_;
  Dtype hard_sample_least_loss = hard_sample_loss_vec_[hard_sample_num];
  for(int i = 0; i < count; i++) {
    Dtype diff_data_tmp = diff_mutable_data[i];
    if (diff_data_tmp < 0) {
      diff_data_tmp = -diff_data_tmp;
    }
    if(diff_data_tmp < hard_sample_least_loss) {
      diff_mutable_data[i] = 0.0;
    }
  }

  //test
  hard_sample_loss_vec_.clear();
  const Dtype* diff_data2 = diff_.cpu_data();
  PushLoss(hard_sample_loss_vec_, diff_data2, count);
  LOG(INFO)<<"forward final Loss#################################:";  
  PrintLoss(hard_sample_loss_vec_);
}



template <typename Dtype>
void EuclideanHardminingLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanHardminingLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanHardminingLossLayer);
REGISTER_LAYER_CLASS(EuclideanHardminingLoss);

}  // namespace caffe
