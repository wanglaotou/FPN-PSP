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
void EuclideanHardminingLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //LOG(INFO) << this->layer_param_.name() << ", bottom[0]->count(): " << bottom[0]->count() << ", num: " << bottom[0]->num();
  //const Dtype *bot_data_1 = bottom[0]->cpu_data();
  //LOG(INFO) << "BOT_DATA_1: " << bot_data_1[0] << " " << bot_data_1[1] << " " << bot_data_1[2] << " " << bot_data_1[3];
  //const Dtype *bot_data_2 = bottom[1]->cpu_data();
  //LOG(INFO) << "BOT_DATA_2: " << bot_data_2[0] << " " << bot_data_2[1] << " " << bot_data_2[2] << " " << bot_data_2[3];

  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  //lyq change for test, for show mean_loss for crowd_count
  top[0]->mutable_cpu_data()[0] = loss;

  //////zt add :for hard sample///////sort diff_cpu_data()
  hard_sample_loss_vec_.clear();
  hard_sample_loss_vec_.resize(count);
  const Dtype* diff_data = diff_.cpu_data();
  Dtype* diff_mutable_data = diff_.mutable_cpu_data();
  PushLoss(hard_sample_loss_vec_, diff_data, count);

  //LOG(INFO)<<"forward :";  
  //LOG(INFO)<<"lossSort Before:";  
  //PrintLoss(hard_sample_loss_vec_);
  std::sort(hard_sample_loss_vec_.begin(), hard_sample_loss_vec_.end(), SortByLoss);
  //LOG(INFO)<<"forward sortLoss####################################1:"; 
  //PrintLoss(hard_sample_loss_vec_);

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
  //hard_sample_loss_vec_.clear();
  //const Dtype* diff_data2 = diff_.cpu_data();
  //PushLoss(hard_sample_loss_vec_, diff_data2, count);
  //LOG(INFO)<<"forward final Loss####################################3:";  
  //PrintLoss(hard_sample_loss_vec_);
}


template <typename Dtype>
void EuclideanHardminingLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanHardminingLossLayer);

}  // namespace caffe
