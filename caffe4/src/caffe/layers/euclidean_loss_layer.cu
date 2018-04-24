#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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
  //top[0]->mutable_cpu_data()[0] = loss/Dtype(count);
  //end of lyq change

  //LOG(INFO) << "=====loss: " << loss << ", count: " << count << ", mean_loss: " << loss/Dtype(count);
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe
