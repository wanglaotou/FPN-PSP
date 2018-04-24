#include <vector>  
  
  
#include "caffe/layers/relative_count_loss_layer.hpp"  
#include "caffe/util/math_functions.hpp"  
  
namespace caffe {  
  
template <typename Dtype>  
void RelativeCountLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,  const vector<Blob<Dtype>*>& top) {  
  //LOG(INFO)<<"start Relative Count Loss  #########################";
  
  int count = bottom[0]->count();
  const Dtype* bottom0_data = bottom[0]->cpu_data();
  const Dtype* bottom1_data = bottom[1]->cpu_data();

  caffe_copy(count, bottom0_data, dis0_.mutable_cpu_data());
  caffe_copy(count, bottom1_data, dis1_.mutable_cpu_data());

  //LOG(INFO) << "dis0_1 count: " << bottom[0]->count();
  //LOG(INFO) << "dis0_1 channel: " << bottom[0]->channels();
  //LOG(INFO) << "dis0_1 num: " << bottom[0]->num();
  //LOG(INFO) << "dis0_1 height: " << bottom[0]->height();
  //LOG(INFO) << "dis0_1 w: " << bottom[0]->width();

  //LOG(INFO) << "dis0_1: " << bottom0_data[0];
  //LOG(INFO) << "dis1_1: " << bottom1_data[0];


  const Dtype* dis0_data = dis0_.cpu_data();
  const Dtype* dis1_data = dis1_.cpu_data();
  // 1> get (bottom[0] +1) -> dis0_
  caffe_add_scalar(
      count,  
      Dtype(1),  
      dis0_.mutable_cpu_data()); 


  // 2> get (bottom[1] +1) 
  caffe_add_scalar(
      count,  
      Dtype(1),  
      dis1_.mutable_cpu_data());  

  //LOG(INFO) << "after dis0_1: " << dis0_data[0];
  //LOG(INFO) << "after dis1_1: " << dis1_data[0];

  const Dtype* tmp0_data = tmp_.cpu_data();
  // 3> do div : (bottom[0]+1) /(bottom[1]+1)
  caffe_div(  
      count,  
      dis0_.cpu_data(), 
      dis1_.cpu_data(),  
      tmp_.mutable_cpu_data());

  //LOG(INFO) << "=====relative count tmp0 : " << tmp0_data[0] ;
  // 4> do sub:
  caffe_add_scalar(  
      count,  
      Dtype(-1),  
      tmp_.mutable_cpu_data());

  //LOG(INFO) << "=====relative count tmp1 : " << tmp0_data[0] ;

  // 5> do dot
  Dtype dot = caffe_cpu_dot(count, tmp_.cpu_data(), tmp_.cpu_data());
  //caffe_gpu_dot(count, tmp_.gpu_data(), tmp_.gpu_data(), &dot);
  
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;

  //LOG(INFO) << "=====relative count loss dot: " << dot ;
  //LOG(INFO)<<"=====loss2:"<<loss;
}  



template <typename Dtype>  
void RelativeCountLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {  
  for (int i = 0; i < 2; ++i) {  
    if (propagate_down[i]) {  
      const Dtype sign = (i == 0) ? 1 : -1;  
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();  

      caffe_div(  
          bottom[i]->count(),  
          tmp_.cpu_data(), 
          dis1_.cpu_data(),  
          tmp_.mutable_cpu_data());

      caffe_gpu_axpby(  
          bottom[i]->count(),                       // count  
          alpha,                             // alpha  
          tmp_.cpu_data(),                        // a  
          Dtype(0),                           // beta  
          bottom[i]->mutable_gpu_diff());                 // b  
    }  
  }  
}  
  
INSTANTIATE_LAYER_GPU_FUNCS(RelativeCountLossLayer);  
  
}  // namespace caffe  