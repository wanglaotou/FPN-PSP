#include <vector>  
#include "caffe/layers/relative_count_loss_layer.hpp"  
#include "caffe/util/math_functions.hpp"  
  
namespace caffe {  
  
template <typename Dtype>  
void RelativeCountLossLayer<Dtype>::Reshape(  
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {  
  LossLayer<Dtype>::Reshape(bottom, top);   
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))  
      << "Inputs must have the same dimension.";  
  dis0_.ReshapeLike(*bottom[0]);
  dis1_.ReshapeLike(*bottom[0]);
  tmp_.ReshapeLike(*bottom[0]);
}  

///// Implemention 2 ///////////////////////
template <typename Dtype>  
void RelativeCountLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,  
    const vector<Blob<Dtype>*>& top) { 
  LOG(INFO)<<"start Relative Count Loss  #########################";

  int count = bottom[0]->count(); 
  const Dtype* bottom0_data = bottom[0]->cpu_data();
  const Dtype* bottom1_data = bottom[1]->cpu_data();

  caffe_copy(count, bottom0_data, dis0_.mutable_cpu_data());
  caffe_copy(count, bottom1_data, dis1_.mutable_cpu_data());

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

  // 3> do div : (bottom[0]+1) /(bottom[1]+1)
  caffe_div(  
      count,  
      dis0_.cpu_data(), 
      dis1_.cpu_data(),  
      tmp_.mutable_cpu_data());
  // 4> do sub:
  caffe_add_scalar(  
      count,  
      Dtype(-1),  
      tmp_.mutable_cpu_data());
  // 5> do dot
  Dtype dot = caffe_cpu_dot(count, tmp_.cpu_data(), tmp_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;

  LOG(INFO)<<"est+1:"<LOG(INFO)<<"est+1:"<<dis0_.cpu_data();
  LOG(INFO)<<"gt+1:"<<dis1_.cpu_data();
  LOG(INFO)<<"loss2:"<<loss;
}  

  
template <typename Dtype>  
void RelativeCountLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,  
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {  
  for (int i = 0; i < 2; ++i) {  
    if (propagate_down[i]) {  
       //label bottom propagate_dowm
      const Dtype sign = (i == 0) ? 1 : -1;  
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();  
      caffe_div(  
          bottom[i]->count(),  
          tmp_.cpu_data(), 
          dis1_.cpu_data(),  
          tmp_.mutable_cpu_data());

      caffe_cpu_axpby(  
          bottom[i]->count(),                       // count  
          alpha,                             // alpha  
          tmp_.cpu_data(),                        // a  
          Dtype(0),                           // beta  
          bottom[i]->mutable_cpu_diff());                 // b  
    }     //bottom[i]->mutable_cpu_diff()) = alpha*dis_.cpu_data()  
  }  
}  
  
#ifdef CPU_ONLY  
STUB_GPU(RelativeCountLossLayer);  
#endif  
  
INSTANTIATE_CLASS(RelativeCountLossLayer);  
REGISTER_LAYER_CLASS(RelativeCountLoss);  
  
}  // namespace caffe  