#include <vector>

#include "caffe/layers/feedback_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FeedbackLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      caffe_copy(bottom[i]->count(), bottom[i]->cpu_data(),
                bottom[i]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SilenceLayer);
#endif

INSTANTIATE_CLASS(FeedbackLayer);
REGISTER_LAYER_CLASS(Feedback);

}  // namespace caffe
