#include <algorithm>
#include <vector>

#include "caffe/layers/hebbian_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void HebbianConvLayer<Dtype>::Add_feedback_gpu(const int size, const Dtype gain, const Dtype* bottom,
		Dtype* feedback) {
		caffe_gpu_axpby<Dtype>(size, (Dtype)1.0, bottom, gain, feedback);
	}

	template <typename Dtype>
	void HebbianConvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* weight = this->blobs_[0]->gpu_data();
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->gpu_data();
			Dtype* feedback = bottom[i]->mutable_gpu_diff();
			Dtype* top_data = top[i]->mutable_gpu_data();
			DISPLAYMEM(bottom[i]->cpu_diff(), bottom[i]->cpu_data(), top[i]->cpu_data(), bottom[i]->mutable_gpu_diff())
				for (int n = 0; n < this->num_; ++n) {
				Add_feedback_gpu(this->bottom_dim_, this->feedback_gain_, bottom_data + n * this->bottom_dim_, feedback + n * this->bottom_dim_);
				this->forward_gpu_gemm(feedback + n * this->bottom_dim_, weight, top_data + n * this->top_dim_);
				if (this->bias_term_) {
					const Dtype* bias = this->blobs_[1]->gpu_data();
					this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
				}
			}
			DISPLAYMEM(bottom[i]->cpu_diff(), bottom[i]->cpu_data(), top[i]->cpu_data(), this->blobs_[0]->gpu_data())
			caffe_gpu_set(this->bottom_dim_, Dtype(0), feedback);
		}
	}

	template <typename Dtype>
	void HebbianConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* weight = this->blobs_[0]->gpu_data();
		Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
		for (int i = 0; i < top.size(); ++i) {
			REGDIFF(bottom[i]) // ensure that CPU mem copy of bottom diff is refreshed
			REGDIFF(top[i]) // ensure that CPU mem copy of top diff is refreshed
			REGDATA(this->blobs_[0])
			DISPLAYMEM(bottom[i]->cpu_diff(), this->blobs_[0]->cpu_data(), this->blobs_[0]->cpu_diff(), top[i]->cpu_diff())

			const Dtype* top_diff = top[i]->gpu_diff(); // top_diff is set by layer above, which must be a pooling layer (in which it is "bottom_diff")

			// Bias gradient, if necessary.
			if (this->bias_term_ && this->param_propagate_down_[1]) {
				Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
				for (int n = 0; n < this->num_; ++n) {
					this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
				}
			}
			if (this->param_propagate_down_[0] || propagate_down[i]) {
				Blob<Dtype>* feedback = new Blob<Dtype>(bottom[i]->shape());
				Dtype* feedback_data = feedback->mutable_gpu_data();
				const Dtype* bottom_data = bottom[i]->gpu_data();
				const Dtype* top_data = top[i]->gpu_data();
				Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
				caffe_gpu_set(feedback->count(), Dtype(0), feedback_data);
				for (int n = 0; n < this->num_; ++n) {
					// feedback to bottom, temporarily store in bottom_diff.
					this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
						bottom_diff + n * this->bottom_dim_);
					// Oja's rule: weight_diff = top_diff * (bottom_data - bottom_diff) where bottom_diff = (top_diff X weight)  <-- see above statement
					// (it's negative because the last step in Update subtracts the weight_diff from the current weight.)
					caffe_gpu_axpy(this->bottom_dim_, Dtype(-1.), bottom_diff + n * this->bottom_dim_, feedback_data + n * this->bottom_dim_);
					caffe_gpu_axpy(this->bottom_dim_, Dtype(1.), bottom_data + n * this->bottom_dim_, feedback_data + n * this->bottom_dim_);
					// why use top diff here instead of top data? because only pooling layer winner should "learn"
					this->weight_gpu_gemm(feedback_data + n * this->bottom_dim_, top_diff + n * this->top_dim_, weight_diff);
				}
				// delete feedback blob
				delete feedback;
			}
			REGDIFF(bottom[i]) // ensure that CPU mem copy of bottom diff is refreshed
			REGDIFF(top[i]) // ensure that CPU mem copy of top diff is refreshed
			REGDIFF(this->blobs_[0])
			DISPLAYMEM(bottom[i]->cpu_diff(), this->blobs_[0]->cpu_data(), this->blobs_[0]->cpu_diff(), top[i]->cpu_diff())
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(HebbianConvLayer);
}