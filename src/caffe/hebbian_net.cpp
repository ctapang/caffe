#include "caffe/layer.hpp"
#include "caffe/hebbian_net.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

	template <typename Dtype>
	HebbianNet<Dtype>::HebbianNet(const NetParameter& param, const HebbianNet* root_net)
		: Net(param, root_net) {}

	template <typename Dtype>
	HebbianNet<Dtype>::HebbianNet(const string& param_file, Phase phase, const HebbianNet* root_net)
		: Net(param_file, phase, root_net) {}

	template <typename Dtype>
	void Net<Dtype>::ClearFeedback(int layerIndex) {
		bottom_vecs()[layerIndex].scale_diff(Dtype(0)); // zero out diff
	}

	INSTANTIATE_CLASS(HebbianNet);

}