#pragma once

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/net.hpp"

namespace caffe {

	/**
	* @brief Connects Layers together into a directed acyclic graph (DAG)
	*        specified by a NetParameter.
	*
	* TODO(dox): more thorough description.
	*/
	template <typename Dtype>
	class HebbianNet : public Net<Dtype> {
	public:
		HebbianNet(const NetParameter& param, const HebbianNet* root_net = NULL);
		HebbianNet(const string& param_file, Phase phase,
			const HebbianNet* root_net = NULL);
		virtual ~HebbianNet() {}

		void ClearFeedback(int layerIndex);
	};

} // namespace caffe