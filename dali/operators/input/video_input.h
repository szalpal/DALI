// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_VIDEO_INPUT_H
#define DALI_VIDEO_INPUT_H

#include <type_traits>
#include "dali/operators/reader/loader/video/frames_decoder.h"
#include "dali/operators/reader/loader/video/frames_decoder_gpu.h"
#include "dali/pipeline/input/input_operator.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

#include "dali/operators/decoder/video/video_decoder_base.h"

namespace dali {

template<typename Backend>
using frames_decoder_t = std::conditional_t<std::is_same<Backend, CPUBackend>::value, FramesDecoder, FramesDecoderGpu>;

template<typename Backend, typename FramesDecoder = frames_decoder_t<Backend>>
class VideoInput : public VideoDecoderBase<Backend, FramesDecoder>, public InputOperator<Backend> {

 public:
  explicit VideoInput(const OpSpec &spec) :
          InputOperator<Backend>(spec),
          frames_per_sequence_(spec.GetArgument<int>("frames_per_sequence")),
          device_id_(spec.GetArgument<int>("device_id")) {
    cout<<"CTOR\n";
  }


  bool CanInferOutputs() const override {
    return true;
  }

  void AssignHorcrux(int64_t horcrux) {
    horcruxes_.push(horcrux);
  }

  std::vector<int64_t> GetHorcruxesBack() {
    auto horcruxes = return_horcruxes_;
    return_horcruxes_.clear();
    return horcruxes;
  }


  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;

  void RunImpl(Workspace &ws) override;


  int NextBatchSize() override {
    return 1;
  }


  void Advance() override {
  }


  template<typename SrcBackend>
  void SetDataSource(const TensorList<SrcBackend> &batch, AccessOrder order = {}) {
    DeviceGuard g(device_id_);
    this->CopyUserData(batch, order, false, false);
  }


 private:
  const int frames_per_sequence_;
  const int device_id_;

  std::queue<int64_t> horcruxes_;
  std::vector<int64_t> return_horcruxes_;

  int curr_frame_;
  bool valid_=false;
  OutputDesc output_desc_;
};

}

#endif //DALI_VIDEO_INPUT_H
