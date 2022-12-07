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

#include "dali/operators/input/video_input.h"

namespace dali {

template<>
bool VideoInput<CPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                       const Workspace &ws) {
  if (!valid_) {
    int batch_size = 1;
    TensorList<CPUBackend> input;
    frames_decoders_.resize(batch_size);
    auto &thread_pool = ws.GetThreadPool();
    this->ForwardCurrentData(input, thread_pool);
    cout << "Input shape: " << input.shape() << endl;
    for (int i = 0; i < batch_size; ++i) {
      auto sample = input[i];
      auto data = reinterpret_cast<const char *>(sample.data<uint8_t>());
      size_t size = sample.shape().num_elements();
      frames_decoders_[i] = std::make_unique<FramesDecoder>(data, size, false);
    }
    output_desc_.shape = ReadOutputShape(frames_per_sequence_, 1);
    output_desc_.type = DALI_UINT8;
    valid_ = true;
  }
  output_desc.resize(1);
  output_desc[0] = output_desc_;
  cout << "SETUP: Output shape " << output_desc[0].shape << " TYPE " << output_desc[0].type
       << endl;
  return true;
}


template<>
void VideoInput<CPUBackend>::RunImpl(Workspace &ws) {
  auto &output = ws.Output<CPUBackend>(0);
  int batch_size = 1;
  for (int s = 0; s < batch_size; ++s) {
    cout << "Curr frame " << curr_frame_ << " fps " << frames_per_sequence_ << endl;
    DecodeFrames(output[s], s, curr_frame_, frames_per_sequence_);
    curr_frame_ += frames_per_sequence_;
  }
  cout << "Output shape: " << output.shape() << endl;
  if (!CanDecode(0)) {
    auto h = horcruxes_.front();
    horcruxes_.pop();
    return_horcruxes_.emplace_back(h);
  }

}


template<>
bool VideoInput<GPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                       const Workspace &ws) {
  DALI_FAIL("Shouldn't be called");
  return true;
}


template<>
void VideoInput<GPUBackend>::RunImpl(Workspace &ws) {
  DALI_FAIL("Shouldn't be called");
}


DALI_SCHEMA(experimental__inputs__Video)
                .DocStr(
                        R"code(...)code")
                .NumInput(0)
                .NumOutput(1)
                .AddArg("frames_per_sequence", R"code(...)code", DALI_INT32);

DALI_REGISTER_OPERATOR(experimental__inputs__Video, VideoInput<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(experimental__inputs__Video, VideoInput<GPUBackend>, GPU);

}