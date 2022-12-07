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

#ifndef DALI_OPERATORS_DECODER_VIDEO_VIDEO_DECODER_BASE_H_
#define DALI_OPERATORS_DECODER_VIDEO_VIDEO_DECODER_BASE_H_

#include <vector>
#include <memory>

#include "dali/pipeline/operator/common.h"

namespace dali {

template <typename Backend, typename FramesDecoder>
class DLL_PUBLIC VideoDecoderBase {
 public:
  using InBackend = CPUBackend;
  using OutBackend = std::conditional_t<std::is_same<Backend, CPUBackend>::value,
                                        CPUBackend,
                                        GPUBackend>;

 protected:
  void ValidateInput(const Workspace &ws) {
    const auto &input = ws.Input<InBackend>(0);
    DALI_ENFORCE(input.type() == DALI_UINT8,
                 "Type of the input buffer must be uint8.");
    DALI_ENFORCE(input.sample_dim() == 1,
                 "Input buffer must be 1-dimensional.");
    for (int64_t i = 0; i < input.num_samples(); ++i) {
      DALI_ENFORCE(input[i].shape().num_elements() > 0,
                   make_string("Incorrect sample at position: ", i, ". ",
                               "Video decoder does not support empty input samples."));
    }
  }

  TensorListShape<4> ReadOutputShape() {
    // to do returns whole file shape, not subsequence
    TensorListShape<4> shape(frames_decoders_.size());
    for (size_t s = 0; s < frames_decoders_.size(); ++s) {
      TensorShape<4> sample_shape;
      sample_shape[0] = frames_decoders_[s]->NumFrames();
      sample_shape[1] = frames_decoders_[s]->Height();
      sample_shape[2] = frames_decoders_[s]->Width();
      sample_shape[3] = frames_decoders_[s]->Channels();
      shape.set_tensor_shape(s, sample_shape);
    }
    return shape;
  }

  TensorListShape<4> ReadOutputShape(int nframes, int batch_size) {
    // to do returns whole file shape, not subsequence
    TensorListShape<4> shape(batch_size);
    for (int s = 0; s < batch_size; ++s) {
      TensorShape<4> sample_shape;
      sample_shape[0] = nframes;
      sample_shape[1] = frames_decoders_[s]->Height();
      sample_shape[2] = frames_decoders_[s]->Width();
      sample_shape[3] = frames_decoders_[s]->Channels();
      shape.set_tensor_shape(s, sample_shape);
    }
    return shape;
  }


  /**
   * @brief Decode sample with index `idx` to `output` tensor.
   */
  void DecodeSample(SampleView<OutBackend> output, int64_t idx) {
    int64_t num_frames = output.shape()[0];
    DecodeFrames(output, idx, 0, num_frames);
  }


  /**
   * to do
   * @param output
   * @param sample_idx
   * @param start_idx
   * @param num_frames
   */
  void DecodeFrames(SampleView<OutBackend> output, int64_t sample_idx, int64_t start_idx,
                    int64_t num_frames) {
    auto &frames_decoder = *frames_decoders_[sample_idx];
    int64_t frame_size = frames_decoder.FrameSize();
    uint8_t *output_data = output.template mutable_data<uint8_t>();
    for (int64_t f = 0; f < num_frames; ++f) {
      frames_decoder.ReadNextFrame(output_data + f * frame_size);
    }
  }


  /**
   * to do
   * @param sample_idx
   * @return
   */
  bool CanDecode(int64_t sample_idx) {
    if (cnt >= 3) return false;
    cnt++;
    return true;
//    return frames_decoders_[sample_idx]->NextFrameIdx() != -1;
  }


  int cnt = 0;  // to do remove

  std::vector<std::unique_ptr<FramesDecoder>> frames_decoders_;
  std::vector<bool> is_valid_;  // to do refactor
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_VIDEO_VIDEO_DECODER_BASE_H_
