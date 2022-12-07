# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
import numpy as np
import nvidia.dali.types as types
import glob
from test_utils import get_dali_extra_path
from nose2.tools import params

filenames = glob.glob(f'{get_dali_extra_path()}/db/video/[cv]fr/*.mp4')
# filter out HEVC because some GPUs do not support it
filenames = filter(lambda filename: 'hevc' not in filename, filenames)
# mpeg4 is not yet supported in the CPU operator itself
filenames = filter(lambda filename: 'mpeg4' not in filename, filenames)

files = [np.fromfile(
    filename, dtype=np.uint8) for filename in filenames]


@pipeline_def
def video_decoder_pipeline(input_name, device='cpu'):
    data = fn.external_source(name=input_name, dtype=types.UINT8, ndim=1)
    vid = fn.experimental.decoders.video(data)
    # vid = fn.resize(vid, size=(15, 20),interp_type=types.INTERP_NN)
    return vid


@pipeline_def
def video_input_pipeline(input_name, frames_per_sequence, device='cpu'):
    vid = fn.experimental.inputs.video(name=input_name, device=device,
                                       frames_per_sequence=frames_per_sequence)
    # vid = fn.resize(vid, size=(15, 20),interp_type=types.INTERP_NN)
    return vid


@params("cpu")
def test_video_input(device):
    input_name = "VIDEO_INPUT"
    frames_per_sequence = 5
    decoder_pipe = video_decoder_pipeline(input_name=input_name, batch_size=1, num_threads=1,
                                          device_id=0, exec_pipelined=False, exec_async=False)
    input_pipe = video_input_pipeline(input_name=input_name, batch_size=1, num_threads=1,
                                      device_id=0, frames_per_sequence=frames_per_sequence,
                                      exec_pipelined=False, exec_async=False)
    decoder_pipe.build()
    decoder_pipe.feed_input(input_name, [files[0]])
    decoder_out = decoder_pipe.run()
    input_pipe.build()
    # input_pipe.serialize(filename="/home/mszolucha/clion_deploy/DALI/dali/test/python/video_input.serialized")
    input_pipe.feed_input(input_name, np.array([[files[0]]]))
    # input_out = input_pipe.run()
    for seq_idx in range(0, 40 - frames_per_sequence, frames_per_sequence):
        # ref_seq = decoder_out[0].as_array()[0]
        ref_seq = decoder_out[0].as_array()[0][seq_idx:seq_idx + frames_per_sequence]
        input_out = input_pipe.run()
        test_seq = input_out[0].as_array()[0]
        # for i in range(seq_idx, seq_idx+frames_per_sequence):
        #     cv2.imwrite(f"frame{i}.jpg", test_seq[i%frames_per_sequence])
        #     cv2.imwrite(f"ref{i}.jpg", ref_seq[seq_idx])
        assert np.all(ref_seq == test_seq)
