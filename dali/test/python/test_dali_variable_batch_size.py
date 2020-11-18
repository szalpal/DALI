# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from nvidia.dali.pipeline import Pipeline
from segmentation_test_utils import make_batch_select_masks
from PIL import Image
from nose.tools import nottest
import nvidia.dali.ops as ops
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np
import test_utils
import inspect
import os
import math

"""
How to test variable (iter-to-iter) batch size for a given op?
-------------------------------------------------------------------------------
The idea is to create a Pipeline that assumes i2i variability, run 2 iterations
and compare them with ad-hoc created Pipelines for given (constant) batch sizes.
This can be easily done using `check_batch` function below.

On top of that, there are some utility functions and routines to help with some
common cases:
1. If the operator is typically processing image-like data (i.e. 3-dim, uint8,
   0-255, with shape like [640, 480, 3]) and you want to test default arguments
   only, please add a record to the `ops_image_default_args` list
2. If the operator is typically processing image-like data (i.e. 3-dim, uint8,
   0-255, with shape like [640, 480, 3]) and you want to specify any number of
   its arguments, please add a record to the `ops_image_custom_args` list
3. If the operator is typically processing audio-like data (i.e. 1-dim, float,
   0.-1.) please add a record to the `float_array_ops` list
4. If your operator case doesn't fit any of the above, please create a nosetest
   function, in which you can define a function, that returns not yet built
   pipeline, and pass it to the `check_batch` function.
"""


def generate_data(max_batch_size, n_iter, sample_shape, lo=0., hi=1., dtype=np.float32):
    """
    Generates an epoch of data, that will be used for variable batch size verification.

    :param max_batch_size: Actual sizes of every batch in the epoch will be less or equal to max_batch_size
    :param n_iter: Number of iterations in the epoch
    :param sample_shape: If sample_shape is callable, shape of every sample will be determined by
                         calling sample_shape. In this case, every call to sample_shape has to
                         return a tuple of integers. If sample_shape is a tuple, this will be a
                         shape of every sample.
    :param lo:
    :param hi:
    :param dtype: 'int' for uint8 or 'float' for float32
    :return: An epoch of data
    """
    batch_sizes = np.array([max_batch_size // 2, max_batch_size // 4, max_batch_size])

    if isinstance(sample_shape, tuple):
        size = sample_shape
    elif inspect.isgeneratorfunction(sample_shape):
        size = sample_shape()
    else:
        raise RuntimeError(
            "`sample_shape` shall be either a tuple or a callable. Provide `(val,)` tuple for 1D shape")

    if np.issubdtype(dtype, np.integer):
        return [np.random.randint(lo, hi, size=(1, bs) + size, dtype=dtype) for bs in
                batch_sizes]
    elif np.issubdtype(dtype, np.float):
        ret = (np.random.random_sample(size=(1, bs) + size) for bs in batch_sizes)
        ret = map(lambda batch: (hi - lo) * batch + lo, ret)
        ret = map(lambda batch: batch.astype(dtype), ret)
        return list(ret)
    else:
        raise RuntimeError("Invalid type argument")


def single_op_pipeline(max_batch_size, input_data, device, /, *, input_layout=None,
                       operator_fn=None, **opfn_args):
    pipe = Pipeline(batch_size=max_batch_size, num_threads=1, device_id=0)
    with pipe:
        input = fn.external_source(source=input_data, num_outputs=1, cycle=False, device=device,
                                   layout=input_layout)
        input = input[0] if operator_fn is None else operator_fn(input, device=device, **opfn_args)
        pipe.set_outputs(input)
    return pipe


def check_pipeline(input_epoch, pipeline_fn, *, devices=['cpu', 'gpu'], eps=1e-7,
                   **pipeline_fn_args):
    """
    Verifies, if given pipeline supports iter-to-iter variable batch size

    :param input_epoch: List of numpy arrays, where every item is a single batch
    :param pipeline_fn: Function, that returns created (but not built) pipeline.
                        Its signature should be (at least):
                        pipeline_fn(max_batch_size, input_data, device, /, ...)
    :param devices: Devices to run the check on
    :param eps: Epsilon for mean error
    :param pipeline_fn_args: Additional args to pipeline_fn
    """
    for device in devices:
        n_iter = len(input_epoch)
        max_bs = max(batch.shape[1] for batch in input_epoch)
        var_pipe = pipeline_fn(max_bs, input_epoch, device, **pipeline_fn_args)
        var_pipe.build()

        for iter_idx in range(n_iter):
            iter_input = input_epoch[iter_idx]
            batch_size = iter_input.shape[1]

            const_pipe = pipeline_fn(batch_size, [iter_input], device, **pipeline_fn_args)
            const_pipe.build()

            test_utils.compare_built_pipelines(var_pipe, const_pipe, batch_size=batch_size,
                                               n_iterations=1, eps=eps)


def image_data_helper(operator_fn, **opfn_args):
    check_pipeline(generate_data(31, 13, (160, 80, 3), lo=0, hi=255, dtype=np.uint8),
                   pipeline_fn=single_op_pipeline, input_layout="HWC", operator_fn=operator_fn,
                   **opfn_args)


def float_array_helper(operator_fn, **opfn_args):
    check_pipeline(generate_data(31, 13, (313,)), pipeline_fn=single_op_pipeline,
                   operator_fn=operator_fn, **opfn_args)


def test_external_source():
    check_pipeline(generate_data(31, 13, (3, 2)), single_op_pipeline)


ops_image_default_args = [
    fn.brightness_contrast,
    fn.hue,
    fn.brightness,
    fn.contrast,
    fn.hsv,
    fn.color_twist,
    fn.saturation,
    fn.shapes,
    fn.reductions.mean,
    fn.reductions.mean_square,
    fn.reductions.rms,
    fn.reductions.min,
    fn.reductions.max,
    fn.reductions.sum,
    fn.crop_mirror_normalize,
    fn.water,
    fn.sphere,
    fn.old_color_twist,
    fn.dump_image,
    fn.copy,
]


def test_ops_image_default_args():
    for op in ops_image_default_args:
        yield image_data_helper, op


ops_image_custom_args = [
    {'operator_fn': fn.rotate, 'angle': 25},
    {'operator_fn': fn.resize, 'resize_x': 50, 'resize_y': 50},
    {'operator_fn': fn.gaussian_blur, 'window_size': 5},
    {'operator_fn': fn.flip, 'horizontal': True},
    {'operator_fn': fn.reinterpret, 'rel_shape': [0.5, 1, -1]},
    {'operator_fn': fn.crop, 'crop': (5, 5)},
    {'operator_fn': fn.erase, 'anchor': [0.3], 'axis_names': "H", 'normalized_anchor': True,
     'shape': [0.1], 'normalized_shape': True},
    {'operator_fn': fn.transpose, 'perm': [2, 0, 1]},
    {'operator_fn': fn.normalize, 'batch': True},
    {'operator_fn': fn.warp_affine, 'matrix': (.1, .9, 10, .8, -.2, -20)},
    {'operator_fn': fn.pad, 'fill_value': -1, 'axes': (0,), 'shape': (10,)},
    {'operator_fn': fn.cast, 'dtype': types.INT32},
    {'operator_fn': fn.color_space_conversion, 'image_type': types.BGR, 'output_type': types.RGB},
    {'operator_fn': fn.fast_resize_crop_mirror, 'crop': [5, 5], 'resize_shorter': 10,
     'devices': ['cpu']},
    {'operator_fn': fn.resize_crop_mirror, 'crop': [5, 5], 'resize_shorter': 10,
     'devices': ['cpu']},
]


def test_ops_image_custom_args():
    for op in ops_image_custom_args:
        print("\nTesting ", op)
        image_data_helper(**op)


float_array_ops = [
    {'operator_fn': fn.preemphasis_filter},
    {'operator_fn': fn.spectrogram, 'nfft': 60, 'window_length': 50, 'window_step': 25},
    {'operator_fn': fn.power_spectrum, 'devices': ['cpu']},
    {'operator_fn': fn.to_decibels},
]


def test_float_array_ops():
    for op in float_array_ops:
        print("\nTesting ", op)
        float_array_helper(**op)


def test_reshape():
    check_pipeline(generate_data(31, 13, (160, 80, 3), lo=0, hi=255, dtype=np.uint8),
                   pipeline_fn=single_op_pipeline, operator_fn=fn.reshape,
                   shape=(160 / 2, 80 * 2, 3))


def test_bb_flip():
    check_pipeline(generate_data(31, 13, (200, 4)), single_op_pipeline, operator_fn=fn.bb_flip)


def test_bbox_paste():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, num_outputs=1, cycle=False, device=device)
        paste_posx = fn.uniform(range=(0, 1))
        paste_posy = fn.uniform(range=(0, 1))
        paste_ratio = fn.uniform(range=(1, 2))
        processed = fn.bbox_paste(data, paste_x=paste_posx, paste_y=paste_posy, ratio=paste_ratio)
        pipe.set_outputs(processed)
        return pipe

    check_pipeline(generate_data(31, 13, (200, 4)), pipe, eps=.5, devices=['cpu'])


def test_coord_flip():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, num_outputs=1, cycle=False, device=device)
        processed = fn.coord_flip(data)
        pipe.set_outputs(processed)
        return pipe

    check_pipeline(generate_data(31, 13, (200, 2)), pipe)


def test_lookup_table():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, num_outputs=1, cycle=False, device=device)
        processed = fn.lookup_table(data, keys=[1, 3], values=[10, 50])
        pipe.set_outputs(processed)
        return pipe

    check_pipeline(generate_data(31, 13, (313,), lo=0, hi=5, dtype=np.uint8), pipe)


def test_reduce():
    reduce_fns = [
        fn.reductions.std_dev,
        fn.reductions.variance
    ]

    def pipe(max_batch_size, input_data, device, /, reduce_fn):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, num_outputs=1, cycle=False, device=device)
        mean = fn.reductions.mean(data)
        reduced = reduce_fn(data, mean)
        pipe.set_outputs(reduced)
        return pipe

    for rf in reduce_fns:
        check_pipeline(generate_data(31, 13, (313, 131, 2)), pipe, reduce_fn=rf)


def test_arithm_ops():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, num_outputs=1, cycle=False, device=device)
        data = data[0] * 2
        data = data + 3
        data = data - 4
        data = data / 5
        data = data // 6
        pipe.set_outputs(data)
        return pipe

    check_pipeline(generate_data(31, 13, (313, 131)), pipe)


def test_sequence_rearrange():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, num_outputs=1, cycle=False, device=device,
                                  layout="FHWC")
        processed = fn.sequence_rearrange(data, new_order=[0, 4, 1, 3, 2])
        pipe.set_outputs(processed)
        return pipe

    check_pipeline(generate_data(31, 13, (5, 10, 20, 3), lo=0, hi=255, dtype=np.uint8), pipe)


def test_element_extract():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, num_outputs=1, cycle=False, device=device,
                                  layout="FHWC")
        processed, _ = fn.element_extract(data, element_map=[0, 3])
        pipe.set_outputs(processed)
        return pipe

    check_pipeline(generate_data(31, 13, (5, 10, 20, 3), lo=0, hi=255, dtype=np.uint8), pipe)


def test_nonsilent_region():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, num_outputs=1, cycle=False, device=device)
        processed, _ = fn.nonsilent_region(data)
        pipe.set_outputs(processed)
        return pipe

    check_pipeline(generate_data(31, 13, (313,), lo=0, hi=255, dtype=np.uint8), pipe,
                   devices=['cpu'])


def test_mel_filter_bank():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        with pipe:
            data = fn.external_source(source=input_data, num_outputs=1, cycle=False, device=device)
            spectrum = fn.spectrogram(data, nfft=60, window_length=50, window_step=25)
            processed = fn.mel_filter_bank(spectrum)
            pipe.set_outputs(processed)
        return pipe

    check_pipeline(generate_data(31, 13, (313,)), pipe)


def test_mfcc():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, num_outputs=1, cycle=False, device=device)
        spectrum = fn.spectrogram(data, nfft=60, window_length=50, window_step=25)
        mel = fn.mel_filter_bank(spectrum)
        dec = fn.to_decibels(mel)
        processed = fn.mfcc(dec)
        pipe.set_outputs(processed)

        return pipe

    check_pipeline(generate_data(31, 13, (313,)), pipe, devices=['cpu'])


def test_audio_decoder():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        encoded = fn.external_source(source=input_data, num_outputs=1, cycle=False, device=device)
        decoded, _ = fn.audio_decoder(encoded, downmix=True, sample_rate=12345)
        pipe.set_outputs(decoded)
        return pipe

    def read_file(filename):
        with open(filename, "rb") as f:
            data = f.read()
        return np.array(list(data)).astype(np.uint8)

    audio_dir = os.path.join(test_utils.get_dali_extra_path(), 'db', 'audio')

    # File reader won't work, so I need to load audio files into external_source manually

    wav_fnames = []  # files, that have .wav extension
    for dir_name, subdir_list, file_list in os.walk(audio_dir):
        wav_list = filter(lambda fname: fname.endswith('.wav'), file_list)
        wav_list = map(lambda fname: os.path.join(dir_name, fname), wav_list)
        wav_fnames.extend(wav_list)

    # Split audio files into batches
    while True:  # ensure, that wavs are split into 2 not equal batches
        split_idx = np.random.randint(1, len(wav_fnames) - 1)
        if not math.isclose(len(wav_fnames) - (split_idx + 1), len(wav_fnames) / 2.):
            break
    _in_ep = [
        list(map(lambda fname: read_file(fname), wav_fnames[:split_idx])),
        list(map(lambda fname: read_file(fname), wav_fnames[split_idx:])),
    ]

    # Since we pack buffers into ndarray, we need to pad samples with 0. Additionally,
    # external_source expects specific shape (due to num_outputs=1), so we add (1,) dimension
    input_epoch = []
    for inp in _in_ep:
        max_len = max(sample.shape[0] for sample in inp)
        inp = map(lambda sample: np.pad(sample, (0, max_len - sample.shape[0])), inp)
        inp = map(lambda sample: np.reshape(sample, (1,) + sample.shape), inp)
        input_epoch.append(np.stack(list(inp)))

    check_pipeline(input_epoch, pipe, devices=['cpu'])


def test_python_function():
    def resize(data):
        data += 13
        return data

    def pipe(max_batch_size, input_data, device, /):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0, exec_async=False,
                        exec_pipelined=False)
        with pipe:
            data = fn.external_source(source=input_data, num_outputs=1, cycle=False, device=device)
            processed = fn.python_function(data[0], function=resize, num_outputs=1)
            pipe.set_outputs(processed)
        return pipe

    check_pipeline(generate_data(31, 13, (313, 131, 2)), pipe, devices=['cpu'])
