"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import tensorrt as trt
import pycuda.driver as cuda
from utils import TimeProfiler
import numpy as np
import os
import time
import torch

from collections import namedtuple, OrderedDict
import glob
import argparse
from dataset import Dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Argument Parser Example')
    parser.add_argument('--infer_dir',
                        type=str,
                        default='./data/COCO2017/val2017',
                        help="Directory for images to perform inference on.")
    parser.add_argument("--session_dir",
                        type=str,
                        default='onnx_engines',
                        help="Directory containing model onnx files.")
    parser.add_argument('--busy',
                        action='store_true',
                        help="Flag to indicate that other processes may be running.")
    args = parser.parse_args()
    return args


time_profile = TimeProfiler()
time_profile_dataset = TimeProfiler()

def warmup(session, blob, n):
    for _ in range(n):
        _ = session(blob)

def speed(session, blob, n, nonempty_process=False):
    times = []
    time_profile_dataset.reset()
    for i in tqdm(range(n), desc="Running Inference", unit="iteration"):
        time_profile.reset()
        with time_profile_dataset:
            img = blob[i]
            if img['images'] is not None:
                img['image'] = img['input'] = img['images'].unsqueeze(0).numpy()
            else:
                img['images'] = img['input'] = img['image'].unsqueeze(0).numpy()
            img['orig_target_sizes'] = img['orig_target_sizes'].numpy() 
        with time_profile:
            _ = session.run(output_names=None, input_feed=img)
        times.append(time_profile.total)

    # end-to-end model only
    times = sorted(times)
    if len(times) > 100 and nonempty_process:
        times = times[:100]

    avg_time = sum(times) / len(times)  # Calculate the average of the remaining times
    return avg_time

def main():
    FLAGS = parse_args()
    dataset = Dataset(FLAGS.session_dir)
    im = torch.ones(1, 3, 640, 640).cuda()
    blob = {
            'image': im,
            'images': im,
            'input': im,
            'im_shape': torch.tensor([640, 640]).to(im.device),
            'scale_factor': torch.tensor([1, 1]).to(im.device),
            'orig_target_sizes': torch.tensor([[640, 640]]).to(im.device),
        }

    engine_files = glob.glob(os.path.join(FLAGS.session_dir, "*.onnx"))
    results = []

    for engine_file in engine_files:
        print(f"Testing engine: {engine_file}")
        # Load the ONNX model
        sess = ort.InferenceSession(args.onnx)
        print(f"Using device: {ort.get_device()}")

        warmup(sess, blob, 400)
        t = []
        for _ in range(1):
            t.append(speed(sess, dataset, 1000, FLAGS.busy))
        avg_latency = 1000 * torch.tensor(t).mean()
        results.append((engine_file, avg_latency))
        print(f"Engine: {engine_file}, Latency: {avg_latency:.2f} ms")

        del model
        torch.cuda.empty_cache()
        time.sleep(1)

    sorted_results = sorted(results, key=lambda x: x[1])
    for engine_file, latency in sorted_results:
        print(f"Session: {engine_file}, Latency: {latency:.2f} ms")

if __name__ == '__main__':
    main()
