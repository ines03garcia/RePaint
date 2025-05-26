# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

"""
Helpers for distributed training.
"""

import io

import blobfile as bf
import torch as th


def dev(device):
    """
    Get the device to use for torch.distributed.
    """
    if device is None:
        if th.cuda.is_available():
            return th.device(f"cuda")
        return th.device("cpu")
    return th.device(device)


def load_state_dict(path, backend=None, **kwargs):
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    print("Loading model...")
    
     # Deserialize with torch
    checkpoint = th.load(io.BytesIO(data), **kwargs)
    
    # Handle case where checkpoint has a nested 'state_dict' key
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Strip "module." prefix if it exists
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[len("module."):] if k.startswith("module.") else k
        new_state_dict[new_key] = v

    print("State dict cleaned and ready.")
    return new_state_dict


