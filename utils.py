# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import numbers
import numpy as np

try:
    from collections.abc import Sequence, Mapping
except:
    from collections import Sequence, Mapping


def default_collate_fn(batch):
    """
    Default batch collating function for :code:`paddle.io.DataLoader`,
    get input data as a list of sample datas, each element in list
    if the data of a sample, and sample data should composed of list,
    dictionary, string, number, numpy array, this
    function will parse input data recursively and stack number,
    numpy array and paddle.Tensor datas as batch datas.

    Args:
        batch(list of sample data): batch should be a list of sample data.

    Returns:
        Batched data: batched each number, numpy array and paddle.Tensor
                      in input data.
    """

    sample = batch[0]

    # ---------------------------------------------------------
    # NumPy array
    # ---------------------------------------------------------
    if isinstance(sample, np.ndarray):
        try:
            batch = np.stack(batch, axis=0)
            return batch

        except ValueError:
            print("\n")
            print("==============================================")
            print("       COLLATE SHAPE ERROR")
            print("==============================================")

            for i, x in enumerate(batch):
                print(
                    "item {} | shape={} | dtype={} | type={}".format(
                        i,
                        getattr(x, "shape", None),
                        getattr(x, "dtype", None),
                        type(x)
                    )
                )

            print("==============================================")
            print("\n")

            raise

    # ---------------------------------------------------------
    # Number
    # ---------------------------------------------------------
    elif isinstance(sample, numbers.Number):
        batch = np.array(batch)
        return batch

    # ---------------------------------------------------------
    # String / bytes
    # ---------------------------------------------------------
    elif isinstance(sample, (str, bytes)):
        return batch

    # ---------------------------------------------------------
    # Dictionary / Mapping
    # ---------------------------------------------------------
    elif isinstance(sample, Mapping):
        result = {}

        for key in sample:
            try:
                result[key] = default_collate_fn(
                    [d[key] for d in batch]
                )

            except ValueError:
                print("\n")
                print("==============================================")
                print("          COLLATE KEY ERROR")
                print("==============================================")
                print("KEY:", key)
                print("==============================================")
                print("\n")

                raise

        return result

    # ---------------------------------------------------------
    # Sequence / list / tuple
    # ---------------------------------------------------------
    elif isinstance(sample, Sequence):

        sample_fields_num = len(sample)

        if not all(
            len(sample) == sample_fields_num
            for sample in iter(batch)
        ):
            raise RuntimeError(
                "fileds number not same among samples in a batch"
            )

        return [
            default_collate_fn(fields)
            for fields in zip(*batch)
        ]

    # ---------------------------------------------------------
    # Unsupported type
    # ---------------------------------------------------------
    raise TypeError(
        "batch data con only contains: tensor, numpy.ndarray, "
        "dict, list, number, but got {}".format(type(sample))
    )
