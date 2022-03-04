

from typing import Tuple



import numpy as np
import pandas as pd

import pytest
from scipy import sparse

from campa.data._data import MPPData
from string import ascii_letters
from campa.data import MPPData
def gen_vstr_recarray(m, n, dtype=None):
    size = m * n
    lengths = np.random.randint(3, 5, size)
    letters = np.array(list(ascii_letters))
    gen_word = lambda l: "".join(np.random.choice(letters, l))
    arr = np.array([gen_word(l) for l in lengths]).reshape(m, n)
    return pd.DataFrame(arr, columns=[gen_word(5) for i in range(n)]).to_records(
        index=False, column_dtypes=dtype
    )


def gen_metadata_df(n, obj_ids, possible_cell_cycles = None, ensure_None=True):
    # TODO: Think about allowing index to be passed for n

    lengths = np.random.randint(3, 5, 6)
    letters = np.array(list(ascii_letters))
    gen_word = lambda l: "".join(np.random.choice(letters, l))
    if possible_cell_cycles is None:
        possible_cell_cycles = [gen_word(l) for l in lengths]

    if ensure_None:
        cell_cycle = np.array([None for i in range(n)])
        non_None_size = int(0.85 * n)
        idx = np.random.choice(np.arange(0, n), non_None_size)
        cell_cycle[idx] = np.random.choice(possible_cell_cycles, non_None_size)
    else:
        cell_cycle = np.random.choice(possible_cell_cycles, n)


    letters = np.fromiter(iter(ascii_letters), "U1")
    if n > len(letters):
        letters = letters[: n // 2]  # Make sure categories are repeated

    metadata_dict=dict(
            mapobject_id=obj_ids,
            cell_cycle=cell_cycle,
            cat=pd.Categorical(np.random.choice(letters, n)),
            int64=np.random.randint(-50, 50, n),
            float64=np.random.random(n),
            uint16=np.random.randint(255, size=n, dtype="uint8"),
        )

    return pd.DataFrame(metadata_dict)

def gen_obj(shape, bounding_box,  num_channels, mpp_dtype):
    mean = np.random.randint(0+bounding_box, shape-bounding_box, 2)
    cov = [[int(bounding_box*0.5), 0], [0, bounding_box]]
    num_values = np.random.randint(100, 200)
    x, y = np.random.multivariate_normal(mean, cov, num_values).T
    x, y = x.astype(np.uint8), y.astype(np.uint8)
    x = x[(x > 0) & (x < shape)]
    y = y[(y > 0) & (y < shape)]
    if len(x!=len(y)):
        num_values=min(len(x), len(y))
        x=x[:num_values]
        y=y[:num_values]
    if np.issubdtype(mpp_dtype, np.floating):
        values=np.array([np.random.random(num_values) for ch in range(num_channels)]).T
    else:
        values=np.array([np.random.randint(100, 5000, num_values) for ch in range(num_channels)]).T
    return x, y, values

def gen_objs(shape, bounding_box, num_channels, obj_ids, mpp_dtype):
    x_all, y_all, values_all = np.empty((0), dtype=np.uint8), np.empty((0), dtype=np.uint8), np.empty((0, num_channels), dtype=np.uint8)
    obj_ids_all=np.empty((0), dtype=obj_ids.dtype)
    for i, obj_id in enumerate(obj_ids):
        x, y, values = gen_obj(shape, bounding_box, num_channels, mpp_dtype)
        obj_ids_all=np.append(obj_ids_all, [obj_id]*len(x))
        x_all = np.append(x_all, x)
        y_all = np.append(y_all, y)
        values_all = np.append(values_all, values, axis=0)
    assert(len(x_all)==len(y_all)==len(values_all)==len(obj_ids_all)), print(len(x_all),len(y_all),len(values_all),len(obj_ids_all))
    return x_all, y_all, values_all, obj_ids_all

def gen_mppdata(
    mpp_dtype=np.uint32,
    shape: int = 100,
    bounding_box:int = 10,
    num_channels: int = 5,
    num_obj_ids: int = 5,
    possible_cell_cycles: list = None,
    channels:list = None,
    data_config:str='TestData',
    **kwargs
) -> MPPData:
    """\
    generate several obj ids, for each - generate X, Y, MPP withing bounding box:
    1. generate obj_ids (number defined)
    2. for each obj_id: generate center of object (shape-bb), then generate X as rand (centerX-bb, centerX +bb) and Y jointly, and rand(0, bb*bb) values for each channel V
    3. generate channel names as var_names = pd.Index(f"gene{i}" for i in range(num_channels))
    4. generate metadata:
        - generate df with cell cycle:
             - generate 5 diff cell cycles, then assign random cols to that
             - generate  TR: float64, from 100 to 1000
        -
    Params
    ------

    """

    obj_ids=np.array([np.uint32(i) for i in range (num_obj_ids)])

    #generate channel names
    lengths = np.random.randint(3, 5, num_channels)
    letters = np.array(list(ascii_letters))
    gen_word = lambda l: "".join(np.random.choice(letters, l))
    if channels is not None:
        num_channels=len(channels)
        channels = pd.DataFrame(np.array(channels), columns=["name"])
    else:
        channels = pd.DataFrame(np.array([gen_word(l) for l in lengths]), columns=["name"])

    channels.index.name = 'channel_id'
    # .reset_index().set_index('name').loc[channels]['channel_id']

    metadata=gen_metadata_df(num_obj_ids, obj_ids, possible_cell_cycles, **kwargs)

    X, Y, mpp, obj_ids=gen_objs(shape, bounding_box, num_channels, obj_ids,  mpp_dtype)

    data={
        "x":X,
        "y":Y,
        "obj_ids":obj_ids,
        "mpp":mpp,
    }
    mppdata=MPPData(metadata, channels, data,  data_config=data_config)
    return mppdata

if __name__=="__main__":
    # tmp1=gen_mppdata()

    mpp_data = gen_mppdata(num_obj_ids=5)
    mpp_data_sub = mpp_data.subsample(frac=-1)

    assert len(mpp_data_sub.unique_obj_ids) == 0
    assert len(mpp_data_sub.x) == 0
    assert len(mpp_data_sub.y) == 0
    assert len(mpp_data_sub.mpp) == 0

