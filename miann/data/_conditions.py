# functions for converting conditions to strings or one-hot encoded vectors
from miann.constants import CONDITIONS
import numpy as np

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def get_combined_one_hot(arrs):
    if len(arrs) != 2:
        raise NotImplementedError(f"combine {len(arrs)} arrs")
    mask = (~np.isnan(arrs[0][:,0]))&(~np.isnan(arrs[1][:,0]))
    n1 = arrs[0].shape[1]
    n2 = arrs[1].shape[1]
    targets = np.zeros(len(arrs[0]), dtype=np.uint8)
    targets[mask] = np.argmax(arrs[0][mask], axis=1) + n1*np.argmax(arrs[1][mask], axis=1)
    res = get_one_hot(targets, n1*n2)
    res[~mask] = np.nan
    return res

def convert_condition(arr, desc, one_hot=False):
    cur_conditions = CONDITIONS[desc]
    # need to go from str to numbers or the other way?
    if np.isin(arr, cur_conditions).any():
        conv_arr = np.zeros(arr.shape, dtype=np.uint8)
        for i,c in enumerate(cur_conditions):
            conv_arr[arr==c] = i
        if one_hot:
            conv_arr = get_one_hot(conv_arr, len(cur_conditions))
    else:
        conv_arr = np.zeros(arr.shape, dtype=np.object)
        for i,c in enumerate(cur_conditions):
            conv_arr[arr==i]=c
    return conv_arr