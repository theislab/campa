import json
import os
from copy import copy
from logging import getLogger
from typing import Dict

import numpy as np
import pandas as pd

from campa.constants import get_data_config
from campa.data._conditions import (
    convert_condition,
    get_bin_3_condition,
    get_combined_one_hot,
    get_lowhigh_bin_2_condition,
    get_one_hot,
    get_zscore_condition,
    process_condition_desc,
)
from campa.data._img_utils import BoundingBox, pad_border
from campa.pl import annotate_img


class ImageData:
    def __init__(self, metadata, channels, imgs=None, seg_imgs=None, **kwargs):
        self.metadata = metadata
        self.channels = channels
        self.seed = kwargs.get("seed", 42)
        self.rng = np.random.default_rng(seed=self.seed)
        self.log = getLogger(self.__class__.__name__)
        self.imgs = imgs
        self.seg_imgs = seg_imgs

    @classmethod
    def from_dir(cls, data_dir):
        # TODO read metadata and images (lazy) and call constructuor
        pass

    def get_obj_img(self, obj_id, kind="data"):
        # TODO crop and return object images for different kinds: data, mask (object mask)
        pass


class MPPData:
    # TODO: in initialisers: make x,y, obj_ids required, but add automatically to keys when not specified
    def __init__(
        self,
        metadata: pd.DataFrame,
        channels: pd.DataFrame,
        data: Dict[str, np.ndarray],
        **kwargs,
    ):
        """
        data: dictionary containing MPPData, at least containing required_keys: x, y, obj_ids.

        If mpp is not present, is replaced with zero-value array of shape: #pixels x 1 x 1 x #channels

        kwargs:
            data_config (default NascentRNA)
            seed (default 42)
        """
        # set up logger
        self.log = getLogger(self.__class__.__name__)
        # set up rng
        self.seed = kwargs.get("seed", 42)
        # TODO change to default_rng as soon as have reproduced legacy datasets
        self.rng = np.random.RandomState(seed=self.seed)
        # save attributes
        self.data_config_name = kwargs.get("data_config", "NascentRNA")
        self.data_config = get_data_config(self.data_config_name)
        self.channels = channels
        self._data = data

        for required_key in ["x", "y", "obj_ids"]:
            assert (
                required_key in data.keys()
            ), f"required key {required_key} missing from data"
        # subset metadata to obj_ids in data
        self.metadata = metadata[
            metadata[self.data_config.OBJ_ID].isin(np.unique(self.obj_ids))
        ]
        # add neighbor dimensions to mpp if not existing
        if len(self.mpp.shape) == 2:
            self._data["mpp"] = self.mpp[:, np.newaxis, np.newaxis, :]
        self.log.info(f"Created new: {self.__str__()}")

    @classmethod
    def from_image_data(self, image_data: ImageData):
        # TODO convert ImageData to MPPData
        pass

    @classmethod
    def from_data_dir(
        cls,
        data_dir,
        mode="r",
        base_dir=None,
        keys=("x", "y", "obj_ids"),
        optional_keys=("mpp", "labels", "latent", "conditions"),
        **kwargs,
    ):
        """
        Read MPPData from directory.

        If mpp_params are present, will first load mpp_data from base_data_dir,
        and add remaining information from data_dir.

        Requires metadata.csv and channels.csv.
        Reads data from key.npy for each key in required_keys.
        If present, will also read key.npy for each key in optional_keys

        Args:
            data_dir: Path to the specific directory containing one set of npy and csv files as described above.
            Note that this path should be relative to the 'base_dir',
            which is either provided implicitly in the function or set
            to data_dir path specified in config.ini otherwise.
            mode: mmap_mode for np.load. Set to None to load data in memory.
            data_config: Name of the constant in the [data] field of config.ini file,
            which contains path to the python file
            with configuration parameters for loading the data.
            base_dir: look for data in base_dir/data_dir. Default in DATA_DIR
        """
        # load data_config
        data_config = get_data_config(kwargs.get("data_config", "NascentRNA"))
        if base_dir is None:
            base_dir = data_config.DATA_DIR

        # mpp_params present?
        if os.path.isfile(os.path.join(base_dir, data_dir, "mpp_params.json")):
            # first, load base_mpp_data
            res_keys, res_optional_keys = _get_keys(keys, optional_keys, None)
            mpp_params = json.load(
                open(os.path.join(base_dir, data_dir, "mpp_params.json"))
            )
            self = cls.from_data_dir(
                mpp_params["base_data_dir"],
                keys=res_keys,
                optional_keys=res_optional_keys,
                base_dir=data_config.DATA_DIR,
                mode=mode,
                **kwargs,
            )
            # second, add mpp_data
            res_keys, res_optional_keys = _get_keys(keys, optional_keys, self)
            self.add_data_from_dir(
                data_dir,
                keys=res_keys,
                optional_keys=res_optional_keys,
                base_dir=base_dir,
                mode=mode,
                subset=mpp_params["subset"],
            )
            self.log.info(
                f"Loaded data from {data_dir}, with base data from {mpp_params['base_data_dir']}"
            )
            self.data_dir = data_dir
            self.base_dir = base_dir
            return self
        else:
            # have reached true base_dir, load data
            # read all data from data_dir
            self = cls._from_data_dir(
                data_dir, mode, base_dir, keys, optional_keys, **kwargs
            )
            self.log.info(f"Loaded data from {data_dir}.")
        return self

    @classmethod
    def _from_data_dir(
        cls,
        data_dir,
        mode="r",
        base_dir=None,
        keys=("x", "y", "obj_ids"),
        optional_keys=("mpp", "labels", "latent", "conditions"),
        **kwargs,
    ):
        """
        Helper function to read MPPData from directory. Ignores mpp_params.json
        """
        # get base_dir
        if base_dir is None:
            data_config = get_data_config(kwargs.get("data_config", "NascentRNA"))
            base_dir = data_config.DATA_DIR

        metadata = pd.read_csv(
            os.path.join(base_dir, data_dir, "metadata.csv"), index_col=0
        ).reset_index(drop=True)
        channels = pd.read_csv(
            os.path.join(base_dir, data_dir, "channels.csv"),
            names=["channel_id", "name"],
            index_col=0,
        ).reset_index(drop=True)
        channels.index.name = "channel_id"
        # read npy data
        data = {}
        for fname in keys:
            data[fname] = _try_mmap_load(
                os.path.join(base_dir, data_dir, f"{fname}.npy"), mmap_mode=mode
            )
        for fname in optional_keys:
            d = _try_mmap_load(
                os.path.join(base_dir, data_dir, f"{fname}.npy"),
                mmap_mode=mode,
                allow_not_existing=True,
            )
            if d is not None:
                data[fname] = d
        # init self
        self = cls(metadata=metadata, channels=channels, data=data, **kwargs)
        self.data_dir = data_dir
        self.base_dir = base_dir
        return self

    @classmethod
    def concat(cls, objs):
        """concatenate the mpp_data objects by concatenating all arrays and return a new one"""
        # channels, data_config, and _data.keys() need to be the same
        for mpp_data in objs:
            assert (mpp_data.channels.name == objs[0].channels.name).all()
            assert mpp_data._data.keys() == objs[0]._data.keys()
            assert mpp_data.data_config_name == objs[0].data_config_name

        channels = objs[0].channels
        # concatenate metadata (pandas)
        metadata = pd.concat(
            [mpp_data.metadata for mpp_data in objs], axis=0, ignore_index=True
        )
        # concatenate numpy arrays
        data = {}
        for key in objs[0]._data.keys():
            data[key] = np.concatenate(
                [mpp_data._data[key] for mpp_data in objs], axis=0
            )

        self = cls(
            metadata=metadata,
            channels=channels,
            data=data,
            seed=objs[0].seed,
            data_config=objs[0].data_config_name,
        )
        self.log.info(f"Created by concatenating {len(objs)} MPPDatas")
        return self

    # --- Properties ---
    @property
    def mpp(self):
        if "mpp" not in self._data.keys():
            self.log.info("Setting mpp to empty array")
            self._data["mpp"] = np.zeros((len(self.x), 1, 1, len(self.channels)))
        return self._data["mpp"]

    @property
    def obj_ids(self):
        return self._data["obj_ids"]

    @property
    def x(self):
        return self._data["x"]

    @property
    def y(self):
        return self._data["y"]

    @property
    def latent(self):
        return self.data("latent")

    def data(self, key):
        """
        Information contained in MPPData.

        Required keys are: x, y, obj_ids.

        Args;
            key: identifier for data that should be returned

        Returns:
            array-like data, or None if data could not be found
        """
        if key in self._data.keys():
            return self._data[key]
        elif key == "center_mpp":
            return self.center_mpp
        return None

    @property
    def conditions(self):
        return self.data("conditions")

    @property
    def has_neighbor_data(self):
        return (self.mpp.shape[1] != 1) and (self.mpp.shape[2] != 1)

    @property
    def center_mpp(self):
        c = self.mpp.shape[1] // 2
        return self.mpp[:, c, c, :]

    @property
    def unique_obj_ids(self):
        return np.unique(self.obj_ids)

    def __str__(self):
        s = f"MPPData for {self.data_config_name} ({self.mpp.shape[0]} mpps with shape {self.mpp.shape[1:]}"
        s += f" from {len(self.metadata)} objects)."
        s += f" Data keys: {list(self._data.keys())}."
        return s

    # --- Saving ---
    def write(self, save_dir, save_keys=None, mpp_params=None):
        """
        Write MPPData to disk.

        Save channels, metadata as csv and one npy file per entry in MPPData.data.

        Args:
            save_dir: full path to directory in which to save MPPData
            save_keys: only save these MPPData.data entries. "x", "y", "obj_ids" are always saved
            mpp_params: params to be saved in save_dir. Use if save_keys is not None. Should contain base_data_dir
            (relative to DATA_DIR), and subset information
                for correct MPPData initialisation.
        Returns:
            Nothing, saves data to disk
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if save_keys is None:
            save_keys = self._data.keys()
        else:
            if mpp_params is None or mpp_params.get("base_data_dir", None) is None:
                self.log.warn(
                    "Saving partial keys of mpp data without a base_data_dir to enable correct loading"
                )
            else:
                # save mpp_params
                mpp_params = {
                    "base_data_dir": mpp_params["base_data_dir"],
                    "subset": mpp_params.get("subset", False),
                }
                json.dump(
                    mpp_params,
                    open(os.path.join(save_dir, "mpp_params.json"), "w"),
                    indent=4,
                )
        # add required save_keys
        save_keys = list(set(save_keys).union(["x", "y", "obj_ids"]))
        self.log.info(f"Saving mpp data to {save_dir} (keys: {save_keys})")
        for key in save_keys:
            np.save(os.path.join(save_dir, f"{key}.npy"), np.array(self._data[key]))
        self.channels.to_csv(os.path.join(save_dir, "channels.csv"), header=None)
        self.metadata.to_csv(os.path.join(save_dir, "metadata.csv"))

    def copy(self):
        import copy

        copy.deepcopy(self._data)
        copy.deepcopy(self.metadata)
        mpp_copy = MPPData(
            metadata=copy.deepcopy(self.metadata),
            channels=copy.deepcopy(self.channels),
            data=copy.deepcopy(self._data),
            seed=copy.deepcopy(self.seed),
            data_config=copy.deepcopy(self.data_config_name),
        )

        return mpp_copy
        # raise(NotImplementedError)
        # pass

    # TODO Nastassya: ensure that each function logs what its doing with log.info
    # --- Modify / Add data ---
    def add_data_from_dir(
        self, data_dir, keys=(), optional_keys=(), subset=False, **kwargs
    ):
        """
        Add data to MPPData from data_dir.

        Expects x, y, and obj_ids to be present in data_dir.
        If not subset, and data in data_dir differs from data in mpp_data, raises a ValueError.

        Assumes that the data in data_dir uses the same data_config as the present object.

        Args:
            data_dir: full path to the dir containing the npy files to be loaded
            keys: filenames of data that should be loaded
            optional_keys: filenames of data that could optionally be loaded if present
            subset: if True, subset MPPData to data present in data_dir.
            kwargs: passes to MPPData.from_data_dir

        Returns:
            Nothing, modifies MPPData in place. Adds `keys` to `MPPData._data`

        """
        mpp_to_add = MPPData._from_data_dir(
            data_dir,
            keys=list(set(["x", "y", "obj_ids"] + list(keys))),
            optional_keys=list(optional_keys),
            data_config=self.data_config_name,
            **kwargs,
        )
        # check that obj_ids from mpp_to_add are the same / a subset of the current obj_ids
        if subset:
            assert set(self.unique_obj_ids).issuperset(mpp_to_add.unique_obj_ids)
            # subset self to the obj_ids in mpp_to_add
            self.subset(obj_ids=mpp_to_add.unique_obj_ids)
        # now, data should be exactly the same
        assert (self.obj_ids == mpp_to_add.obj_ids).all()
        assert (self.x == mpp_to_add.x).all()
        assert (self.y == mpp_to_add.y).all()
        # finally, add keys
        added_mpp = False
        for key in keys:
            self._data[key] = mpp_to_add.data(key)
            added_mpp = added_mpp if key != "mpp" else True
        for key in optional_keys:
            if mpp_to_add.data(key) is not None:
                # check if mpp is not only zeros
                if (
                    key == "mpp"
                    and (
                        mpp_to_add.data("mpp") == np.zeros_like(mpp_to_add.data("mpp"))
                    ).all()
                ):
                    continue
                self._data[key] = mpp_to_add.data(key)
                added_mpp = added_mpp if key != "mpp" else True
        # update channels if added mpp
        if added_mpp:
            self.channels = mpp_to_add.channels
        self.log.info(f"Updated data to keys {list(self._data.keys())}")

    def train_val_test_split(self, train_frac=0.8, val_frac=0.1):
        """split along obj_ids for train/val/test split"""
        # TODO (maybe) adapt and ensure that have even val/test fractions from each well
        ids = self.unique_obj_ids.copy()
        self.rng.shuffle(ids)
        num_train = int(len(ids) * train_frac)
        num_val = int(len(ids) * val_frac)
        train_ids = ids[:num_train]
        val_ids = ids[num_train : num_train + num_val]
        test_ids = ids[num_train + num_val :]
        self.log.info(
            f"Splitting data in {len(train_ids)} train, {len(val_ids)} val, and {len(test_ids)} test objects"
        )
        splits = []
        for split_ids in (train_ids, val_ids, test_ids):
            ind = np.in1d(self.obj_ids, split_ids)
            splits.append(self.apply_mask(ind, copy=True))
        return splits

    def prepare(self, params):
        """prepare MPP data according to given params.

        Ignores data_dirs, split, subsampling, and neighborhood.
        after self.prepare(), you might want to call self.subsample() and
        self.add_neighorhood()
        """
        # subset channels - should be done first, as normalise writes params wrt channel ordering
        if params.get("channels", None) is not None:
            self.subset_channels(params["channels"])
        # normalise
        if params["normalise"]:
            self.normalise(**params["normalise_kwargs"])
        # add conditions
        if params.get("condition", None) is not None:
            self.add_conditions(params["condition"], **params["condition_kwargs"])
        # subset (after conditions, bc subsetting might need conditions)
        if params["subset"]:
            self.subset(**params["subset_kwargs"])

    # --- Detailed fns used in prepare() ---
    # TODO: Nastassya write tests for this
    def add_conditions(self, cond_desc, cond_params=None):
        """
        Add conditions informations by aggregating over channels (per cell) or reading data from cell cycle file.

        Withing a framework of conditional VAE training, a conditional vector is passed alongside the input vector.
        This vector could represent e.g. perturbation or a cell cycle.
        The vector is constructed for each input sample (here, we refer to an input sample is a pixel or a
        pixel's neighbourhood)
        from a metadata table as a concatenation of vectors each representing separate condition.

        In order to create a conditions table, method add_conditions(), which takes the following parameters:

        cond_desc: a list of conditions descriptions, represented as string values. Conditions can be represented in
        one of the following formats:
            - columnName - if this column is specified in data_config.CONDITIONS,
              then a value in that column gets numerically encoded into a class value (1, 2, ...).
              Note that the class is assigned to the value only if the values is contained in range of values
              that column could take (as specified in data_config.CONDITIONS[columnName]).
              Note that one can also provide a special entrie "UNKNOWN" there.
              In that case, all values with conditions that are not listed are mapped to the additional last class.
              If the column name is not provided in the data_config.CONDITIONS,
              values are assumed to be continious and stored as they are in the conditions table.
            - columnName_postprocess, where postprocess can be set to one of the following values:
                - lowhigh_bin_2: bins all values in the specified column in 4 quantiles, encodes values in
                  the lowest quantile as one class and values in the high quantile as teh second class
                  (one-hot encoded), and all values in-between sets to None
                - bin_3: bin values in .33 and .66 quantiles and one-hot encodes each of them into 3 classes according
                  to quantile where the value belongs to
                - zscore: normalizes a continuous set of values by mean and std of train subset,
                  supposed to be used only after a train-test split up
                - one_hot: same as when only ***columnName*** is provided as a condition description,
                  but additionally one-hot encodes the class values.

        cond_params: Dictionary with parameters needed for some conditions, e.g. for classyfying values into bins.
            If empty, these values are calculated and stored into this dict (where appropriate).

        NOTE: Operation is performed inplace.
        """
        self.log.info(f"Adding conditions: {cond_desc}")
        conditions = []
        for desc in cond_desc:
            cond = self.get_condition(desc, cond_params)
            conditions.append(cond)
        self._data["conditions"] = np.concatenate(conditions, axis=-1)

    # TODO: Nastassya: write tests for this
    def subset(
        self,
        frac=None,
        num=None,
        obj_ids=None,
        nona_condition=False,
        copy=False,
        **kwargs,
    ):
        """
        Restrict objects to those with specified value(s) for key in the metadata table

        Args:
            frac (float): fraction of objects to randomly subsample.
                Applied after the other subsetting
            num (int): number of objects to randomly subsample. frac takes precedence
                Applied after the other subsetting.
            obj_ids (list of str): ids of objects to subset to
            nona_condition: if set to True,  all values having nan conditions will be filtered out. Note that the way
                condition's creation
             is implemented allows one to e.g. leave only entries which values in the specified column were in the low
                and high quantilies,
             and filter out everything else.
            copy: return new MPPData object or modify in place

            kwargs:
                key: name of column used to select objects
                value (str or list of str): allowed entries for selected objects,
                    NO_NAN is special token selecting all values except nan

        Returns:
            if copy, subsetted MPPData, otherwise nothing, modifies Data object in place
        """
        selected_obj_ids = np.array(self.metadata[self.data_config.OBJ_ID])
        self.log.info(f"Before subsetting: {len(selected_obj_ids)} objects")
        # select ids based on pre-defined ids
        if obj_ids is not None:
            selected_obj_ids = np.array(obj_ids)
            self.log.info(f"Subsetting to {len(obj_ids)} objects")
        # select ids based on kwargs
        for key, value in kwargs.items():
            cur_metadata = self.metadata.set_index(self.data_config.OBJ_ID).loc[
                selected_obj_ids
            ]
            assert (
                key in cur_metadata.columns
            ), f"provided column {key} was not found in the metadata table!"
            if value == "NO_NAN":
                mask = ~cur_metadata[key].isnull()  # TODO check!
                self.log.info(f"Subsetting to NO_NAN {key}: {sum(mask)} objects")
            else:
                if not isinstance(value, list):
                    value = [value]
                mask = cur_metadata[key].isin(value)
                self.log.info(f"Subsetting to {key}={value}: {sum(mask)} objects")
            selected_obj_ids = selected_obj_ids[mask]
        # select ids based on nona conditions
        if nona_condition:  # TODO test this!
            if self.conditions is not None:
                mask = ~np.isnan(self.conditions).any(axis=1)
                mpp_df = self.metadata.set_index(self.data_config.OBJ_ID).loc[
                    self.obj_ids
                ][mask]
                nona_obj_ids = (
                    mpp_df.reset_index().groupby(self.data_config.OBJ_ID).first().index
                )
                selected_obj_ids = np.array(
                    list(set(selected_obj_ids).intersection(nona_obj_ids))
                )
                self.log.info(
                    f"Subsetting to objects with NO_NAN condition: {len(selected_obj_ids)}"
                )
        # select ids based on ramdom subset
        if frac is not None:
            num = int(len(selected_obj_ids) * frac)
        if num is not None:
            self.log.info(f"Subsetting to {num} random objects")
            self.rng.seed(
                self.seed
            )  # TODO: remove seeding if have duplicated existing datasets
            selected_obj_ids = selected_obj_ids[
                self.rng.choice(len(selected_obj_ids), size=num, replace=False)
            ]
        # create mpp mask for applying subset
        obj_mask = self.metadata[self.data_config.OBJ_ID].isin(selected_obj_ids)
        mpp_mask = self._get_per_mpp_value(obj_mask)
        # apply mask
        return self.apply_mask(mpp_mask, copy=copy)

    # TODO Nastassya: write tests for this
    def subset_channels(self, channels):
        """
        Restrict self.mpp to defined channels. Channels are given as string values.
        Updates self.mpp and self.channels
        """
        # TODO need copy argument?
        self.log.info(f"Subsetting from {len(self.channels)} channels")
        channels_overlap = np.intersect1d(self.channels.name.values, channels)
        assert (
            len(np.intersect1d(self.channels.name.values, channels)) != 0
        ), "mpp object does not contain provided channels!"
        cids = list(
            self.channels.reset_index()
            .set_index("name")
            .loc[channels_overlap]["channel_id"]
        )
        raw_channels = self.channels
        self.channels = self.channels.loc[cids].reset_index(drop=True)
        self.channels.index.name = "channel_id"
        self._data["mpp"] = self.mpp[:, :, :, cids]
        subset_mpp_channels = self.channels
        self.log.info(f"Restricted channels to {len(self.channels)} channels")
        channels_diff = np.setdiff1d(raw_channels, subset_mpp_channels)
        if len(channels_diff) > 0:
            self.log.info(f"The following channels were excluded {channels_diff}")
        else:
            self.log.info("None of the channels were excluded ")

    # TODO Nastassya: write tests for this
    def subsample(
        self,
        frac=None,
        frac_per_obj=None,
        num=None,
        num_per_obj=None,
        add_neighborhood=False,
        neighborhood_size=3,
    ):
        """
        Performs MPP-level subsampling by selecting mpps (could be thought of as pixels)
        All other information is updated accordingly (to save RAM/HDD-memory).
        Additionally, can extend mpps' representations by their neighbourhoods before subsampling.
        Note that several conditions for subsampling can be provided,
        and the resulting MPP Dataset will be computed as combination of all the provided conditions.

        args:
            frac: subsample a random number of mpps (pixels) from the whole dataset by a specified fraction.
            Should be in range [0, 1].
            num: subsample a random number of mpps (pixels) from the whole dataset by a specified number of mpps
                to be chosen.
            frac_per_obj: allows to subsample a fraction of mpps on the object level - that is,
                for each object (cell) to subsample
                the same fraction of mpps independently.
            num_per_obj: same as frac_per_obj, but a number of mpps to be left instead of fraction is provided
            add_neighborhood: if set to True, extends mpp representation with a square neighbourhood around it
            neighborhood_size: size of the neighborhood

        Note that at least one of four parameters that indicate subsampling (frac, num, frac_per_obj, num_per_obj)
        size should be provided

        Returns: new subsampled MPPData. Operation cannot be done in place
        """
        assert (
            sum(
                [
                    frac is not None,
                    frac_per_obj is not None,
                    num is not None,
                    num_per_obj is not None,
                ]
            )
            == 1
        ), "set only one of the params to a value"
        assert not (
            self.has_neighbor_data and add_neighborhood
        ), "cannot add neighborhood, already has neighbor data"
        if (frac is not None) or (num is not None):
            if frac is not None:
                num = int(len(self.mpp) * frac)
            self.log.info(f"Subsampling from {len(self.mpp)} to {num}")
            # randomly select num mpps
            # TODO: replace with self.rng as soon as have reproduced data
            rng = np.random.default_rng(seed=self.seed)
            selected = rng.choice(len(self.mpp), num, replace=False)
            # select other information accordingly
            mpp_data = self.apply_mask(selected, copy=True)

        else:
            if frac_per_obj is not None:
                self.log.info(f"Subsampling each object to {frac_per_obj*100}%")
            else:
                self.log.info(f"Subsampling each object to {num_per_obj}")
            # iterate over all obj_ids
            mask = np.zeros(len(self.mpp), dtype=bool)
            idx = np.arange(len(mask))
            for obj_id in self.unique_obj_ids:
                obj_mask = self.obj_ids == obj_id
                if frac_per_obj is not None:
                    cur_num = int(obj_mask.sum() * frac_per_obj)
                else:
                    cur_num = num_per_obj
                # TODO replace with self.rng if have reproduced datasets
                rng = np.random.default_rng(seed=self.seed)
                selected = rng.choice(
                    len(obj_mask.nonzero()[0]), cur_num, replace=False
                )
                selected_idx = idx[obj_mask][selected]
                mask[selected_idx] = True
            mpp_data = self.apply_mask(mask, copy=True)
        if add_neighborhood:
            self.log.info(f"Adding neighborhood of size {neighborhood_size}")
            neighbor_mpp = self._get_neighborhood(
                mpp_data.obj_ids, mpp_data.x, mpp_data.y, size=neighborhood_size
            )
            mpp_data._data["mpp"] = neighbor_mpp
        return mpp_data

    def add_neighborhood(self, size=3, copy=False):
        """
        Extends mpp representation with a square neighbourhood around it.

        args:
            neighborhood_size: size n of the neighborhood. The resulting mpp representation will then be set
                to a nXn square around each pixel.
            copy: return new MPPData object or modify in place
        """
        # TODO need copy flag?
        self.log.info(f"Adding neighborhood of size {size}")
        mpp = self._get_neighborhood(self.obj_ids, self.x, self.y, size=size)
        assert (mpp[:, size // 2, size // 2, :] == self.center_mpp).all()

        if copy:
            data = self._data.copy()
            data["mpp"] = mpp
            return MPPData(
                metadata=self.metadata,
                channels=self.channels,
                data=data,
                seed=self.seed,
                data_config=self.data_config_name,
            )
        else:
            assert (
                not self.has_neighbor_data
            ), "cannot add neighborhood, already has neighbor data"
            # TODO Nastassya: this should be in a unittest of self.get_neighborhood
            self._data["mpp"] = mpp

    def normalise(self, background_value=None, percentile=None, rescale_values=()):
        """
        To prepare the data, pixel values can be normalized. For that, pixel values can be e.g. shifted and rescaled so
        that they end up ranging between 0 and 1:

        args:
            background_value: shifts all values (contained in .data field) by that value. Note that all resulting
            negative values are cut off to zero. Several shift options are possible, depending on a background_value,
            which could be in one of the following formats:
               - single value (float): then data is shifted by this number in every channels
               - list of predifined values (floats): data in each channel is shifted separately by a corresponding
                 value in that list. Note: in that case, number of shifted
                 values should be the same as number of channels.
               - string: list of values for shifting are loaded from the channels_metadata table
                 (should be located in the folder data_config.DATA_DIR in a file self.data_config.CHANNELS_METADATA).
                 The values are then loaded from the column corresponding to a provided string value.

            percentile: after (optional) shifting data can be rescaled by a specified value separately for each channel.
            The value could be in one of the following formats:
                - float: value from 0 to 100, which represents a quantile by which data should be rescaled.
                  Quantiles are then calculated for each channel separately, and data is rescaled independently in
                  each channel by the calculated respective quantile value.
                - list of predifined values (floats): data in each channel is rescaled separately for each channel by a
                  corresponding value in that list.

            rescale_values: A dictionary with values used for normalization. If empty, parameters calculated for
                rescaling would be optionally be stored into this dictionary.

        NOTE: this operation is performed **inplace**.
        """
        if background_value is not None:
            self._subtract_background(background_value)
        if percentile is not None:
            self._rescale_intensities_per_channel(percentile, rescale_values)

    # --- Helper functions ---
    def apply_mask(self, mask, copy=False):
        """
        return new MPPData with masked self._data values
        """
        data = {}
        for key in self._data.keys():
            data[key] = self._data[key][mask]

        if copy is False:
            self._data = data
            self.metadata = self.metadata[
                self.metadata[self.data_config.OBJ_ID].isin(np.unique(self.obj_ids))
            ]
        else:
            return MPPData(
                metadata=self.metadata,
                channels=self.channels,
                data=data,
                seed=self.seed,
                data_config=self.data_config_name,
            )

    def _subtract_background(self, background_value):
        """
        background_value: float value to be subtracted, or name of column in CHANNELS_METADATA
        NOTE: code copied/adapted from Scott Berry's MCU package
        NOTE: mpp is converted to float
        """
        if isinstance(background_value, float):
            self.log.info(
                f"Subtracting constant value {background_value} from all channels"
            )
            self._data["mpp"] = self.mpp.astype(np.float32) - background_value
        elif isinstance(background_value, list):
            assert len(background_value) == len(
                self.channels
            ), "specify one background value per channel"
            self.log.info("Subtracting predefined channel specific background value")
            self._data["mpp"] = self.mpp.astype(np.float32) - np.array(background_value)
        else:  # is column name
            self.log.info(
                f"Subtracting channel-specific background value defined in column {background_value}"
            )
            channels_metadata = pd.read_csv(
                os.path.join(
                    self.data_config.DATA_DIR, self.data_config.CHANNELS_METADATA
                )
            )
            background_value = channels_metadata.set_index("name").loc[
                self.channels.name
            ][background_value]
            # replace nan with 0
            for ch, val in zip(self.channels.name, background_value):
                if pd.isna(val):
                    self.log.warning(f"Missing background value for channel {ch}")
            background_value = background_value.fillna(0)
            self.log.debug(f"use background_value: {background_value}")
            self._data["mpp"] = self.mpp.astype(np.float32) - np.array(background_value)
        # cut off at 0 (no negative values)
        self._data["mpp"][self.mpp < 0] = 0

    def _rescale_intensities_per_channel(self, percentile=98.0, rescale_values=None):
        """
        TODO add more info
        NOTE converts self.mpp to float32
        """
        # TODO need copy flag?
        if rescale_values is None:
            rescale_values = []
        if len(rescale_values) == 0:
            self.log.info(
                f"Rescaling MPP intensities per channel with {percentile} percentile"
            )
            rescale_values.extend(np.percentile(self.center_mpp, percentile, axis=0))
        else:
            assert len(rescale_values) == len(
                self.channels
            ), f"need {len(self.channels)} rescale values, got {len(rescale_values)}"
            self.log.info(
                "Rescaling MPP intensities per channels with predefined values"
            )
        self._data["mpp"] = self.mpp / rescale_values
        self._data["mpp"] = self.mpp.astype(np.float32)

    # ---- getting functions -----
    def _get_per_mpp_value(self, per_obj_value):
        """takes list of values corresponding to self.metadata[OBJ_ID]
        and propagates them to self.obj_id

        Args:
            per_obj_value: list of values, or dict containing multiple value lists

        Returns:
            pd.Series or pd.DataFrame
        """
        if isinstance(per_obj_value, dict):
            val = per_obj_value.keys()
            per_obj_value.update({"MERGE_KEY": self.metadata[self.data_config.OBJ_ID]})
            per_obj_df = pd.DataFrame(per_obj_value)
        else:
            val = "val"
            per_obj_df = pd.DataFrame(
                {
                    "val": per_obj_value,
                    "MERGE_KEY": self.metadata[self.data_config.OBJ_ID],
                }
            )
        df = pd.DataFrame({"MERGE_KEY": self.obj_ids})
        per_mpp_value = df.merge(
            per_obj_df, left_on="MERGE_KEY", right_on="MERGE_KEY", how="left"
        )[val]
        return per_mpp_value

    def get_condition(self, desc, cond_params=None):
        """
        return condition based on desc (used by add_conditions).
        If cond is a list of conditions, return unique one-hot encoded vector combining multiple conditions
        (only possible when all sub-conditions are one-hot encoded)

        NOTE: cond_params are modified inplace
        """
        if cond_params is None:
            cond_params = {}
        cond = None
        if isinstance(desc, list):
            # check if combining is possible
            assert np.all(
                [("one_hot" in d) for d in desc]
            ), f"all of {desc} need to be one_hot encoded"
            conds = [self.get_condition(d, cond_params) for d in desc]
            cond = get_combined_one_hot(conds)
        else:
            # desc is one column in metadata
            # get correct desc name (match to column names)
            desc, postprocess = process_condition_desc(desc)
            self.log.info(
                f"Looking up condition {desc}, with postprocess {postprocess}"
            )
            cond = self._get_per_mpp_value(np.array(self.metadata[desc]))
            if postprocess == "one_hot":
                cond = convert_condition(
                    np.array(cond),
                    desc=desc,
                    one_hot=True,
                    data_config=self.data_config,
                )
            elif postprocess == "bin_3":
                cond, _ = get_bin_3_condition(cond, desc, cond_params)
                cond = get_one_hot(cond, nb_classes=3)
            elif postprocess == "lowhigh_bin_2":
                cond_bin, q = get_lowhigh_bin_2_condition(cond, desc, cond_params)
                cond_one_hot = get_one_hot(cond_bin, nb_classes=2)
                cond_one_hot[np.logical_and(cond > q[0], cond < q[1])] = np.nan
                cond = cond_one_hot
            elif postprocess == "zscore":
                cond, _ = get_zscore_condition(cond, desc, cond_params)
                # add empty axis for concatenation
                cond = np.array(cond[:, np.newaxis])
            else:
                cond = convert_condition(
                    np.array(cond)[:, np.newaxis],
                    desc=desc,
                    data_config=self.data_config,
                )
        return cond

    def get_adata(self, X="mpp", obsm=(), obs=()):
        """
        Create adata from information contained in MPPData.

        channels are put in adata.var
        metadata is put in adata.obs

        Args:
            X: key in MPPData.data that should be in adata.X
            obsm: keys in MPPData.data that should be in adata.obsm.
                Can be dict with keys the desired names in adata, and values the keys in MPPData.data
            obs: keys from MPPData.data that should be in adata.obs in addition to all information form metadata
        """
        import anndata as ad

        if isinstance(obsm, list) or isinstance(obsm, tuple):
            obsm = {o: o for o in obsm}
        obsm = {k: self.data(v).astype(np.float32) for k, v in obsm.items()}
        if X == "mpp":
            var = self.channels
            X = self.center_mpp
        else:
            var = None
            X = self.data(X)
        # add spatial coords as obsm['spatial']
        obsm["spatial"] = np.stack([self.x, self.y]).T.astype(np.float32)
        # get per-pixel obs
        obs_ = self._get_per_mpp_value(self.metadata.to_dict(orient="list"))
        for o in obs:
            obs_[o] = self.data(o)
        adata = ad.AnnData(X=X.astype(np.float32), obs=obs_, var=var, obsm=obsm)
        # set var_names if possible
        if var is not None:
            adata.var_names = adata.var["name"]
        return adata

    def extract_csv(self, data="mpp", obs=()):
        """
        extract mpp_data.data into csv file.

        Per default, files contain data, x and y coordinates, object_id.
        Args:
            obs: name of metadat columns that should be additionally added
        """
        # TODO: column names are tuples
        columns = None
        if data == "mpp":
            columns = self.channels
            X = self.center_mpp
        else:
            X = self.data(data)
        df = pd.DataFrame(data=X, columns=columns)
        # add x,y,obj_id
        df["x"] = self.x
        df["y"] = self.y
        obs_ = self._get_per_mpp_value(
            {o: self.metadata[o] for o in list(obs) + [self.data_config.OBJ_ID]}
        )
        df[self.data_config.OBJ_ID] = obs_[self.data_config.OBJ_ID]
        return df

    # TODO nastassya: test
    def _get_neighborhood(self, obj_ids, xs, ys, size=3, border_mode="center"):
        """return neighborhood information for given obj_ids + xs + ys"""
        data = np.zeros(
            (len(obj_ids), size, size, len(self.channels)), dtype=self.mpp.dtype
        )
        for obj_id in np.unique(obj_ids):
            mask = obj_ids == obj_id
            img, (xoff, yoff) = self.get_object_img(obj_id, data="mpp", pad=size // 2)
            vals = []
            for x, y in zip(xs[mask], ys[mask]):
                idx = tuple(
                    slice(pp - size // 2, pp + size // 2 + 1)
                    for pp in [y - yoff, x - xoff]
                )
                vals.append(pad_border(img[idx], mode=border_mode))
            data[mask] = np.array(vals)
        return data

    def get_channel_ids(self, to_channels, from_channels=None):
        """
        for a list of channels, return their ids in mpp_data

        Args:
            from_channels: get channel_ids assuming ordering of from_channels.
                This is useful for models that are trained with different output than input channels
        """
        if from_channels is None:
            from_channels = self.channels.copy()
            from_channels = from_channels.reset_index().set_index("name")
        if not isinstance(from_channels, pd.DataFrame):
            from_channels = pd.DataFrame(
                {"name": from_channels, "channel_id": np.arange(len(from_channels))}
            )
            from_channels = from_channels.set_index("name")
        from_channels = from_channels.reindex(to_channels)
        return list(from_channels["channel_id"])

    # --- image plotting (only for non-subsampled MPPData) ---
    @staticmethod
    def _get_img_from_data(x, y, data, img_size=None, pad=0):
        """
        Create image from x and y coordinates and fill with data.
        Args:
            x, y: 1-d array of x and y corrdinates
            data: shape (n_coords,n_channels), data for the image
            img_size: size of returned image
            pad: amount of padding added to returned image (only used when img_size is None)
        """
        x_coord = x - x.min() + pad
        y_coord = y - y.min() + pad
        # create image
        img = np.zeros(
            (y_coord.max() + 1 + pad, x_coord.max() + 1 + pad, data.shape[-1]),
            dtype=data.dtype,
        )
        img[y_coord, x_coord] = data
        # resize
        if img_size is not None:
            c = BoundingBox(0, 0, img.shape[0], img.shape[1]).center
            bbox = BoundingBox.from_center(c[0], c[1], img_size, img_size)
            img = bbox.crop(img)
            return img
        else:
            # padding info is only relevant when not cropping img to shape
            return img, (x.min() - pad, y.min() - pad)

    def get_object_img(
        self, obj_id, data="mpp", channel_ids=None, annotation_kwargs=None, **kwargs
    ):
        """
        Calculate data image of given object id.
        args:
            obj_id: an ID of the object (cell)
            data: key in self._data that should be plotted on the image
            channel_ids: only if key == 'mpp'. Channels that the image should have.
                If None, all channels are returned.
            annotation_kwargs: arguments for self.annotate_img. (annotation, to_col, color):
                img_size: size of returned image. If provided, image is padded or cropped to the desired size,
                    and returns resulting image.
                    If not provided, returns image and pad parameters from padding (pad set to 0 by default)
                pad: amount of padding added to returned image (only used when img_size is None). If  provided,
                 returns image and a list of padding parameters used for image padding.
            kwargs: arguments for self._get_img_from_data

        """
        mask = self.obj_ids == obj_id
        x = self.x[mask]
        y = self.y[mask]
        if data == "mpp":
            values = self.center_mpp[mask]
            if channel_ids is not None:
                values = values[:, channel_ids]
        else:
            values = self._data[data][mask]
        if len(values.shape) == 1:
            values = values[:, np.newaxis]
        # color image if necessary
        img = MPPData._get_img_from_data(x, y, values, **kwargs)
        if annotation_kwargs is not None:
            if isinstance(img, tuple):  # have padding info in img
                img_ = annotate_img(img[0], from_col=data, **annotation_kwargs)
                img = (img_, img[1])
            else:
                img = annotate_img(img, from_col=data, **annotation_kwargs)
        return img

    def get_object_imgs(
        self, data="mpp", channel_ids=None, annotation_kwargs=None, **kwargs
    ):
        """
        Return images for each obj_id in current data.
        Args: arguments for get_object_img
        """
        imgs = []
        for obj_id in self.metadata[self.data_config.OBJ_ID]:
            res = self.get_object_img(
                obj_id=obj_id,
                data=data,
                channel_ids=channel_ids,
                annotation_kwargs=annotation_kwargs,
                **kwargs,
            )
            if kwargs.get("img_size", None) is None:
                # fn also returns padding info, which we don't need here
                res = res[0]
            imgs.append(res)
        return imgs

    def get_img(self, data="mpp", channel_ids=None, **kwargs):
        """
        Calculate data image of all mpp data.
        data: key in self._data that should be plotted on the image
        channel_ids: only if key == 'mpp'. Channels that the image should have.
            If None, all channels are returned.
        kwargs: arguments for self._get_img_from_data
        """
        if data == "mpp":
            values = self.center_mpp
            if channel_ids is not None:
                values = values[:, channel_ids]
        else:
            values = self._data[data]
        if len(values.shape) == 1:
            values = values[:, np.newaxis]
        return MPPData._get_img_from_data(self.x, self.y, values, **kwargs)

    def compare(self, obj):

        assert np.array_equal(list(self._data.keys()), list(obj._data.keys()))
        same_data = {}
        for key in self._data.keys():
            issame = np.array_equal(self._data[key], obj._data[key])
            same_data[key] = issame

        same_data["channels"] = self.channels.equals(obj.channels)
        if len(self.metadata) == len(obj.metadata):
            same_data["metadata"] = self.metadata.reset_index(drop=True).equals(
                obj.metadata.reset_index(drop=True)
            )
        else:
            same_data["metadata"] = False
        return np.all(list(same_data.values())), same_data


def _try_mmap_load(fname, mmap_mode="r", allow_not_existing=False):
    """
    use np.load to read fname. If mmap loading fails, load with mmap_mode=None.

    Args:
        allow_not_existing: if fname does not exist, do not raise and exception, just return None

    Returns:
        np.ndarray: loaded numpy array
    """
    try:
        try:
            res = np.load(fname, mmap_mode=mmap_mode)
        except ValueError as e:
            if "Array can't be memory-mapped: Python objects in dtype." in str(e):
                print("Cannot read with memmap: ", fname)
                res = np.load(fname, mmap_mode=None, allow_pickle=True)
            else:
                raise e
    except FileNotFoundError as e:
        if allow_not_existing:
            return None
        else:
            raise e
    return res


def _get_keys(keys, optional_keys, base_mpp_data=None):
    """
    keys and optional keys to use for loading base_mpp_data / mpp_data,
    when reading parts of the data from other dirs
    """
    if base_mpp_data is None:
        # loading of base_mpp_data
        # return all keys as optional, as might still be present in mpp_data_dir
        res_keys = ["x", "y", "obj_ids"]
        res_optional_keys = list(set(optional_keys).union(keys).difference(res_keys))
    else:
        # loading of mpp_data
        # check which of required keys are already loaded, and move them to optional
        res_keys = ["x", "y", "obj_ids"]
        res_optional_keys = copy(optional_keys)
        for k in list(set(keys).difference(res_keys)):
            if base_mpp_data.data(k) is not None:
                res_optional_keys.append(k)
            else:
                res_keys.append(k)
    return res_keys, res_optional_keys
