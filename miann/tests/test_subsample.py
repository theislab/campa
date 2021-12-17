import pytest

from miann.tests.helpers import gen_mppdata
import numpy as np
# content of test_subset.py
import logging
import os
import math
LOGGER = logging.getLogger(__name__)

class TestSubset:
    @pytest.mark.parametrize("frac", (np.linspace(0.1, 0.9, 3)))
    @pytest.mark.parametrize("num_objects", np.arange(10, 200, 70))
    def test_random_frac_nonemptyRes(self, frac, num_objects):
        mpp_data = gen_mppdata(num_obj_ids=num_objects)
        # fracs=np.random.rand(20)
        mpp_data_sub = mpp_data.subsample(frac=frac)
        assert math.isclose(len(mpp_data_sub.mpp) / len(mpp_data.mpp),
                          frac, rel_tol=0.05), f"test failed for frac {frac}"

    @pytest.mark.parametrize("frac_per_object", np.linspace(0.1, 0.9, 3))
    def test_random_frac_per_obj_nonemptyRes(self, frac_per_object):
        mpp_data = gen_mppdata(num_obj_ids=6)
        # fracs=np.random.rand(20)
        mpp_data_sub = mpp_data.subsample(frac_per_obj=frac_per_object)
        for obj_id in mpp_data_sub.unique_obj_ids:
            assert math.isclose(len(mpp_data_sub.subset(obj_ids=[obj_id], copy=True).mpp) / len(mpp_data.subset(obj_ids=[obj_id], copy=True).mpp),
                                frac_per_object, rel_tol=0.05, abs_tol=0.05), f"test failed for frac {frac_per_object}"



    @pytest.mark.parametrize("num_ids", np.arange(1,10,5))
    def test_frac_1(self, num_ids):
        mpp_data = gen_mppdata(num_obj_ids=num_ids)
        mpp_data_sub = mpp_data.subsample(frac=1)
        assert np.array_equal(mpp_data.mpp.shape, mpp_data_sub.mpp.shape)
        assert np.array_equal(list(mpp_data._data.keys()), list(mpp_data_sub._data.keys()))


    def test_frac_per_obj_1(self):
        mpp_data = gen_mppdata(num_obj_ids=5)
        mpp_data_sub = mpp_data.subsample(frac_per_obj=1)
        assert np.array_equal(mpp_data.mpp.shape, mpp_data_sub.mpp.shape)
        assert np.array_equal(list(mpp_data._data.keys()), list(mpp_data_sub._data.keys()))


    def test_frac_0(self):
        mpp_data = gen_mppdata(num_obj_ids=5)
        mpp_data_sub = mpp_data.subsample(frac=0)

        assert len(mpp_data_sub.unique_obj_ids)==0
        assert len(mpp_data_sub.x) == 0
        assert len(mpp_data_sub.y) == 0
        assert len(mpp_data_sub.mpp) == 0


    @pytest.mark.parametrize("frac", (np.linspace(0.1, 0.9, 3)))
    def test_frac_same_seed(self, frac):
        mpp_data = gen_mppdata(num_obj_ids=5)
        mpp_data_orig=mpp_data.copy()
        mpp_data_subset_1=mpp_data.subsample(frac=frac)
        mpp_data_subset_2=mpp_data_orig.subsample(frac=frac)

        isequal, isequal_dict=mpp_data_subset_1.compare(mpp_data_subset_2)
        assert isequal

    @pytest.mark.parametrize("frac", (np.linspace(0.1, 0.9, 3)))
    def test_frac_per_obj_same_seed(self, frac):
        mpp_data = gen_mppdata(num_obj_ids=5)
        mpp_data_orig = mpp_data.copy()
        mpp_data_subset_1 = mpp_data.subsample(frac_per_obj=frac)
        mpp_data_subset_2 = mpp_data_orig.subsample(frac_per_obj=frac)

        isequal, isequal_dict = mpp_data_subset_1.compare(mpp_data_subset_2)
        assert isequal
