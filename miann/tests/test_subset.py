import pytest

from miann.tests.helpers import gen_mppdata
import numpy as np
# content of test_subset.py
import logging
import os
LOGGER = logging.getLogger(__name__)


class TestSubset:
    def test_different_nans(self):
        pass

    @pytest.mark.parametrize("frac", (np.linspace(0.1, 0.9, 5)))
    @pytest.mark.parametrize("num_objects", np.arange(20, 200, 50))
    def test_random_fraction_nonemptyRes(self, frac, num_objects):
        mpp_data = gen_mppdata(num_obj_ids=num_objects)
        # fracs=np.random.rand(20)

        mpp_data_sub = mpp_data.subset(frac=frac, copy=True)
        assert np.isclose(len(mpp_data_sub.unique_obj_ids)/len(mpp_data.unique_obj_ids), frac), f"test failed for frac {frac}"

    # ToDo: test_random_fraction_emptyRes
    # # @pytest.mark.parametrize("frac", (np.linspace(0.1, 0.9, 5)))
    # @pytest.mark.parametrize("num_objects", np.arange(20, 200, 50))
    # def test_random_fraction_emptyRes(self, num_objects):
    #     mpp_data = gen_mppdata(num_obj_ids=num_objects)
    #     frac=1/num_objects-1e-2
    #     # fracs=np.random.rand(20)
    #
    #     mpp_data_sub = mpp_data.subset(frac=frac, copy=True)
    #     caplog.set_level(logging.WARNING):
    #     run_function()
    #     assert 'Something bad happened!' in caplog.text
    #
    #     assert np.isclose(len(mpp_data_sub.unique_obj_ids)/len(mpp_data.unique_obj_ids), frac), f"test failed for frac {frac}"

    def test_metadata_keys(self):
        pass

    @pytest.mark.parametrize("num_ids", np.arange(1, 5))
    def test_existing_obj_ids(self, num_ids):
        num_objects=10
        mpp_data = gen_mppdata(num_obj_ids=num_objects)
        ids=np.random.randint(0, num_objects-1, num_ids)
        mpp_data_sub = mpp_data.subset(obj_ids=ids, copy=True)
        unique_ids=list(set(ids))
        assert np.array_equal(np.sort(mpp_data_sub.unique_obj_ids), np.sort(unique_ids))


    @pytest.mark.parametrize("num_objects", np.arange(1, 10))
    def test_all_existing_obj_ids(self, num_objects):
        mpp_data = gen_mppdata(num_obj_ids=num_objects)
        ids=np.arange(0, num_objects)
        mpp_data_sub = mpp_data.subset(obj_ids=ids, copy=True)
        assert np.array_equal(np.sort(mpp_data_sub.unique_obj_ids), np.sort(ids))

    @pytest.mark.parametrize("num_objects", np.arange(1, 10))
    def test_nonExisting_obj_ids(self, num_objects):
        mpp_data = gen_mppdata(num_obj_ids=num_objects)
        ids=[num_objects+10]
        mpp_data_sub = mpp_data.subset(obj_ids=ids, copy=True)
        assert len(mpp_data_sub.unique_obj_ids)==0
        assert len(mpp_data_sub.x) == 0
        assert len(mpp_data_sub.y) == 0
        assert len(mpp_data_sub.mpp) == 0

if __name__ == '__main__':

    pytest.main(args=[os.path.abspath(__file__)])

