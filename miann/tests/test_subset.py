import pytest

from miann.tests.helpers import gen_mppdata
import numpy as np
# content of test_subset.py
import logging
import os
LOGGER = logging.getLogger(__name__)


class TestSubset:

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

    @pytest.mark.parametrize("num_ids", np.arange(1,10,5))
    def test_frac_1(self, num_ids):
        mpp_data = gen_mppdata(num_obj_ids=num_ids)
        mpp_data_sub = mpp_data.subset(frac=1, copy=True)
        isequal, isequal_dict=mpp_data.compare(mpp_data_sub)
        assert isequal

    def test_frac_trueCopy_flag(self):
        mpp_data = gen_mppdata(num_obj_ids=10)
        mpp_data_orig=mpp_data.copy()
        mpp_data_sub = mpp_data.subset(frac=0.5, copy=True)
        isequal, isequal_dict=mpp_data.compare(mpp_data_orig)
        assert isequal
        assert isequal_dict["channels"]

    def test_frac_falseCopy_flag_(self):
        num_objects_orig_intended=10
        mpp_data = gen_mppdata(num_obj_ids=num_objects_orig_intended)
        mpp_data_orig=mpp_data.copy()
        num_objects_orig=len(mpp_data_orig.unique_obj_ids)
        mpp_data.subset(frac=0.5, copy=False)
        isequal, isequal_dict=mpp_data.compare(mpp_data_orig)
        assert isequal==False
        assert num_objects_orig_intended==num_objects_orig==len(mpp_data_orig.unique_obj_ids)
        assert num_objects_orig!=len(mpp_data.unique_obj_ids)

    @pytest.mark.parametrize("frac", (np.linspace(0.1, 0.9, 10)))
    def test_frac_same_seed(self, frac):
        num_objects_orig_intended=10
        mpp_data = gen_mppdata(num_obj_ids=num_objects_orig_intended)
        mpp_data_orig=mpp_data.copy()
        mpp_data_subset_1=mpp_data.subset(frac=frac, copy=True)
        mpp_data_subset_2=mpp_data.subset(frac=frac, copy=True)

        isequal, isequal_dict=mpp_data_subset_1.compare(mpp_data_subset_2)
        assert isequal

    # @pytest.mark.parametrize("frac", (np.linspace(0.1, 0.9, 10)))
    # def test_frac_diff_seed(self, frac):
    #     num_objects_orig_intended = 10
    #
    #     mpp_data = gen_mppdata(num_obj_ids=num_objects_orig_intended)
    #     mpp_data_orig = mpp_data.copy()
    #     mpp_data_subset_1 = mpp_data.subset(frac=frac, copy=True)
    #
    #     mpp_data.seed=mpp_data.seed+2
    #     mpp_data_subset_2 = mpp_data.subset(frac=frac, copy=True)
    #
    #     isequal, isequal_dict = mpp_data_subset_1.compare(mpp_data_subset_2)
    #     assert isequal==False

    def test_subset_by_metadataKey_selectAll(self):
        cell_cycle=[str(i) for i in range(5)]
        mpp_data = gen_mppdata(num_obj_ids=20, possible_cell_cycles=cell_cycle)
        cell_cycle=list(mpp_data.metadata.cell_cycle.unique())
        mpp_data_subset = mpp_data.subset(cell_cycle=cell_cycle, copy=True)

        isequal, isequal_dict = mpp_data.compare(mpp_data_subset)
        assert isequal

    @pytest.mark.parametrize("num_keys", np.arange(1,4))
    def test_subset_by_metadataKey_differentKeys(self, num_keys):
        cell_cycle=[str(i) for i in range(5)]
        mpp_data = gen_mppdata(num_obj_ids=20, possible_cell_cycles=cell_cycle)
        cell_cycle=list(mpp_data.metadata.cell_cycle.unique())[:num_keys]
        mpp_data_subset = mpp_data.subset(cell_cycle=cell_cycle, copy=True)
        isequal, isequal_dict = mpp_data.compare(mpp_data_subset)
        assert isequal==False
        assert np.array_equal(mpp_data_subset.metadata.cell_cycle.unique(),cell_cycle)

    def test_subset_by_metadataKey_unknownKey(self):
        mpp_data = gen_mppdata()
        new_key="new_key"
        with pytest.raises(AssertionError) as exc:
            mpp_data_copy = mpp_data.subset(new_key=['1', '2'], copy=True)
        assert f'provided column {new_key} was not found in the metadata table!' in str(exc.value)


    def test_subset_by_metadataKey_NO_NAN_woNone(self):
        cell_cycle = [str(i) for i in range(5)]
        mpp_data = gen_mppdata(num_obj_ids=20,  possible_cell_cycles=cell_cycle, ensure_None=False)

        mpp_data_subset = mpp_data.subset(cell_cycle='NO_NAN', copy=True)
        isequal, isequal_dict = mpp_data.compare(mpp_data_subset)
        assert isequal==True
        assert not mpp_data_subset.metadata.cell_cycle.isnull().values.any()
        assert (None not in mpp_data_subset.metadata.cell_cycle.unique())

    def test_subset_by_metadataKey_NO_NAN(self):
        mpp_data = gen_mppdata(num_obj_ids=20,  ensure_None=True)

        mpp_data_subset = mpp_data.subset(cell_cycle='NO_NAN', copy=True)
        isequal, isequal_dict = mpp_data.compare(mpp_data_subset)
        assert isequal==False
        assert not mpp_data_subset.metadata.cell_cycle.isnull().values.any()
        assert (None not in mpp_data_subset.metadata.cell_cycle.unique())


class TestSubsetChannels:
    def test_nonExistingChannels(self):
        channels=[str(i) for i in range(5)]
        mpp_data = gen_mppdata(num_obj_ids=20,channels=channels)
        channels=str(len(channels)+1)
        with pytest.raises(AssertionError) as exc:
            mpp_data.subset_channels(channels)
        assert f'mpp object does not contain provided channels!' in str(exc.value)




if __name__ == '__main__':

    pytest.main(args=[os.path.abspath(__file__)])

