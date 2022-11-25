Overview
========

CAMPA is a framework for analysis of subcellular multi-channel imaging data.
It consists of a workflow that generates consistent subcellular landmarks (CSLs)
and several plotting functions. The output of the CAMPA workflow is a :class:`anndata.AnnData` object
that contains per-cell features about the molecular composition and spatial arrangement
of CSLs inside each cell.

CAMPA both contains a high-level API for easy access to the workflow, a low-level class-based access
to all functions for more detailed control, and a CLI for all steps except the manual cluster annotation step.
Depending on your data size, you might want to run long-running steps of the CAMPA workflow on a HPC cluster.

.. _campa-config:

CAMPA config
------------

To find data and experiment folders, and dataset specific parameters,
CAMPA uses a config, :attr:`campa.constants.campa_config`.
Config values can directly be set in memory. For more information on
this see the `Setup and download data tutorial`_.

In order to provide a persistent config, CAMPA attempts to read the
config values from a ``campa.ini`` config file
from the following locations:

- the current directory
- ``$HOME/.config/campa``

and will use the first file that it finds.

There is an example config file in
`campa/campa.ini.example <https://github.com/theislab/campa/blob/main/campa/campa.ini.example>`_.
Create ``campa.ini`` with system-specific paths to experiment and
data folders in by running::

    campa setup

This creates a default config in ``$HOME/.config/campa/campa.ini``.
Note that for following the tutorials it is not necessary complete this step and write a config file to disk,
as all tutorials set the necessary config values at the start.


Data format
-----------

The pixel-level multi-channel data is represented as a :class:`campa.data.MPPData`
(Multiplexed Pixel Profile Data) object.

It is represented on disk by a folder containing ``npy`` and ``csv`` files.
Let :math:`n_o` the number of objects (cells) in the data,
:math:`n_p` the number of pixels, and :math:`m` the number of measurements per pixel
(e.g. number of protein channels), the data is saved in:

- ``mpp.npy``: :math:`n_p \times m`, per-pixel values
- ``x.npy, y.npy``: :math:`n_p \times 1`, spatial coordinates of each pixel
- ``obj_ids.npy``: :math:`n_p \times 1`, object id (cell) that each pixel belongs to
- ``channels.csv``: :math:`m`, protein channel names
- ``metadata.csv``: :math:`n_o`, metadata information (perturbation, cell cycle, etc) for each object (cell)

During processing of the data, additional numpy files might be created.
For example, after training a cVAE with :math:`c` conditions and clustering its latent space
of dimension :math:`l` into :math:`k` CSLs, the results directory will also contain:

- ``conditions.npy``: :math:`n_p \times c`, conditions used to train the cVAE
- ``latent.npy``: :math:`n_p \times l`, latent space of the cVAE used for clustering CSLs
- ``clustering.npy``: :math:`n_p \times k`, clustering of ``latent.npy``
- ``cluster_annotation.csv``: :math:`k`, mapping of clusters to cluster names

For more information on the data representation, see also the `Data representation tutorial`_.
More data formats, including reading data from (multiple) image files will be supported in the future.


Data config
-----------

To tell CAMPA how to load specific data and set up the conditions contained in specific datasets,
a ``data_config`` is used. All available data configs are in :attr:`campa.constants.CAMPAConfig.data_configs`.
E.g., for the example data used in the tutorials, the ``data_config`` is identified by ``ExampleData``
and points to the `example data config file`_.

The data config file is specific per dataset but has to contain the following variables:

- ``DATA_DIR``: path to the folder containing the data, can use
  :attr:`campa.constants.CAMPAConfig.BASE_DATA_DIR` to be device-agnostic.
- ``DATASET_DIR``: path to the folder where training/testing datasets derived from this data should be stored;
- ``OBJ_ID``: name of column in ``metadata.csv`` that contains a unique object identifier.
- ``CHANNELS_METADATA``: name of CSV file containing channels metadata (relative to ``DATA_DIR``).
  Is expected to contain channel names in column "name".
- ``CONDITIONS``: dictionary of conditions to be used for cVAE models.
  Keys are column names in ``metadata.csv``, and values are all possible values for this condition.
  This will be used to convert conditions to one-hot encoded vector.

For using a different dataset, simply create a new constants file with dataset specific settings,
and add a new entry to :attr:`campa.constants.campa_config` using :func:`campa.constants.CAMPAConfig.add_data_config`.

.. _workflow:

Workflow
--------

CAMPA contains a high-level API that can be easily used to create datasets, train models, and extract features.
Settings for the different stages of the workflow are communicated via parameter files.
These are python files usually containing a dictionary of settings that are used by the individual steps.
You can find a complete set of example parameter files `here <https://github.com/theislab/campa/tree/main/notebooks/params>`_.

The workflow consists of the following steps:

- Setup up the config and download data by following along with the `Setup and download data tutorial`_.

- Create a subsampled pixel-level dataset for neural network training.
  This is done either by using the API function :func:`campa.data.create_dataset` or by using the CLI::

    campa create_dataset ...

  For more information, see the `Dataset for training models tutorial`_.

- Train a conditional variational autoencoder to generate a condition-independent latent representation.
  This is done either by using the API function :func:`campa.tl.run_experiments` or by using the CLI::

    campa train ...

  For more information, see the `Train and evaluate models tutorial`_.

- Cluster cVAE latent representation into CSLs.
  This is done in three steps:

    - First, the data is subsampled and clustered, because we would like the clustering
      to be interactive and feasible to compute on a laptop.
      If you have more time or access to GPUs, you could also consider to skip the subsampling
      step and cluster all data directly.
      Use the API function :func:`campa.tl.create_cluster_data` or the CLI::

        campa cluster <EXPERIMENT> create ...

      Optionally, after this step a manual re-clustering or annotation of clusters can be done.
      See the `Cluster data into CSLs tutorial`_ for more details

    - To project the clustering to the entire dataset, the model needs to be used to predict the
      latent representation on all data.
      It is recommended to run this step in a script, as this might take a while for large datasets.
      Use the API function :func:`campa.tl.prepare_full_dataset` or the CLI::

        campa cluster <EXPERIMENT> prepare-full ...

    - Finally, the clustering can be projected to the entire dataset.
      Use the API function :func:`campa.tl.project_cluster_data` or the CLI::

        campa cluster <EXPERIMENT> project ...

  For more information, see the `Cluster data into CSLs tutorial`_.

- Extract features from CSLs to quantitatively compare molecular intensity differences and
  spatial re-localisation of proteins in different conditions.
  Use the API function :func:`campa.tl.extract_features` or the CLI::

    campa extract_features ...

  For more information, see the `Extract features from CSLs tutorial`_.

.. _Data representation tutorial: notebooks/mpp_data.ipynb
.. _Setup and download data tutorial: notebooks/setup.ipynb
.. _example data config file: https://github.com/theislab/campa/blob/main/notebooks/params/ExampleData_constants.py
.. _Dataset for training models tutorial: notebooks/nn_dataset.ipynb
.. _Train and evaluate models tutorial: notebooks/train.ipynb
.. _Cluster data into CSLs tutorial: notebooks/cluster.ipynb
.. _Extract features from CSLs tutorial: notebooks/extract_features.ipynb
