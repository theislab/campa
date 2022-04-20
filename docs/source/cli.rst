CLI
===

Command line interface for CAMPA.

Use this to execute long-running steps of the CAMPA workflow 
(e.g. clustering large data or calculating co-ocurrence features) in a script on e.g. a HPC system.

See the :ref:`workflow` documentation for a short introduction to the CLI commands and the :ref:`tutorials` for the 
CLI commands needed to reproduce the example workflow.

Basic usage
-----------

.. argparse::
    :module: campa.cli.main
    :func: _get_base_parser
    :prog: campa

setup
-----

.. argparse::
    :module: campa.cli.main
    :func: _get_setup_parser
    :prog: campa setup

create_dataset
--------------

.. argparse::
    :module: campa.cli.main
    :func: _get_create_dataset_parser
    :prog: campa create_dataset

train
-----

.. argparse::
    :module: campa.cli.main
    :func: _get_train_parser
    :prog: campa train

cluster
-------

.. argparse::
    :module: campa.cli.main
    :func: _get_cluster_parser
    :prog: campa cluster

extract_features
----------------

.. argparse::
    :module: campa.cli.main
    :func: _get_extract_features_parser
    :prog: campa extract_features

