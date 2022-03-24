API
===

High-level functions for processing data, training and evaluating models and plotting results.

Data
~~~~

.. module:: campa.data
.. currentmodule:: campa

.. autosummary::
    :toctree: api

    data.create_dataset

Tools
~~~~~

.. module:: campa.tl
.. currentmodule:: campa

.. autosummary::
    :toctree: api

    tl.run_experiments
    tl.create_cluster_data
    tl.prepare_full_dataset
    tl.project_cluster_data
    tl.extract_features

Plotting
~~~~~~~~

.. module:: campa.pl
.. currentmodule:: campa

.. autosummary::
    :toctree: api

    pl.plot_mean_intensity
    pl.plot_intensity_change
    pl.plot_mean_size
    pl.plot_size_change
    pl.plot_object_stats
    pl.plot_co_occurrence
    pl.plot_co_occurrence_grid
    pl.annotate_img
