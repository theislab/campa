CAMPA - Conditional Autoencoder for Multiplexed Pixel Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CAMPA is a framework for quantiative analysis of subcellular multi-channel imaging data.
It consists of a workflow that generates consistent subcellular landmarks (CSLs)
using conditional Variational Autoencoders (cVAE).
The output of the CAMPA workflow is an `anndata`_ object that contains interpretable
per-cell features summarizing the molecular composition
and spatial arrangement of CSLs inside each cell.

.. image:: https://raw.githubusercontent.com/theislab/campa/master/docs/source/_static/img/Figure1ab.jpg
    :alt: CAMPA title figure
    :width: 400px
    :align: center
    :target: .

Visit our `documentation`_ for installation and usage examples.


Manuscript
----------
Please see our preprint 
*"Quantifying condition-dependent changes in subcellular organisation using multiplexed protein maps"* 
(`Spitzer, Berry et al. (2022)`_) to learn more.


Installation
------------

Install CAMPA by running::

    pip install campa

Setup `data and experiment paths`_ with::

    campa setup


Contributing
------------
We are happy about any contributions! Before you start, check out our `contributing guide`_.

.. _anndata: https://anndata.readthedocs.io/en/stable/
.. _documentation: https://campa.readthedocs.io/en/stable/
.. _`data and experiment paths`: https://campa.readthedocs.io/en/stable/overview.html#campa-config
.. _`Spitzer, Berry et al. (2022)`: .
.. _contributing guide: https://github.com/theislab/campa/blob/main/CONTRIBUTING.rst
