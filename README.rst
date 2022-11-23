CAMPA - Conditional Autoencoder for Multiplexed Pixel Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CAMPA is a framework for quantiative analysis of subcellular multi-channel imaging data.
It consists of a workflow that generates consistent subcellular landmarks (CSLs)
using conditional Variational Autoencoders (cVAE).
The output of the CAMPA workflow is an `anndata`_ object that contains interpretable
per-cell features summarizing the molecular composition
and spatial arrangement of CSLs inside each cell.

.. raw:: html

    <p align="center">
        <a href="https://www.biorxiv.org/content/10.1101/2022.05.07.490900v1">
            <img src="https://raw.githubusercontent.com/theislab/campa/main/docs/source/_static/img/Figure1ab.jpg"
             width="600px" alt="CAMPA title figure">
        </a>
    </p>

Visit our `documentation`_ for installation and usage examples.


Manuscript
----------
Please see our preprint
*"Learning consistent subcellular landmarks to quantify changes in multiplexed protein maps"*
(`Spitzer, Berry et al. (2022)`_) to learn more.


Installation
------------

Install CAMPA by running::

    pip install campa


Contributing
------------
We are happy about any contributions! Before you start, check out our `contributing guide <CONTRIBUTING.rst>`_.

.. _anndata: https://anndata.readthedocs.io/en/stable/
.. _documentation: https://campa.readthedocs.io/en/stable/
.. _`data and experiment paths`: https://campa.readthedocs.io/en/stable/overview.html#campa-config
.. _`Spitzer, Berry et al. (2022)`: https://www.biorxiv.org/content/10.1101/2022.05.07.490900v1
