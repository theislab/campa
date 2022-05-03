Contributing guide
~~~~~~~~~~~~~~~~~~

Contributing to CAMPA
---------------------
Clone CAMPA from source as::

    git clone https://github.com/theislab/campa
    cd campa

Install the test and development mode::

    pip install -e'.[dev,test]'

Optionally install pre-commit. This will ensure that
the pushed code passes the linting steps::

    pre-commit install

Although the last step is not necessary, it is highly recommended,
since it will help you to pass the linting step
(see `Code style guide`_). If you did install ``pre-commit``
but are unable in deciphering some flags, you can
still commit using the ``--no-verify``.

To build documentation for ipython notebooks, you will also have to install
`pandoc <https://pandoc.org/installing.html>`_. This is possible for example using conda::

    conda install -c conda-forge pandoc


Code style guide
----------------
We rely on ``black`` and ``isort`` to do the most of the formatting
- both of them are integrated as pre-commit hooks.
We use ``flake8`` and ``mypy`` to further analyze the code.
Use ``# noqa: <error1>,<error2>`` to ignore certain ``flake8`` errors and
``# type: ignore[error1,error2]`` to ignore specific ``mypy`` errors.

You can use ``tox`` to check the changes::

    tox -e lint


Testing
-------
We use ``tox`` to automate our testing, as well as linting and
documentation creation.
To run the tests, run::

    tox -e py39

If needed, recreate the ``tox`` environment::

    tox -e py39 --recreate

Writing documentation
---------------------
We use ``numpy``-style docstrings for the documentation.

In order to build the documentation, run::

    tox -e docs

To validate the links inside the documentation, run::

    tox -e check-docs

If you need to clean the artifacts from previous documentation builds, run::

    tox -e clean-docs

Creating a relase
-----------------

Before creating a new relase, ensure that all tests pass and that the documentation
is build successfully.
For easy uploading to pypi set up a `token <https://test.pypi.org/help/#apitoken>`_ in
`~/.pypirc <https://truveris.github.io/articles/configuring-pypirc/>`_
and ensure that for TestPyPi a ``repository-url`` is set.

1. Create a new branch for this release::

    git checkout -b release/VERSION

2. Check that `README_pypi.rst <README_pypi.rst>`_ renders correctly on pypi::

    tox -e readme

3. Create a new version using ``bump2version`` (``pip install bump2version``)::

    bump2version {major,minor,patch}

4. Build and test the package::

    rm -r dist
    python setup.py sdist
    twine check dist/*

   Check out the created ``*.tar`` file in ``dist`` and ensure that all new auxiliary files
   needed for this relase are included. Otherwise, include them in `MANIFEST.in <MANIFEST.in>`_
   and try again.

5. Optional: 

    1. Upload to TestPyPi::

        twine upload --repository pypi dist/*

    2. Install and test package from TestPyPi::

        pip install -i https://test.pypi.org/simple/ campa 

6. Release package on PyPi (``pip install twine``)::

    twine upload dist/*

7. Merge release branch to main::

    git co main
    git merge release/VERSION

8. Create relase on github by adding a tag::

    git tag VERSION
    git push --tags

   Edit and publish the relase on github and add release notes.

9. Readthedocs should be updated automatically once the new tag is pushed.
