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

TODO build documentation with tox
