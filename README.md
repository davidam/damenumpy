
# Table of Contents

1.  [Check Test](#org569e0c3)
2.  [Pypi](#org290af36)
    1.  [You can install from Internet in a python virtual environment to check:](#org8bb217d)
    2.  [To install from local:](#orgab790a6)
    3.  [To install create tar.gz in dist directory:](#org4e104bb)
    4.  [To upload to pypi:](#orgbb19390)

Learning Numpy from Tests by David Arroyo Menéndez


<a id="org569e0c3"></a>

# Check Test

-   Execute all tests:

    $ pytest tests

-   Execute one file:

    $ pytest tests/test_basics.py

-   Execute one test:

    $ pytest tests/test_basics.py:TestBasics.test_indexing


<a id="org290af36"></a>

# Pypi


<a id="org8bb217d"></a>

## You can install from Internet in a python virtual environment to check:

    $ python3 -m venv /tmp/funny
    $ cd /tmp/funny
    $ source bin/activate
    $ pip3 install damenumpy


<a id="orgab790a6"></a>

## To install from local:

    $ pip install -e .


<a id="org4e104bb"></a>

## To install create tar.gz in dist directory:

    $ python3 -m build


<a id="orgbb19390"></a>

## To upload to pypi:

    $ twine upload dist/damenumpy-0.1.tar.gz

