
# Table of Contents

1.  [Check Test](#orgf7e42f0)
2.  [Pypi](#orge94fd50)
    1.  [You can install from Internet in a python virtual environment to check:](#org01e63af)
    2.  [To install from local:](#org6fb219a)
    3.  [To install create tar.gz in dist directory:](#orga3f9d4d)
    4.  [To upload to pypi:](#orgddd2aeb)

Learning Numpy from Tests by David Arroyo Menéndez


<a id="orgf7e42f0"></a>

# Check Test

-   Execute all tests:

    $ pytest tests

-   Execute one file:

    $ pytest tests/test_basics.py

-   Execute one test:

    $ pytest tests/test_basics.py::TestBasics::test_arange


<a id="orge94fd50"></a>

# Pypi


<a id="org01e63af"></a>

## You can install from Internet in a python virtual environment to check:

    $ mkdir /tmp/funny  
    $ python3.14 -m venv /tmp/funny
    $ cd /tmp/funny
    $ source bin/activate
    $ python3.14 -m pip install --upgrade pip  
    $ python3.14 -m pip install damenumpy 


<a id="org6fb219a"></a>

## To install from local:

    $ pip install -e .


<a id="orga3f9d4d"></a>

## To install create tar.gz in dist directory:

    $ python3 -m build


<a id="orgddd2aeb"></a>

## To upload to pypi:

    $ twine upload dist/damenumpy-0.1.tar.gz

