Discovery Behavioral Tools
#############################

This project looks to help in the building of tools that require data that has behavioral
characteristics.

.. class:: no-web no-pdf

|pypi| |rdt| |license| |wheel|


.. contents::

.. section-numbering::

Main features
=============

* Data Generation
* Data Correlation
* Feature association
* Characteristic analysis

Installation
============

package install
---------------

The best way to install this package is directly from the Python Package Index repository using pip

.. code-block:: bash

    $ pip install discovery-behavioral-utils

if you want to upgrade your current version then using pip

.. code-block:: bash

    $ pip install --upgrade discovery-behavioral-utils

env setup
---------
Other than the dependant python packages indicated in the ``requirements.txt`` there are
no special environment setup needs to use the package. The package should sit as an extension to
your current data science and discovery packages.

Using the Behavioral Synthetic Data Generator
=============================================

Package Structure
-----------------

Within the Discovery Transitioning Utils are a set
of\ ``simulator package`` that contains the DataBuilder,
DataBuilderPropertyManager and the DataBuilderTools class

DataBuilder
~~~~~~~~~~~

-  is a Data Builder management instance that allows the building of
   datasets to be repeatable by saving a configuration of the build
   definition

DataBuilderPropertyManager
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  manages the configuration property values and saves the build
   templates to regenerate the synthetic data

DataBuilderTools:
~~~~~~~~~~~~~~~~~

-  is a set of static methods that generate the different data types
   ``int``, ``float``, ``string``, ``category`` and ``date``. and define
   the randomness and patterns of the values.

Firstly we need to import the ``DataBuilder`` class and create a
**named** instance to identify this instance from other instances we
might create. Normally the name would be representative of the dataset
you are trying to create such as ``customer``, ``accounts`` or
``transactions`` as an example

.. code:: python

    from ds_discovery.simulators.data_builder import DataBuilder

.. code:: python

    builder = DataBuilder('SimpleExample')

Building a basic dataset
------------------------

with this example we will firstly look at the tools that are avaialbe
and produce a ``Pandas DataFrame`` on the fly

.. code:: python

    builder.tool_dir

.. parsed-literal::

    ['associate_custom',
     'associate_dataset',
     'correlate_categories',
     'correlate_dates',
     'correlate_numbers',
     'get_category',
     'get_column_csv',
     'get_custom',
     'get_datetime',
     'get_distribution',
     'get_names',
     'get_number',
     'get_string_pattern',
     'unique_date_seq',
     'unique_identifiers',
     'unique_numbers',
     'unique_str_tokens']

Here we can see the methods are broken down into four categories:
``get``, ``unique``, ``correlate``, ``associate``.

We can also look at the contextual help for each of the methods calling
the ``tools`` property and using the ``help`` build-in

.. code:: python

    help(builder.tools.get_number)

.. parsed-literal::

    Help on function get_number in module ds_discovery.simulators.data_builder:
    
    get_number(to_value: , from_value: = None, weight_pattern: list = None, precision: int = None, size: int = None,
               quantity: float = None, seed: int = None)
        returns a number in the range from_value to to_value. if only to_value given from_value is zero
        
        **:param to_value:** highest integer value, if from_value provided must be one above this value
        **:param from_value:** optional, (signed) integer to start from. Default is zero (0)
        **:param weight_pattern:** a weighting pattern or probability that does not have to add to 1
        **:param precision:** the precision of the returned number. if None then assumes int value else float
        **:param size:** the size of the sample
        **:param quantity:** a number between 0 and 1 representing data that isn't null
        **:param seed:** a seed value for the random function: default to None
        **:return:** a random number
    
From here we can now play with some of the ``get`` methods

.. code:: python

    # get an integer between 0 and 9
    builder.tools.get_number(10, size=5)

.. parsed-literal::

    **$>** [6, 5, 3, 2, 3]

.. code:: python

    # get a float between -1 and 1, notice by passing an float it assumes the output to be a float
    builder.tools.get_number(from_value=-1.0, to_value=1.0, precision=3, size=5)

.. parsed-literal::

    **$>** [0.283, 0.296, -0.958, 0.185, 0.831]

.. code:: python

    # get a currency by setting the 'currency' parameter to a currency symbol.
    # Note this returns a list of strings
    builder.tools.get_number(from_value=1000.0, to_value=2000.0, size=5, currency='$', precision=2)

.. parsed-literal::

    **$>** ['$1,286.00', '$1,858.00', '$1,038.00', '$1,944.00', '$1,250.00']

.. code:: python

    # get a timestamp between two dates
    builder.tools.get_datetime(start='01/01/2017', until='31/12/2018')

.. parsed-literal::

    **$>** [Timestamp('2018-02-11 02:23:32.733296768')]

.. code:: python

    # get a formated date string between two numbers
    builder.tools.get_datetime(start='01/01/2017', until='31/12/2018', size=4, date_format='%d-%m-%Y')

.. parsed-literal::

    **$>** ['06-06-2017', '05-11-2017', '28-09-2018', '04-11-2017']

.. code:: python

    # get categories from a selection
    builder.tools.get_category(['Red', 'Blue', 'Green', 'Black', 'White'], size=4)

.. parsed-literal::

    **$>** ['Green', 'Blue', 'Blue', 'White']

.. code:: python

    # get unique categories from a selection
    builder.tools.get_category(['Red', 'Blue', 'Green', 'Black', 'White'], size=4, replace=False)

.. parsed-literal::

    **$>** ['Blue', 'White', 'Green', 'Black']


Building a DataFrame
--------------------

With these lets build a quick Synthetic DataFrame. For ease of code we
will redefine the 'builder.tools' call

.. code:: python

    tools = builder.tools

.. code:: python

    # the dataframe has a unique id, a float value between 0.0 and 1.0and a date formtted as a text string
    df = pd.DataFrame()
    df['id'] = tools.unique_numbers(start=10, until=100, size=10)
    df['values'] = tools.get_number(to_value=1.0, size=10)
    df['date'] = tools.get_datetime(start='12/05/2018', until='30/11/2018', date_format='%d-%m-%Y %H:%M:%S', size=10)


Data quantity
~~~~~~~~~~~~~

to show representative data we can adjust the quality of the data we
produce. Here we only get about 50% of the telephone numbers

.. code:: python

    # using the get string pattern we can create part random and part static data elements. see the inline docs for help on customising choices
    df['mobile'] = tools.get_string_pattern("(07ddd) ddd ddd", choice_only=False, size=10, quantity=0.5)
    df

.. image:: https://raw.githubusercontent.com/Gigas64/discovery-behavioral-utils/master/docs/img/output_26_0.png

Weighted Patterns
-----------------

Now we can get a bit more controlled in how we want the random numbers
to be generated by using the weighted patterns. Weighted patterns are
similar to probability but don't need to add to 1 and also don't need to
be the same size as the selection. Lets see how this works through an
example.

lets generate an array of 100 and then see how many times each category
is selected

.. code:: python

    selection = ['M', 'F', 'U']
    gender = tools.get_category(selection, weight_pattern=[5,4,1], size=100)
    dist = [0]*3
    for g in gender:
        dist[selection.index(g)] += 1
    
    print(dist)

.. parsed-literal::

    **$>** [51, 40, 9]

.. code:: python

    fig = plt.figure(figsize=(8,3))
    sns.set(style="whitegrid")
    g = sns.barplot(selection, dist)

.. image:: https://raw.githubusercontent.com/Gigas64/discovery-behavioral-utils/master/docs/img/output_25_0.png


It can also be used to create more complex distribution. In this example
we want an age distribution that has peaks around 35-40 and 55-60 with a
significant tail off after 60 but don't want a probability for every
age.

.. code:: python

    # break the pattern into every 5 years
    pattern = [3,5,6,10,6,5,7,15,5,2,1,0.5,0.2,0.1]
    age = tools.get_number(20, 90, weight_pattern=pattern, size=1000)
    
    fig = plt.figure(figsize=(10,4))
    _ = sns.set(style="whitegrid")
    _ = sns.kdeplot(age, shade=True)

.. image:: https://raw.githubusercontent.com/Gigas64/discovery-behavioral-utils/master/docs/img/output_27_0.png


Complex Weighting patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~

Weighting patterns acn be multi dimensial representing controlling
distribution over time.

In this example we don't want there to be any values below 50 in the
first half then only values below 50 in the second

.. code:: python

    split_pattern = [[0,1],[1,0]]
    numbers = tools.get_number(100, weight_pattern=split_pattern, size=100)
    
    fig = plt.figure(figsize=(8,4))
    plt.style.use('seaborn-whitegrid')
    plt.plot(list(range(100)), numbers);
    _ = plt.axhline(y=50, linewidth=0.75, color='red')
    _ = plt.axvline(x=50, linewidth=0.75, color='red')

.. image:: https://raw.githubusercontent.com/Gigas64/discovery-behavioral-utils/master/docs/img/output_29_1.png


we can even build more complex numbering where we always get numbers
around the middle but first 3rd and last 3rd additionally high and low
numbers respectively

.. code:: python

    mid_pattern = [[0,0,1],1,[1,0,0]]
    numbers = tools.get_number(100, weight_pattern=mid_pattern, size=100)
    fig = plt.figure(figsize=(8,4))
    _ = plt.plot(list(range(100)), numbers);
    _ = plt.axhline(y=33, linewidth=0.75, color='red')
    _ = plt.axhline(y=67, linewidth=0.75, color='red')
    _ = plt.axvline(x=33, linewidth=0.75, color='red')
    _ = plt.axvline(x=67, linewidth=0.75, color='red')


.. image:: https://raw.githubusercontent.com/Gigas64/discovery-behavioral-utils/master/docs/img/output_31_0.png


Random Seed
~~~~~~~~~~~

in this example we are using seeding to fix predictability of the
randomness of both the weighted pattern and the numbers generated. We
can then look for a good set of seeds to generate different spike
patterns we can predict.

.. code:: python

    fig = plt.figure(figsize=(12,15))
    right=False
    for i in range(0,10): 
        ax = plt.subplot2grid((5,2),(int(i/2), int(right)))
        result = tools.get_number(100, weight_pattern=np.sin(range(10)), size=100, seed=i+10)
        g = plt.plot(list(range(100)), result);
        t = plt.title("seed={}".format(i+10))
        right = not right
    plt.tight_layout()
    plt.show()

.. image:: https://raw.githubusercontent.com/Gigas64/discovery-behavioral-utils/master/docs/img/output_33_0.png


Dates
-----

Dates are an important part of most datasets and need flexibility in all
theri multidimensional elements

.. code:: python

    # creating a set of randome dates and a set of unique dates
    df = pd.DataFrame()
    df['dates'] =  tools.get_datetime('01/01/2017', '21/01/2017', size=20, date_format='%d-%m-%Y')
    df['seq'] = tools.unique_date_seq('01/01/2017', '21/01/2017', size=20, date_format='%d-%m-%Y')
    print("{}/20 dates and {}/20 unique date sequence".format(df.dates.nunique(), df.seq.nunique()))

.. parsed-literal::

    **$>** 11/20 dates and 20/20 unique date sequence


Date patterns
~~~~~~~~~~~~~

Get Data has a number of different weighting patterns that can be
applied - accross the daterange - by year - by month - by weekday - by
hour - by minutes

Or by a combination of any of them.

.. code:: python

    from ds_discovery.transition.discovery import Visualisation as visual

.. code:: python

    # Create a month pattern that has no data in every other month
    pattern = [1,0]*6
    selection = ['Rigs', 'Office']
    
    df_rota = pd.DataFrame()
    df_rota['rota'] = tools.get_category(selection, size=300)
    df_rota['dates'] =  tools.get_datetime('01/01/2017', '01/01/2018', size=300, month_pattern=pattern)
    
    df_rota = cleaner.to_date_type(df_rota, headers='dates')
    df_rota = cleaner.to_category_type(df_rota, headers='rota')

.. code:: python

    visual.show_cat_time_index(df_rota, 'dates', 'rota')

.. image:: https://raw.githubusercontent.com/Gigas64/discovery-behavioral-utils/master/docs/img/output_39_0.png


Quite often dates need to have specific pattern to represent real
working times, in this example we only want dates that occur in the
working week.

.. code:: python

    # create dates that are only during the working week
    pattern = [1,1,1,1,1,0,0]
    selection = ['Management', 'Staff']
    
    df_seating = pd.DataFrame()
    df_seating['position'] = tools.get_category(selection, weight_pattern=[7,3], size=100)
    df_seating['dates'] =  tools.get_datetime('14/01/2019', '22/01/2019', size=100, weekday_pattern=pattern)
    
    df_seating = cleaner.to_date_type(df_seating, headers='dates')
    df_seating = cleaner.to_category_type(df_seating, headers='position')

.. code:: python

    visual.show_cat_time_index(df_seating, 'dates', 'position')

.. image:: https://raw.githubusercontent.com/Gigas64/discovery-behavioral-utils/master/docs/img/output_36_0.png

What Next
~~~~~~~~~
These are only the starter building blocks that give the foundation to more comple rule
and behaviour. Have a play with:

    :correlate:
        creates data that correlates to another set of values giving an offset value
        based on the original. This applies to Dates, numbers and categories
    :associate:
        allows the construction of complex rule based actions nd behavior
    :builder instance:
        explore the ability to configure and save a template so you can repeat the build

but the library is being built out all the time so keep it updated.


Python version
--------------

Python 2.6 and 2.7 are not supported. Although Python 3.x is supported, it is recommended to install
``discovery-behavioral-utils`` against the latest Python 3.6.x whenever possible.
Python 3 is the default for Homebrew installations starting with version 0.9.4.

GitHub Project
--------------
Discovery-Behavioral-Utils: `<https://github.com/Gigas64/discovery-behavioral-utils>`_.

Change log
----------

See `CHANGELOG <https://github.com/doatridge-cs/discovery-behavioral-utils/blob/master/CHANGELOG.rst>`_.


Licence
-------

BSD-3-Clause: `LICENSE <https://github.com/doatridge-cs/discovery-behavioral-utils/blob/master/LICENSE.txt>`_.


Authors
-------

`Gigas64`_  (`@gigas64`_) created discovery-behavioral-utils.


.. _pip: https://pip.pypa.io/en/stable/installing/
.. _Github API: http://developer.github.com/v3/issues/comments/#create-a-comment
.. _Gigas64: http://opengrass.io
.. _@gigas64: https://twitter.com/gigas64


.. |pypi| image:: https://img.shields.io/pypi/pyversions/Django.svg
    :alt: PyPI - Python Version

.. |rdt| image:: https://readthedocs.org/projects/discovery-behavioral-utils/badge/?version=latest
    :target: http://discovery-behavioral-utils.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |license| image:: https://img.shields.io/pypi/l/Django.svg
    :target: https://github.com/Gigas64/discovery-behavioral-utils/blob/master/LICENSE.txt
    :alt: PyPI - License

.. |wheel| image:: https://img.shields.io/pypi/wheel/Django.svg
    :alt: PyPI - Wheel

