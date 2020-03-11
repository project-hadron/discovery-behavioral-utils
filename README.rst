AI-STAC Behavioral Synthetic Generator
######################################

What is AI-STAC
===============

Augmented Intent - Single Task Adaptive Components (AI-STAC) is a disruptive innovator for data recovery, discovery,
synthesis, feature cataloging and machine Learning that changes the approach to data science and it's transition to
production.

it's origins came from an incubator project that shadowed a team of Ph.D. data scientists in connection with the
development and delivery of machine learning initiatives to define measurable benefit propositions for customer success.
From this, a number of observable 'capabilities' were identified as unique and separate concerns. The challenges of the
data scientist, and in turn the production teams, were to effectively leverage these separation of concerns and
distribute and loosely couple the specialist capability needs to the appropriate skills set.

In addition the need to remove the opaque nature of the machine learning end-to-end required better transparency and
traceability, to better inform the broadest of interested parties and be able to adapt without leaving being the
code 'sludge' of redundant ideas. AI-STAC is a disruptive innovation, changing the way we approach the challenges of
Machine Learning and Augmented Inelegance, introduces the ideas of 'Single Task Adaptive Components' around the
core concept of 'Parameterised Intent'


.. class:: no-web no-pdf

|pypi| |rdt| |license| |wheel|


.. contents::

.. section-numbering::

Main features
=============

* Probability Waiting
* Correlation and Association
* Behavioral Analytics

Overview
========

The Behavioral Synthetic Data Generator was developed as a solution to the current challenges of data accessibility
and the early mobilization of machine learning discovery, feature cataloging and model build.

AI-STAC Behavioral Synthetic Data Generator takes on, what is, a sceptically viewed and challenging problem area of
the generation of data that is synthetic in nature, but is still representative of its intended real-life counterpart.
In short, The AI-STAC package needed to develop rich data sets to demonstrate the capabilities across a number of
different disciplines from simple volumetrics to more demanding and complex data needs

Value Proposition
-----------------
Within the Machine learning discipline, and as a broader challenge, the accessibility of data and its relevance to
early engagement and customer success, is an industry problem with many variants available on the market.
Though competent in their delivery, their ability to flex and enrich across multiple examples of need and particularly
the high demands of pattern and associative recognition, pertaining to machine learning, is limited and cynically
considered within the machine learning community.

The Behavioral Synthetic Data Generator improves representation of data appropriate to Use Case discovery,
Machine Learning outcomes, and disclosure mitigation, removing personal DNA, while presenting representative data that
continues to represetnt eh behavioralk characteristics of the original.

The ability to engage with the customer before the availability of or access to organisational data sets is a vital
part of an organisations challenge to prove value add early and build customer success. The Behavioural Synthetic Data
Generator provides extended tooling for stress, volume and boundary testing and is also highly effective at
presentation enrichment modelling to bring product, UI and ideation to life.

With the need for Use Case early success signals, Hypothesis modelling and Business Intent Injection, as critical tools
for the success and innovation of a department, business or organisation, AI-STAC Behavioral Synthetic Generator
facilitates the mechanism to build higher order thinking. The ability to generate highly sophisticated business critical
behavioural data, particularly in today's heavily restricted and controlled environments, allows for early validation
of customer success, and more particularly, early failure to build towards more productive successes

Techniques and Methods
----------------------

To achieve this, AI-STAC Behavioral Synthetic Generator highlighted three constructs;

1.  Probability Waiting - Is an algorithm based on probability weighting patterns, dominance distributions and type
characteristics, allowing fine grain and complex behavioral characteristics and patterns to be added, across multiple
data types, to the distribution of data points within a data set.

2.  Correlation and Association – Through advanced programming techniques and a deep knowledge of component modelling
and code reuse, the project developed a finite set of data point generation tooling that implements method chaining
and rules-based association against action techniques. This approach and its techniques provide the ability to capture
complex hypothesis and business intent and generate specialized output against those outcomes.

3.  Behavioral Analytics – In addition to the data point generators, the tooling provides data analytics and behavioral
extraction, against existing data sets, that can be replayed to quickly create behavioral patterns within existing
data sets, without compromising or disclosing sensitive, or protected information. Though considered for the
regeneration of sample code or for experimental train/predict continuous data, this can be particularly valuable
with today’s concerns of data protection and disclosure mitigation strategies.

Installation
============

package install
---------------
The best way to install AI-STAC component packages is directly from the Python Package Index repository using pip.
All AI-STAC components are based on a pure python foundation package ``aistac-foundation``, but this also takes
advantage of other AI-STAC components  ``discovery-connectors`` providing extended connectivity, and the
``discovery-transition-ds`` package providing, amongst other things, data anylitics. The pip install is:

.. code-block:: bash

    $ pip install aistac-foundation
    $ pip install discovery-connectors
    $ pip install discovery-transition-ds

The AI-STAC component package for the Behavioral Synthetic is ``discovery-behavioral-utils`` and pip installed with:

.. code-block:: bash

    $ pip install discovery-behavioral-utils

if you want to upgrade your current version then using pip

.. code-block:: bash

    $ pip install --upgrade discovery-behavioral-utils

First Time Env Setup
--------------------
In order to ease the startup of tasks a number of environment variables are available to pre-assign where and how
configuration and data can be collected. This can considerable improve the burden of setup and help in the migration
of the outcome contracts between environments.

In this section we will look at a couple of primary environment variables and demonstrate later how these are used
in the Component. In the following example we are assuming a local file reference but this is not the limit of how one
can use the environment variables to locate date from multiple different connection mediums. Examples of other
connectors include AWS S3, Hive, Redis, MongoDB, Azure Blob Storage, or specific connectors can be created very
quickly using the AS-STAC foundation abstracts.

If you are on linux or MacOS:

1. Open the current user's profile into a text editor.

.. code-block:: bash

    $> vi ~/.bash_profile.

2. Add the export command for each environment variable setting your preferred paths in this example I am setting
them to a demo projects folder

.. code-block:: bash

    # where to find the properties contracts
    export AISTAC_PM_PATH=~/projects/demo/contracts

    # The default path for the source and the persisted data
    export AISTAC_DEFAULT_PATH=~/projects/demo/data

3. In addition to the default environment variables you can set specific component environment variables. This is
particularly useful with the Synthetic component where output might vary from the default path structure.
For Synthetic persist you replace the ``DEFAULT`` with ``SYNTHETIC``, and in this case specify the ``PERSIST`` path

.. code-block:: bash

    # specific to the synthetic component persist path
    export AISTAC_SYNTHETIC_SOURCE_PATH=/tmp/data/sftp

4. save your changes
5. re-run your bash_profile and check the variables have been set

.. code-block:: bash

    $> source ~/.bash_profile.
    $> env

SyntheticBuilder Task - Setup
=============================
The SyntheticBuilder Component is a 'Capability' component and a 'Separation of Concern' dealing specifically with the
generation of synthetic data.

In the following example we are assuming a local file reference and are using the default AI-STAC Connector Contracts
for Data Sourcing and Persisting, but this is not the limit of how one can use connect to data retrieval and storage.
Examples of other connectors include AWS S3, Hive, Redis, MongoDB, Azure Blob Storage, or specific connectors can be
created very quickly using the AS-STAC foundation abstracts.

Instantiation
-------------
The ``SyntheticBuilder`` class is the encapsulating class for the Synthetic Capability, providing a wrapper for
synthetic builder functionality. and imported as:

.. code-block:: python

    from ds_behavioral import SyntheticBuilder

The easiest way to instantiate the ``SyntheticBuilder`` class is to use Factory Instantiation method ``.from_env(...)``
that takes advantage of our environment variables set up in the previous section. in order to differentiate each
instance of the SyntheticBuilder Component, we assign it a ``Task`` name that we can use going forward to retrieve
or re-create our SyntheticBuilder instance with all its 'Intent'

.. code-block:: python

    builder = SyntheticBuilder.from_env(task_name='demo')

Augmented Knowledge
-------------------
Once you have instantiated the SyntheticBuilder Task it is important to add a description of the task as a future remind,
for others using this task and when using the MasterLedger component (not covered in this tutorial) it allows for a
quick reference overview of all the tasks in the ledger.

.. code-block:: python

    builder.set_description("A Demo task as a tutorial in building synthetic data")

Note: the description should be a short summary of the task. If we need to be more verbose, and as good practice,
we can also add notes, that are timestamped and cataloged, to help augment knowledge about this
task that is carried as part of the Property Contract.

in the SyntheticBuilder Component notes are cataloged within five named sections:
* source - notes about the source data that help in what it is, where it came from and any SME knowledge of interest
* schema - data schemas to capture and report on the outcome data set
* observations - observations of interest or enhancement of the understanding of the task
* actions - actions needed, to be taken or have been taken within the task

each ``catalog`` can have multiple ``labels`` whick in tern can have multiple text entries, each text keyed by
timestamp. through the catalog set is fixed, ``labels`` can be any reference label

the following example adds a description to the source catalogue

.. code-block:: python

    tr.add_notes(catalog='source', label='describe', text="The source of this demo is a synthetic data set"

To retrieve the list of allowed ``catalog`` sections we use the property method:

.. code-block:: python

    builder.notes_catalog


One-Time Connectors Settings
----------------------------
With each component task we need to set up its connectivity defining an outcome ``Connector Contract`` which control
the loose coupling of where data is persisted to the code that uses it. Though we can define the Connect Contract in
full, it is easier to take advantage of template connectors set up as part of the Factory initialisation method.

Though we can define as many Connector Contract as we like, by its nature, the SyntheticBuilder task has a single
outcome connector contract that need to be set up as a 'one-off' task. Once this is set it is stored in the Property
Contract and thus do not need to be set again.

Outcome Contract
~~~~~~~~~~~~~~~~
We need to specify where we are going to persist our data once we have synthesised it. Here we are going to take
advantage of what our Factory Initialisation method set up for us and allow the SyntheticBuilder task to define our
output based on constructed template Connector Contracts. With this the file will be placed in predefined persist path

.. code-block:: python

    builder.set_outcome(uri_file='synthetic_demo.csv')

We are ready to go. The SyntheticBuilder task is ready to use.

SyntheticBuilder Task - Intent
==============================

Instantiate the Task
--------------------

The easiest way to instantiate the ``SyntheticBuilder`` class is to use Factory Instantiation method ``.from_env(...)``
that takes advantage of our environment variables set up in the previous section. in order to differentiate each
instance of the SyntheticBuilder Component, we assign it a ``Task`` name that we can use going forward to retrieve
or re-create our SyntheticBuilder instance with all its 'Intent'

.. code-block:: python

    builder = SyntheticBuilder.from_env(task_name='demo')

Parameterised Intent
--------------------
Parameterised intent is a core concept and represents the intended action and defining functions of the component.
Each method is known as a component intent and the parameters the task parameterisation of that intent. The intent
and its parameters are saved and can be replayed using the ``run_intent_pipeline(size=1000)`` method

The following sections are a brief description of the intent and its parameters. To retrieve the list of available
intent methods in code run:

.. code-block:: python

    tr.intent_model.__dir__()

We can also look at the contextual help for each of the methods calling
the ``intent_model`` property and using the ``help`` build-in

.. code:: python

    help(builder.intent_model.get_number)

.. parsed-literal::

    def get_number(self, from_value: [int, float], to_value: [int, float]=None, weight_pattern: list=None,
                   label: str=None, offset: int=None, precision: int=None, currency: str=None,
                   bounded_weighting: bool=True, at_most: int=None, dominant_values: [float, list]=None,
                   dominant_percent: float=None, dominance_weighting: list=None, size: int = None, quantity: float=None,
                   seed: int=None, save_intent: bool=None, intent_level: [int, str]=None,
                   replace_intent: bool=None) -> list:
        """ returns a number in the range from_value to to_value. if only to_value given from_value is zero

        :param from_value: range from_value to_value if to_value is used else from 0 to from_value if to_value is None
        :param to_value: optional, (signed) integer to end from.
        :param weight_pattern: a weighting pattern or probability that does not have to add to 1
        :param label: a unique name to use as a label for this column
        :param precision: the precision of the returned number. if None then assumes int value else float
        :param offset: an offset multiplier, if None then assume 1
        :param currency: a currency symbol to prefix the value with. returns string with commas
        :param bounded_weighting: if the weighting pattern should have a soft or hard boundary constraint
        :param at_most: the most times a selection should be chosen
        :param dominant_values: a value or list of values with dominant_percent. if used MUST provide a dominant_percent
        :param dominant_percent: a value between 0 and 1 representing the dominant_percent of the dominant value(s)
        :param dominance_weighting: a weighting of the dominant values
        :param size: the size of the sample
        :param quantity: a number between 0 and 1 representing data that isn't null
        :param seed: a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) a level to place the intent
        :param replace_intent: (optional) replace strategy for the same intent found at that level
        :return: a random number
        """
    
From here we can now play with some of the ``get`` methods

.. code:: python

    # get an integer between 0 and 9
    builder.intent_model.get_number(10, size=5)

.. parsed-literal::

    **$>** [6, 5, 3, 2, 3]

.. code:: python

    # get a float between -1 and 1, notice by passing an float it assumes the output to be a float
    builder.intent_model.get_number(from_value=-1.0, to_value=1.0, precision=3, size=5)

.. parsed-literal::

    **$>** [0.283, 0.296, -0.958, 0.185, 0.831]

.. code:: python

    # get a currency by setting the 'currency' parameter to a currency symbol.
    # Note this returns a list of strings
    builder.intent_model.get_number(from_value=1000.0, to_value=2000.0, size=5, currency='$', precision=2)

.. parsed-literal::

    **$>** ['$1,286.00', '$1,858.00', '$1,038.00', '$1,944.00', '$1,250.00']

.. code:: python

    # get a timestamp between two dates
    builder.intent_model.get_datetime(start='01/01/2017', until='31/12/2018')

.. parsed-literal::

    **$>** [Timestamp('2018-02-11 02:23:32.733296768')]

.. code:: python

    # get a formated date string between two numbers
    builder.intent_model.get_datetime(start='01/01/2017', until='31/12/2018', size=4, date_format='%d-%m-%Y')

.. parsed-literal::

    **$>** ['06-06-2017', '05-11-2017', '28-09-2018', '04-11-2017']

.. code:: python

    # get categories from a selection
    builder.intent_model.get_category(['Red', 'Blue', 'Green', 'Black', 'White'], size=4)

.. parsed-literal::

    **$>** ['Green', 'Blue', 'Blue', 'White']

.. code:: python

    # get unique categories from a selection
    builder.intent_model.get_category(['Red', 'Blue', 'Green', 'Black', 'White'], size=4, replace=False)

.. parsed-literal::

    **$>** ['Blue', 'White', 'Green', 'Black']


Building a DataFrame
--------------------

With these lets build a quick Synthetic DataFrame. For ease of code we
will redefine the 'builder.intent_model' call

.. code:: python

    tools = builder.intent_model

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

