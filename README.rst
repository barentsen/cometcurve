CometCurve
==========

**A fuzzy-tailed package for fitting the light curves of comets.**

**CometCurve** is an open source Python package which utilizes data from the
[Comet Observation Database](https://www.cobs.si/) (COBS) to fit a canonical
model to the light curves of comets.


Installation
------------

.. code-block:: bash

  pip install planetpixel


Example
-------

.. code-block:: python

  import cometcurve as cc
  model = cc.CometModel(comet="2019 Y4", start="2020-02-01", stop="2020-07-01")
  model.plot()
