CometCurve
==========

**A fuzzy-tailed package for fitting the light curves of comets.**

*CometCurve* is an open source Python package which utilizes data from the
`Comet Observation Database <https://www.cobs.si>`_ (COBS) to fit a canonical
model to the light curves of comets.


Example
-------

.. code-block:: python

  import cometcurve as cc
  model = cc.CometModel(comet="2019 Y4", start="2020-02-01", stop="2020-07-01")
  model.plot()

.. image:: https://raw.githubusercontent.com/barentsen/cometcurve/master/examples/example-2019y4.png


Installation
------------

(coming soon)

.. code-block:: bash

  pip install cometcurve
