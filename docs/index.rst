.. pYSOVAR documentation master file, created by
   sphinx-quickstart on Thu Jan 24 14:07:23 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pYSOVAR's documentation!
===================================

This the the collection of python moduls that we (Katja and Moritz) use for our
analysis of the YSOVAR data of those two clusters we are currently working on.
While we attempt to write things general and applicable to more datasets, in 
some cases it's still very specific to our two clusters. So, don't be surprised
if it does not work out-of-the-box for you, but feel free to try.
You find the entire code on https://github.com/YSOVAR/YSOVAR .

First, we show how we process one of our clusters:

.. toctree::
   :maxdepth: 2
   
   workflow

Here is a detailed documentation and API for all modules (possibly incomplete,
since we document methods only after we know they work):

.. toctree::
   :maxdepth: 2

   ysovar_atlas
   registry
   plot_atlas
   ysovar_lombscargle
   lightcurves
   great_circle_dist

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

