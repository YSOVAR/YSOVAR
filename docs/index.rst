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
   misc

License
=======
Copyright (C) 2013 H. M. Guenther & K. Poppenhaeger

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
:download:`License.txt <License.txt>` for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/ .



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

