Generate atlas
==============

This module collects all procedures that are required to make the
atlas. This starts with reading in the csv file from the YSOVAR2
database and includes the calculation of the some fits and quantities.
More specific tasks for the analysis of the lightcurve can be found in
:mod:`YSOVAR.lightcurves`, more stuff for plotting in :mod:`YSOVAR.plot`.

The basic structure for the YSOVAR analysis is the
:class:`YSOVAR.atlas.YSOVAR_atlas`. 
To initialize an atlas object pass is a numpy array wich all the lightcurves::
    
    from YSOVAR import atlas
    data = atlas.dict_from_csv('/path/tp/my/irac.csv', match_dist = 0.)
    MyRegion = atlas.YSOVAR_atlas(lclist = data)

The :class:`YSOVAR.atlas.YSOVAR_atlas` is build on top of a `astropy.table.Table
(documentation here)
<http://docs.astropy.org/en/v0.2/table/index.html>`_ object. See that
documentation for the syntax on how to acess the data or add a column.

This :class:`YSOVAR.atlas.YSOVAR_atlas` auto-generates some content in the 
background, so I really encourage you to read the documentation (I promise it's
only a few lines because I am too lazy to type much more).

.. automodule:: YSOVAR.atlas
   :members:
   :undoc-members:
