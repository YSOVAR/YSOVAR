How we process a cluster
========================

First, we need to import the separate modules::

    import astropy.io.ascii as ascii
    import ysovar_atlas as atlas
    import plot_atlas
    import lightcurves as lc

(Yes, we could have found better names for those modules. Maybe we will overhaul that
at some point and rename everything.)

We did not write our papers yet, so we don't know everything we'll eventually do in
the analysis. The list of processing steps below shows what we do, but we don't
discuss every parameter value. For that, please check the documentation of the 
individual routines.

Reading the data
----------------
We start with the csv file that we downlaod from the YSOVAR2 database.
``match_dist = 0.`` means that no source merging will be performed at this stage,
i.e. we assume that this was done to the data in the database already.
The routine reads a csv files (with data for both IRAC bands). 
Also, it calls :func:`ysovar_atlas.dict_cleanup`, which performs three important tasks:

1. It throws out sources with few entries
2. It constrcts a merged lightcurve with IRAC 1 and IRAC 2 values that are close in time.
3. It adds Scotts error floor value to all errors in the data.

All this is done as part of the read in::

    stars = atlas.dict_from_csv('myinputfile.csv', match_dist = 0.)

After reading, it's a good idea to do a few sanity checks on the data.
Errors can happen, either in the database or in the sqlquery or elsewhere::

    atlas.check_dataset(stars)

This routine performs a few simple tests, e.g. it looks for sources that are so close
together, that they probably should be merged. Obviously, this is not going to find
all possible problems, but every time we run into something we add a test here
to make sure we find is next time.

Then, we make an :class:`ysovar_atlas.YSOVAR_atlas` object. These
object can auto-generate some column based on the lightcurves we put
in, so check out the documentation for that::

    mycloud = atlas.YSOVAR_atlas(lclist = stars)

The :class:`ysovar_atlas.YSOVAR_atlas` is build on top of a `astropy.table.Table
(documentation here)
<http://docs.astropy.org/en/v0.2/table/index.html>`_ object. See that
documentation for the syntax on how to acess the data or add a column.


Merging with auxillary data
---------------------------
There are all sorts of things that `Astropy
<http://docs.astropy.org/en/v0.2/index.html>`_ can read automatically
(fits, votable, ascii), I just show two examples here.

The first is a machine-readable table that I downloaded from the ApJ website::

    import astropy.io.ascii
    # in this example from Barsony et al. (2012)
    bar12 = astropy.io.ascii.read('apj426650t3_mrt.txt')

The second is a votable. I opended an image of my region in ds9, then
did Analysis -> Catalogs -> SIMBAD and exported the SIMBAD catalog in the
catalog window as a votable::
    
    import asctropy.io.votable
    simbad = astropy.io.votable.parse_single_table('SIMBAD.xml').to_table()

Unfortunately, in both cases the (RA, DEC) in the tables are not in
degrees. pYSOVAR provides two methods of fixing that (or you could
always compute the values by hand and add them to the table::

    # by hand
    mycloud.add_column(astropy.table.Column(name = 'RAdeg', data = RA_by_Hand))

    # if RA, DEC are in CDS format, i.e. there are several columns
    # `RAh`, `RAm`, `RAs`, `DE-`, `DEd`, `DEm` and `DEs`
    atlas.coord_CDS2RADEC(bar12)

    # if RA and DEC are string in the form
    # hh:mm:ss.ss and dd:mm:ss.sss  
    atlas.coord_str2RADEC(mytable, ra='myRAcol', dec = 'myDEcol')

Then, we want to merge the auxillary data with our YSOVAR data. To
make sure that there is no name clash, I first add an `bar12_` to the
name of each column::

    for col in bar12.colnames:
        bar12.rename_column(col, 'bar12_'+col)

Then, I decide which column should be copied to `mycloud`. By defauld
the matching is done by position and matched within 1 arcsec are
accepted. `mycloud` objects with no counterpart in `bar12` get an empty
value in the column, `bar12` objects with no counterpart in `mycloud`
are ignored::

    bar12_names = ['bar12_AV', 'bar12_Teff', 'bar12_Jmag']
    mycloud.add_catalog_data(bar12, names = bar12_names, ra1='ra', dec1='dec', ra2='bar12_RAdeg', dec2='bar12_DEdeg')


Calculating properties
----------------------
This is where it gets interesting. 
Ths :class:`ysovar_atlas.YSOVAR_atlas` auto-generates some content in the background, so I really
encourage you to read the documentation (I promise it's only a few
lines because I am too lazy to type much more), e.g.::

    print mycloud['median_45']

will calculate the median for all lightcurves, add a column called
`median_45` to the `mycloud` table and print the numbers to the screen.

Then, we search for periods. The paramters in the call are the maximum period in days, the oversampling factor, and a multiplier for the maximum frequncy (see :func:`ysovar_atlas.calc_ls` for details)::

    mycloud.calc_ls('36', 300)
    mycloud.calc_ls('45', 300)

    mycloud.is_there_a_good_period(20, 1,100)

Try fitting x vs. y (this is NOT like bisector, because I always use both x and y errors). This will show differences if the slope is almost vertical in one of the coordinate systems::

    mycloud.cmd_slope_odr()

    lc.calc_poly_chi(mycloud)

Make all the pretty plots
-------------------------
We write all the stuff in `outroot` and determine that a pdf will be good format.
All matplotlib output formats are supported. Then, we do all the cool plots::

    outroot = '/my/directory/'

    # set output file type to pdg (for pdflatex)
    plot_atlas.filetype = ['.pdf']

    plot_atlas.get_stamps(mycloud, outroot_stars)
    plot_atlas.plot_polys(mycloud, outroot_stars)
    plot_atlas.make_lc_plots(mycloud, outroot_stars) 
    plot_atlas.make_cmd_plots(cat, outroot_stars)
    plot_atlas.make_ls_plots(cat, outroot_stars, 300, 4, 1)
    plot_atlas.make_phased_lc_cmd_plots(cat, outroot_stars)
    plot_atlas.make_info_plots(cat, outroot_overview)
    plot_atlas.make_sed_plots(cat, outroot_stars, title = 'SED')

Write latex files for atlas. In this case we select for YSOs (numerical class < 4) only::

    ind_ysos = np.where(mycloud['ysoclass'] < 4)[0]
    plot_atlas.make_latexfile(mycloud, outroot_stars, 'atlas_ysos', ind_ysos)


Write (parts of) catalog to file
--------------------------------
`astropy.table.Table
<http://docs.astropy.org/en/v0.2/table/index.html>`_ 
offers several methods to quickly look at the data,
e.g. `mycloud.more()` to receive a more-style output in your
terminal. See the documentation for details.

Here, is one example of output to LaTeX::

    import astropy.io.ascii as ascii

    f42 = format_or_string('%4.2f')

    ascii.write(mycloud, sys.stdout, Writer = ascii.Latex,
          names = mycloud.colnames,
          include_names=['ra', 'dec','YSOVAR2_id', 'median_45'],
          formats = {'ra':'%10.5f', 'dec':'%10.5f', 'median_45': '%4.2f'})













