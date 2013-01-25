How we process a cluster
========================

First, we need to import the separate modules::

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
i.e. we assume that this was done to the data in the database already::

    stars = atlas.dict_from_csv('myinputfile.csv', match_dist = 0.)

For out clusters we also have some auxiallry data, in this example it's the data
table for Guenther et al 2012, which was downlaoded from the AJ website as
machine-readable table::

    (guenther_data_all, guenther_data) = atlas.readguentherlist('Guenther2012_table.txt')

After reading, it's a good idea to do a few sanity checks on the data.
Errors can happen, either in the database or in the sqlquery or elsewhere::

    atlas.check_dataset(stars)

This routine performs a few simple tests, e.g. it looks for sources that are so close
together, that they probably should be merged. Obviously, this is not going to find
all possible problems, but every time we run into something we add a test here
to make sure we find is next time.

Merging with auxillary data
---------------------------
Get ra and dec in a neat array::

    radec = atlas.radec_from_dict(stars, RA = 'ra', DEC = 'dec')

And then, we cross-match the Guenther et al list with the YSOVAR targets::

    ids = atlas.makecrossids(atlas.radec_from_dict(stars), guenther_data, 1./3600., ra1='RA', dec1='DEC', ra2='RAdeg', dec2='DEdeg')

    for d, i in zip(stars, ids):
        if i != -99999:
            d['id_guenther'] = guenther_data[i]['ID']
            d['index_guenther'] = guenther_data[i]['ID'] - 1

Keep only those stars that have an id::

    stars = stars[np.where(ids != -99999)[0]]

Calculating properties
----------------------
This is where it gets interesting. Most of the magic is hidden in 
:func:`ysovar_atlas.initialize_info_array` and in 
:func:`ysovar_atlas.make_stats`::

    guenther_class = atlas.makeclassinteger(guenther_data)
    infos = atlas.initialize_info_array(stars, guenther_data, guenther_class)
    infos = atlas.make_stats(stars, infos)


Fit the CMD slope properly::
    
    atlas.add_twocolor_fits_to_infos(stars, infos, outroot_stars, 1000, ifplot=False, ifbootstrap=False, xyswitch=False)

Try fitting x vs. y (this is NOT like bisector, because I always use both x and y errors). This will show differences if the slope is almost vertical in one of the coordinate systems::

    atlas.add_twocolor_fits_to_infos(stars, infos, outroot_stars, 1000, ifplot=False, ifbootstrap=False, xyswitch=True)
    atlas.good_slope_angle(infos)

Then, we search for periods. The paramters in the call are the maximum period in days, the oversampling factor, and the maximum frequency (not angular frequency)::
    
    atlas.calc_ls(stars, infos, 40, 4, 1)


.. warning::
     THERE IS SOME PROBLEM WITH THE LONG-TERM PERIODS. CHECK THIS.

So, here we do that::

    atlas.is_there_a_good_period(stars,infos, 10, 1, 30)

Phase-fold data if period is found::

    atlas.phase_fold_data(stars,infos)

    infos.cmd_dominated[:] = -99999
    atlas.cmd_dominated_by(infos)


Write (parts of) catalog to file
--------------------------------
Here, we use the `asciitable <cxc.harvard.edu/contrib/asciitable/>`_ module. First,
we need a short function that helps us format the output nicely in the file,
then we write the LaTeX table::

    import asciitable
    def format_or_string(format_str):
        def func(val):
            if isinstance(val, basestring):
                return val
            else:
                return format_str % val
        return func

    f42 = format_or_string('%4.2f')

    asciitable.write(infos[15:25], sys.stdout, Writer = asciitable.Latex,
    names = infos.dtype.names, include_names=['id_guenther', 'ra_spitzer',
    'dec_spitzer', 'ysoclass', 'median_36', 'delta_36', 'median_45', 'delta_45',
    'stetson', 'good_period', 'cmd_dominated'],
    formats = {'id_guenther':'%4.0f',  'ra_spitzer':'%10.5f', 'dec_spitzer':'%10.5f',
    'ysoclass':'%1.0f', 'median_36':f42, 'delta_36':f42, 'median_45':f42,
    'delta_45':f42, 'stetson':f42, 'good_period':f42, 'cmd_dominated':'%10s'},
    fill_values=[(-99999., ' -- ')])

Make all the pretty plots
-------------------------
We write all the stuff in `outroot` and determine that a pdf will be good format.
All matplotlib output formats are supported. Then, we do all the cool plots::

    outroot = '/my/directory/'
    plot_atlas.filetype = ['.pdf']

    plot_atlas.make_lc_plots(stars, outroot)
    plot_atlas.make_cmd_plots(stars, infos, outroot)
    plot_atlas.make_phased_lc_cmd_plots(stars,infos,outroot)
    plot_atlas.make_sed_plots(infos, outroot)
    plot_atlas.plot_polys(stars, outroot)

Write latex files for atlas. In this case we select for YSOs (numerical class < 4) only::

    ind_ysos = np.where(infos.ysoclass < 4)[0]
    atlas.make_latexfile(stars, infos, outroot_stars, 'atlas_ysos', ind_ysos )













