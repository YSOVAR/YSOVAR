# -*- coding: utf-8 -*-
# Copyright (C) 2013 H.M.Guenther & K.Poppenhaeger. See Licence.rst for details.
'''This module holds specialized plotting function for YSOVAR data.

This module hold some plotting functions for YSOVAR data, i.e. lightcurves
and color-color or color-magnitude diagrams.
All the plotting is done with matplotlib.

Module level variables
----------------------

There are several module level variables, that define defaults for plots.

Default offset of x-axis for lightcurve plots::

    YSOVAR.plot.mjdoffset = 55000

List of all formats for output for those routines that make e.g. the 
lightcurve plot for each source. This has to be a list *even if it contains 
only one element*. List multiple formats to obtain each image in each format::

    YSOVAR.plot.filetype  = ['.eps']

This routine can connect to the internet and download thumbnail images from
the YSOVAR database. To do so, you need to set the username and password
for the YSOVAR database::

    YSOVAR.plots.YSOVAR_USERNAME = 'username'
    YSOVAR.plots.YSOVAR_PASSWORD = 'password'


Plotting functions
------------------

The most important purpose of this module is to provide functions that can
generate a big atlas, which holds some key diagnostic plots (e.g. the
lightcurve, color-mag diagram) for every star in a :class:`YSOVAR.atlas.YSOVAR_atlas` 
object. However, some of the functions are also useful for stand-alone plots.
Functions that generate multiple plots often start with ``make`` and have
a plural name (e.g. :func:`YSOVAR.plot.make_lc_plots`). These function then
call a function that makes the individual plot (and sometimes that is broken
down again into one function that sets up the figure and the axis and a 
second one that executes the actual plot command), see
:func:`YSOVAR.plot.plot_lc` and :func:`YSOVAR.plot.lc_plot`.

Use those functions to plot individual lightcurves e.g. for a paper.
'''

import urllib
import urllib2
import math
import os.path

import numpy as np
from copy import deepcopy
import string

import matplotlib
import matplotlib.pyplot as plt
import pylab

from . import lombscargle
from . import lightcurves as lc
from .autofuncs import redvecs

from .atlas import *

filetype = ['.eps']
mjdoffset = 55000
YSOVAR_USERNAME = None
YSOVAR_PASSWORD = None

def multisave(fig, filename):
    for t in filetype:
        fig.savefig(filename + t)
        
def make_latexfile(atlas, outroot, name, ind = None, plotwidth = '0.45\\textwidth',
                   output_figs = [['_lc', '_color'],
                                  ['_ls', '_sed'],
                                  ['_lc_phased', '_color_phased'],
                                  ['_stamp', '_lcpoly']],
                   output_cols = {'YSOVAR2_id': 'ID in YSOVAR 2 database',
                                  'simbad_MAIN_ID': 'ID Simbad',
                                  'simbad_SP_TYPE': 'Simbad Sp type',
                                  'IRclass': 'Rob class',
                                  'median_36': 'median [3.6]',
                                  'mad_36': 'medium abs dev [3.6]',
                                  'stddev_36': 'stddev [3.6]',
                                  'median_45': 'median [4.5]',
                                  'mad_45': 'medium abs dev [4.5]',
                                  'stddev_45': 'stddev [4.5]',
                                  'stetson_36_45': 'Stetson [3.6] vs. [4.5]'},
                   pdflatex = True):
    '''make output LeTeX file that produces an atlas

    This procedure actually checks the directory `outroot` and only includes
    Figures in the LaTeX document that are present there.
    For some stars (e.g. if they only have one lightcurve) certain plots
    may never be produced, so this strategy will ensure that the LaTeX
    document compiles in any case. It also means that files, that are not
    present because you forgot to produce them, will not be present.

    Parameters
    ----------
    atlas : ysovar_atlas.YSOVAR_atlas
        This is the atlas with the data to be plotted
    outroot : string
        path to directory where all figures are found. The LeTeX file will
        be written in the same directory.
    name : string
        filename of the atlas file (without the `.tex`)
    ind : index array
        Only objects in this index array will be included in the LaTeX file.
        Use `None` to output the entire atlas.
    plotwidth : string
        width of the plots in LeTeX notation. It is the users responsibility
        to ensure that the plots chosen with `output_figs` fit on the page.
    output_figs : list of lists
        List of file extensions of plots to be included. Filenames will be of
        format `i + fileextension`.
        This is a list of lists in the form::

           `[[fig1_row1, fig2_row1, fig3_row1], [fig1_row2, ...]]`
            
        Each row in the figure grid can have a different number of figures,
        but it is the users responsibility to choose `plotwidth` so that they all
        fit on a page.
    output_cols : dictionary
        Select columns in the table to print out below the figures.
        Format is `{'colname': 'label'}`, where label is what will appear
        in the LaTeX document.
    pdflatex : bool
        if `True` check for files that pdflatex uses (jpg, png, pdf), otherwise
        for fiels LaTeX uses (ps, eps).
    '''
    def fig_if_exists(filename, figname, plotwidth, fileextensions):
        filepresent = False
        for ext in fileextensions:
            #print filepresent, figname + ext
            if os.path.exists(figname+ext): filepresent = True
        if filepresent: filename.write('\\includegraphics[width=' + plotwidth  + ']{' + os.path.basename(figname) + '}' + '\n')

    if ind is None:
        ind = np.arange(len(atlas), dtype = np.int)

    if pdflatex:
        fileextensions = ['.png','.jpg','.jepg','.pdf']
    else:
        fileextensions = ['.eps', '.ps']

    # write selected data and figures for sources into latex file
    with open(os.path.join(outroot, name + '.tex'), 'wb') as f:
        f.write('\\documentclass[letterpaper,12pt]{article}\n')
        f.write('\\usepackage{graphicx}\n')
        f.write('\\begin{document}\n')
        f.write('\\setlength{\parindent}{0pt}\n')
        f.write('\\oddsidemargin 0.0in\n')
        f.write('\\evensidemargin 0.0in\n')
        f.write('\\textwidth 6.5in\n')
        f.write('\\topmargin -0.5in\n')
        f.write('\\textheight 9in\n')
        f.write('\n')
        f.write('\\newpage \n')
        f.write('\n')
        f.write('\\small \n')

        for i in ind:
            f.write('\\newpage \n')
            f.write('\\begin{minipage}[l]{6.5in} \n')
            for row in output_figs:
                for fig in row:
                    fig_if_exists(f, os.path.join(outroot, atlas['YSOVAR2_id'][i] + fig), plotwidth, fileextensions)
                f.write('~\\newline\n\n')

            f.write('Index number in Atlas: ' + str(i) + '\\\ \n')
            for col in output_cols:
                f.write(output_cols[col]+': ' + str(atlas[col][i]) + '\\\ \n')

            f.write('\\end{minipage} \n')

        f.write('\\end{document}')
        f.close()



def get_stamps(data, outroot, verbose = True):
    '''Retrieve stamp images from the YSOVAR2 database

    The database requires a login. To get past that set the following variables
    before calling this routine:
    - YSOVAR.plot.YSOVAR_USERNAME
    - YSOVAR.plot.YSOVAR_PASSWORD

    Parameters
    ----------
    data : ysovar_atlas.YSOVAR_atlas
        needs to contain a column called `YSOVAR2_id`
    outroot : string
        directory where the downloaded files will end up
    
    '''
    url = 'http://cosmos.physast.uga.edu/YSOVAR2/cgi/stamps.py'
    top_level_url = "http://cosmos.physast.uga.edu/YSOVAR2"
    # create a password manager
    password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
    # Add the username and password.
    if (YSOVAR_USERNAME is None) or (YSOVAR_PASSWORD is None):
        raise ValueError("Need to set plot_atlas.YSOVAR_USERNAME and plot_atlas.YSOVAR_PASSWORD before automated retrival.")
    # If we knew the realm, we could use it instead of None.
    password_mgr.add_password(None, top_level_url, YSOVAR_USERNAME , YSOVAR_PASSWORD)
    
    handler = urllib2.HTTPBasicAuthHandler(password_mgr)
    # create "opener" (OpenerDirector instance)
    opener = urllib2.build_opener(handler)
    # use the opener to fetch a URL
    opener.open(url)
    # Install the opener.
    # Now all calls to urllib2.urlopen use our opener.
    urllib2.install_opener(opener)

    for i in range(len(data)):
        if np.mod(i, 100) == 0:
            print 'Fetching stamp ' + str(i) + ' of ' + str(len(data)) + ' from YSOVAR2 Server.'
        dat = urllib.urlencode({'source_id' : data['YSOVAR2_id'][i]})
        req = urllib2.Request(url, dat)
        response = urllib2.urlopen(req)
        try:   # with statement would be shorter, but is not available in python 2.6
            f = open(os.path.join(outroot, data['YSOVAR2_id'][i]+'_stamp.png'), 'w')
            f.write(response.read())
        finally:
            f.close()

def make_reddeningvector_for_plot(x1, x2, y1, y2):
    # calculates the coordinates of the reddening evctor in convenient plot coordinates.
    slope = redvecs['36_45']
    
    if (x2-x1 <= 0.1):
        AV = 0.25
        vec_ylength = slope[1]*AV
    elif (x2-x1 > 0.1) & (x2-x1 <= 0.2):
        AV = 0.5
        vec_ylength = slope[1]*AV
    elif (x2-x1 > 0.2) & (x2-x1 <= 0.4):
        AV = 1.
        vec_ylength = slope[1]*AV
    elif (x2-x1 > 0.4):
        AV = 2.
        vec_ylength = slope[1]*AV
    
    vec_startpoint = np.array([ x1 + 0.55*np.abs(x2-x1) , y1 + 0.1*np.abs(y2-y1) ]) # note: y-axis is inverted in plots!
    vec_endpoint = vec_startpoint + np.array([ vec_ylength/slope[0] , vec_ylength ])
    vec_info = np.array([ vec_startpoint[0], vec_startpoint[1], vec_endpoint[0], vec_endpoint[1], vec_ylength, AV ])
    #print vec_info
    
    return vec_info



def make_slope_plot(infos, outroot):
	# makes plot of slopes in CMDs, one panel for each object class
	#color definition:
	color = [(1,0.35,0.35), (1,0.95,0.35), (0.5,0.9,0.25), (0.25,0.45,1), (0.9,0.35,1)]
	plt.clf()
	ind0 = np.where((infos.cmd_alpha > -99999) & (infos.ysoclass == 0))[0]
	ind1 = np.where((infos.cmd_alpha > -99999) & (infos.ysoclass == 1))[0]
	ind2 = np.where((infos.cmd_alpha > -99999) & (infos.ysoclass == 2))[0]
	ind3 = np.where((infos.cmd_alpha > -99999) & (infos.ysoclass == 3))[0]
	ind4 = np.where((infos.cmd_alpha > -99999) & (infos.ysoclass == 4))[0]
	binning = np.arange(0,180,8)
	
	alpha_shift = deepcopy(infos.cmd_alpha)
        bad = (alpha_shift < 0) & (alpha_shift > -99999)
        alpha_shift[bad] = (alpha_shift[bad] + np.pi)
        
        alpha_red = 58.891
        alpha_accr = 90.
        
	f = plt.figure()
	plt.subplots_adjust(hspace=0.001)
	ax1 = plt.subplot(511)
	n0, bins, patches = plt.hist(alpha_shift[ind0]/(2*np.pi) * 360, bins=binning, facecolor = color[0], alpha = 1)
	plt.legend(['XYSOs'], 'upper right')
	plt.yticks(np.arange(1,max(n0),max(np.trunc(max(n0)/4),1) ))
	plt.plot([alpha_red,alpha_red],[0,max(n0)],"k--", lw=2)
	plt.plot([alpha_accr,alpha_accr],[0,max(n0)],"k", lw=2, ls='dotted')
	plt.annotate('standard \nreddening', xy =(alpha_red, 0.7), xytext=(15,1.1), arrowprops=dict(facecolor='black', shrink=0.1, width=1., frac=0.1))
	plt.text(118,0.4,'accretion-like')
	plt.annotate('', xy =(90, 0.5), xytext=(116,0.5), arrowprops=dict(facecolor='black', shrink=0.1, width=1., frac=0.1))
	plt.annotate('', xy =(180, 0.5), xytext=(154,0.5), arrowprops=dict(facecolor='black', shrink=0.1, width=1., frac=0.1))
	
	ax2 = plt.subplot(512, sharex=ax1)
	n1, bins, patches = plt.hist(alpha_shift[ind1]/(2*np.pi) * 360, bins=binning, facecolor = color[1], alpha = 1)
	plt.legend(['class 1'], 'upper right')
	plt.yticks(np.arange(1,max(n1),max(np.trunc(max(n1)/4),1) ))
	plt.plot([alpha_red,alpha_red],[0,max(n1)],"k--", lw=2)
	plt.plot([alpha_accr,alpha_accr],[0,max(n1)],"k", lw=2, ls='dotted')
	
	ax3 = plt.subplot(513, sharex=ax1)
	n2, bins, patches = plt.hist(alpha_shift[ind2]/(2*np.pi) * 360, bins=binning, facecolor = color[2], alpha = 1)
	plt.legend(['class 2'], 'upper right')
	plt.ylabel('number of objects')
	plt.yticks(np.arange(1,max(n2),max(np.trunc(max(n2)/4),1) ))
	plt.plot([alpha_red,alpha_red],[0,max(n2)],"k--", lw=2)
	plt.plot([alpha_accr,alpha_accr],[0,max(n2)],"k", lw=2, ls='dotted')
	
	ax4 = plt.subplot(514, sharex=ax1)
	n3, bins, patches = plt.hist(alpha_shift[ind3]/(2*np.pi) * 360, bins=binning, facecolor = color[3], alpha = 1)
	plt.legend(['class 3'], 'upper right')
	plt.yticks(np.arange(1,max(n3),max(np.trunc(max(n3)/4),1) ))
	plt.plot([alpha_red,alpha_red],[0,max(n3)],"k--", lw=2)
	plt.plot([alpha_accr,alpha_accr],[0,max(n3)],"k", lw=2, ls='dotted')
	
	ax5 = plt.subplot(515, sharex=ax1)
	n4, bins, patches = plt.hist(alpha_shift[ind4]/(2*np.pi) * 360, bins=binning, facecolor = color[4], alpha = 1)
	plt.legend(['stars'], 'upper right')
	plt.yticks(np.arange(1,max(n4),max(np.trunc(max(n4)/4),1) ))
	plt.plot([alpha_red,alpha_red],[0,max(n4)],"k--", lw=2)
	plt.plot([alpha_accr,alpha_accr],[0,max(n4)],"k", lw=2, ls='dotted')
	
	xticklabels = ax1.get_xticklabels()+ax2.get_xticklabels()+ax3.get_xticklabels()+ax4.get_xticklabels()
	plt.setp(xticklabels, visible=False)
	
	plt.xlabel('CMD slope angle (degrees)')
	multisave(plt.gcf(), os.path.join(outroot, 'ysovar_slope_new.png'))
	plt.clf()



def make_info_plots(infos, outroot, bands = ['36', '45'], bandlabels=['[3.6]', '[4.5]']):
    # makes some overview histograms of object properties
    #color definition:
    color = [(1,0.35,0.35), (1,0.95,0.35), (0.5,0.9,0.25), (0.25,0.45,1), (0.9,0.35,1)]
    ysoclass = infos['ysoclass']

    for band, bandlabel in zip(bands, bandlabels):
        mads = infos['mad_'+band]
        ind0 = np.where((mads >= 0) & (ysoclass == 0))[0]
        ind1 = np.where((mads >= 0) & (ysoclass == 1))[0]
        ind2 = np.where((mads >= 0) & (ysoclass == 2))[0]
        ind3 = np.where((mads >= 0) & (ysoclass == 3))[0]
        ind4 = np.where((mads >= 0) & (ysoclass == 4))[0]
        binning = np.arange(0,0.2,0.005)

        plt.figure()
        plt.subplots_adjust(hspace=0.001)
        ax1 = plt.subplot(511)
        n0, bins, patches = plt.hist(mads[ind0], bins=binning, facecolor = color[0])
        plt.legend(['XYSOs'])
        plt.yticks(np.arange(1,max(n0),max(np.trunc(max(n0)/4),1) ))
	
        ax2 = plt.subplot(512, sharex=ax1)
        n1, bins, patches = plt.hist(mads[ind1], bins=binning, facecolor = color[1])
        plt.legend(['class 1'])
        plt.yticks(np.arange(1,max(n1),max(np.trunc(max(n1)/4),1) ))
	
        ax3 = plt.subplot(513, sharex=ax1)
        n2, bins, patches = plt.hist(mads[ind2], bins=binning, facecolor = color[2])
        plt.legend(['class 2'])
        plt.ylabel('number of objects')
        plt.yticks(np.arange(1,max(n2),max(np.trunc(max(n2)/4),1) ))
	
        ax4 = plt.subplot(514, sharex=ax1)
        n3, bins, patches = plt.hist(mads[ind3], bins=binning, facecolor = color[3])
        plt.legend(['class 3'])
        plt.yticks(np.arange(1,max(n3),max(np.trunc(max(n3)/4),1) ))
	
        ax5 = plt.subplot(515, sharex=ax1)
        n4, bins, patches = plt.hist(mads[ind4], bins=binning, facecolor = color[4])
        plt.legend(['stars'])
        plt.yticks(np.arange(1,max(n4),max(np.trunc(max(n4)/4),1) ))

        xticklabels = ax1.get_xticklabels()+ax2.get_xticklabels()+ax3.get_xticklabels()+ax4.get_xticklabels()
        plt.setp(xticklabels, visible=False)
        plt.xlabel('MAD in '+bandlabel)
        plt.show()
        multisave(plt.gcf(), os.path.join(outroot, 'ysovar_mad'+band))
	
	
    # STETSON PLOT:	
    plt.clf()
    stetson = infos['stetson_36_45']
    ind0 = np.where((stetson > -99999) & (ysoclass == 0))[0]
    ind1 = np.where((stetson > -99999) & (ysoclass == 1))[0]
    ind2 = np.where((stetson > -99999) & (ysoclass == 2))[0]
    ind3 = np.where((stetson > -99999) & (ysoclass == 3))[0]
    ind4 = np.where((stetson > -99999) & (ysoclass == 4))[0]
    binning = np.arange(-10,50,1)
	
    plt.figure()
    plt.subplots_adjust(hspace=0.001)
    ax1 = plt.subplot(511)
    n0, bins, patches = plt.hist(stetson[ind0], bins=binning, facecolor = color[0], alpha = 1)
    plt.legend(['XYSOs'])
    plt.yticks(np.arange(1,max(n0),max(np.trunc(max(n0)/4),1) ))
    plt.plot([1.,1.],[0,max(n0)],"k--", lw=3)
	
    ax2 = plt.subplot(512, sharex=ax1)
    n1, bins, patches = plt.hist(stetson[ind1], bins=binning, facecolor = color[1], alpha = 1)
    plt.legend(['class 1'])
    plt.yticks(np.arange(1,max(n1),max(np.trunc(max(n1)/4),1) ))
    plt.plot([1.,1.],[0,max(n1)],"k--", lw=3)
    plt.text(30, 1, 'has longer tail')
	
    ax3 = plt.subplot(513, sharex=ax1)
    n2, bins, patches = plt.hist(stetson[ind2], bins=binning, facecolor = color[2], alpha = 1)
    plt.legend(['class 2'])
    plt.ylabel('number of objects')
    plt.yticks(np.arange(1,max(n2),max(np.trunc(max(n2)/4),1) ))
    plt.plot([1.,1.],[0,max(n2)],"k--", lw=3)
    plt.text(30, 1, 'has longer tail')
	
    ax4 = plt.subplot(514, sharex=ax1)
    n3, bins, patches = plt.hist(stetson[ind3], bins=binning, facecolor = color[3], alpha = 1)
    plt.legend(['class 3'])
    plt.yticks(np.arange(1,max(n3),max(np.trunc(max(n3)/4),1) ))
    plt.plot([1.,1.],[0,max(n3)],"k--", lw=3)
	
    ax5 = plt.subplot(515, sharex=ax1)
    n4, bins, patches = plt.hist(stetson[ind4], bins=binning, facecolor = color[4], alpha = 1)
    plt.legend(['stars'])
    plt.yticks(np.arange(1,max(n4),max(np.trunc(max(n4)/4),1) ))
    plt.plot([1.,1.],[0,max(n4)],"k--", lw=3)
	
    xticklabels = ax1.get_xticklabels()+ax2.get_xticklabels()+ax3.get_xticklabels()+ax4.get_xticklabels()
    plt.setp(xticklabels, visible=False)
	
    plt.text(3, 30,"varying (S.I. > 1)")
	
    plt.xlabel('Stetson index of [3.6], [4.5]')
	
    multisave(plt.gcf(), os.path.join(outroot, 'ysovar_stetson'))
	
	
    # Periodicity plot:
    good_period = infos['good_period']
    i0 = len(np.where((good_period > -99999) & (ysoclass == 0))[0])
    i00 = len(np.where( (ysoclass == 0))[0])
    i1 = len(np.where((good_period > -99999) & (ysoclass == 1))[0])
    i01 = len(np.where( (ysoclass == 1))[0])
    i2 = len(np.where((good_period > -99999) & (ysoclass == 2))[0])
    i02 = len(np.where( (ysoclass == 2))[0])
    i3 = len(np.where((good_period > -99999) & (ysoclass == 3))[0])
    i03 = len(np.where((ysoclass == 3))[0])
    i4 = len(np.where((good_period > -99999) & (ysoclass == 4))[0])
    i04 = len(np.where((ysoclass == 4))[0])
	
    f = plt.figure()
    plt.clf()
    x = np.arange(0,5)
    n_var = np.array([float(i0), float(i1), float(i2), float(i3), float(i4)])
    n_tot = np.array([float(i00), float(i01), float(i02), float(i03), float(i04)])
    y = n_var/n_tot
    print y
    y_err = np.sqrt( n_var)/n_tot
    print y_err
    plt.bar(x,y, color=color, yerr=y_err)
    plt.xlim(-0.5,5)
    plt.xticks(x+0.4, ('XYSOs', 'class 1', 'class 2', 'class 3', 'stars' ))
    plt.ylabel('fraction with significant periods*')
    plt.text(0, 0.6, '*periods > 2d and < 20d')
    plt.text(0, 0.57, 'with peak power > 10')
    plt.show()
    multisave(plt.gcf(), os.path.join(outroot, 'ysovar_period'))



def plot_lc(ax, data, mergedlc):
    ''' plot lc in a given axes container
    
    Parameters
    ----------
    data : dictionary
        contains 't36' and / or 't45' as time for lightcurves and 
        'm36' and / or 'm45' as magnitues for lightcurves
    '''
    if 't36' in data.keys():
        ax.scatter(data['t36']-mjdoffset, data['m36'], lw=0, s=20, marker='o', color='k', label = '[3.6]')
    if 't45' in data.keys():
        ax.scatter(data['t45']-mjdoffset, data['m45'], lw=1, s=20, marker='+', color='k', label = '[4.5]')
    if len(mergedlc['t']) > 0:
        ax.scatter(mergedlc['t']-mjdoffset, mergedlc['m36'], lw=0, s=30, marker='o', c=mergedlc['t'])
        ax.scatter(mergedlc['t']-mjdoffset, mergedlc['m45'], lw=2, s=40, marker='+', c=mergedlc['t'])

def setup_lcplot_axes(data, xlim, twinx = True, fig = None):
    '''set up axis containers for one or two lcs for a single object.

    This function checks the xlim and divides the space in the figure
    so that the xaxis has the same scale in each subplot.
    
    Parameters
    ----------
    data : dict
        This lightcurve is inspected for the number of bands present
    xlim : None or list
        None auto scales the x-axis
        list of [x0, x1] scales the xaxis form x1 to x2
        list of lists [[x00,x01], [x10, x11], ...] splits in multiple panels and
        each [x0, x1] pair gives the limits for one panel.
    twinx : boolean
        if true make seperate y axes for IRAC1 and IRAC2 if both are present
    fig: matplotlib figure instance or ``None``
        If ``None``, it creates a figure with the matplotlib defaults. Pass in a figure
        instance to customize e.g. the figure size.

    Returns
    -------
    fig : matplotlib figure instance
    axes : list of matplotlib axes instances
        This list holds the default axes (axis labels at the left and bottom).
    taxes : list of matplotlib axes instances
        This list holds the twin axes (labels on bottom and right).
    '''
    # twin axis only if really t1 and t2 are present
    if not(('t36' in data) and ('t45' in data)): twinx=False
    if xlim is None:
        # make an xlim for min(time) to max(time)
        if ('t36' in data): xlim = [data['t36'][0], data['t36'][-1]]
        if ('t45' in data): xlim = [data['t45'][0], data['t45'][-1]]
        if ('t36' in data) and ('t45' in data):
            xlim = [min(data['t36'][0], data['t45'][0]), max(data['t36'][-1], data['t45'][-1])]
    # test if xlim is a list of lists. If not, add one layer of [ ]
    try:  
        temp = xlim[0][0]
    except (TypeError, IndexError):
        xlim = [xlim]
    
    xlen = np.array([x[1]-x[0] for x in xlim], dtype = np.float)
    xtot = xlen.sum()
    x0 = .13 #leave enough space for labels on the left
    if twinx:
        x1 = .87 
    else:
        x1 = .95
    y0 = .15 #leave space for label on bottom
    y1 = .95 #leave space for title

    if fig is None:
        fig = plt.figure()
    axes = []
    taxes = []
    for i,xl in enumerate(xlim):
        xl = np.array(xl, dtype = float) - mjdoffset # ensure it's float for divisions
        axpos = [x0 + xlen[0:i].sum() / xtot * (x1-x0), y0, (xl[1] - xl[0]) / xtot * (x1-x0), y1-y0]
        if i == 0:
            ax = fig.add_axes(axpos, xlim = xl)
            ax.set_ylabel('mag')
        else:
            ax = fig.add_axes(axpos, xlim = xl,sharey = axes[0])
        axes.append(ax)
        if twinx:
            tax = ax.twinx()
            tax.ticklabel_format(useOffset=False, axis='y') 
            tax.set_xlim(xl)
            taxes.append(tax)
            tax.tick_params(axis='y', colors='r')
        else:
            taxes.append(None)
        # Special cases where e.g. first axes is treated differently
        if np.mod(i,2) == 0: ax.set_xlabel('time (MJD - '+str(mjdoffset)+' )')
        if i !=0: 
            plt.setp(ax.get_yticklabels(), visible=False)
        if twinx and i!= len(xlim)-1:
            plt.setp(tax.get_yticklabels(), visible=False)
    # some really fancy stuff to make the ylim connected on the twinx axes...
    # works partially...
    if twinx:
        taxes = np.array(taxes)
        # careful - connection works only in one direction!!!
        def update_twinx(tax1):
            y1, y2 = tax1.get_ylim()
            for tax in taxes[1:]:
                tax.set_ylim(y1,y2)

        taxes[0].callbacks.connect("ylim_changed", update_twinx)
        taxes[0].set_ylim([np.max(data['m45']), np.min(data['m45'])])
        taxes[-1].set_ylabel('[4.5]', color = 'r')
        axes[0].set_ylabel('[3.6]')
        taxes[0].invert_yaxis()
    axes[0].invert_yaxis()
    return fig, axes, taxes


def lc_plot(catalog, xlim = None, twinx = True):
    '''plot one or two lcs for a single object
    
    Parameters
    ----------
    catalog : single row from YSOVAR.atlas.YSOVAR_atlas
        contains 't1' and / or 't2' as time for lightcurves and 
        'm1' and / or 'm2' as magnitues for lightcurves
    xlim : None or list
        None auto scales the x-axis
        list of [x0, x1] scales the xaxis form x1 to x2
        list of lists [[x00,x01], [x10, x11], ...] splits in multiple panels and
        each [x0, x1] pair gives the limits for one panel.
    twinx : boolean
        if true make seperate y axes for IRAC1 and IRAC2 if both are present
    '''
    data = catalog.lclist[0]
    fig, axes, taxes = setup_lcplot_axes(data, xlim=xlim, twinx=twinx)

    mergedlc = merge_lc(data, ['36','45'])
    for ax, tax in zip(axes, taxes):
        if twinx and ('t36' in data) and ('t45' in data):
            # for each channel check that there is actually data there
            ax.scatter(data['t36']-mjdoffset, data['m36'], lw=0, s=20, marker='o', color='k', label = '[3.6], symbol: o')
            tax.scatter(data['t45']-mjdoffset, data['m45'], lw=1, s=20, marker='+', color='k', label = '[4.5], symbol: +')            
            if len(mergedlc) > 0:
                ax.scatter(mergedlc['t']-mjdoffset, mergedlc['m36'], lw=0, s=30, marker='o', c=mergedlc['t'])
                tax.scatter(mergedlc['t']-mjdoffset, mergedlc['m45'], lw=2, s=30, marker='+', c=mergedlc['t'])
            tax.tick_params(axis='y', colors='r')
        else:
            plot_lc(ax, data, mergedlc)

    if tax is None:
        ax.legend()
    else:
        taxes[0].invert_yaxis() # required here again, because ylim was reset
                                # in plotting in between

    return fig


def make_lc_plots(atlas, outroot, verbose = True, xlim = None, twinx = False, ind = None,
                  filedescription='_lc'):
    '''plot lightcurves into files for all objects in `atlas`
    
    Parameters
    ----------
    atlas : ysovar_atlas.YSOVAR_atlas
        contains dict with 't36' and / or 't45' as time for lightcurves and 
        'm36' and / or 'm45' as magnitues for lightcurves
    verbose : boolean
        if true print progress in processing
    xlim : None or list
        None auto scales the x-axis
        list of [x0, x1] scales the xaxis form x1 to x2
        list of lists [[x00,x01], [x10, x11], ...] splits in multiple panels and
        each [x0, x1] pais gives teh limits for one panel.
    twinx : boolean
        if true make separate y axes for IRAC1 and IRAC2 if both are present
    ind : list of integers
        index numbers of elements, only for those elements a lightcurve is created.
        If None, make lightcurve for all sources.
    filedescription : string
        Output files are named ``YSOVAR2_id + filedescription + extension``. The extension(s)
        is specified in ``YSOVAR.plots.filetype``.
        Use the ``filedescription`` parameters if this method is called more than once 
        per star.
    '''
    if ind is None: ind = np.arange(len(atlas))
    for i in ind:
        #print i
        if verbose and np.mod(i,100) == 0: 
            print 'lightcurve plots: ' + str(i) + ' of ' + str(len(atlas))
            plt.close("all")
        # make lightcurve plot:
        fig = lc_plot(atlas[i], xlim = xlim, twinx = twinx)
        filename = os.path.join(outroot, atlas['YSOVAR2_id'][i] + filedescription)
        multisave(fig, filename)
        plt.close("all")

def cmd_plot(atlas, mergedlc, redvec = None, verbose = True):
    '''
    Parameters
    ----------
    atlas : ysovar_atlas.YSOVAR_atlas with one row only 
    mergedlc : np.ndarray
        contains ``'t'`` as time for merged lightcurves and 
        ``'m36'`` and ``'m45'`` as magnitues for lightcurves
    redvec : float
        slope of reddening vector in the CMD. If ``None`` use default.

    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    p1 = ax.scatter(mergedlc['m36']-mergedlc['m45'], mergedlc['m36'], lw=0, s=40, marker='^', c=mergedlc['t'])
    ax.set_xlabel('[3.6] - [4.5]')
    ax.set_ylabel('[3.6]')
    # get x and y coordinates of plot
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[1], ylim[0]) # invert y axis!
    xlim = np.array(ax.get_xlim())
    x1 = xlim[0]
    x2 = xlim[1]
    y1 = ax.get_ylim()[0]
    y2 = ax.get_ylim()[1]
    # plot line for fit to data:
    m = atlas['cmd_m_36_45']
    b = atlas['cmd_b_36_45']
    line_y = m*xlim+b
    ax.plot(xlim, line_y, 'k-', label = 'measured slope')
    
    # plot line for shifted reddening vector to data:
    if redvec is None:
        m = redvecs['36_45'][0]
    else:
        m = redvec
    b = np.mean(ax.get_ylim()) - m*np.mean(xlim)    
    line_y = m*xlim+b
    ax.plot(xlim, line_y, 'k--', label = 'standard reddening')
    
    # plot reddening vector: (with length somewhat adjusted to the plot size)
    vector = make_reddeningvector_for_plot(x1, x2, y1, y2)
    if verbose: print vector
    ax.arrow(vector[0],m*vector[0]+b, vector[2]-vector[0],vector[3]-vector[1],fc="k", ec="k", head_width=0.025*(x2-x1))
    plot_angle = math.atan( vector[4]*(x2-x1)*3.2 / ((vector[2]-vector[0])*(y2-y1)*4) )/(2*np.pi)*360 # the 3.2 and the 4 comes from the actual size of the figure (angle does not work in data coordinates)
    #print plot_angle
    ax.text(vector[0],m*vector[0]+b, "$A_V = $ " + str(vector[5]) , rotation = plot_angle   )
    
    ax.set_title('CMD color-coded by time')
    
    # plot typical error bars in lower left corner
    y_err = np.median(mergedlc['m36_error'])
    x_err = np.sqrt( y_err**2 + (np.median(mergedlc['m45_error']))**2  )
    x1 = ax.get_xlim()[0]
    y1 = ax.get_ylim()[0]
    pos_x = x1 + 1.5*x_err
    pos_y = y1 - 1.5*y_err
    ax.errorbar(pos_x, pos_y, xerr = x_err, yerr = y_err, label='typical errors')
    ax.legend(prop={'size':12})
    
    return fig

def make_cmd_plots(atlas, outroot, verbose = True):
    '''plot cmds into files for all objects in `atlas`
    
    Parameters
    ----------
    atlas : :class:`~YSOVAR.atlas.YSOVAR_atlas`
        contains dict with ``'t36'`` and / or ``'t45'`` as time for lightcurves and 
        ``'m36'`` and / or ``'m45'`` as magnitues for lightcurves
    '''
    for i in range(len(atlas)):
        if verbose and np.mod(i,100) == 0: 
            print 'cmd plot: ' + str(i) + ' of ' + str(len(atlas))
        # make cmd plot:
        mergedlc = merge_lc(atlas.lclist[i], ['36','45'])
        if len(mergedlc) > 5:
            fig = cmd_plot(atlas[i], mergedlc)
            filename = os.path.join(outroot, atlas['YSOVAR2_id'][i] + '_color')
            multisave(fig, filename)
            plt.close(fig)

def make_lc_cmd_plots(atlas, outroot, lc_xlim = None, lc_twinx = False):
    '''plot CMDs and lightcurves

    See :meth:`make_lc_plots` and :meth:`make_cmd_plots` for documentation. 
    
    Parameters
    ----------
    atlas : :class:`~YSOVAR.atlas.YSOVAR_atlas`
        contains dict with ``'t36'`` and / or ``'t45'`` as time for lightcurves and 
        ``'m36'`` and / or ``'m45'`` as magnitues for lightcurves
    '''
    # basic lc plots and CMD
    make_lc_plots(atlas, outroot, verbose = True, xlim = lc_xlim)
    make_cmd_plots(atlas, outroot, verbose = True)

def plot_polys(atlas, outroot, verbose = True):
    '''plot lightcurves into files for all objects in data
    
    Parameters
    ----------
    atlas : :class:`~YSOVAR.atlas.YSOVAR_atlas`
        each ls in the atlas contains ``'t36'`` and / or ``'t45'`` as time
        for lightcurves and 
        ``'m36'`` and / or ``'m45'`` as magnitues for lightcurves
    outroot : string
        data path for saving resulting files
    verbose : boolean
        if true print progress in processing
    '''
    for i, d in enumerate(atlas.lclist):
        if verbose and np.mod(i,100) == 0: print 'lightcurve plots: ' + str(i) + ' of ' + str(len(atlas))
        # make lightcurve plot:
        if ('t36' in d.keys()) and (len(d['t36']) > 15):
            fig = lc.plot_all_polys(d['t36'], d['m36'], d['m36_error'], 'IRAC 1')
            filename = os.path.join(outroot, atlas['YSOVAR2_id'][i] + '_lcpoly')
            multisave(fig, filename)
            plt.close(fig)
        elif ('t45' in d.keys()) and (len(d['t45']) > 15):
            fig = lc.plot_all_polys(d['t45'], d['m45'], d['m45_error'], 'IRAC 2')
            filename = os.path.join(outroot, atlas['YSOVAR2_id'][i] + '_lcpoly')
            multisave(fig, filename)
            plt.close(fig)



def make_plot_skyview(outroot, infos):
	'''only for IRAS 20050: plots positions of identified YSOs over all detected sources
        This code is specific to the cluster IRAS 20050+2720 and should not be used
        for other regions.
        '''
	plt.clf()
	p1, = plt.plot(infos['ra'], infos['dec'], '.', color='0.75', markeredgecolor='0.75')
	
	i0 = np.where(infos['ysoclass'] == 0)[0]
	i1 = np.where(infos['ysoclass'] == 1)[0]
	i2 = np.where(infos['ysoclass'] == 2)[0]
	i3 = np.where(infos['ysoclass'] == 3)[0]

        
	
	p5, = plt.plot(infos['ra'][i2], infos['dec'][i2], 'o', markersize=5, color=(0.5,0.9,0.25))
	p6, = plt.plot(infos['ra'][i3], infos['dec'][i3], 'o', markersize=5, color=(0.25,0.45,1))
	p4, = plt.plot(infos['ra'][i1], infos['dec'][i1], 'o', markersize=5, color=(1,1,0.25))
	p3, = plt.plot(infos['ra'][i0], infos['dec'][i0], 'o', markersize=5, color=(1,0.35,0.35))
	
	plt.legend([p1,p3,p4,p5,p6],['time-resolved data','XYSOs','class 1', 'class 2', 'class 3'], 'lower right',prop={'size':10})
	
	ax = plt.gca()
	plt.axis([ax.get_xlim()[1], ax.get_xlim()[0], ax.get_ylim()[0], ax.get_ylim()[1]])
	plt.xlabel('RA')
	plt.ylabel('DEC')
	
	plt.savefig(os.path.join(outroot,  'skyview_iras20050.pdf'))
	plt.clf()



def make_ls_plots(atlas, outroot, maxper, oversamp, maxfreq, verbose = True):
    '''calculates & plots Lomb-Scargle periodogram for each source 
    
    Parameters
    ----------
    atlas : :class:`YSOVAR.atlas.YSOVAR_atlas`
        input atlas, which includes lightcurves
    outroot : string
        data path for saving resulting files
    maxper : float
        maximum period to be used for periodogram
    oversamp : integer
        oversampling factor
    maxfreq : float
        maximum frequency to be used for periodogram
    verbose : bool
        Show progress as output?
    '''
    fig = plt.figure()
    for i in np.arange(0,len(atlas)):
        if verbose and np.mod(i,100) == 0: print 'LS plot: ' + str(i) + ' of ' + str(len(atlas))
        fig.clf()
        data = atlas.lclist[i]
        ax = fig.add_subplot(111)
        if 't36' in data.keys():
            t1 = data['t36']
            m1 = data['m36']
            if len(t1) > 2:
                test1 = lombscargle.fasper(t1,m1,oversamp,maxfreq)
                ax.plot(1/test1[0],test1[1],linewidth=2, label = r'3.6 $\mu$m')
        if 't45' in  data.keys():
            t2 = data['t45']
            m2 = data['m45']
            if len(t2) > 2:
                test2 = lombscargle.fasper(t2,m2,oversamp,maxfreq)
                ax.plot(1/test2[0],test2[1],linewidth=2, label = r'4.5 $\mu$m')

        if ('t36' in data.keys() and (np.isfinite(test1[1]).sum() > 1)) or ('t45' in data.keys() and (np.isfinite(test2[1]).sum() > 1)):
            # The isfinite is here for very, VERY pathological cases
            # like constant lightcurves
            ax.legend(loc = 'upper right')
            ax.set_xscale('log')
            ax.set_xlabel('Period (d)')
            ax.set_ylabel('Periodogram power')
            ax.set_title('Lomb-Scargle Periodogram')
            multisave(fig, os.path.join(outroot, atlas['YSOVAR2_id'][i] + '_ls'))


def make_phased_lc_cmd_plots(atlas, outroot, bands = ['36','45'], marker = ['o', '+'], lw = [0,1], colorphase=True, lc_name = '_lc_phased'):
    '''plots phased lightcurves and CMDs for all sources
    
    Parameters
    ----------
    atlas : :class:`~YSOVAR.atlas.YSOVAR_atlas`
        input atlas, which includes lightcurves
    outroot : string
        data path for saving resulting files
    bands : list of strings
        band identifiers
    marker : list of valid matplotlib markers (e.g. string)
        marker for each band
    lw : list of floats
        linewidth for each band
    lc_name : string
        filenames for phased lightcurve plots
    colorphase : bool
        If true, entries in the lightcurves will be color coded by phase, if not, by time (to
        see if there are e.g. phase shifts over time). 
    '''
    # make phase-folded lightcurve plots for sources with "good" detected periods
    if len(marker) < len(bands): raise ValueError('Need one marker type per band')
    if len(lw) < len(bands): raise ValueError('Need to give linewidth for each band')
    fig = plt.figure()
    for i in np.arange(0,len(atlas)):
        data = atlas.lclist[i]
        if np.mod(i, 100) == 0:
            print 'phase plots: ' + str(i)
        x1 = 0
        x2 = 0
        y1 = 0
        y2 = 0
        fig.clf()
        ax = fig.add_subplot(111)
        plotdone = False
        if atlas['good_period'][i] > 0:
            period = atlas['good_period'][i]
            for j, band in enumerate(bands):
                if atlas['n_'+band][i] > 0:
                    plotdone = True
                    p = phase_fold(atlas.lclist[i]['t'+band], period)
                    if colorphase:
                        c = p
                    else:
                        c = atlas.lclist[i]['t'+band]
                    ax.scatter(p, atlas.lclist[i]['m'+band], marker=marker[j], lw = lw[j], s=40, c=c)
            if plotdone:
                ax.set_xlabel('phase')
                ax.set_title('phase-folded lightcurve, period = ' + str( ("%.2f" % period) ) + ' d')
                ax.set_ylim(ax.get_ylim()[::-1])
                multisave(fig, os.path.join(outroot, atlas['YSOVAR2_id'][i] + lc_name))
            
            # make phased color-magnitude plot
            fig.clf()
            ax = fig.add_subplot(111)
            mergedlc = merge_lc(data, ['36','45'])
            p = phase_fold(mergedlc['t'], period)
            if len(mergedlc) > 1:
                p1 = plt.scatter(mergedlc['m36']-mergedlc['m45'], mergedlc['m36'], lw=0, s=40, marker='^', c = p)
                ax.set_xlabel('[3.6] - [4.5]')
                ax.set_ylabel('[3.6]')
                # get x and y coordinates of plot
                x1 = ax.get_xlim()[0]
                x2 = ax.get_xlim()[1]
                y1 = ax.get_ylim()[0]
                y2 = ax.get_ylim()[1]
                ax.set_ylim([y2, y1]) # invert y axis!
                
                ax.set_title('CMD color-coded by phase, period = ' + str( ("%.2f" % atlas['good_period'][i]) ) + ' d')
                
                filename = os.path.join(outroot, atlas['YSOVAR2_id'][i] + '_color_phased')
                multisave(fig, filename)



def make_sed_plots(infos, outroot, title = 'SED (data from Guenther+ 2012)', sed_bands = {}):
    ''' Plot SEDs for all objects in ``infos``

    Parameters
    ----------
    infos : :class:`~YSOVAR.atlas.YSOVAR_atlas` or :class:`astropy.table.Table`
        This input table holds the magnitudes in different bands that form the
        points of the SED.
    outroot : string
        path to an output directory where the plots are saved. Individual
        files will automatically named according to the YSOVAR id number.
    title : string
        Title for each plot. Can be an empty string.
    sed_bands : dictionary
        This dictionary specifies which bands are to be plotted in the SED.
        See :meth:`YSOVAR.atlas.get_sed` for a description of the 
        format for ``sed_bands``.
    '''
    if not sed_bands:
        raise ValueError('sed_bands needs to be set. An empty dictionary would produce an empty plot.')
    fig = plt.figure()
    for i in np.arange(0,len(infos)):
        print 'SED plots: ' + str(i)
        fig.clf()
        ax = fig.add_subplot(111)
        try:
            plot_sed = deepcopy(infos.fluxes[i])
            lambdas = infos.lambdas[i]
        except AttributeError:
            (lambdas, mags, mags_error, plot_sed) = get_sed(infos[i], sed_bands = sed_bands) 
        flag36 = 0
        flag45 = 0
        ax.plot(lambdas, plot_sed, 'o')
        ax.set_xlabel('wavelength ($\mu m$)')
        ax.set_ylabel('flux (erg s$^{-1}$ cm$^{-2}$ $\mu$m$^{-1}$)')
        ax.set_xlim(0.3,30)
        ax.set_title(title)
        ax.set_yscale('log')
        ax.set_xscale('log')
        multisave(fig, os.path.join(outroot, infos['YSOVAR2_id'][i] + '_sed'))


def extraplots_1():
	'''some random plots I was interested in; may or may not be relevant to others.
        Read the code if you want to know more about this function.
        '''
	# m1 min/max plot
	i1 = np.where(info_ysos[:,30] > -99999)[0]
	i10 = np.where( (info_ysos[:,30] > -99999) & (info_ysos[:,1] == 0) )[0]
	i11 = np.where( (info_ysos[:,30] > -99999) & (info_ysos[:,1] == 1) )[0]
	i12 = np.where( (info_ysos[:,30] > -99999) & (info_ysos[:,1] == 2) )[0]
	i13 = np.where( (info_ysos[:,30] > -99999) & (info_ysos[:,1] == 3) )[0]
	i14 = np.where( (info_stars[:,30] > -99999))[0]
	color = [(1,0.35,0.35), (1,0.95,0.35), (0.5,0.9,0.25), (0.25,0.45,1), (0.9,0.35,1)]
	plt.clf()
	p4 = plt.scatter( info_stars[i14,4], info_stars[i14,31] - info_stars[i14,30], marker='.', color = color[4] )
	p0 = plt.scatter( info_ysos[i10,4], info_ysos[i10,31] - info_ysos[i10,30], marker='o', color = color[0] )
	p1 = plt.scatter( info_ysos[i11,4], info_ysos[i11,31] - info_ysos[i11,30], marker='o', color = color[1] )
	p2 = plt.scatter( info_ysos[i12,4], info_ysos[i12,31] - info_ysos[i12,30], marker='o', color = color[2] )
	p3 = plt.scatter( info_ysos[i13,4], info_ysos[i13,31] - info_ysos[i13,30], marker='o', color = color[3] )
	plt.xlabel('mean i1')
	plt.ylabel('i1$_{max}$ - i1$_{min}$  ')
	plt.legend([p0, p1, p2, p3, p4], ['XYSOs', 'class 1', 'class 2', 'class 3', 'stars'], 'upper left')
	plt.savefig(outroot_overview + 'delta_mag_36.eps')
	
	# m2 min/max plot
	i1 = np.where(info_ysos[:,32] > -99999)[0]
	i10 = np.where( (info_ysos[:,32] > -99999) & (info_ysos[:,1] == 0) )[0]
	i11 = np.where( (info_ysos[:,32] > -99999) & (info_ysos[:,1] == 1) )[0]
	i12 = np.where( (info_ysos[:,32] > -99999) & (info_ysos[:,1] == 2) )[0]
	i13 = np.where( (info_ysos[:,32] > -99999) & (info_ysos[:,1] == 3) )[0]
	i14 = np.where( (info_stars[:,32] > -99999))[0]
	color = [(1,0.35,0.35), (1,0.95,0.35), (0.5,0.9,0.25), (0.25,0.45,1), (0.9,0.35,1)]
	plt.clf()
	p4 = plt.scatter( info_stars[i14,9], info_stars[i14,33] - info_stars[i14,32], marker='.', color = color[4] )
	p0 = plt.scatter( info_ysos[i10,9], info_ysos[i10,33] - info_ysos[i10,32], marker='o', color = color[0] )
	p1 = plt.scatter( info_ysos[i11,9], info_ysos[i11,33] - info_ysos[i11,32], marker='o', color = color[1] )
	p2 = plt.scatter( info_ysos[i12,9], info_ysos[i12,33] - info_ysos[i12,32], marker='o', color = color[2] )
	p3 = plt.scatter( info_ysos[i13,9], info_ysos[i13,33] - info_ysos[i13,32], marker='o', color = color[3] )
	plt.xlabel('mean i2')
	plt.ylabel('i2$_{max}$ - i2$_{min}$  ')
	plt.legend([p0, p1, p2, p3, p4], ['XYSOs', 'class 1', 'class 2', 'class 3', 'stars'], 'upper left')
	plt.savefig(outroot_overview + 'delta_mag_45.eps')
	
	# chisquared plot for m1
	i1 = np.where(info_ysos[:,28] > -99999)[0]
	i10 = np.where( (info_ysos[:,28] > -99999) & (info_ysos[:,1] == 0) )[0]
	i11 = np.where( (info_ysos[:,28] > -99999) & (info_ysos[:,1] == 1) )[0]
	i12 = np.where( (info_ysos[:,28] > -99999) & (info_ysos[:,1] == 2) )[0]
	i13 = np.where( (info_ysos[:,28] > -99999) & (info_ysos[:,1] == 3) )[0]
	i14 = np.where( (info_stars[:,28] > -99999))[0]
	color = [(1,0.35,0.35), (1,0.95,0.35), (0.5,0.9,0.25), (0.25,0.45,1), (0.9,0.35,1)]
	plt.clf()
	p4 = plt.scatter( info_stars[i14,4], log10(info_stars[i14,28]), marker='.', color = color[4] )
	p0 = plt.scatter( info_ysos[i10,4], log10(info_ysos[i10,28]), marker='o', color = color[0] )
	p1 = plt.scatter( info_ysos[i11,4], log10(info_ysos[i11,28]), marker='o', color = color[1] )
	p2 = plt.scatter( info_ysos[i12,4], log10(info_ysos[i12,28]), marker='o', color = color[2] )
	p3 = plt.scatter( info_ysos[i13,4], log10(info_ysos[i13,28]), marker='o', color = color[3] )
	plt.xlabel('mean [3.6] mag')
	plt.ylabel('log ( $\chi^2_{reduced}$ ) [3.6] ')
	plt.legend([p0, p1, p2, p3, p4], ['XYSOs', 'class 1', 'class 2', 'class 3', 'stars'])
	plt.savefig(outroot_overview + 'chisq_mag_36.eps')
	
	# chisquared plot for m2
	i1 = np.where(info_ysos[:,29] > -99999)[0]
	i10 = np.where( (info_ysos[:,29] > -99999) & (info_ysos[:,1] == 0) )[0]
	i11 = np.where( (info_ysos[:,29] > -99999) & (info_ysos[:,1] == 1) )[0]
	i12 = np.where( (info_ysos[:,29] > -99999) & (info_ysos[:,1] == 2) )[0]
	i13 = np.where( (info_ysos[:,29] > -99999) & (info_ysos[:,1] == 3) )[0]
	i14 = np.where( (info_stars[:,29] > -99999))[0]
	color = [(1,0.35,0.35), (1,0.95,0.35), (0.5,0.9,0.25), (0.25,0.45,1), (0.9,0.35,1)]
	plt.clf()
	p4 = plt.scatter( info_stars[i14,9], log10(info_stars[i14,29]), marker='.', color = color[4] )
	p0 = plt.scatter( info_ysos[i10,9], log10(info_ysos[i10,29]), marker='o', color = color[0] )
	p1 = plt.scatter( info_ysos[i11,9], log10(info_ysos[i11,29]), marker='o', color = color[1] )
	p2 = plt.scatter( info_ysos[i12,9], log10(info_ysos[i12,29]), marker='o', color = color[2] )
	p3 = plt.scatter( info_ysos[i13,9], log10(info_ysos[i13,29]), marker='o', color = color[3] )
	plt.xlabel('mean [4.5] mag')
	plt.ylabel('log ( $\chi^2_{reduced}$ ) [4.5] ')
	plt.legend([p0, p1, p2, p3, p4], ['XYSOs', 'class 1', 'class 2', 'class 3', 'stars'])
	plt.savefig(outroot_overview + 'chisq_mag_45.eps')
	
	# Stetson plot vs. m1
	i1 = np.where(info_ysos[:,12] > -99999)[0]
	i10 = np.where( (info_ysos[:,12] > -99999) & (info_ysos[:,1] == 0) )[0]
	i11 = np.where( (info_ysos[:,12] > -99999) & (info_ysos[:,1] == 1) )[0]
	i12 = np.where( (info_ysos[:,12] > -99999) & (info_ysos[:,1] == 2) )[0]
	i13 = np.where( (info_ysos[:,12] > -99999) & (info_ysos[:,1] == 3) )[0]
	i14 = np.where( (info_stars[:,12] > -99999))[0]
	color = [(1,0.35,0.35), (1,0.95,0.35), (0.5,0.9,0.25), (0.25,0.45,1), (0.9,0.35,1)]
	plt.clf()
	p4 = plt.scatter( info_stars[i14,4], info_stars[i14,12], marker='.', color = color[4] )
	p0 = plt.scatter( info_ysos[i10,4], info_ysos[i10,12], marker='o', color = color[0] )
	p1 = plt.scatter( info_ysos[i11,4], info_ysos[i11,12], marker='o', color = color[1] )
	p2 = plt.scatter( info_ysos[i12,4], info_ysos[i12,12], marker='o', color = color[2] )
	p3 = plt.scatter( info_ysos[i13,4], info_ysos[i13,12], marker='o', color = color[3] )
	#plt.yscale('log')
	plt.xlabel('mean [3.6] mag')
	plt.ylabel('Stetson index [3.6]/[4.5] ')
	#plt.axis([6,18,-3, 50])
	plt.axis([6,18,-3, 5])
	plt.show()
	plt.legend([p0, p1, p2, p3, p4], ['XYSOs', 'class 1', 'class 2', 'class 3', 'stars'], 'upper left')
	#plt.savefig('stetson_mag_36.eps')
	plt.savefig(outroot_overview + 'stetson_mag_36_zoom.eps')
	
	# stetson vs. chisquared plot
	i1 = np.where(info_ysos[:,12] > -99999)[0]
	i10 = np.where( (info_ysos[:,12] > -99999) & (info_ysos[:,1] == 0) )[0]
	i11 = np.where( (info_ysos[:,12] > -99999) & (info_ysos[:,1] == 1) )[0]
	i12 = np.where( (info_ysos[:,12] > -99999) & (info_ysos[:,1] == 2) )[0]
	i13 = np.where( (info_ysos[:,12] > -99999) & (info_ysos[:,1] == 3) )[0]
	i14 = np.where( (info_stars[:,12] > -99999))[0]
	color = [(1,0.35,0.35), (1,0.95,0.35), (0.5,0.9,0.25), (0.25,0.45,1), (0.9,0.35,1)]
	plt.clf()
	p4 = plt.scatter( log10(info_stars[i14,28]), info_stars[i14,12], marker='.', color = color[4] )
	p0 = plt.scatter( log10(info_ysos[i10,28]), info_ysos[i10,12], marker='o', color = color[0] )
	p1 = plt.scatter( log10(info_ysos[i11,28]), info_ysos[i11,12], marker='o', color = color[1] )
	p2 = plt.scatter( log10(info_ysos[i12,28]), info_ysos[i12,12], marker='o', color = color[2] )
	p3 = plt.scatter( log10(info_ysos[i13,28]), info_ysos[i13,12], marker='o', color = color[3] )
	#plt.yscale('log')
	plt.xlabel('log ( $\chi^2_{reduced}$ ) [3.6] ')
	plt.ylabel('Stetson index of [3.6] and [4.5] ')
	#plt.axis([-1,4,-3, 50])
	plt.axis([-0.5,1.5,-3, 10])
	plt.show()
	plt.legend([p0, p1, p2, p3, p4], ['XYSOs', 'class 1', 'class 2', 'class 3', 'stars'])
	#plt.savefig('stetson_chisq_36.eps')
	plt.savefig(outroot_overview + 'stetson_chisq_36_zoom.eps')
	
	# stetson vs. chisquared plot in logscale
	i1 = np.where(info_ysos[:,12] > -99999)[0]
	i10 = np.where( (info_ysos[:,12] > -99999) & (info_ysos[:,1] == 0) )[0]
	i11 = np.where( (info_ysos[:,12] > -99999) & (info_ysos[:,1] == 1) )[0]
	i12 = np.where( (info_ysos[:,12] > -99999) & (info_ysos[:,1] == 2) )[0]
	i13 = np.where( (info_ysos[:,12] > -99999) & (info_ysos[:,1] == 3) )[0]
	i14 = np.where( (info_stars[:,12] > -99999))[0]
	color = [(1,0.35,0.35), (1,0.95,0.35), (0.5,0.9,0.25), (0.25,0.45,1), (0.9,0.35,1)]
	plt.clf()
	p4 = plt.scatter( log10(info_stars[i14,28]), log10(info_stars[i14,12]), marker='.', color = color[4] )
	p0 = plt.scatter( log10(info_ysos[i10,28]), log10(info_ysos[i10,12]), marker='o', color = color[0] )
	p1 = plt.scatter( log10(info_ysos[i11,28]), log10(info_ysos[i11,12]), marker='o', color = color[1] )
	p2 = plt.scatter( log10(info_ysos[i12,28]), log10(info_ysos[i12,12]), marker='o', color = color[2] )
	p3 = plt.scatter( log10(info_ysos[i13,28]), log10(info_ysos[i13,12]), marker='o', color = color[3] )
	#plt.yscale('log')
	plt.xlabel('log ( $\chi^2_{reduced}$ ) [3.6] ')
	plt.ylabel('log (Stetson index) of [3.6] and [4.5] ')
	#plt.axis([-1,4,-3, 50])
	#plt.axis([-0.5,1.5,-3, 10])
	plt.show()
	plt.legend([p0, p1, p2, p3, p4], ['XYSOs', 'class 1', 'class 2', 'class 3', 'stars'])
	#plt.savefig('stetson_chisq_36.eps')
	plt.savefig(outroot_overview + 'stetson_chisq_36_log.eps')


def extraplots_2(data, infos, outroot_overview):
    '''some more random plots I was interested in; may or may not be relevant to others.

    Read the code if you want to know more about this function.
    '''
    # delta plot
    plt.clf()
    i1 = np.where(infos.delta_36 > -99999)[0]
    i10 = np.where( (infos.delta_36 > -99999) & (infos.ysoclass == 0) )[0]
    i11 = np.where( (infos.delta_36 > -99999) & (infos.ysoclass == 1) )[0]
    i12 = np.where( (infos.delta_36 > -99999) & (infos.ysoclass == 2) )[0]
    i13 = np.where( (infos.delta_36 > -99999) & (infos.ysoclass == 3) )[0]
    i14 = np.where( (infos.delta_36 > -99999) & (infos.ysoclass == 4) )[0]
    color = [(1,0.35,0.35), (1,0.95,0.35), (0.5,0.9,0.25), (0.25,0.45,1), (0.9,0.35,1)]
    plt.clf()
    p4 = plt.scatter( infos.median_36[i14], infos.delta_36[i14], marker='.', color = color[4] )
    p0 = plt.scatter( infos.median_36[i10], infos.delta_36[i10], marker='o', color = color[0] )
    p1 = plt.scatter( infos.median_36[i11], infos.delta_36[i11], marker='o', color = color[1] )
    p2 = plt.scatter( infos.median_36[i12], infos.delta_36[i12], marker='o', color = color[2] )
    p3 = plt.scatter( infos.median_36[i13], infos.delta_36[i13], marker='o', color = color[3] )
    plt.xlabel('median $3.6\mu\mathrm{m}$ flux (mag)')
    plt.ylabel('$3.6\mu\mathrm{m}$ variability (mag)')
    plt.legend([p0, p1, p2, p3, p4], ['XYSOs', 'class 1', 'class 2', 'class 3', 'stars'], 'upper left')
    plt.axis([6,18,0,0.3])
    plt.savefig(outroot_overview + 'delta_mag_36.pdf')
    
    # stetson plot
    plt.clf()
    i1 = np.where(infos.stetson > -99999)[0]
    i10 = np.where( (infos.stetson > -99999) & (infos.ysoclass == 0) )[0]
    i11 = np.where( (infos.stetson > -99999) & (infos.ysoclass == 1) )[0]
    i12 = np.where( (infos.stetson > -99999) & (infos.ysoclass == 2) )[0]
    i13 = np.where( (infos.stetson > -99999) & (infos.ysoclass == 3) )[0]
    i14 = np.where( (infos.stetson > -99999) & (infos.ysoclass == 4) )[0]
    color = [(1,0.35,0.35), (1,0.95,0.35), (0.5,0.9,0.25), (0.25,0.45,1), (0.9,0.35,1)]
    plt.clf()
    p4 = plt.scatter( infos.median_36[i14], infos.stetson[i14], marker='.', color = color[4] )
    p0 = plt.scatter( infos.median_36[i10], infos.stetson[i10], marker='o', color = color[0] )
    p1 = plt.scatter( infos.median_36[i11], infos.stetson[i11], marker='o', color = color[1] )
    p2 = plt.scatter( infos.median_36[i12], infos.stetson[i12], marker='o', color = color[2] )
    p3 = plt.scatter( infos.median_36[i13], infos.stetson[i13], marker='o', color = color[3] )
    plt.xlabel('median $3.6\mu\mathrm{m}$ flux (mag)')
    plt.ylabel('Stetson index of $3.6\mu\mathrm{m}$ and $4.5\mu\mathrm{m}$')
    plt.legend([p0, p1, p2, p3, p4], ['XYSOs', 'class 1', 'class 2', 'class 3', 'stars'], 'upper left')
    plt.axis([6,18,-10,60])
    plt.savefig(outroot_overview + 'stetson_mag_36.pdf')
     
    # stetson plot (logarithmic of absolute)
    plt.clf()
    i1 = np.where(infos.stetson > -99999)[0]
    i10 = np.where( (infos.stetson > -99999) & (infos.ysoclass == 0) )[0]
    i11 = np.where( (infos.stetson > -99999) & (infos.ysoclass == 1) )[0]
    i12 = np.where( (infos.stetson > -99999) & (infos.ysoclass == 2) )[0]
    i13 = np.where( (infos.stetson > -99999) & (infos.ysoclass == 3) )[0]
    i14 = np.where( (infos.stetson > -99999) & (infos.ysoclass == 4) )[0]
    color = [(1,0.35,0.35), (1,0.95,0.35), (0.5,0.9,0.25), (0.25,0.45,1), (0.9,0.35,1)]
    plt.clf()
    p4 = plt.scatter( infos.median_36[i14], np.abs(infos.stetson[i14]), marker='.', color = color[4] )
    p0 = plt.scatter( infos.median_36[i10], np.abs(infos.stetson[i10]), marker='o', color = color[0] )
    p1 = plt.scatter( infos.median_36[i11], np.abs(infos.stetson[i11]), marker='o', color = color[1] )
    p2 = plt.scatter( infos.median_36[i12], np.abs(infos.stetson[i12]), marker='o', color = color[2] )
    p3 = plt.scatter( infos.median_36[i13], np.abs(infos.stetson[i13]), marker='o', color = color[3] )
    plt.xlabel('median $3.6\mu\mathrm{m}$ flux (mag)')
    plt.ylabel('|Stetson index| of $3.6\mu\mathrm{m}$ and $4.5\mu\mathrm{m}$')
    plt.legend([p0, p1, p2, p3, p4], ['XYSOs', 'class 1', 'class 2', 'class 3', 'stars'], 'upper left')
    plt.yscale('log')
    plt.savefig(outroot_overview + 'stetson_log_mag_36.pdf')
    
    # periods plot
    plt.clf()
    i1 = np.where(infos.good_period > -99999)[0]
    i10 = np.where( (infos.good_period > -99999) & (infos.median_36 > 0) & (infos.ysoclass == 0) )[0]
    i11 = np.where( (infos.good_period > -99999) & (infos.median_36 > 0) & (infos.ysoclass == 1) )[0]
    i12 = np.where( (infos.good_period > -99999) & (infos.median_36 > 0) & (infos.ysoclass == 2) )[0]
    i13 = np.where( (infos.good_period > -99999) & (infos.median_36 > 0) & (infos.ysoclass == 3) )[0]
    i14 = np.where( (infos.good_period > -99999) & (infos.median_36 > 0) & (infos.ysoclass == 4) )[0]
    color = [(1,0.35,0.35), (1,0.95,0.35), (0.5,0.9,0.25), (0.25,0.45,1), (0.9,0.35,1)]
    plt.clf()
    p4 = plt.scatter( infos.median_36[i14], infos.good_period[i14], marker='.', color = color[4] )
    p0 = plt.scatter( infos.median_36[i10], infos.good_period[i10], marker='o', color = color[0] )
    p1 = plt.scatter( infos.median_36[i11], infos.good_period[i11], marker='o', color = color[1] )
    p2 = plt.scatter( infos.median_36[i12], infos.good_period[i12], marker='o', color = color[2] )
    p3 = plt.scatter( infos.median_36[i13], infos.good_period[i13], marker='o', color = color[3] )
    plt.xlabel('median $3.6\mu\mathrm{m}$ flux (mag)')
    plt.ylabel('Detected period in lightcurve (d)')
    plt.legend([p0, p1, p2, p3, p4], ['XYSOs', 'class 1', 'class 2', 'class 3', 'stars'], 'lower left')
    plt.axis([6,17,1,25])
    plt.savefig(outroot_overview + 'period_mag_36.pdf')
    
    # slope plot
    plt.clf()
    alpha_shift = deepcopy(infos.cmd_alpha)
    bad = (alpha_shift < 0) & (alpha_shift > -99999)
    alpha_shift[bad] = (alpha_shift[bad] + np.pi)
    i1 = np.where(infos.cmd_alpha > -99999)[0]
    i10 = np.where( (infos.cmd_alpha > -99999) & (infos.ysoclass == 0) )[0]
    i11 = np.where( (infos.cmd_alpha > -99999) & (infos.ysoclass == 1) )[0]
    i12 = np.where( (infos.cmd_alpha > -99999) & (infos.ysoclass == 2) )[0]
    i13 = np.where( (infos.cmd_alpha > -99999) & (infos.ysoclass == 3) )[0]
    i14 = np.where( (infos.cmd_alpha > -99999) & (infos.ysoclass == 4) )[0]
    color = [(1,0.35,0.35), (1,0.95,0.35), (0.5,0.9,0.25), (0.25,0.45,1), (0.9,0.35,1)]
    plt.clf()
    p4 = plt.scatter( infos.median_36[i14], alpha_shift[i14]/(2*np.pi) * 360, marker='.', color = color[4] )
    p0 = plt.scatter( infos.median_36[i10], alpha_shift[i10]/(2*np.pi) * 360, marker='o', color = color[0] )
    p1 = plt.scatter( infos.median_36[i11], alpha_shift[i11]/(2*np.pi) * 360, marker='o', color = color[1] )
    p2 = plt.scatter( infos.median_36[i12], alpha_shift[i12]/(2*np.pi) * 360, marker='o', color = color[2] )
    p3 = plt.scatter( infos.median_36[i13], alpha_shift[i13]/(2*np.pi) * 360, marker='o', color = color[3] )
    plt.xlabel('median $3.6\mu\mathrm{m}$ flux (mag)')
    plt.ylabel('Slope angle in color-magnitude diagram (deg)')
    plt.legend([p0, p1, p2, p3, p4], ['XYSOs', 'class 1', 'class 2', 'class 3', 'stars'], 'upper left')
    plt.axis([6,22,0,180])
    plt.plot([16.8,16.8],[53,63],'k-')
    plt.text(17,56,'extinction dominated')
    plt.plot([16.8,16.8],[20,50],'k-')
    plt.text(17,40,'hot & cool spots')
    plt.plot([16.8,16.8],[95,175],'k-')
    plt.text(17,135,'accretion dominated')
    plt.plot([16.5,16.5],[5,95],'k-')
    plt.text(16.7,83,'mix of processes')
    plt.text(16.7,75,'(accr./extinc./spots)')
    plt.savefig(outroot_overview + 'alpha_mag_36.pdf')
    
    
    # slope plot (without stars)
    plt.clf()
    alpha_shift = deepcopy(infos.cmd_alpha)
    bad = (alpha_shift < 0) & (alpha_shift > -99999)
    alpha_shift[bad] = (alpha_shift[bad] + np.pi)
    i1 = np.where(infos.cmd_alpha > -99999)[0]
    i10 = np.where( (infos.cmd_alpha > -99999) & (infos.ysoclass == 0) )[0]
    i11 = np.where( (infos.cmd_alpha > -99999) & (infos.ysoclass == 1) )[0]
    i12 = np.where( (infos.cmd_alpha > -99999) & (infos.ysoclass == 2) )[0]
    i13 = np.where( (infos.cmd_alpha > -99999) & (infos.ysoclass == 3) )[0]
    i14 = np.where( (infos.cmd_alpha > -99999) & (infos.ysoclass == 4) )[0]
    color = [(1,0.35,0.35), (1,0.95,0.35), (0.5,0.9,0.25), (0.25,0.45,1), (0.9,0.35,1)]
    plt.clf()
    p0 = plt.scatter( infos.median_36[i10], alpha_shift[i10]/(2*np.pi) * 360, marker='o', color = color[0] )
    p1 = plt.scatter( infos.median_36[i11], alpha_shift[i11]/(2*np.pi) * 360, marker='o', color = color[1] )
    p2 = plt.scatter( infos.median_36[i12], alpha_shift[i12]/(2*np.pi) * 360, marker='o', color = color[2] )
    p3 = plt.scatter( infos.median_36[i13], alpha_shift[i13]/(2*np.pi) * 360, marker='o', color = color[3] )
    plt.xlabel('median $3.6\mu\mathrm{m}$ flux (mag)')
    plt.ylabel('Slope angle in color-magnitude diagram (deg)')
    plt.legend([p0, p1, p2, p3, p4], ['XYSOs', 'class 1', 'class 2', 'class 3'], 'upper left')
    plt.axis([6,22,0,180])
    plt.plot([16.8,16.8],[53,63],'k-')
    plt.text(17,56,'extinction dominated')
    plt.plot([16.8,16.8],[20,50],'k-')
    plt.text(17,40,'hot & cool spots')
    plt.plot([16.8,16.8],[95,175],'k-')
    plt.text(17,135,'accretion dominated')
    plt.plot([16.5,16.5],[5,95],'k-')
    plt.text(16.7,83,'mix of processes')
    plt.text(16.7,75,'(accr./extinc./spots)')
    plt.savefig(outroot_overview + 'alpha_mag_36_new.pdf')
    
    # Periodicity plot:
    i0 = len(np.where((infos.good_period > -99999) & (infos.ysoclass == 0))[0])
    i00 = len(np.where( (infos.ysoclass == 0))[0])
    i1 = len(np.where((infos.good_period > -99999) & (infos.ysoclass == 1))[0])
    i01 = len(np.where( (infos.ysoclass == 1))[0])
    i2 = len(np.where((infos.good_period > -99999) & (infos.ysoclass == 2))[0])
    i02 = len(np.where( (infos.ysoclass == 2))[0])
    i3 = len(np.where((infos.good_period > -99999) & (infos.ysoclass == 3))[0])
    i03 = len(np.where((infos.ysoclass == 3))[0])
    i4 = len(np.where((infos.good_period > -99999) & (infos.ysoclass == 4))[0])
    i04 = len(np.where((infos.ysoclass == 4))[0])
    plt.clf()
    x = np.arange(0,5)
    n_var = np.array([float(i0), float(i1), float(i2), float(i3), float(i4)])
    n_tot = np.array([float(i00), float(i01), float(i02), float(i03), float(i04)])
    y = n_var/n_tot
    print y
    y_err = np.sqrt( n_var)/n_tot
    print y_err
    x = x[0:4]
    y = y[0:4]
    plt.bar(x,y, color=color[0:4], yerr=y_err[0:4])
    plt.axis([-0.5,4.5,0,1])
    #plt.xticks(x+0.4, ('XYSOs', 'class 1', 'class 2', 'class 3', 'stars' ))
    plt.xticks(x+0.4, ('XYSOs', 'class 1', 'class 2', 'class 3'))
    plt.ylabel('fraction of objects with detected periods')
    #plt.text(0, 0.6, '*periods > 2d and < 20d')
    #plt.text(0, 0.57, 'with peak power > 10')
    plt.savefig(outroot_overview + 'ysovar_period.pdf')
    
    # Variability plot:
    i0 = len(np.where((infos.stetson > 0.9) & (infos.ysoclass == 0))[0])
    i00 = len(np.where((infos.stetson > -99999) & (infos.ysoclass == 0))[0])
    i1 = len(np.where((infos.stetson > 0.9) & (infos.ysoclass == 1))[0])
    i01 = len(np.where((infos.stetson > -99999) & (infos.ysoclass == 1))[0])
    i2 = len(np.where((infos.stetson > 0.9) & (infos.ysoclass == 2))[0])
    i02 = len(np.where((infos.stetson > -99999) & (infos.ysoclass == 2))[0])
    i3 = len(np.where((infos.stetson > 0.9) & (infos.ysoclass == 3))[0])
    i03 = len(np.where((infos.stetson > -99999) &(infos.ysoclass == 3))[0])
    i4 = len(np.where((infos.stetson > 0.9) & (infos.ysoclass == 4))[0])
    i04 = len(np.where((infos.stetson > -99999) &(infos.ysoclass == 4))[0])
    plt.clf()
    x = np.arange(0,5)
    n_var = np.array([float(i0), float(i1), float(i2), float(i3), float(i4)])
    n_tot = np.array([float(i00), float(i01), float(i02), float(i03), float(i04)])
    y = n_var/n_tot
    print y
    y_err = np.sqrt( n_var)/n_tot
    print y_err
    #x = x[0:4]
    #y = y[0:4]
    plt.bar(x,y, color=color, yerr=y_err)
    plt.axis([-0.5,5.5,0,1.1])
    plt.xticks(x+0.4, ('XYSOs', 'class 1', 'class 2', 'class 3', 'stars' ))
    #plt.xticks(x+0.4, ('XYSOs', 'class 1', 'class 2', 'class 3'))
    plt.ylabel('fraction of objects with detected variability')
    #plt.text(0, 0.6, '*periods > 2d and < 20d')
    #plt.text(0, 0.57, 'with peak power > 10')
    plt.savefig(outroot_overview + 'ysovar_variable.pdf')




def fancyplot(x, y, outroot, filename, colors, ind,  xlabel, ylabel, marker, xlog=False, ylog=False, legendtext='', legendpos='upper right'):
    plt.clf()
    if xlog == True:
	plt.xscale('log')
    
    if ylog == True:
	plt.yscale('log')
    
    pids = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6']
    for i in np.arange(0, len(colors)):
	pids[i] = plt.scatter(x[ind[i]], y[ind[i]], marker=marker[i], color=colors[i] )
    
    if legendtext != '':
	plt.legend(pids, legendtext, legendpos)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.axis(axis)
    plt.savefig(outroot+filename)
        
