# -*- coding: utf-8 -*-
import urllib
import urllib2
import math

import numpy as np
from copy import deepcopy
import string

import matplotlib
import matplotlib.pyplot as plt
import pylab

import ysovar_lombscargle
import lightcurves as lc

from ysovar_atlas import *

filetype = ['.eps']
mjdoffset = 55000
YSOVAR_USERNAME = None
YSOVAR_PASSWORD = None

def multisave(fig, filename):
    for t in filetype:
        fig.savefig(filename + t)
        

def get_stamps(data, outroot, verbose = True):
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

    for i, d in enumerate(data):
        if np.mod(i, 100) == 0:
            print 'Fetching stamp ' + str(i) + ' of ' + str(len(data)) + ' from YSOVAR2 Server.'
        dat = urllib.urlencode({'source_id' : d['YSOVAR2_id'][0]})
        req = urllib2.Request(url, dat)
        response = urllib2.urlopen(req)
        try:   # with statement would be shorter, but is not available in python 2.6
            f = open(outroot + str(i)+'_stamp.png', 'w')
            f.write(response.read())
        finally:
            f.close()

def make_reddeningvector_for_plot(x1, x2, y1, y2):
    # calculates the coordinates of the reddening evctor in convenient plot coordinates.
    slope = calc_reddening()
    
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


def make_info_plots(infos, outroot):
    # makes overview statistics plots
    #color definition:
    color = [(1,0.35,0.35), (1,0.95,0.35), (0.5,0.9,0.25), (0.25,0.45,1), (0.9,0.35,1)]
    
    # MAD 3.6 PLOT:
    ind0 = np.where((infos.mad_36 > -99999) & (infos.ysoclass == 0))[0]
    ind1 = np.where((infos.mad_36 > -99999) & (infos.ysoclass == 1))[0]
    ind2 = np.where((infos.mad_36 > -99999) & (infos.ysoclass == 2))[0]
    ind3 = np.where((infos.mad_36 > -99999) & (infos.ysoclass == 3))[0]
    ind4 = np.where((infos.mad_36 > -99999) & (infos.ysoclass == 4))[0]
    binning = np.arange(0,0.2,0.005)
    
    plt.figure()
    plt.subplots_adjust(hspace=0.001)
    ax1 = plt.subplot(511)
    n0, bins, patches = plt.hist(infos.mad_36[ind0], bins=binning, facecolor = color[0])
    plt.legend(['XYSOs'])
    plt.yticks(np.arange(1,max(n0),max(np.trunc(max(n0)/4),1) ))
    
    ax2 = plt.subplot(512, sharex=ax1)
    n1, bins, patches = plt.hist(infos.mad_36[ind1], bins=binning, facecolor = color[1])
    plt.legend(['class 1'])
    plt.yticks(np.arange(1,max(n1),max(np.trunc(max(n1)/4),1) ))
    
    ax3 = plt.subplot(513, sharex=ax1)
    n2, bins, patches = plt.hist(infos.mad_36[ind2], bins=binning, facecolor = color[2])
    plt.legend(['class 2'])
    plt.ylabel('number of objects')
    plt.yticks(np.arange(1,max(n2),max(np.trunc(max(n2)/4),1) ))
    
    ax4 = plt.subplot(514, sharex=ax1)
    n3, bins, patches = plt.hist(infos.mad_36[ind3], bins=binning, facecolor = color[3])
    plt.legend(['class 3'])
    plt.yticks(np.arange(1,max(n3),max(np.trunc(max(n3)/4),1) ))
    
    ax5 = plt.subplot(515, sharex=ax1)
    n4, bins, patches = plt.hist(infos.mad_36[ind4], bins=binning, facecolor = color[4])
    plt.legend(['stars'])
    plt.yticks(np.arange(1,max(n4),max(np.trunc(max(n4)/4),1) ))
    
    xticklabels = ax1.get_xticklabels()+ax2.get_xticklabels()+ax3.get_xticklabels()+ax4.get_xticklabels()
    plt.setp(xticklabels, visible=False)
    plt.xlabel('MAD in [3.6]')
    plt.show()
    plt.savefig(outroot + 'ysovar_mad36.eps')
    
    
    
    # MAD 4.5 PLOT:
    ind0 = np.where((infos.mad_45 > -99999) & (infos.ysoclass == 0))[0]
    ind1 = np.where((infos.mad_45 > -99999) & (infos.ysoclass == 1))[0]
    ind2 = np.where((infos.mad_45 > -99999) & (infos.ysoclass == 2))[0]
    ind3 = np.where((infos.mad_45 > -99999) & (infos.ysoclass == 3))[0]
    ind4 = np.where((infos.mad_45 > -99999) & (infos.ysoclass == 4))[0]
    binning = np.arange(0,0.2,0.005)
    
    plt.figure()
    plt.subplots_adjust(hspace=0.001)
    ax1 = plt.subplot(511)
    n0, bins, patches = plt.hist(infos.mad_45[ind0], bins=binning, facecolor = color[0])
    plt.legend(['XYSOs'])
    plt.yticks(np.arange(1,max(n0),max(np.trunc(max(n0)/4),1) ))
    
    ax2 = plt.subplot(512, sharex=ax1)
    n1, bins, patches = plt.hist(infos.mad_45[ind1], bins=binning, facecolor = color[1])
    plt.legend(['class 1'])
    plt.yticks(np.arange(1,max(n1),max(np.trunc(max(n1)/4),1) ))
    
    ax3 = plt.subplot(513, sharex=ax1)
    n2, bins, patches = plt.hist(infos.mad_45[ind2], bins=binning, facecolor = color[2])
    plt.legend(['class 2'])
    plt.ylabel('number of objects')
    plt.yticks(np.arange(1,max(n2),max(np.trunc(max(n2)/4),1) ))
    
    ax4 = plt.subplot(514, sharex=ax1)
    n3, bins, patches = plt.hist(infos.mad_45[ind3], bins=binning, facecolor = color[3])
    plt.legend(['class 3'])
    plt.yticks(np.arange(1,max(n3),max(np.trunc(max(n3)/4),1) ))
    
    ax5 = plt.subplot(515, sharex=ax1)
    n4, bins, patches = plt.hist(infos.mad_45[ind4], bins=binning, facecolor = color[4])
    plt.legend(['stars'])
    plt.yticks(np.arange(1,max(n4),max(np.trunc(max(n4)/4),1) ))
    
    xticklabels = ax1.get_xticklabels()+ax2.get_xticklabels()+ax3.get_xticklabels()+ax4.get_xticklabels()
    plt.setp(xticklabels, visible=False)
    plt.xlabel('MAD in [4.5]')
    plt.show()
    plt.savefig(outroot + 'ysovar_mad45.eps')
    
    
    # STETSON PLOT:    
    plt.clf()
    ind0 = np.where((infos.stetson > -99999) & (infos.ysoclass == 0))[0]
    ind1 = np.where((infos.stetson > -99999) & (infos.ysoclass == 1))[0]
    ind2 = np.where((infos.stetson > -99999) & (infos.ysoclass == 2))[0]
    ind3 = np.where((infos.stetson > -99999) & (infos.ysoclass == 3))[0]
    ind4 = np.where((infos.stetson > -99999) & (infos.ysoclass == 4))[0]
    binning = np.arange(-10,50,1)
    
    plt.figure()
    plt.subplots_adjust(hspace=0.001)
    ax1 = plt.subplot(511)
    n0, bins, patches = plt.hist(infos.stetson[ind0], bins=binning, facecolor = color[0], alpha = 1)
    plt.legend(['XYSOs'])
    plt.yticks(np.arange(1,max(n0),max(np.trunc(max(n0)/4),1) ))
    plt.plot([1.,1.],[0,max(n0)],"k--", lw=3)
    
    ax2 = plt.subplot(512, sharex=ax1)
    n1, bins, patches = plt.hist(infos.stetson[ind1], bins=binning, facecolor = color[1], alpha = 1)
    plt.legend(['class 1'])
    plt.yticks(np.arange(1,max(n1),max(np.trunc(max(n1)/4),1) ))
    plt.plot([1.,1.],[0,max(n1)],"k--", lw=3)
    plt.text(30, 1, 'has longer tail')
    
    ax3 = plt.subplot(513, sharex=ax1)
    n2, bins, patches = plt.hist(infos.stetson[ind2], bins=binning, facecolor = color[2], alpha = 1)
    plt.legend(['class 2'])
    plt.ylabel('number of objects')
    plt.yticks(np.arange(1,max(n2),max(np.trunc(max(n2)/4),1) ))
    plt.plot([1.,1.],[0,max(n2)],"k--", lw=3)
    plt.text(30, 1, 'has longer tail')
    
    ax4 = plt.subplot(514, sharex=ax1)
    n3, bins, patches = plt.hist(infos.stetson[ind3], bins=binning, facecolor = color[3], alpha = 1)
    plt.legend(['class 3'])
    plt.yticks(np.arange(1,max(n3),max(np.trunc(max(n3)/4),1) ))
    plt.plot([1.,1.],[0,max(n3)],"k--", lw=3)
    
    ax5 = plt.subplot(515, sharex=ax1)
    n4, bins, patches = plt.hist(infos.stetson[ind4], bins=binning, facecolor = color[4], alpha = 1)
    plt.legend(['stars'])
    plt.yticks(np.arange(1,max(n4),max(np.trunc(max(n4)/4),1) ))
    plt.plot([1.,1.],[0,max(n4)],"k--", lw=3)
    
    xticklabels = ax1.get_xticklabels()+ax2.get_xticklabels()+ax3.get_xticklabels()+ax4.get_xticklabels()
    plt.setp(xticklabels, visible=False)
    
    plt.text(3, 30,"varying (S.I. > 1)")
    
    plt.xlabel('Stetson index of [3.6], [4.5]')
    
    plt.savefig(outroot + 'ysovar_stetson.eps')
    
    
    # SLOPE PLOT
    plt.clf()
    ind0 = np.where((infos.cmd_m > -99999) & (infos.ysoclass == 0))[0]
    ind1 = np.where((infos.cmd_m > -99999) & (infos.ysoclass == 1))[0]
    ind2 = np.where((infos.cmd_m > -99999) & (infos.ysoclass == 2))[0]
    ind3 = np.where((infos.cmd_m > -99999) & (infos.ysoclass == 3))[0]
    ind4 = np.where((infos.cmd_m > -99999) & (infos.ysoclass == 4))[0]
    binning = np.arange(-3,4,0.25)
    
    f = plt.figure()
    plt.subplots_adjust(hspace=0.001)
    ax1 = plt.subplot(511)
    n0, bins, patches = plt.hist(infos.cmd_m[ind0], bins=binning, facecolor = color[0], alpha = 1)
    plt.legend(['XYSOs'], 'upper left')
    plt.yticks(np.arange(1,max(n0),max(np.trunc(max(n0)/4),1) ))
    plt.plot([1.65,1.65],[0,max(n0)],"k--", lw=3)
    
    ax2 = plt.subplot(512, sharex=ax1)
    n1, bins, patches = plt.hist(infos.cmd_m[ind1], bins=binning, facecolor = color[1], alpha = 1)
    plt.legend(['class 1'], 'upper left')
    plt.yticks(np.arange(1,max(n1),max(np.trunc(max(n1)/4),1) ))
    plt.plot([1.65,1.65],[0,max(n1)],"k--", lw=3)
    
    ax3 = plt.subplot(513, sharex=ax1)
    n2, bins, patches = plt.hist(infos.cmd_m[ind2], bins=binning, facecolor = color[2], alpha = 1)
    plt.legend(['class 2'], 'upper left')
    plt.ylabel('number of objects')
    plt.yticks(np.arange(1,max(n2),max(np.trunc(max(n2)/4),1) ))
    plt.plot([1.65,1.65],[0,max(n2)],"k--", lw=3)
    
    ax4 = plt.subplot(514, sharex=ax1)
    n3, bins, patches = plt.hist(infos.cmd_m[ind3], bins=binning, facecolor = color[3], alpha = 1)
    plt.legend(['class 3'], 'upper left')
    plt.yticks(np.arange(1,max(n3),max(np.trunc(max(n3)/4),1) ))
    plt.plot([1.65,1.65],[0,max(n3)],"k--", lw=3)
    
    ax5 = plt.subplot(515, sharex=ax1)
    n4, bins, patches = plt.hist(infos.cmd_m[ind4], bins=binning, facecolor = color[4], alpha = 1)
    plt.legend(['stars'], 'upper left')
    plt.yticks(np.arange(1,max(n4),max(np.trunc(max(n4)/4),1) ))
    plt.plot([1.65,1.65],[0,max(n4)],"k--", lw=3)
    
    xticklabels = ax1.get_xticklabels()+ax2.get_xticklabels()+ax3.get_xticklabels()+ax4.get_xticklabels()
    plt.setp(xticklabels, visible=False)
    
    plt.xlabel('color-magnitude slope')
    plt.annotate('standard reddening', xy = (1.69, 10), xytext=(2.0,20), arrowprops=dict(facecolor='black', shrink=0.1, width=2, frac=0.25))
    plt.text(-2,10,'accretion-like')
    plt.text(-0.5,20,'colorless')
    plt.savefig(outroot + 'ysovar_slope.eps')
    plt.clf()
    
    
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
    plt.savefig(outroot + 'ysovar_period.eps')
    plt.savefig(outroot + 'ysovar_period.png')

def plot_lc(ax, data):
    ''' plot lc in a given axes container
    
    Parameters
    ----------
    data : dictionary
        contains 't1' and / or 't2' as time for lightcurves and 
        'm1' and / or 'm2' as magnitues for lightcurves
    '''
    if 't1' in data.keys():
        ax.scatter(data['t1']-mjdoffset, data['m1'], lw=0, s=20, marker='o', color='k', label = '[3.6]')
    if 't2' in data.keys():
        ax.scatter(data['t2']-mjdoffset, data['m2'], lw=1, s=20, marker='+', color='k', label = '[4.5]')
    if len(data['t']) > 0:
        ax.scatter(data['t']-mjdoffset, data['m36'], lw=0, s=30, marker='o', c=data['t'])
        ax.scatter(data['t']-mjdoffset, data['m45'], lw=2, s=40, marker='+', c=data['t'])

def lc_plot(data, xlim = None, twinx = True):
    '''make the plot an one or two lcs for a single object
    
    Parameters
    ----------
    data : dictionary
        contains 't1' and / or 't2' as time for lightcurves and 
        'm1' and / or 'm2' as magnitues for lightcurves
    xlim : None or list
        None auto scales the x-axis
        list of [x0, x1] scales the xaxis form x1 to x2
        list of lists [[x00,x01], [x10, x11], ...] splits in multiple panels and
        each [x0, x1] pais gives teh limits for one panel.
    twinx : boolean
        if true make seperate y axes for IRAC1 and IRAC2 if both are present
    '''
    # twin axis only if really t1 and t2 are present
    if not(('t1' in data.keys()) and ('t2' in data.keys())): twinx=False
    if xlim is None:
        # make an xlim for min(time) to max(time)
        if ('t1' in data.keys()): xlim = [data['t1'][0], data['t1'][-1]]
        if ('t2' in data.keys()): xlim = [data['t2'][0], data['t2'][-1]]
        if ('t1' in data.keys()) and ('t2' in data.keys()):
            xlim = [min(data['t1'][0], data['t2'][0]), max(data['t1'][-1], data['t2'][-1])]
    # make an xlim for min(time) to max(time)
    if ('t1' in data.keys()): ylim = [np.max(data['m1']), np.min(data['m1'])]
    if ('t2' in data.keys()) and not twinx: ylim = [np.max(data['m2']), np.min(data['m2'])]
    if ('t1' in data.keys()) and ('t2' in data.keys()) and not twinx:
        allmagvals = np.hstack([data['m1'], data['m2']])
        ylim = [np.max(allmagvals), np.min(allmagvals)]
    # test is xlim is a list of lists. If not, add one layer of [ ]
    try:  
        temp = xlim[0][0]
    except (TypeError, IndexError):
        xlim = [xlim]
    
    xlen = np.array([x[1]-x[0] for x in xlim], dtype = np.float)
    xtot = xlen.sum()
    x0 = .1 #leave enough space for labels on the left
    x1 = .9 
    y0 = .15 #leave space for label on botton
    y1 = .95 #leave space for title

    fig = plt.figure()
    axes = []
    taxes = []
    for i,xl in enumerate(xlim):
        xl = np.array(xl, dtype = float) - mjdoffset # ensure it's float for devisions
        axpos = [x0 + xlen[0:i].sum() / xtot * (x1-x0), y0, (xl[1] - xl[0]) / xtot * (x1-x0), y1-y0]
        if i == 0:
            ax = fig.add_axes(axpos, xlim = xl, ylim = ylim)
            ax.set_ylabel('mag')
        else:
            ax = fig.add_axes(axpos, xlim = xl,sharey = axes[0])
        axes.append(ax)
        if twinx:
            tax = ax.twinx()
            tax.ticklabel_format(useOffset=False, axis='y') 
            tax.set_xlim(xl)
            taxes.append(tax)
            ax.scatter(data['t1']-mjdoffset, data['m1'], lw=0, s=20, marker='o', color='k', label = '[3.6], symbol: o')
            tax.scatter(data['t2']-mjdoffset, data['m2'], lw=1, s=20, marker='+', color='k', label = '[4.5], symbol: +')
            if len(data['t']) > 0:
                ax.scatter(data['t']-mjdoffset, data['m36'], lw=0, s=30, marker='o', c=data['t'])
                tax.scatter(data['t']-mjdoffset, data['m45'], lw=2, s=30, marker='+', c=data['t'])
            tax.tick_params(axis='y', colors='r')
        else:
            plot_lc(ax, data)
        # Special cases where e.g. first axes is treated different
        if np.mod(i,2) == 0: ax.set_xlabel('time (MJD - '+str(mjdoffset)+' )')
        if i ==0: 
            if not twinx: ax.legend()
        else:
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
                #tax.figure.canvas.draw()

        taxes[0].callbacks.connect("ylim_changed", update_twinx)
        taxes[0].set_ylim([np.max(data['m2']), np.min(data['m2'])])
        taxes[-1].set_ylabel('[4.5]', color = 'r')
        axes[0].set_ylabel('[3.6]')
    return fig


def make_lc_plots(data, outroot, verbose = True, xlim = None, twinx = False, ind = None):
    '''plot lightcurves into files for all objects in data
    
    Parameters
    ----------
    data : list of dictionary
        each dict contains 't1' and / or 't2' as time for lightcurves and 
        'm1' and / or 'm2' as magnitues for lightcurves
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
    '''
    if ind is None:
        ind = np.arange(len(data))
    for i in ind:
        print i
        if verbose and np.mod(i,100) == 0: 
            print 'lightcurve plots: ' + str(i) + ' of ' + str(len(data))
            plt.close("all")
        # make light curve plot:
        fig = lc_plot(data[i], xlim = xlim, twinx = twinx)
        filename = outroot + str(i) + '_lc'
        multisave(fig, filename)
        plt.close(fig)

def cmd_plot(data, infos):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    p1 = ax.scatter(data['m36']-data['m45'], data['m36'], lw=0, s=40, marker='^', c=data['t'])
    ax.set_xlabel('[3.6] - [4.5]')
    ax.set_ylabel('[3.6]')
    # get x and y coordinates of plot
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[1], ylim[0]) # invert y axis!
    x1 = ax.get_xlim()[0]
    x2 = ax.get_xlim()[1]
    y1 = ax.get_ylim()[0]
    y2 = ax.get_ylim()[1]
    # plot line for fit to data:
    m = infos['cmd_m']
    b = infos['cmd_b']
    line_x = np.array([x1, x2])
    line_y = np.array([m*x1+b, m*x2+b])
    ax.plot(line_x, line_y, 'k-', label = 'measured slope')
    
    # plot line for shifted reddening vector to data:
    m = infos['cmd_m_redvec']
    b = infos['cmd_b_redvec']
    line_x = np.array([x1, x2])
    line_y = np.array([m*x1+b, m*x2+b])
    ax.plot(line_x, line_y, 'k--', label = 'standard reddening')
    
    # plot reddening vector: (with length somewhat adjusted to the plot size)
    vector = make_reddeningvector_for_plot(x1, x2, y1, y2)
    print vector
    ax.arrow(vector[0],m*vector[0]+b, vector[2]-vector[0],vector[3]-vector[1],fc="k", ec="k", head_width=0.025*(x2-x1))
    plot_angle = math.atan( vector[4]*(x2-x1)*3.2 / ((vector[2]-vector[0])*(y2-y1)*4) )/(2*np.pi)*360 # the 3.2 and the 4 comes from the actual size of the figure (angle does not work in data coordinates)
    #print plot_angle
    ax.text(vector[0],m*vector[0]+b, "$A_V = $ " + str(vector[5]) , rotation = plot_angle   )
    
    ax.set_title('CMD color-coded by time')
    
    # plot typical error bars in lower left corner
    y_err = np.median(data['m36_error'])
    x_err = np.sqrt( y_err**2 + (np.median(data['m45_error']))**2  )
    x1 = ax.get_xlim()[0]
    y1 = ax.get_ylim()[0]
    pos_x = x1 + 1.5*x_err
    pos_y = y1 - 1.5*y_err
    ax.errorbar(pos_x, pos_y, xerr = x_err, yerr = y_err, label='typical errors')
    ax.legend(prop={'size':12})
    
    return fig

def make_cmd_plots(data, infos, outroot, verbose = True):
    for i, d in enumerate(data):
        if verbose and np.mod(i,100) == 0: 
            print 'cmd plot: ' + str(i) + ' of ' + str(len(data))
        # make cmd plot:
        if ('t' in d.keys()) and (len(d['t']) > 5):
            fig = cmd_plot(d, infos[i])
            filename = outroot + str(i) + '_color'
            multisave(fig, filename)
            plt.close(fig)

def make_lc_cmd_plots(data, infos, outroot, lc_xlim = None, lc_twinx = False):
        # basic lc plots and CMD
        make_lc_plots(data, outroot, verbose = True, xlim = lc_xlim)
        make_cmd_plots(data, infos, outroot, verbose = True)

def plot_polys(data, outroot, verbose = True):
    '''plot lightcurves into files for all objects in data
    
    Parameters
    ----------
    data : list of dictionary
        each dict contains 't1' and / or 't2' as time for lightcurves and 
        'm1' and / or 'm2' as magnitues for lightcurves
    outroot : string
        data path for saving resulting files
    verbose : boolean
        if true print progress in processing
    '''
    for i, d in enumerate(data):
        if verbose and np.mod(i,100) == 0: print 'lightcurve plots: ' + str(i) + ' of ' + str(len(data))
        # make light curve plot:
        if ('t1' in d.keys()) and (len(d['t1']) > 15):
            fig = lc.plot_all_polys(d['t1'], d['m1'], d['m1_error'], 'IRAC 1')
            filename = outroot + str(i) + '_lcpoly'
            multisave(fig, filename)
            plt.close(fig)
        elif ('t2' in d.keys()) and (len(d['t2']) > 15):
            fig = lc.plot_all_polys(d['t2'], d['m2'], d['m2_error'], 'IRAC 2')
            filename = outroot + str(i) + '_lcpoly'
            multisave(fig, filename)
            plt.close(fig)

def check_time_obs(outroot, ysovar1):
    
    plt.clf()
    x = np.array(ysovar1['ra'])
    y = np.array(ysovar1['dec'])
    t = np.array(ysovar1['hmjd1'])
    t_max = np.zeros(len(x))
    for i in np.arange(0,len(x)):
        t_max[i] = max(t[i])
    
    plt.clf()
    plt.scatter(x, y, lw=0, s=40, marker='.',  c=t_max)
    plt.savefig(outroot + 'FOV_time.eps')
    plt.clf()





def make_plot_skyview(outroot, ysovar1, ysovar2, infos):
    plt.clf()
    p1, = plt.plot(ysovar1['ra'], ysovar1['dec'], '.', color='0.75', markeredgecolor='0.75')
    p2, = plt.plot(ysovar2['ra'], ysovar2['dec'], '.', color='0.75', markeredgecolor='0.75')
    
    i0 = np.where(infos.ysoclass == 0)[0]
    i1 = np.where(infos.ysoclass == 1)[0]
    i2 = np.where(infos.ysoclass == 2)[0]
    i3 = np.where(infos.ysoclass == 3)[0]
    
    #p7, = plt.plot(guenther_data_stars['ra'], guenther_data_stars['dec'], 'mo')
    p5, = plt.plot(infos.ra_spitzer[i2], infos.dec_spitzer[i2], 'o', markersize=5, color=(0.5,0.9,0.25))
    p6, = plt.plot(infos.ra_spitzer[i3], infos.dec_spitzer[i3], 'o', markersize=5, color=(0.25,0.45,1))
    p4, = plt.plot(infos.ra_spitzer[i1], infos.dec_spitzer[i1], 'o', markersize=5, color=(1,1,0.25))
    p3, = plt.plot(infos.ra_spitzer[i0], infos.dec_spitzer[i0], 'o', markersize=5, color=(1,0.35,0.35))
    
    plt.legend([p1,p3,p4,p5,p6],['time-resolved data','Guenther+ 2012 XYSOs','Guenther+ 2012 class 1', 'Guenther+ 2012 class 2', 'Guenther+ 2012 class 3'], 'lower right',prop={'size':10})
    
    plt.xlabel('RA')
    plt.ylabel('DEC')
    
    plt.savefig(outroot + 'skyview_iras20050.eps')
    plt.clf()


def make_ls_plots(ysos, outroot, maxper, oversamp, maxfreq):
    fig = plt.figure()
    for i in np.arange(0,len(ysos)):
        print 'LS plot: ' + str(i)
        fig.clf()
        ax = fig.add_subplot(111)
        if 't1' in ysos[i].keys():
            t1 = ysos[i]['t1']
            m1 = ysos[i]['m1']
            if len(t1) > 2:
                test1 = ysovar_lombscargle.fasper(t1,m1,oversamp,maxfreq)
                max1 = np.where(test1[1] == np.max(test1[1][np.where(test1[0] > 1/maxper)]))[0] # be sensitive only to periods shorter than maxper
                sig1 = test1[1][max1]
                period1 = 1/test1[0][max1]
                ysos[i]['period_36'] = period1
                ysos[i]['peak_36'] = sig1
                ax.plot(1/test1[0],test1[1],linewidth=2, label = r'3.6 $\mu$m')
        if 't2' in  ysos[i].keys():
            t2 = ysos[i]['t2']
            m2 = ysos[i]['m2']
            if len(t2) > 2:
                test2 = ysovar_lombscargle.fasper(t2,m2,oversamp,maxfreq)
                max2 = np.where(test2[1] == np.max(test2[1][np.where(test2[0] > 1/maxper)]))[0] # be sensitive only to periods shorter than maxper
                sig2 = test2[1][max2]
                period2 = 1/test2[0][max2]
                ysos[i]['period_45'] = period2
                ysos[i]['peak_45'] = sig2
                ax.plot(1/test2[0],test2[1],linewidth=2, label = r'4.5 $\mu$m')
        
        ax.legend(loc = 'upper right')
        ax.set_xscale('log')
        #plt.xlim([1./maxfreq,maxper,0])
        #plt.axis([1./maxfreq,maxper,0,1.2*np.max(np.array([test1[1][max1], test2[1][max2]]))])
        ax.set_xlabel('Period (d)')
        ax.set_ylabel('Periodogram power')
        #plt.text(1.1*x1, 0.9*y2, 'period [3.6]: ' + str( ("%.2f" % period1) ) + ' d' )
        #plt.text(1.1*x1, 0.85*y2, 'FAP [3.6]: ' + str( ("%.2e" % test1[4]) ) )
        #plt.text(1.1*x1, 0.8*y2, 'period [4.5]: ' + str( ("%.2f" % period2) ) + ' d' )
        #plt.text(1.1*x1, 0.75*y2, 'FAP [4.5]: ' + str( ("%.2e" % test2[4]) ) )
        ax.set_title('Lomb-Scargle Periodogram')
        multisave(fig, outroot + str(i) + '_ls')



def make_phased_lc_cmd_plots(data, infos, outroot):
    # make phase-folded light curve plots
    fig = plt.figure()
    for i in np.arange(0,len(data)):
        print 'phase plots: ' + str(i)
        x1 = 0
        x2 = 0
        y1 = 0
        y2 = 0
        fig.clf()
        ax = fig.add_subplot(111)
        if ('p1' in data[i].keys()) and ('p2' in data[i].keys()):
            ax.scatter(data[i]['p1'], data[i]['m1'], marker='o', lw=0, s=40,  c=data[i]['p1'])
            ax.scatter(data[i]['p2'], data[i]['m2'], marker='+', lw=1, s=40, c=data[i]['p2'])
            ax.set_xlabel('phase')
            ax.set_title('phase-folded light curve, period = ' + str( ("%.2f" % infos.good_period[i]) ) + ' d')
            ax.set_ylim(ax.get_ylim()[::-1])
            multisave(fig, outroot + str(i) + '_lc_phased')
        elif ('p1' in data[i].keys()):
            ax.scatter(data[i]['p1'], data[i]['m1'], marker='o', lw=0, s=40,  c=data[i]['p1'])
            ax.set_xlabel('phase')
            ax.set_title('phase-folded light curve, period = ' + str( ("%.2f" % infos.good_period[i]) ) + ' d')
            ax.set_ylim(ax.get_ylim()[::-1])
            multisave(fig, outroot + str(i) + '_lc_phased')
        elif ('p2' in data[i].keys()):
            ax.scatter(data[i]['p2'], data[i]['m2'], marker='+', lw=1, s=40, c=data[i]['p2'])
            ax.set_xlabel('phase')
            ax.set_title('phase-folded light curve, period = ' + str( ("%.2f" % infos.good_period[i]) ) + ' d')
            ax.set_ylim(ax.get_ylim()[::-1])
            multisave(fig, outroot + str(i) + '_lc_phased')
                
        # make phased color-magnitude plot
        if 'p' in data[i].keys():
            fig.clf()
            ax = fig.add_subplot(111)
            if ( (len(data[i]['t']) > 1) & (infos.good_period[i] > 0) ):
                p1 = plt.scatter(data[i]['m36']-data[i]['m45'], data[i]['m36'], lw=0, s=40, marker='^', c=data[i]['p'])
                ax.set_xlabel('[3.6] - [4.5]')
                ax.set_ylabel('[3.6]')
                # get x and y coordinates of plot
                x1 = ax.get_xlim()[0]
                x2 = ax.get_xlim()[1]
                y1 = ax.get_ylim()[0]
                y2 = ax.get_ylim()[1]
                ax.set_ylim([y2, y1]) # invert y axis!
                
                ## plot line for fit to data:
                #m = data[i]['fit_twocolor'][0]
                #b = data[i]['fit_twocolor'][1]
                #line_x = np.array([x1, x2])
                #line_y = np.array([m*x1+b, m*x2+b])
                #plt.plot(line_x, line_y, 'k-')
                
                ## plot line for shifted reddening vector to data:
                #m = data[i]['fit_twocolor'][2]
                #b = data[i]['fit_twocolor'][3]
                #line_x = np.array([x1, x2])
                #line_y = np.array([m*x1+b, m*x2+b])
                #plt.plot(line_x, line_y, 'k--')
                #plt.legend(['measured slope','standard reddening'])
                
                ## plot reddening vector:
                #vector = make_reddeningvector_for_plot(x1, x2, y1, y2)
                #print vector
                #plt.arrow(vector[0],m*vector[0]+b, vector[2]-vector[0],vector[3]-vector[1],fc="k", ec="k", head_width=0.025*(x2-x1))
                #plot_angle = - np.atan( vector[4]*(x2-x1)*3.2 / ((vector[2]-vector[0])*(y2-y1)*4) )/(2*np.pi)*360 # the 3.2 and the 4 comes from the actual size of the figure (angle does not work in data coordinates)
                ##print plot_angle
                #plt.text(vector[0],m*vector[0]+b, "$A_V = $ " + str(vector[5]) , rotation = plot_angle   )
                
                ax.set_title('CMD color-coded by phase, period = ' + str( ("%.2f" % infos.good_period[i]) ) + ' d')
                
                filename = outroot + str(i) + '_color_phased'
                multisave(fig, filename)



def make_sed_plots(infos, outroot, title = 'SED (data from Guenther+ 2012)'):
    # plots SED from Guenther+ 2012, and adds m1 and m2 values from the Spitzer monitoring (marked with a different symbol) if the source does not have a Spitzer 3.6 or 4.5 datapoint in Guenther+ 2012.
    fig = plt.figure()
    for i in np.arange(0,len(infos)):
        print 'SED plots: ' + str(i)
        fig.clf()
        ax = fig.add_subplot(111)
        try:
            plot_sed = deepcopy(infos.fluxes[i])
            lambdas = infos.lambdas[i]
        except AttributeError:
            (lambdas, mags, mags_error, plot_sed) = get_sed(infos[i]) 
        flag36 = 0
        flag45 = 0
        ax.plot(lambdas, plot_sed, 'o')
        #Attenion: min(magnitude) = max(flux)
        m36 = 2.5**(-np.array([infos.median_36[i], infos.min_36[i], infos.max_36[i]])) * 6.50231481e-08
        m45 = 2.5**(-np.array([infos.median_45[i], infos.min_45[i], infos.max_45[i]])) * 2.66222222e-08
        ax.errorbar([3.6,4.5], [m36[0],m45[0]], yerr = [[m36[0]-m36[2], m45[0]-m45[2]],[m36[1]-m36[0], m45[1]-m45[0]]], fmt='^')
        #if (np.isnan(plot_sed[8]) & (infos.median_36[i] > -99999) ):
            #plot_sed[8] = 2.5**(-infos.median_36[i]) * 6.50231481e-08 # zero point flux for 3.6mu
            #plt.plot(lambdas[8], plot_sed[8], '^')
        #if (np.isnan(plot_sed[9]) & (infos.median_45[i] > -99999) ):
            #plot_sed[9] = 2.5**(-infos.median_45[i]) * 2.66222222e-08 # zero point flux for 4.5mu
            #plt.plot(lambdas[9], plot_sed[9], '^')
        ax.set_xlabel('wavelength ($\mu m$)')
        ax.set_ylabel('flux (erg s$^{-1}$ cm$^{-2}$ $\mu$m$^{-1}$)')
        ax.set_xlim(0.3,30)
        ax.set_title(title)
        ax.set_yscale('log')
        ax.set_xscale('log')
        multisave(fig, outroot + str(i) + '_sed')


def extraplots_1():
    
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
    #plt.title('')
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
    #plt.title('')
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
    #plt.axis([6,18,-10,60])
    plt.yscale('log')
    #plt.title('')
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
    plt.ylabel('Detected period in light curve (d)')
    plt.legend([p0, p1, p2, p3, p4], ['XYSOs', 'class 1', 'class 2', 'class 3', 'stars'], 'lower left')
    plt.axis([6,17,1,25])
    #plt.title('')
    #plt.yscale('log')
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
    #plt.legend([p0, p1, p2, p3, p4], ['XYSOs', 'class 1', 'class 2', 'class 3'], 'upper left')
    plt.axis([6,22,0,180])
    #plt.title('')
    #plt.yscale('log')
    plt.plot([16.8,16.8],[53,63],'k-')
    plt.text(17,56,'extinction dominated')
    plt.plot([16.8,16.8],[75,85],'k-')
    plt.text(17,77,'hot & cool spots')
    plt.plot([16.8,16.8],[95,175],'k-')
    plt.text(17,135,'accretion dominated')
    plt.plot([16.5,16.5],[5,95],'k-')
    plt.text(16.7,30,'mix of processes')
    plt.text(16.7,22,'(accr./extinc./spots)')
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
    #p4 = plt.scatter( infos.median_36[i14], alpha_shift[i14]/(2*np.pi) * 360, marker='.', color = color[4] )
    p0 = plt.scatter( infos.median_36[i10], alpha_shift[i10]/(2*np.pi) * 360, marker='o', color = color[0] )
    p1 = plt.scatter( infos.median_36[i11], alpha_shift[i11]/(2*np.pi) * 360, marker='o', color = color[1] )
    p2 = plt.scatter( infos.median_36[i12], alpha_shift[i12]/(2*np.pi) * 360, marker='o', color = color[2] )
    p3 = plt.scatter( infos.median_36[i13], alpha_shift[i13]/(2*np.pi) * 360, marker='o', color = color[3] )
    plt.xlabel('median $3.6\mu\mathrm{m}$ flux (mag)')
    plt.ylabel('Slope angle in color-magnitude diagram (deg)')
    #plt.legend([p0, p1, p2, p3, p4], ['XYSOs', 'class 1', 'class 2', 'class 3', 'stars'], 'upper left')
    plt.legend([p0, p1, p2, p3, p4], ['XYSOs', 'class 1', 'class 2', 'class 3'], 'upper left')
    plt.axis([6,22,0,180])
    #plt.title('')
    #plt.yscale('log')
    plt.plot([16.8,16.8],[53,63],'k-')
    plt.text(17,56,'extinction dominated')
    plt.plot([16.8,16.8],[75,85],'k-')
    plt.text(17,77,'hot & cool spots')
    plt.plot([16.8,16.8],[95,175],'k-')
    plt.text(17,135,'accretion dominated')
    plt.plot([16.5,16.5],[5,95],'k-')
    plt.text(16.7,30,'mix of processes')
    plt.text(16.7,22,'(accr./extinc./spots)')
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



