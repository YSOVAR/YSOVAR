# Copyright (C) 2013 H.M.Guenther & K.Poppenhaeger. See Licence.rst for details.
'''Generate an atlas of YSOVAR lightcurves

This module collects all procedures that are required to make the
atlas. This starts with reading in the csv file from the YSOVAR2
database and includes the calculation of the some fits and quantities.
More specific tasks for the analysis of the lightcurve can be found in
:mod:`YSOVAR.lightcurves`, more stuff for plotting in :mod:`YSOVAR.plot`.

The basic structure for the YSOVAR analysis is the
:class:`YSOVAR.atlas.YSOVAR_atlas`. 
To initialize an atlas object pass in a numpy array with all the lightcurves::
    
    from YSOVAR import atlas
    data = atlas.dict_from_csv('/path/to/my/irac.csv', match_dist = 0.)
    MyRegion = atlas.YSOVAR_atlas(lclist = data)

The :class:`YSOVAR.atlas.YSOVAR_atlas` is build on top of a `astropy.table.Table
(documentation here)
<http://docs.astropy.org/en/stable/table/index.html>`_ object. See that
documentation for the syntax on how to acess the data or add a column.

This :class:`YSOVAR.atlas.YSOVAR_atlas` auto-generates some content in the 
background, so I really encourage you to read the documentation (We promise it's
only a few lines because we are too lazy to type much more).
'''
import os.path
from collections import defaultdict
import math
import re
from copy import deepcopy
import string
from warnings import warn

import numpy as np
import scipy
import scipy.odr
import scipy.stats
import scipy.stats.mstats
import matplotlib.pyplot as plt
import pylab

import astropy.io.fits as pyfits
import astropy.io.ascii as asciitable
import astropy.table

from .great_circle_dist import dist_radec, dist_radec_fast
from . import lightcurves
from . import registry

### Helper functions, simple one liners to do some math needed later on etc. ###
def coord_add_RADEfromhmsdms(dat, rah, ram, ras, design, ded, dem, des):
    '''transform RA and DEC in table from hms, dms to degrees

    Parameters
    ----------
    dat : :class:`YSOVAR.atlas.YSOVAR_atlas` or :class:`astropy.table.Table`
        with columns in the CDS format (e.g. from reading a CDS table with
        :mod:`astropy.io.ascii`)
    rah, ram, ras, ded, dem, des: np.ndarray
        RA and DEC hms, dms values
    design: +1 or -1
        Sign of the DE coordinate (integer or float, not string)
    '''

    radeg = rah*15. + ram / 4. + ras/4./60.
    dedeg = design*(np.abs(ded + dem / 60. + des/3600.))

    coltype = type(dat.columns[0])  # could be Column or MaskedColumn
    dat.add_column(coltype(name = 'RAdeg', data = radeg, unit='deg', description='Right ascension'))
    dat.add_column(coltype(name = 'DEdeg', data = dedeg, unit='deg', description='Declination'))
  

def coord_CDS2RADEC(dat):
    '''transform RA and DEC from CDS table to degrees

    CDS tables have a certain format of string columns to store coordinates
    (``RAh``, ``RAm``, ``RAs``, ``DE-``, ``DEd``, ``DEm``, ``DEs``). 
    This procedure
    parses that and calculates new values for RA and DEC in degrees.
    These are added to the Table as ``RAdeg`` and ``DEdeg``.

    Parameters
    ----------
    dat : :class:`YSOVAR.atlas.YSOVAR_atlas` or :class:`astropy.table.Table`
        with columns in the CDS format (e.g. from reading a CDS table with
        :mod:`astropy.io.ascii`)
    '''
    coord_add_RADEfromhmsdms(dat, dat['RAh'], dat['RAm'], dat['RAs'],
                             (dat['DE-'] !='-')*2-1, dat['DEd'], dat['DEm'], dat['DEs'])

def coord_hmsdms2RADEC(dat, ra=['RAh', 'RAm', 'RAs'], dec=['DEd', 'DEm','DEs']):
    '''transform RA and DEC from table to degrees

    Tables where RA and DEC are encoded as three numeric columns each like
    ``hh:mm:ss`` and ``dd:mm:ss`` can be converted into decimal deg.
    This procedure parses that and calculates new values for RA and DEC in degrees.
    These are added to the Table as ``RAdeg`` and ``DEdeg``.
    
    .. warning::
       This format is ambiguous for sources with dec=+/-00:xx:xx, because 
       python does not differentiate between ``+0`` and ``-0``.

    Parameters
    ----------
    dat :  :class:`YSOVAR.atlas.YSOVAR_atlas` or :class:`astropy.table.Table`
        with columns in the format given above
    ra : list of three strings
        names of RA column names for hour, min, sec
    dec : list of three strings
        names of DEC column names for deg, min, sec
    '''
    sign = [-1 if d < 0 else 1 for d in dat[dec[0]]]
    coord_add_RADEfromhmsdms(dat, dat[ra[0]], dat[ra[1]], dat[ra[2]],
                             sign, np.abs(dat[dec[0]]), dat[dec[1]], dat[dec[2]])

def coord_strhmsdms2RADEC(dat, ra = 'RA', dec = 'DEC', delimiter=':'):
    '''transform RA and DEC from table to degrees

    Tables where RA and DEC are encoded as string columns each like
    `hh:mm:ss` `dd:mm:ss` can be converted into decimal deg.
    This procedure parses that and calculates new values for RA and DEC in degrees.
    These are added to the Table as `RAdeg` and `DEdeg`.

    Parameters
    ----------
    dat :  :class:`YSOVAR.atlas.YSOVAR_atlas` or :class:`astropy.table.Table`
        with columns in the format given above
    ra : string
        name of RA column names for hour, min, sec
    dec : string
        name of DEC column names for deg, min, sec
    delimiter : string
        delimiter between elements, e.g. ``:`` in ``01:23:34.3``.
    '''
    raarr = astropy.io.ascii.read(dat[ra], Reader=astropy.io.ascii.NoHeader,
                               delimiter=delimiter, names=['h','m','s'])
    dearr = astropy.io.ascii.read(dat[dec], Reader=astropy.io.ascii.NoHeader,
                               delimiter=delimiter, names=['d','m','s'])
    sign = [-1 if d[0]=='-' else 1 for d in dat[dec]]
    coord_add_RADEfromhmsdms(dat, raarr['h'], raarr['m'], raarr['s'],
                             sign, np.abs(dearr['d']), dearr['m'], dearr['s'])


### Everything that deals with the lightcurve dictionaries and only with those ###

def radec_from_dict(data, RA = 'ra', DEC = 'dec'):
    '''return ra dec numpy array for list of dicts
    
    Parameters
    ----------
    data : list of several `dict`
    RA, DEC : strings
        keys for RA and DEC in the dictionary
    
    Returns
    -------
    radec : np record array with RA, DEC columns
    '''
    radec = np.zeros(len(data), dtype=[('RA', np.float),('DEC',np.float)])
    for i, d in enumerate(data):
        radec[i]['RA'] = d[RA]
        radec[i]['DEC'] = d[DEC]
    return radec

def val_from_dict(data, name):
    '''return ra dec numpy array for list of dicts
    
    Parameters
    ----------
    data : list of dict
    name : strings
        keys for entry in the dictionary
    
    Returns
    -------
    col : list of values
    '''
    col = []
    for d in data:
        col.append(d[name])
    return col
    


def makecrossids(data1, data2, radius, ra1='RAdeg', dec1='DEdeg', ra2='ra', dec2='dec', double_match = False):
    '''Cross-match two lists of coordinates, return closest match

    This routine is not very clever and not very fast. It should be fine
    up to a hundred thousand entries per list. 

    Parameters
    ----------
    data1 : :class:`astropy.table.Table` or np.recarray
        This is the master data, i.e. for each element in data1, the
        results wil have one (or zero) index numbers in data2, that provide
        the best match to this entry in data1.
    data2 : astropt.table.Table or np.recarray
        This data is matched to data1.
    radius : np.float or array
       maximum radius to accept a match (in degrees); either a scalar or same
       length as data2
    ra1, dec1, ra2, dec2 : string
        key for access RA and DEG (in degrees) the the data, i.e. the routine
        uses `data1[ra1]` for the RA values of data1.
    double_match : bool
        If true, one source in data2 could be matched to several sources in data1.
        This can happen, if a source in data2 lies between two sources of data1, 
        which are both within ``radius``.
        If this switch is set to ``False``, then a strict one-on-one matching 
        is enforced, selecting the closest pair in the situation above.

    Returns
    -------
    cross_ids : np.ndarray
        Will have len(data1). For each elelment it contains the index of data2
        that provides the best match. If no match within `radius` is found,
        then entry will be -99999.
    '''
    if not (np.isscalar(radius) or (len(radius)==len(data2))):
        raise ValueError("radius must be scalar or have same number of elements as data2")
    cross_ids = np.ones(len(data1),int) * -99999
    
    for i in np.arange(0,len(data1)):
        # Pick out only those that are close in dec
        ind = np.where(np.abs(data1[dec1][i] - data2[dec2]) < radius)[0]
        # and calculate the full dist_radec only for those that are close enough
        # since dist_radec includes several sin, cos, that speeds it up a lot
        if len(ind) > 0:
            distance = dist_radec(data1[ra1][i], data1[dec1][i], data2[ra2][ind], data2[dec2][ind], unit ='deg')
            if np.isscalar(radius):
                if np.any(distance < radius):
                    cross_ids[i] = ind[np.argmin(distance)]
            else:
                if np.any(distance < radius[ind]):
                    cross_ids[i] = ind[np.argmin(distance/radius[ind])]

    if not double_match:
        matched = (cross_ids >=0)
        multmatch = np.bincount(cross_ids[matched])
        for i, m in enumerate(multmatch): 
            if m > 1:
                ind = (cross_ids == i)
                distance = dist_radec(data1[ra1][ind], data1[dec1][ind], data2[ra2][i], data2[dec2][i], unit ='deg')
                cross_ids[ind.nonzero()[0][~(distance == np.min(distance))]] = -99999
                

        
    return cross_ids

def makecrossids_all(data1, data2, radius, ra1='RAdeg', dec1='DEdeg', ra2='ra', dec2='dec', return_distances=False):
    '''Cross-match two lists of coordinates, return all matches within radius

    This routine is not very clever and not very fast. If should be fine
    up to a hundred thousand entries per list. 

    Parameters
    ----------
    data1 : :class:`astropy.table.Table` or np.recarray
        This is the master data, i.e. for each element in data1, the
        results wil have the index numbers in data2, that provide
        the best match to this entry in data1.
    data2 : :class:`astropy.table.Table` or np.recarray
        This data is matched to data1.
    radius : np.float or array
       maximum radius to accept a match (in degrees) 
    ra1, dec1, ra2, dec2 : string
        key for access RA and DEG (in degrees) the the data, i.e. the routine
        uses `data1[ra1]` for the RA values of data1.
    return_distances : bool
        decide if distances should be returned

    Returns
    -------
    cross_ids : list of lists
        Will have len(data1). For each elelment it contains the indices of data2
        that are within `radius`. If no match within `radius` is found,
        then the entry will be `[]`.
    distances : list of lists
        If ``return_distances==True`` this has the same format a ``cross_ids``
        and contains the distance to the match in degrees.
    '''
    cross_ids = []
    distances = []

    for i in range(len(data1)):
        # Pick out only those that are close in dec
        ind = np.where(np.abs(data1[dec1][i] - data2[dec2]) <= radius)[0]
        # and calculate the full dist_radec only for those that are close enough
        # since dist_radec includes several sin, cos, that speeds it up a lot
        if len(ind) > 0:
            distance = dist_radec(data1[ra1][i], data1[dec1][i], data2[ra2][ind], data2[dec2][ind], unit ='deg') 
            cross_ids.append(ind[distance <= radius])
            distances.append(distance[distance <= radius])
        else:
            cross_ids.append([])
            distances.append([])
    if return_distances:
        return cross_ids, distances
    else:
        return cross_ids


''' Format:
    dictionary of bands, where the name of the band mag is the key
    entries are lists of::

        [name of error field, 
         wavelength in micron, 
         zero_magnitude_flux_freq in Jy = 1e-23 erg s-1 cm-2 Hz-1]

using Sloan wavelengths for r and i
http://casa.colorado.edu/~ginsbura/filtersets.htm
http://coolwiki.ipac.caltech.edu/index.php/Central_wavelengths_and_zero_points
http://arxiv.org/pdf/1011.2020.pdf for SLoan u', r', i' (Umag, Rmag, Imag)

'''
sed_bands = {'Umag': ['e_Umag', 0.355, 1500], 
             'Bmag': ['e_Bmag', 0.430, 4000.87], 
             'Vmag': ['e_Vmag', 0.623, 3597.28], 
             'Rmag': ['e_Rmag', 0.759, 3182], 
             'Imag': ['e_Imag', 0.798, 2587], 
             'Jmag': ['e_Jmag', 1.235, 1594], 
             'Hmag': ['e_Hmag', 1.662, 1024], 
             'Kmag': ['e_Kmag', 2.159, 666.7], 
             '3.6mag': ['e_3.6mag', 3.6, 280.9], 
             '4.5mag': ['e_4.5mag', 4.5, 179.7], 
             '5.8mag': ['e_5.8mag', 5.8, 115.0], 
             '8.0mag': ['e_8.0mag', 8.0, 64.13], 
             '24mag': ['e_24mag', 24.0, 7.14], 
             'Hamag': ['e_Hamag', 0.656, 2974.4], 
             'rmag': ['e_rmag', 0.622, 3173.3], 
             'imag': ['e_imag', 0.763, 2515.7], 
             'nomad_Bmag': [None, 0.430, 4000.87], 
             'nomad_Vmag': [None, 0.623, 3597.28], 
             'nomad_Rmag': [None, 0.759, 3182], 
             'simbad_B': [None, 0.430, 4000.87], 
             'simbad_V': [None, 0.623, 3597.28],
             'mean_36': ['e_3.6mag', 3.6, 280.9], 
             'mean_45': ['e_4.5mag', 4.5, 179.7], 
            }

def sed_slope(data, sed_bands=sed_bands):
    '''fit the SED slope to data for all bands in ``data`` and ``sed_bands``

    Parameters
    ----------
    data :  :class:`YSOVAR.atlas.YSOVAR_atlas` or :class:`astropy.table.Table`
        input data that has arrays of magnitudes for different bands
    sed_bands : dict 
        keys must be the name of the field that contains the magnitudes in each
        band,  entries are lists of [name of error field, wavelength in micron,
        zero_magnitude_flux_freq in Jy]
    
    Returns
    -------
    slope : float
        slope of the SED determined with a least squares fit.
        Return ``np.nan`` if there is too little data.
    '''
    sed = get_sed(data, sed_bands, valid=True)
    slope = np.nan
    if len(sed[0]) >= 2:
        out = np.polyfit(np.log10(sed[0]), np.log10(sed[0]*sed[3]), 1, full=True)
        # check rank of matrix. If less than 2, fit is ill-conditioned
        # e.g. one data point only
        if out[2] == 2:
            slope = out[0][0] # slope
    return slope
    
def get_sed(data, sed_bands = sed_bands, valid = False):
    '''make SED by collecting info from the input data
    
    Parameters
    ----------
    data :  :class:`YSOVAR.atlas.YSOVAR_atlas` or :class:`astropy.table.Table`
        input data that has arrays of magnitudes for different bands
    sed_bands : dict 
        keys must be the name of the field that contains the magnitudes in each
        band,  entries are lists of [name of error field, wavelength in micron,
        zero_magnitude_flux_freq in Jy]
    valid : bool
        If true, return only bands with finite flux, otherwise return all bands
        that exist in both ``data`` and ``sed_bands``.
        
    Returns
    -------
    wavelen : np.ndarray
        central wavelength of bands in micron
    mags : np.ndarray
        magnitude in band
    mags_error : np.ndarray
        error on magnitude
    sed : np.ndarray
        flux in Jy
    '''
    #make list of bands available in input data
    try:
        # data is np.record array
        bands  = [band for band in sed_bands.keys() if band in data.dtype.names]
    except AttributeError:
        # data is dictionary or atpy.table
        bands  = [band for band in sed_bands.keys() if band in data.keys()]
    
    mags = np.zeros(len(bands))
    mags_error = np.zeros_like(mags)
    wavelen = np.zeros_like(mags)
    zero_magnitude_flux_freq = np.zeros_like(mags)
    for i, b in enumerate(bands):
        mags[i] = data[b]
        if sed_bands[b][0] is None:
            mags_error[i] = np.nan
        else:
            mags_error[i] = data[sed_bands[b][0]]
        wavelen[i] = sed_bands[b][1]
        zero_magnitude_flux_freq[i] = sed_bands[b][2]

    freq = 3e14 / wavelen # with c in microns because wavlen is in microns.
    # from flux density per frequency to flux density per wavelength:
    # f_nu dnu = f_lambda dlambda, with nu = c/lambda ->
    # nu * f_nu = lambda * f_lambda 
    zero_magnitude_flux_wavlen = zero_magnitude_flux_freq * 1e-23 * freq / wavelen
    sed = 2.5**(-mags)*zero_magnitude_flux_wavlen

    # sort by wavelength
    ind = np.argsort(wavelen)
    wavelen = wavelen[ind]
    mags = mags[ind]
    mags_error = mags_error[ind]
    sed = sed[ind]
    if valid:
        ind = np.isfinite(sed)
        wavelen = wavelen[ind]
        mags = mags[ind]
        mags_error = mags_error[ind]
        sed = sed[ind]
    return (wavelen, mags, mags_error, sed)

def dict_cleanup(data, channels, min_number_of_times = 0, floor_error = {}):
    '''Clean up dictionaries after add_ysovar_mags
    
    Each object in the `data` list can be constructed from multiple sources per
    band in multiple bands. This function averages the coordinates over all
    contributing sources, converts and sorts lists of times and magnitudes
    and makes multiband lightcurves.
    
    Parameters
    ----------
    data : list of dictionaries
        as obtained from :func:`YSOVAR.atlas.add_ysovar_mags`
    channels : dictionary
        This dictionary traslantes the names of channels in the csv file to
        the names in the output structure, e.g.
        that for `'IRAC1'` will be `'m36'` (magnitudes) and `'t36'` (times).
    min_number_of_times : integer
        Remove all lightcurves with less than `min_number_of_times` datapoints
        from the list
    floor_error : dict
        Floor errors will be added in quadrature to all error values.
        The keys in the dictionary should be the same as in the channels
        dictionary.
        
    Returns
    -------
    data : list of dictionaries
        individual dictionaries are cleaned up as decribed above
    '''
    n_datapoints = np.zeros(len(data), dtype = np.int)
    for i, d in enumerate(data):
        for channel in channels:
            if 'm'+channels[channel] in d:
                n = len(d['m'+channels[channel]])
                n_datapoints[i] = max(n_datapoints[i], n)
                if n < min_number_of_times:
                    del d['m'+channels[channel]]
                    del d['t'+channels[channel]]
    # go in reverse direction, otherwise each removal would change index numbers of
    # later entries
    data = data[n_datapoints >= min_number_of_times]
    if len(floor_error) > 0:
        print 'Adding floor error in quadrature to all error values'
        print 'Floor values ', floor_error

    for j, d in enumerate(data):
        d['id'] = j
        # set RA, DEC to the mean values of all sources, which make up an entry
        d['ra'] = np.mean(d['ra'])
        d['dec'] = np.mean(d['dec'])
        # take the set of those two things and convert to a string
        d['IAU_NAME'] = set(d['IAU_NAME'])
        d['IAU_NAME'] = ', '.join(d['IAU_NAME'])
        d['YSOVAR2_id'] = set(d['YSOVAR2_id'])
        d['YSOVAR2_id'] = ', '.join([str(n) for n in d['YSOVAR2_id']])
        for channel in channels:
            c = channels[channel]
            if 't'+c in d.keys():
                # sort lightcurves by time.
                # required if multiple original sources contribute to this entry
                ind = np.array(d['t'+c]).argsort()
                d['t'+c] = np.array(d['t'+c])[ind]
                d['m'+c] = np.array(d['m'+c])[ind]
                d['m'+c+'_error'] = np.array(d['m'+c+'_error'])[ind]
                if channel in floor_error:
                    d['m'+c+'_error'] = np.sqrt(d['m'+c+'_error']**2 + floor_error[channel]**2)
    return data

def merge_lc(d, bands, t_simul=0.01):
    '''merge lightcurves from several bands

    This returns a lightcurve that contains only entries for those times,
    where *all* required bands have an entry.

    Parameters
    ----------
    d : dictionary
        as obtained from :func:`YSOVAR.atlas.add_ysovar_mags`
    bands : list of strings
        labels of the spectral bands to be merged, e.g. ['36','45']
    t_simul : float
        max distance in days to accept datapoints in band 1 and 2 as simultaneous
        In L1688 and IRAS 20050+2720 the distance between band 1 and 2 coverage
        is within a few minutes, so a small number is sufficent to catch 
        everything and to avoid false matches.

    Returns
    -------
    tab : astropy.table.Table
        This table contains the merged lightcurve and contains times,
        fluxes and errors.
    '''
    if not isinstance(d, dict):
        raise ValueError('d must be a dictionary that contains lightcurves.')

    if len(bands) ==1:
        band = bands[0]
        # no matching necessary
        # The else block would also work for one band only,
        # but the matching process is very slow.
        tab = astropy.table.Table({'t': d['t'+band],
                                   'm'+band: d['m'+band],
                                   'm'+band+'_error': d['m'+band+'_error']})
    else:
        tab = astropy.table.Table()
        names = ['t']
        for band in bands:
            names.extend(['m'+band,'m'+band+'_error'])

        for name in names:
            tab.add_column(astropy.table.Column(name=name, length=0, dtype=np.float))

        if set(['m'+b for b in bands]).issubset(d.keys()):
            for i, t in enumerate(d['t'+bands[0]]):
                minind = np.zeros(len(bands), dtype = np.int)
                diff_t = np.zeros(len(bands))
                for j, band in enumerate(bands):
                    deltat = np.abs(t - d['t'+band])
                    #index of closest matching time in each band
                    minind[j] = np.argmin(deltat)
                    diff_t[j] = np.min(deltat)

                if np.all(diff_t < t_simul):
                    tlist = [ d['t'+band][minind[j]] for j,band in enumerate(bands) ]
                    newrow = [np.mean(tlist)]
                    for j, band in enumerate(bands):
                        newrow.extend([d['m'+band][minind[j]],
                                       d['m'+band+'_error'][minind[j]]])
                    tab.add_row(newrow)
    return tab

def phase_fold(time, period):
    '''Phase fold a set of time on a period

    Parameters
    ----------
    time : np.ndarray
        array of times
    period : np.float    
    '''
    return np.mod(time, period) / period



def IAU2radec(isoy):
    '''convert IAU name do decimal degrees.
    
    Parameters
    ----------
    isoy : string
        IAU Name 

    Returns
    -------
    ra, dec : float
        ra and dec in decimal degrees
    '''
    s = (isoy.split('J'))[-1]
    ra = int(s[0:2]) *15. + int(s[2:4])/4. + float(s[4:9])/4./60.
    dec = np.sign(float(s[9:])) * (float(s[10:12]) + int(s[12:14]) /60. + float(s[14:])/3600.)
    return ra, dec


def dict_from_csv(csvfile,  match_dist = 1.0/3600., min_number_of_times = 5, channels = {'IRAC1': '36', 'IRAC2': '45'}, data = [], floor_error = {'IRAC1': 0.01, 'IRAC2': 0.007}, mag = 'mag1', emag = 'emag1', time = 'hmjd', bg = None, source_name = 'sname',  verbose = True, readra = 'ra', readdec = 'de', sourceid = 'ysovarid', channelcolumn='fname'):
    '''Build YSOVAR lightcurves from database csv file
    
    Parameters
    ----------
    cvsfile : sting or file object
        input csv file
    match_dist : float
        maximum distance to match two positions as one sorce
    min_number_of_times : integer
        Remove all sources with less than min_number_of_times datapoints
        from the list
    channels : dictionary
        This dictionary translates the names of channels in the csv file to
        the names in the output structure, e.g.
        that for ``IRAC1`` will be ``m36`` (magnitudes) and ``t36`` (times).
    data : list of dicts
        New entries will be added to data. It can be empty (the default).
    mag : string
        name of magnitude column
    emag : string
        name of column holding the error on the mag
    time : string
        name of column holding the time of observation
    bg : string or None
        name of column holding the bg for each observations (None indicates
        that the bg column is not present).
    floor_error : dict
        Floor errors will be added in quadrature to all error values.
        The keys in the dictionary should be the same as in the channels
        dictionary.   
    verbose : bool
       If True, print progress status.
        
    Returns
    -------
    data : empty list or list of dictionaries
        structure to hold all the information
        
    TBD: Still need to deal with double entries in lightcurve (and check manually...)
    '''
    if verbose: print 'Reading csv file - This may take a few minutes...'
    tab = asciitable.read(csvfile)
    radec = {'RA':[], 'DEC': []}
    for i, n in enumerate(set(tab[source_name])):
        if verbose and (np.mod(i,100)==0):
            print 'Processing dict for source ' + str(i) + ' of ' + str(len(set(tab[source_name])))
        ind = (tab[source_name] == n)
        ra = tab[ind][0][readra]
        dec = tab[ind][0][readdec] 
        if len(data) > 0:
            distance = dist_radec_fast(ra, dec, np.array(radec['RA']), np.array(radec['DEC']), scale = match_dist, unit ='deg')
        if len(data) > 0 and min(distance) <= match_dist:
            import pdb; pdb.set_trace()
            dict_temp = data[np.argmin(distance)]
        else:
            dict_temp = defaultdict(list)
            data = np.append(data, dict_temp)
            radec['RA'].append(ra)
            radec['DEC'].append(dec)
        dict_temp['ra'].extend([ra] * ind.sum())
        dict_temp['dec'].extend([dec] * ind.sum())
        dict_temp['IAU_NAME'].append(n.replace('ISY_', 'SSTYSV '))
        dict_temp['YSOVAR2_id'].append(tab[sourceid][ind][0])
        for channel in channels.keys():
            good = ind & (tab[time] >= 0.) & (tab[channelcolumn] == channel)
            if np.sum(good) > 0:
                dict_temp['t'+channels[channel]].extend((tab[time][good]).tolist())
                dict_temp['m'+channels[channel]].extend((tab[mag][good]).tolist())
                dict_temp['m'+channels[channel]+'_error'].extend((tab[emag][good]).tolist())
                if bg is not None:
                    dict_temp['m'+channels[channel]+'_bg'].extend((tab[bg][good]).tolist())
        
    if verbose: print 'Cleaning up dictionaries'
    data = dict_cleanup(data, channels = channels, min_number_of_times = min_number_of_times, floor_error = floor_error)
    return data

def check_dataset(data, min_number_of_times = 5, match_dist = 1./3600.):
    '''check dataset for anomalies, cross-match problems etc.
    
    Of course, not every problem can be detected here, but every time I find
    something I add a check so that next time this routine will warn me of the 
    same problem.
    
    Parameters
    ----------
    data : list of dicts
        as read in with e.e. `dict_from_csv`
    '''
    print 'Number of sources in datset: ', len(data)
    print 'The following entries have less than ', min_number_of_times,' datapoints in both IRAC bands:'
    for i, d in enumerate(data):
        if (not ('t36' in d.keys()) or (len(d['t36']) < min_number_of_times)) and  (not ('t45' in d.keys()) or (len(d['t45']) < min_number_of_times)): print i, list(d['IAU_NAME'])
    print '----------------------------------------------------------------------------'
    # IRAC1==IRAC2 ?
    for i, d in enumerate(data):
        if 't36' in d.keys() and 't45' in d.keys():
            ind1 = np.argsort(d['t36'])
            ind2 = np.argsort(d['t45'])
            if (len(d['t36']) == len(d['t45'])) and np.all(d['m36'][ind1] == d['m45'][ind2]):
                print 'IRAC1 and IRAC2 are identical in source', i
                print 'Probably error in SQL query made to retrieve this data'
    ### Find entries with two mags in same lightcurve
    print 'The following lightcurves have two or more entries with almost identical time stamps'
    for band in ['36', '45']:
        for i, d in enumerate(data):
            if 't'+band in d.keys():
                dt = np.diff(d['t'+band])
                dm = np.diff(d['m'+band])
                ind = np.where(dt < 0.00001)[0]
                if len(ind) > 0:
                    print 'IRAC'+band+ ', source ', i, 'at times: ', d['t'+band][ind], ' mag diff is: ', dm[ind]
    print '----------------------------------------------------------------------------'
    print 'The following entries are combined from multiple sources:'
    for i, d in enumerate(data):
        if len(d['IAU_NAME'].split(',')) > 1: print i, d['IAU_NAME']
    print '----------------------------------------------------------------------------'
    print 'The following sources are less than ', match_dist, ' deg apart'
    radec = radec_from_dict(data)
    for i in range(len(data)-1):
        dist = dist_radec_fast(radec['RA'][i], radec['DEC'][i],
                               radec['RA'][i+1:], radec['DEC'][i+1:], scale = match_dist, unit = 'deg')
        if np.min(dist) < match_dist:
            print 'distance', i, i+np.argmin(dist), ' is only ', np.min(dist)
    print '----------------------------------------------------------------------------'



#### The big table / Atlas class that holds the data and does some cool processing ##


class YSOVAR_atlas(astropy.table.Table):
    '''
    The basic structure for the YSOVAR analysis is the
    :class:`YSOVAR_atlas`. 
    To initialize an atlas object pass in a numpy array with all the lightcurves::

        from YSOVAR import atlas
        data = atlas.dict_from_csv('/path/tp/my/irac.csv', match_dist = 0.)
        MyRegion = atlas.YSOVAR_atlas(lclist = data)

    The :class:`YSOVAR_atlas` is build on top of a `astropy.table.Table
    (documentation here)
    <http://docs.astropy.org/en/stable/table/index.html>`_ object. See that
    documentation for the syntax on how to acess the data or add a column.

    Some columns are auto-generated, when they are first
    used. Some examples are
    
        - median
        - mean
        - stddev
        - min
        - max
        - mad (median absolute deviation)
        - delta (90% quantile - 10% quantile)
        - redchi2tomean
        - wmean (uncertainty weighted average).

    When you ask for ``MyRegion['min_36']`` it first checks if that column is already
    present. If not, if adds the new column called ``min_36`` and calculates
    the minimum of the lightcurve in band ``36`` for each object in the
    atlas, that has ``m36`` and ``t36`` entries (for magnitude and time in
    band ``36`` respectively. Data read with :meth:`dict_from_csv`
    atomatically has the required format.

    More functions may be added to this magic list later. Check::

        import YSOVAR.registry 
        YSOVAR.registry.list_lcfuncs()

    to see which functions are implemented.
    More function can be added.

    Also, table columns can be added to an ``YSOVAR_atlas`` object manually,
    giving you all the freedom to do arbitrary calculations to arrive at those
    vales.
    '''
    def __init__(self, *args, **kwargs):
        self.t_simul=0.01
        if 'lclist' in kwargs:
            self.lclist = kwargs.pop('lclist')
            #self.add_column(astropy.table.Column(name = 'id', data = np.arange(1,len(self)+1)))
        else:
            raise ValueError('Need to pass a list of lightcurves with lclist=')
        super(YSOVAR_atlas, self).__init__(*args, **kwargs)
        for name in ['ra', 'dec', 'YSOVAR2_id','IAU_NAME']:
             col = astropy.table.Column(name=name, data=val_from_dict(self.lclist, name))
             self.add_column(col)
        self['IAU_NAME'].description = 'J2000.0 IAU designation within the YSOVAR program'
        self['ra'].unit = 'deg'
        self['dec'].unit = 'deg'
        self['ra'].description = 'J2000.0 Right ascension'
        self['dec'].description = 'J2000.0 Declination'
        self['YSOVAR2_id'].description = 'ID in YSOVAR database'
    
    def __getitem__(self, item):

        if isinstance(item, basestring):
            if item not in self.colnames:
                self.autocalc_newcol(item)
                # safeguard against infinite loops, because there is no direct
                # control, if the new column is really called item
                if item in self.colnames:
                    return self[item]
                else:
                    raise Exception('Attempted to calculate {0}, but the columns added are not called {1}'.format(item, item))
            else:
                return self.columns[item]
        elif isinstance(item, int):
            # In thiscase astropy.table.Table would return a Row object
            # not a new Table - we change this behavior here
            return self._new_from_slice([item])
        elif isinstance(item, tuple):
            if all(isinstance(x, np.ndarray) for x in item):
                # Item is a tuple of ndarrays as in the output of np.where, e.g.
                # t[np.where(t['a'] > 2)]
                return self._new_from_slice(item)
            else:
                for x in item:
                    if x not in self.colnames:
                        # column with name x does not exist.
                        # Try to autogenerate.
                        temp = self[x]
                # Now item should be a tuple of strings that are valid column names
                return astropy.table.Table([self[x] for x in item], meta=deepcopy(self.meta))
        elif (isinstance(item, slice) or isinstance(item, np.ndarray)
              or isinstance(item, list)):
            return self._new_from_slice(item)
        else:
            raise ValueError('Illegal type {0} for table item access'
                             .format(type(item)))


    def _new_from_slice(self, slice_):
        """Create a new table as a referenced slice from self."""

        table = YSOVAR_atlas(lclist = self.lclist[slice_])
        # delete the columns that are autogenerated and just copy everything
        table.remove_columns(['ra', 'dec', 'YSOVAR2_id','IAU_NAME'])
        table.meta.clear()
        table.meta.update(deepcopy(self.meta))
        cols = self.columns.values()
        names = [col.name for col in cols]
        data = self._data[slice_]

        self._update_table_from_cols(table, data, cols, names)

        return table

    def sort(self, keys):
        '''
        Sort the table according to one or more keys. This operates
        on the existing table and does not return a new table.

        Parameters
        ----------
        keys : str or list of str
            The key(s) to order the table by
        '''
        # kind of ugly to make sure lclist get the same reordering
        self.add_column(astropy.table.Column(data = self.lclist, name = 'randomnamethatisneverusedagain'))
        super(YSOVAR_atlas, self).sort(keys)
        self.lclist = self['randomnamethatisneverusedagain'].data
        self.remove_column('randomnamethatisneverusedagain')
        
    def autocalc_newcol(self, name):
        '''automatically calcualte some columns on the fly'''
        splitname = name.split('_')
        try:
            self.calc(splitname[0], re.findall('_([A-Za-z0-9]+(?:_error)?)', name), colnames = [splitname[0]])
        except:
            raise ValueError('Column '+ name + ' not found and cannot be autogenerated. Use ``calc`` to explictly calculate')

    def calc(self, name, bands, timefilter = None, data_preprocessor = None, colnames = [], colunits=[], coldescriptions=[], coltypes=[], overwrite = True, t_simul = None, **kwargs):
        '''calculate some quantity for all sources

        This is a very general interface to the catalog that allows the user
        to initiate calculations with any function defined in the :mod:`registry`.
        A new column is added to the datatable that contains the result.
        (If the column exists before, it is overwritten).
    
        Parameters
        ----------
        name : string
            name of the function in the function registry for YSOVAR
            (see :mod:`registry`)
        bands : list of strings
            Band identifiers
            In some cases, it can be useful to calculate a quantity for the
            error (e.g. the mean error). In this case, just give the band
            as e.g. ``36_error``.
            (This only works for simple functions.)
        timefilter : function or None
            If not  ``None``, this function accepts a np.ndarray of observation times and
            it should return an index array selecteing those time to be
            included in the calculation.
            The default function selects all times. An example how to use this
            keyword include certain times only is shown below::

                cat.calc('mean', '36', timefilter = lambda x : x < 55340)

        data_preprocessor : function or None
            If not None, for each source, a row from the present table is
            extracted as ``source = self[id]``. This yields a :class:`YSOVAR_atlas`
            object with one row. This row is passed to
            ``data_preprocessor``, which can modify the data (e.g. smooth
            a lightcurve), but should keep the structure of the object
            intact. Here is an example for a possible ``data_preprocessor``
            function::

                def smooth(source):
                    lc = source.lclist[0]
                    w = np.ones(5)
                    if 'm36' in lc and len(lc['m36'] > 10):
                        lc['m36'] = np.convolve(w/w.sum(),lc['m36'],mode='valid')
                        # need to keep m36 and t36 same length
                        lc['t36'] = lc['t36'][0:len(lc['m36'])]
                    return source
                
        colnames : list of strings
            Basenames of columns to be hold the output of the calculation.
            If not present already, the bands are added automatically with
            ``dtype = np.float``::

                cat.calc('mean', '36', colnames = ['stuff'])

            would add the column ``stuff_36``.
            If this is left empty, its default is set based on the function.
            If ``colnames`` has less elements than the function returns, only
            the first few are kept.
        colunits : list of strings
            Units for the output columns.
            If this is left empty, its default is set based on the function.
            If ``colunits`` has fewer elements than there are new columns,
            the unit of the remaining columns will be ``None``.
        coldescriptions : list of strings
            Descriptions for the output columns.
            If this is left empty, its default is set based on the function.
            If ``coldescriptions`` has fewer elements than there are new columns,
            the description of the remaining columns will be ``''``.
        coltypes: list
            list of dtypes for autogenerated columns. If the list is empty,
            its set to the default of the function called.
        overwrite : bool
            If True, values in existing columns are silently overwritten.
        t_simul : float
            max distance in days to accept datapoints in band 1 and 2 as simultaneous
            In L1688 and IRAS 20050+2720 the distance between band 1 and 2 coverage
            is within a few minutes, so a small number is sufficent to catch 
            everything and to avoid false matches.
            If ``None`` is given, this defaults to ``self.t_simul``.

        All remaining keywords are passed to the function identified by ``name``.
        '''
        t_simul = t_simul or self.t_simul
        if name not in registry.lc_funcs:
            raise ValueError('{0} not found in registry of function for autogeneration of columns'.format(name))
        # If band or colnames is just a string, make it a list
        if isinstance(bands, basestring): bands = [bands]
        if isinstance(colnames, basestring): colnames = [colnames]
        
        func = registry.lc_funcs[name]
        if func.n_bands != len(bands):
            raise ValueError('{0} requires {1} bands, but input was {2}'.format(name, func.n_bands, bands))
                             
        if colnames == []:
            colnames = func.default_colnames.keys()
        if coltypes == []:
            coltypes = func.default_colnames.values()

        colnames = [c + '_' + '_'.join(bands) for c in colnames]
        tband = set(['t'+b.replace('_error','') for b in bands])
        # bands could be '36_error' if we want the mean of the error or something
        # like that. merge_lc always return a band and its error, so make a 
        # list of names that contains only the real bandname
        basebands = [b.replace('_error','') for b in bands]
        if not overwrite and not set(colnames).isdisjoint(set(self.colnames)):
            raise Excpetion('overwrite = False, but the following columns exist and would be overwritten: {0}'.format(set(colnames).intersection(set(self.colnames))))

        #add new columns as required
        for c,d in zip(colnames, coltypes):
            if c not in self.colnames:
                self.add_column(astropy.table.Column(name = c, dtype = d, length = len(self)))
                if d == np.float:
                    self[c][:] = np.nan
        if colunits == []:
            colunits = func.default_colunits
        if coldescriptions == []:
            coldescriptions = func.default_coldescriptions
        for c, n in zip(colnames, colunits):
            self[c].unit = n
        for c, n in zip(colnames, coldescriptions):
            self[c].description = n

        for i in np.arange(0,len(self)):
            if tband.issubset(set(self.lclist[i].keys())):
                if data_preprocessor:
                    # need to make copy before we change the content
                    source  = self[i]
                    source.lclist = np.array([deepcopy(self.lclist[i])])
                    source = data_preprocessor(source)
                    lc = merge_lc(source.lclist[0], basebands, t_simul = t_simul)
                else:
                    lc = merge_lc(self.lclist[i], basebands, t_simul = t_simul)
                if timefilter:
                    ind = timefilter(lc['t'])
                    lc = lc[ind]
                    # Check if there is anything left in the lightcurve after
                    # timefilter
                    if ind.sum()==0:
                        continue

                args = []
                if func.time: args.append(lc['t'])
                for b in bands:
                    args.append(lc['m'+b])
                if func.error:
                    for b in bands:
                        args.append(lc['m'+b+'_error'])
                out = func(*args, **kwargs)
                # If out contains several values, it is a tuple. If not, make it one
                if not isinstance(out, tuple): out = (out, )
                for res, col in zip(out, colnames):
                    self[col][i] = res 


    def add_catalog_data(self, catalog, radius = 1./3600., names = None, **kwargs):
        '''add information from a different Table

        The tables are automatically cross matched and values are copied only
        for objects that have a counterpart in the current table.

        Parameters
        ----------
        catalog : astropy.table.Table
            This is the table where the new information is provided.
        radius : np.float
            matching radius in degrees
        names : list of strings
            List column names that should be copied. If this is `None` (the default)
            copy all columns. Column names have to be unique. Thus, make sure that
            no column of the same name aleady exisits (this will raise an exception).
        
        All other keywords are passed to :func:`YSOVAR.atlas.makecrossids` (see there
        for the syntax).
        '''
        names = names or catalog.colnames
        ids = makecrossids(self, catalog, radius, **kwargs) 
        matched = (ids >=0)
        multmatch = np.where(np.bincount(ids[matched]) > 1)[0]
        if len(multmatch) > 0:
            warn('add_catalog_data: The following sources in the input catalog are matched to more than one source in this atlas: {0}'.format(multmatch), UserWarning)

        for n in names:
            self.add_column(astropy.table.Column(name = n, length  = len(self), dtype=catalog[n].dtype, format=catalog[n].format, unit=catalog[n].unit, description=catalog[n].description, meta=catalog[n].meta))
            # select all types of floats
            if np.issubdtype(catalog[n].dtype, np.inexact):
                self[n][:] = np.nan
            elif np.issubdtype(catalog[n].dtype, np.int):
                self[n][:] = -99999

        for i in range(len(self)):
            if ids[i] >=0:
                for n in names:
                    self[n][i] = catalog[n][ids[i]]

    def add_mags(self, data, cross_ids, band, channel):
        '''Add lightcurves to some list of dictionaries


        Parameters
        ----------
        data : astropy.table.Table or np.rec.array
            data table with new mags
        cross_ids : list of lists
            for each elements in self, ``cross_ids`` says which row in `data`
            should be included for this object
        band : list of strings
            [name of mag, name or error, name of time]
        channel : string
            name of this channel in the lightcurve. Should be short and unique.

        Example
        -------
        In this example ``cat`` is a :class:`YSOVAR_atlas` and `pairitel.csv`
        looks like this::

            2MASSRA,2MASSDEC,Jmag,JmagE,HJDATE
            13.111, 14.111,12.,0.1,55000.
            13.111, 14.111,12.1,0.1,55001.
            <...>
            14.000, 15.000, 13.1, 0.12, 55000.
            <...>

        Then, this code will perform add thos J mags to ``cat``::

            jhk  = ascii.read('pairitel.csv', fill_values = ('', 'nan'))
            cross_ids = atlas.makecrossids_all(cat, jhk, 1./3600.,
                    ra1 = 'ra', dec1 = 'dec', ra2 = '2MASSRA', dec2 = '2MASSDEC')
            cat.add_mags(jhk, cross_ids, ['Jmag','JmagE','HJDATE'], 'J')

        
        '''
        if len(self) != len(cross_ids):
            raise ValueError('cross_ids needs to have same length as master table')
        for k in range(len(self)):
            if len(cross_ids[k]) > 0:
                ind = np.array(cross_ids[k])
                #self.lclist[k]['t'+channel].extend(data[band[2]][ind].tolist())
                #self.lclist[k]['m'+channel].extend(data[band[0]][ind].tolist())
                #self.lclist[k]['m'+channel+'_error'].extend(data[band[1]][ind].tolist())
                self.lclist[k]['t'+channel] = data[band[2]][ind]
                self.lclist[k]['m'+channel] = np.array(data[band[0]][ind])
                self.lclist[k]['m'+channel+'_error'].extend(data[band[1]][ind])

    def calc_allstats(self, band):
        '''calcualte all simple statistical descriptors for a single band

        This function calculates all simple statistical quantities that can
        be autogenerated for a certain band. The required columns are
        added to the data table.

        This does no include periodicity, which requires certain user selected
        parameters.

        Parameters
        ----------
        band : string
            name of band for which the calcualtion should be performed
        '''
        for f in registry.lc_funcs:
            func = registry.lc_funcs[f]
            if func.n_bands ==1:
                self.calc(f, band)

    def classify_SED_slope(self, bands=['mean_36', 'mean_45', 'Kmag', '3.6mag', '4.5mag', '5.8mag', '8.0mag'], colname = 'IRclass'):
        '''Classify the SED slope of an object

        This function calculates the SED slope for each object according
        to the prescription outlined by Luisa in the big data paper.

        It uses all available datapoints in the IR from the bands given
        If no measurement is present (e.g. missing or upper limit only)
        this band is ignored. The procedure performs a least-squares fit
        with equal weight for each band and then classifies the resulting
        slope into class I, flat-spectrum, II and III sources.

        Parameters
        ----------
        bands : list of strings
            List of names for all bands to be used. Bands must be defined in
            ``YSOVAR.atlas.sed_bands``.
        colname : string
            The classification will be placed in this column. If it exists
            it is overwritten.
        '''
        sed_ir = {}

        for b in bands:
            sed_ir[b] = sed_bands[b]

        if colname in self.colnames:
            self.remove_column(colname)
        self.add_column(astropy.table.Column(
                       name=colname, length=len(self), dtype='S3',
                       description='IR class according to SED slope'))
        for i, star in enumerate(self):
            slope = sed_slope(star, sed_ir)
            if slope >= 0.3:
                irclass = 'I'
            elif slope >= -0.3:
                irclass = 'F'
            elif slope >= -1.6:
                irclass = 'II'
            elif np.isfinite(slope):
                irclass = 'III'
            else:
                irclass = ''
            self[colname][i] = irclass

    def is_there_a_good_period(self, power, minper, maxper, bands=['36','45'], FAP=False):
        '''check if a strong periodogram peak is found

        This method checks if a period exisits with the required
        power and period in any of of the bands given in ``bands``. If
        the peaks in several bands fullfill the criteria, then the band with
        the peak of highest power is selected. Output is placed in the
        columns ``good_peak``, ``good_FAP`` and ``good_period``.

        New columns are added to the datatable that contains the result.
        (If a column existed before, it is overwritten).
    
        Parameters
        ----------
        power : float
            minimum power or maximal FAP for "good" period
        minper : float
            lowest period which is considered
        maxper : float
            maximum period which is considered
        bands : list of strings
            Band identifiers, e.g. ``['36', '45']``, can also be a list with one
            entry, e.g. ``['36']``
        FAP : boolean
            If ``True``, then ``power`` is interpreted as maximal FAP for a good
            period; if ``False`` then ``power`` means the minimum power a peak in
            the periodogram must have.
            
        '''
        if FAP:
            if not 0. <= power <= 1.:
                raise ValueError('FAP=True, so parameter power means the maximal FAP. This value must be in the range 0..1')
        else:
            if power <= 1.:
                raise ValueError('FAP=False, so parameter power gives the minimum power in the peak of the periodogram. Values <1 do not make sense.')
        if 'good_period' not in self.colnames:
            self.add_column(astropy.table.Column(name = 'good_period',
                            dtype=np.float, length = len(self), unit='d',
                            description='most significant period in object'))
        if 'good_peak' not in self.colnames:
            self.add_column(astropy.table.Column(name = 'good_peak',
                            dtype=np.float, length = len(self),
                            description='highest peak in periodogram'))
        if 'good_FAP' not in self.colnames:
            self.add_column(astropy.table.Column(name = 'good_FAP',
                            dtype=np.float, length = len(self),
                            description='FAP for most significant period'))
        self['good_period'][:] = np.nan
        self['good_peak'][:] = np.nan
        self['good_FAP'][:] = np.nan


        # numpy 1.7.1 introduced a bug in view for arrays that contain objects
        # see:
        # https://github.com/numpy/numpy/issues/3253
        # So, for now, make a workaround until this problem is fixed
        # peaknames = ['peak_'+band for band in bands]
        # periodnames = ['period_'+band for band in bands]
        # peaks = self._data[peaknames].view((np.float, len(bands)))
        # periods = self._data[periodnames].view((np.float, len(bands)))

        peaks = np.zeros((len(self), len(bands)))
        periods = np.zeros_like(peaks)
        FAPs = np.zeros_like(peaks)
        for i, band in enumerate(bands):
            peaks[:,i] = self['peak_'+band]
            periods[:,i] = self['period_'+band]
            FAPs[:,i] = self['FAP_'+band]
        
        if FAP:
            good = (FAPs < power) & (periods > minper) & (periods < maxper)
            bestpeak = np.argmax(np.ma.masked_where(~good, FAPs), axis=1)
        else:
            good = (peaks > power) & (periods > minper) & (periods < maxper)
            bestpeak = np.argmax(np.ma.masked_where(~good, peaks), axis=1)
        anygood = np.any(good, axis=1)
        self['good_period'][anygood] = periods[np.arange(len(self)),bestpeak][anygood]
        self['good_peak'][anygood] = peaks[np.arange(len(self)),bestpeak][anygood]
        self['good_FAP'][anygood] = FAPs[np.arange(len(self)),bestpeak][anygood]
