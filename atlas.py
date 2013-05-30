# -*- coding: utf-8 -*-
import os.path
from collections import defaultdict
import math
from copy import deepcopy
import string

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
           

def coord_CDS2RADEC(dat):
    '''transform RA and DEC from CDS table to degrees

    CDS tables have a certain format of string columns to store coordinates
    (`RAh`, `RAm`, `RAs`, `DE-`, `DEd`, `DEm`, `DEs`). This procedure
    parses that and calculates new values for RA and DEC in degrees.
    These are added to the Table as `RAdeg` and `DEdeg`.

    Parameters
    ----------
    dat : astropy.table.Table
        with columns in the CDS format (e.g. from reading a CDS table with
        `astropy.io.ascii`)
    '''
    radeg = dat['RAh']*15. + dat['RAm'] / 4. + dat['RAs']/4./60.
    dedeg = ((dat['DE-'] !='-')*2-1) * (dat['DEd'] + dat['DEm'] / 60. + dat['DEs']/3600.)
    coltype = type(dat.columns[0])  # could be Column or MaskedColumn
    dat.add_column(coltype(name = 'RAdeg', data = radeg))
    dat.add_column(coltype(name = 'DEdeg', data = dedeg))

def coord_hmsdms2RADEC(dat, ra = ['RAh', 'RAm', 'RAs'],dec = ['DEd', 'DEm','DEs']):
    '''transform RA and DEC from table to degrees

    Tables where RA and DEC are encoded as three numeric columns each like
    `hh:mm:ss` `dd:mm:ss` can be converted into decimal deg.
    This procedure parses that and calculates new values for RA and DEC in degrees.
    These are added to the Table as `RAdeg` and `DEdeg`.

    Parameters
    ----------
    dat : astropy.table.Table
        with columns in the format given above
    ra : list of three strings
        names of RA column names for hour, min, sec
    dec : list of three strings
        names of DEC column names for deg, min, sec
    '''
    radeg = dat[ra[0]]*15. + dat[ra[1]] / 4. + dat[ra[2]]/4./60.
    dedeg = dat[dec[0]] + dat[dec[1]] / 60. + dat[dec[2]]/3600.
    dat.add_column(astropy.table.Column(name = 'RAdeg', data = radeg))
    dat.add_column(astropy.table.Column(name = 'DEdeg', data = dedeg))

### Everything that deals with the lightcurve dictionaries and only with those ###

def radec_from_dict(data, RA = 'ra', DEC = 'dec'):
    '''return ra dec numpy array for list of dicts
    
    Parameters
    ----------
    data : list of dict
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
    


def makecrossids(data1, data2, radius, ra1='RAdeg', dec1='DEdeg', ra2='ra', dec2='dec'):
    '''Cross-match two lists of coordinates, return closest match

    This routine is not very clever and not very fast. If should be fine
    up to a few thousand entries per list. 

    Parameters
    ----------
    data1 : astropy.table.Table or np.recarray
        This is the master data, i.e. for each element in data1, the
        results wil have one (or zero) index numbers in data2, that provide
        the best match to this entry in data1.
    data2 : astropt.table.Table or np.recarray
        This data is matched to data1.
    radius : np.float
       maximum radius to accept a match (in degrees) 
    ra1, dec1, ra2, dec2 : string
        key for access RA and DEG (in degrees) the the data, i.e. the routine
        uses `data1[ra1]` for the RA values of data1.

    Results
    -------
    cross_ids : np.ndarray
        Will have len(data1). For each elelment it contains the index of data2
        that provides the best match. If no match within `radius` is found,
        then entry will be -99999.
    '''
    cross_ids = np.ones(len(data1),int) * -99999
    
    for i in np.arange(0,len(data1)):
        # Pick out only those that are close in dec
        ind = np.where(np.abs(data1[dec1][i] - data2[dec2]) < radius)[0]
        # and calculate the full dist_radec only for those that are close enough
        # since dist_radec includes several sin, cos, that speeds it up a lot
        if len(ind) > 0:
            distance = dist_radec(data1[ra1][i], data1[dec1][i], data2[ra2][ind], data2[dec2][ind], unit ='deg') 
            if min(distance) < radius:
                cross_ids[i] = ind[np.argmin(distance)]
        
    return cross_ids

def makecrossids_all(data1, data2, radius, ra1='RAdeg', dec1='DEdeg', ra2='ra', dec2='dec'):
    '''Cross-match two lists of coordinates, return all matches within radius

    This routine is not very clever and not very fast. If should be fine
    up to a few thousand entries per list. 

    Parameters
    ----------
    data1 : astropy.table.Table or np.recarray
        This is the master data, i.e. for each element in data1, the
        results wil have one (or zero) index numbers in data2, that provide
        the best match to this entry in data1.
    data2 : astropt.table.Table or np.recarray
        This data is matched to data1.
    radius : np.float
       maximum radius to accept a match (in degrees) 
    ra1, dec1, ra2, dec2 : string
        key for access RA and DEG (in degrees) the the data, i.e. the routine
        uses `data1[ra1]` for the RA values of data1.

    Results
    -------
    cross_ids : list of lists
        Will have len(data1). For each elelment it contains the indices of data2
        that are within `radius`. If no match within `radius` is found,
        then entry will be `[]`.
    '''
    cross_ids = []
    
    for i in range(len(data1)):
        # Pick out only those that are close in dec
        ind = np.where(np.abs(data1[dec1][i] - data2[dec2]) <= radius)[0]
        # and calculate the full dist_radec only for those that are close enough
        # since dist_radec includes several sin, cos, that speeds it up a lot
        if len(ind) > 0:
            distance = dist_radec(data1[ra1][i], data1[dec1][i], data2[ra2][ind], data2[dec2][ind], unit ='deg') 
            cross_ids.append(ind[distance <= radius])
        else:
            cross_ids.append([])

    return cross_ids


''' Format:
    dictionary of bands, where the name of the band mag is the key
    entries are lists of [name of error field, wavelength in micron, zero_magnitude_flux_freq in Jy = 1e-23 erg s-1 cm-2 Hz-1]'''
sed_bands = {'Umag': ['e_Umag', 0.355, 1500], 'Bmag': ['e_Bmag', 0.430, 4000.87], 'Vmag': ['e_Vmag', 0.623, 3597.28], 'Rmag': ['e_Rmag', 0.759, 3182], 'Imag': ['e_Imag', 0.798, 2587], 'Jmag': ['e_Jmag', 1.235, 1594], 'Hmag': ['e_Hmag', 1.662, 1024], 'Kmag': ['e_Kmag', 2.159, 666.7], '3.6mag': ['e_3.6mag', 3.6, 280.9], '4.5mag': ['e_4.5mag', 4.5, 179.7], '5.8mag': ['e_5.8mag', 5.8, 115.0], '8.0mag': ['e_8.0mag', 8.0, 64.13], '24mag': ['e_24mag', 24.0, 7.14], 'Hamag': ['e_Hamag', 0.656, 2974.4], 'rmag': ['e_rmag', 0.622, 3173.3], 'imag': ['e_imag', 0.763, 2515.7], 'nomad_Bmag': [None, 0.430, 4000.87], 'nomad_Vmag': [None, 0.623, 3597.28], 'nomad_Rmag': [None, 0.759, 3182], 'simbad_B': [None, 0.430, 4000.87], 'simbad_V': [None, 0.623, 3597.28]}
# using Sloan wavelengths for r and i
# compare: http://casa.colorado.edu/~ginsbura/filtersets.htm
# from: http://coolwiki.ipac.caltech.edu/index.php/Central_wavelengths_and_zero_points
# from: http://arxiv.org/pdf/1011.2020.pdf for SLoan u', r', i' (Umag, Rmag, Imag)

def get_sed(data, sed_bands = sed_bands):
    '''make SED by collecting info from the input data
    
    Parameters
    ----------
    data : np.rec.array or atpy.table or dict
        input data that has arrays of magnitudes for different bands
    sed_bands : dict 
        keys must be the name of the field that contains the magnitudes in each band
        entries are lists of [name of error field, wavelength in micron,
        zero_magnitude_flux_freq in Jy]
        
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
        as obtained from :func:`add_ysovar_mags`
    channels : dictionary
        This dictionary traslantes the names of channels in the csv file to
        the names in the output structure, e.g.
        that for `'IRAC1'` will be `'m36'` (magnitudes) and `'t36'` (times).
    min_number_of_times : integer
        Remove all sources with less than `min_number_of_times` datapoints
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
                n_datapoints[i] = max(n_datapoints[i], len(d['m'+channels[channel]]))
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
        d['ISOY_NAME'] = set(d['ISOY_NAME'])
        d['ISOY_NAME'] = ', '.join(d['ISOY_NAME'])
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
        as obtained from :func:`add_ysovar_mags`
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



def Isoy2radec(isoy):
    '''convert ISOY name do decimal degrees.
    
    Parameters
    ----------
    isoy : string
        ISOY Name 

    Returns
    -------
    ra, dec : float
        ra and dec in decimal degrees
    '''
    s = (isoy.split('J'))[-1]
    ra = int(s[0:2]) *15. + int(s[2:4])/4. + float(s[4:9])/4./60.
    dec = np.sign(float(s[9:])) * (float(s[10:12]) + int(s[12:14]) /60. + float(s[14:])/3600.)
    return ra, dec

def dict_from_csv(csvfile,  match_dist = 1.0/3600., min_number_of_times = 5, channels = {'IRAC1': '36', 'IRAC2': '45'}, data = [], floor_error = {'IRAC1': 0.01, 'IRAC2': 0.007}, mag = 'mag1', emag = 'emag1', time = 'hmjd', bg = None, source_name = 'sname',  verbose = True):
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
        This dictionary traslantes the names of channels in the csv file to
        the names in the output structure, e.g.
        that for `'IRAC1'` will be `'m36'` (magnitudes) and `'t36'` (times).
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
    for i, n in enumerate(set(tab['sname'])):
        if verbose and (np.mod(i,100)==0):
            print 'Processing dict for source ' + str(i) + ' of ' + str(len(set(tab['sname'])))
        ind = (tab['sname'] == n)
        ra = tab[ind][0]['ra']
        dec = tab[ind][0]['de'] 
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
        dict_temp['ISOY_NAME'].append(n)
        dict_temp['YSOVAR2_id'].append(tab['ysovarid'][ind][0])
        for channel in channels.keys():
            good = ind & (tab[time] >= 0.) & (tab['fname'] == channel)
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
        if (not ('t36' in d.keys()) or (len(d['t36']) < min_number_of_times)) and  (not ('t45' in d.keys()) or (len(d['t45']) < min_number_of_times)): print i, list(d['ISOY_NAME'])
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
        if len(d['ISOY_NAME'].split(',')) > 1: print i, d['ISOY_NAME']
    print '----------------------------------------------------------------------------'
    print 'The following sources are less than ', match_dist, ' deg apart'
    radec = radec_from_dict(data)
    for i in range(len(data)-1):
        dist = dist_radec_fast(radec['RA'][i], radec['DEC'][i], radec['RA'][i+1:], radec['DEC'][i+1:], scale = match_dist, unit = 'deg')
        if np.min(dist) < match_dist:
            print 'distance', i, i+np.argmin(dist), ' is only ', np.min(dist)
    print '----------------------------------------------------------------------------'



#### The big table / Atlas class that holds the data and does some cool processing ##


class YSOVAR_atlas(astropy.table.Table):
    '''
    The basic structure for the YSOVAR analysis is the
    :class:`YSOVAR_atlas`. 
    To initialize an atlas object pass is a numpy array wich all the lightcurves::

        import ysovar_atlas as atlas
        data = atlas.dict_from_csv('/path/tp/my/irac.csv', match_dist = 0.)
        MyRegion = atlas.YSOVAR_atlas(lclist = data)

    The :class:`YSOVAR_atlas` is build on top of a `astropy.table.Table
    (documentation here)
    <http://docs.astropy.org/en/v0.21/table/index.html>`_ object. See that
    documentation for the syntax on how to acess the data or add a column.

    Some columns are auto-generated, when they are first
    used. Specifically, these are the 
    - median
    - mean
    - stddev
    - min
    - max
    - mad (median absolute deviation)
    - delta (90% quantile - 10% quantile)
    - redchi2tomean
    - wmean (uncertainty weighted average).

    When you ask for `MyRegion['min_36']` it first checks if that column is already
    present. If not, if adds the new column called `min_36` and calculates
    the minimum of the lightcurve in band `36` for each object in the
    atlas, that has `m36` and `t36` entries (for magnitude and time in
    band `36` respectively. Data read with :meth:`dict_from_csv`
    atomatically has the required format.

    More functions may be added to this magic list later. Check::

        import YSOVAR.registry 
        YSOVAR.registry.list_lcfuncs()

    to see which functions are implemented.
    More function can be added.

    Also, table columns can be added to an ``YSOVAR_atlas`` object manually,
    giving you all the freedome to to arbitrary calculations to arrive at those
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
        for name in ['ra', 'dec', 'YSOVAR2_id','ISOY_NAME']:
             col = astropy.table.Column(name = name, data = val_from_dict(self.lclist, name))
             self.add_column(col)
    
    def __getitem__(self, item):
        
        if isinstance(item, basestring) and item not in self.colnames:
            self.autocalc_newcol(item)
            # safeguard against infinite loops, because there is no direct
            # control, if the new column is really called item
            if item in self.colnames:
                return self[item]
            else:
                raise Exception('Attempted to calculate {0}, but the columns added are not called {1}'.format(item, item))

        # In thiscase astropy.table.Table would return a Row object
        # not a new Table
        if isinstance(item, int):
            item = [item]
         
        return super(YSOVAR_atlas, self).__getitem__(item)

    def _new_from_slice(self, slice_):
        """Create a new table as a referenced slice from self."""

        table = YSOVAR_atlas(lclist = self.lclist[slice_])
        # delete the columns that are autogenerated and just copy everything
        table.remove_columns(['ra', 'dec', 'YSOVAR2_id','ISOY_NAME'])
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
            self.calc(splitname[0], splitname[1:], colnames = [splitname[0]])
        except:
            raise ValueError('Column '+ name + ' not found and cannot be autogenerated. Use ``calc`` to explictly calculate')

    def calc(self, name, bands, timefilter = None, data_preprocessor = None, colnames = [], overwrite = True, t_simul = None, **kwargs):
        '''calculate Lomb-Scagle periodograms for all sources

        A new column is added to the datatable that contains the result.
        (If the column exists before, it is overwritten).
    
        Parameters
        ----------
        name : string
            name of the function in the function registry for YSOVAR
            (see :mod:`registry`)
        bands : list of strings
            Band identifiers
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
            If colnames has less elements than the function returns, only the
            first few are kept.
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
            coltypes = func.default_colnames.values()
        else:
            coltypes = [np.float] * len(colnames)
        colnames = [c + '_' + '_'.join(bands) for c in colnames]
        tband = set(['t'+b for b in bands])
        if not overwrite and not set(colnames).isdisjoint(set(self.colnames)):
            raise Excpetion('overwrite = False, but the following columns exist and would be overwritten: {0}'.format(set(colnames).intersection(set(self.colnames))))

        #add new columns as required
        for c,d in zip(colnames, coltypes):
            if c not in self.colnames:
                self.add_column(astropy.table.Column(name = c, dtype = d, length = len(self)))
                if d == np.float:
                    self[c][:] = np.nan

        for i in np.arange(0,len(self)):
            if tband.issubset(set(self.lclist[i].keys())):
                # need to make copy before we change the content
                source  = self[i]
                source.lclist = np.array([deepcopy(self.lclist[i])])
                if data_preprocessor:
                    source = data_preprocessor(source)
                lc = merge_lc(source.lclist[0], bands, t_simul = t_simul)
                if timefilter:
                    ind = timefilter(lc['t'])
                    lc = lc[ind]

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


    def add_catalog_data(self, catalog, radius = 1./3600., names = None, ra1 = 'RA', dec1 = 'DE', ra2 = 'RA', dec2 = 'DE'):
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
        ra1, dec1, ra2, dec2 : string
            key for access RA and DEG (in degrees) the the data, i.e. the routine
            uses `data1[ra1]` for the RA values of data1.    
        '''
        names = names or catalog.colnames
        ids = makecrossids(self, catalog, radius, ra1 = ra1 , dec1 = dec1, ra2 = ra2, dec2 = dec2) 

        for n in names:
            self.add_column(astropy.table.Column(name = n, length  = len(self), dtype = catalog[n].dtype))

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
                self.lclist[k]['t'+channel].extend(data[band[2]][ind].tolist())
                self.lclist[k]['m'+channel].extend(data[band[0]][ind].tolist())
                self.lclist[k]['m'+channel+'_error'].extend(data[band[1]][ind].tolist())

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

    def is_there_a_good_period(self, power, minper, maxper, bands=['36','45']):
        '''check if a strong periodogram peak is found

        This method checks if a period exisits with the required
        power and period in any of of the bands given in `bands`. If
        the peaks in several bands fullfill the criteria, then the band with
        the peak of highest power is selected. Output is placed in the
        columns `good_peak` and `good_period`.

        New columns are added to the datatable that contains the result.
        (If a column existed before, it is overwritten).
    
        Parameters
        ----------
        power : float
            required power threshold for "good" period
        minper : float
            lowest period which is considered
        maxper : float
            maximum period which is considered
        bands : list of strings
            Band identifiers, e.g. ['36', '45'], can also be a list with one
            entry, e.g. ['36']
        '''
        if 'good_period' not in self.colnames:
            self.add_column(astropy.table.Column(name = 'good_period',
                            dtype=np.float, length = len(self)))
        if 'good_peak' not in self.colnames:
            self.add_column(astropy.table.Column(name = 'good_peak',
                            dtype=np.float, length = len(self)))
        self['good_period'][:] = np.nan
        self['good_peak'][:] = np.nan

        peaknames = tuple(['peak_'+band for band in bands])
        periodnames = tuple(['period_'+band for band in bands])

        peaks = self[peaknames]._data.view((np.float, len(bands)))
        periods = self[periodnames]._data.view((np.float, len(bands)))
        
        good = (peaks > power) & (periods > minper) & (periods < maxper)
        bestpeak = np.argmax(np.ma.masked_where(~good, peaks), axis=1)
        anygood = np.any(good, axis=1)
        self['good_period'][anygood] = periods[np.arange(peaks.shape[0]),bestpeak][anygood]
        self['good_peak'][anygood] = peaks[np.arange(peaks.shape[0]),bestpeak][anygood]