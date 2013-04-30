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

import ysovar_lombscargle
from great_circle_dist import dist_radec, dist_radec_fast
import lightcurves


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


def mad(data):
    '''calculate median absolute deviation'''
    return np.median(np.abs(data - np.median(data)))

def redchi2tomean(data, error):
    '''reduced chi^2 to mean'''
    return np.sum( (data - np.mean(data))**2/(error**2) )/(len(data)-1)

def delta(data):
    '''with of distribution form 10%-90%'''
    return (scipy.stats.mstats.mquantiles(data, prob=0.9) - scipy.stats.mstats.mquantiles(data, prob=0.1))/2.

def wmean(data, error):
    '''error weighted mean'''
    return np.average(data, weights=1./error**2.)

def stetson(data1, data1_error,data2, data2_error):
    '''Calculates the Stetson index for a two-band light curve.

    According to eqn (1) in Stetson 1996, PSAP, 108, 851.
    This procedure uses on the matched lightcurves
    (not frames with one band only) and assignes a weight (g_i) in
    Stetson (1996) of 1 to each datapoint.

    Parameters
    ----------
    data1 : np.array
        single light curve of band 1 in magnitudes
    data1_error : np.array
        error on data points of band 1 in magnitudes    
    data2 : np.array
        single light curve of band 2 in magnitudes
    data2_error : np.array
        error on data points of band 2 in magnitudes          
        
    Returns
    -------
    stetson : float
        Stetson value for the provided two-band light curve
        
    '''
    # number of datapoints:
    N = float(len(data1))
    
    if (len(data2) != N) or (len(data1_error) !=N) or (len(data2_error) !=N):
        raise ValueError('All input arrays must have the same length')
    if N > 1:
        # weighted mean magnitudes in each passband:
        wmean1 = wmean(data1, data1_error)
        wmean2 = wmean(data2, data2_error)
        # normalized residual from the weighted mean for each datapoint:
        res_1 = (data1 - wmean1) / data1_error
        res_2 = (data2 - wmean2) / data2_error
        P_ik = res_1 * res_2
        return np.sqrt(1./(N*(N-1))) * np.sum( np.sign(P_ik) * np.sqrt(np.abs(P_ik)) )
    else:
        return np.nan

#def redvec_36_45():
#    ''' Rieke & Lebofsky 1984:  I take the extinctions from the L and M band (3.5, 5.0).'''
#    A36 = 0.058
#    A45 = 0.023
#    R36 = - A36/(A45 - A36)
#    return np.array([R36, A36])
redvec_36_45 = np.array([-2.58, 0.058])

def fit_cmdslope_simple(data1, data1_error,data2, data2_error, redvec):
    '''measures the slope of the data points in the color-magnitude diagram
    
    This is just fitted with ordinary least squares, using the analytic formula.
    This is then used as a first guess for an orthogonal least squares fit with simultaneous treatment of errors in x and y (see fit_twocolor_odr)

    Parameters
    ----------
    data1 : np.array
        single light curve of band 1 in magnitudes
    data1_error : np.array
        error on data points of band 1 in magnitudes    
    data2 : np.array
        single light curve of band 2 in magnitudes
    data2_error : np.array
        error on data points of band 2 in magnitudes
    redvec : np.array with two elements
        theoretical reddening vector for the two bands chosen
        
    Returns
    -------
    m : float
        slope of fit in color-magnitude diagram
    b : float
        axis intercept of fit
    m2 : float
        slope of the input theoretical reddening vector `redvec`
    b2 : float
        axis intercept of fit forcin the slope to `m2`
    redchi2 : float
        reduced chi^2 of fit of `[m,b]`
    redchi2_2 : float
        reduced chi^2 of fit of `b2`
    
        
    '''
    # number of datapoints:
    N = float(len(data1))
    
    if (len(data2) != N) or (len(data1_error) !=N) or (len(data2_error) !=N):
        raise ValueError('All input arrays must have the same length')

    x = data1 - data2
    y = data1
    x_error = np.sqrt( data1_error**2 + data2_error**2 )
    y_error = data1_error
    # calculate the different sums:
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x**2)
    sum_xy = np.sum(x*y)
    # now get b and m from analytic formula:
    m = (-sum_x*sum_y + N*sum_xy) / (N*sum_xx - sum_x*sum_x)
    b = (-sum_x*sum_xy + sum_xx*sum_y) / (N*sum_xx - sum_x*sum_x)
    # now calculate chisquared for this line:
    redchi2 = np.sum( (y - (m*x+b))**2/ y_error**2)/(N-2)
    
    # now fit theoretical reddening vector to data, for plotting purposes (i.e. just shifting it in y:)
    m2 = redvec[0] # the sign is okay, because the y axis is inverted in the plots
    b2 = 1/N * ( sum_y - m2 * sum_x )
    redchi2_2 = np.sum( (y - (m2*x+b2))**2/y_error**2 )/(N-2)
    
    return m,b,m2,b2,redchi2,redchi2_2





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
    

def readguentherlist(filename):  
    # reads the data from Table 3 in Guenther+ 2012 and returns both (a) all data and (b) only ysos and stars as one subset. The data from the paper needs to be stored locally as ascii file.
    print 'reading data from Guenther+ 2012 ...'
    a = asciitable.read(filename)
    guenther_data = a.data
    
    guenther_data_subset = guenther_data[np.where( (guenther_data['Class'] == 'XYSO') | (guenther_data['Class'] == 'I*') | (guenther_data['Class'] == 'I') | (guenther_data['Class'] == 'II*') | (guenther_data['Class'] == 'II') | (guenther_data['Class'] == 'III') | (guenther_data['Class'] == 'III*')  | (guenther_data['Class'] == 'star')  )[0]]
    
    return (guenther_data, guenther_data_subset)


def makeclassinteger(guenther_data_yso):
    # assigns an integer to the Guenther+ 2012 classes. 0=XYSO, 1=I+I*, 2=II+II*, 3=III, 4=star
    guenther_class = np.ones(len(guenther_data_yso),int)*-9
    guenther_class[np.where(guenther_data_yso['Class'] == 'XYSO' )[0]] = 0
    guenther_class[np.where((guenther_data_yso['Class'] == 'I*') | (guenther_data_yso['Class'] == 'I'))[0]] = 1
    guenther_class[np.where((guenther_data_yso['Class'] == 'II*') | (guenther_data_yso['Class'] == 'II'))[0]] = 2
    guenther_class[np.where((guenther_data_yso['Class'] == 'III') | (guenther_data_yso['Class'] == 'III*'))[0]] = 3
    guenther_class[np.where(guenther_data_yso['Class'] == 'star')[0]] = 4
    return guenther_class




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

    allbandsthere = True
    for band in bands:
        allbandsthere = allbandsthere and ('t'+band in d.keys())
    if allbandsthere:
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

def dict_from_csv(csvfile,  match_dist = 1.0/3600., min_number_of_times = 5, channels = {'IRAC1': '36', 'IRAC2': '45'}, data = [], floor_error = {'IRAC1': 0.01, 'IRAC2': 0.008}, mag = 'mag1', emag = 'emag1', time = 'hmjd', bg = None, source_name = 'sname',  verbose = True):
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
    something I add a check sp that next time this routine will warn me of the 
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

valfuncdict = {'mean': np.mean, 'median': np.median, 'stddev': lambda x: np.std(x, ddof = 1), 'min': np.min, 'max': np.max, 'mad': mad, 'delta': delta, 'skew': scipy.stats.skew, 'kurtosis': scipy.stats.kurtosis}
'''
mad: median absolute deviation
stddev: standard deviation calculated fron non-biased variance
skew:  biased (no correction for dof) skew
kurtosis:  biased (no correction for dof) Fischer kurtosis
'''

valerrfuncdict = {'redchi2tomean': redchi2tomean, 'wmean': wmean}
'''
redchi2tomean - reduced chi^2 to mean value
wmean - error weighted mean
'''

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
    <http://docs.astropy.org/en/v0.2/table/index.html>`_ object. See that
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

    More function may be added to this magic list later. Check::

        import ysovar_atlas as atlas
        atlas.valfuncdict
        atlas.valerrfuncdict

    to see which functions are implemented.

    :class:`YSOVAR_atlas` also includes some more functions (which need to
    be called explicitly) to add columns that contain period, LS peaks,
    etc.  See the documentation of the individual methods.
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
            newcol = self.autocalc_newcol(item)
            self.add_column(astropy.table.Column(data = newcol, name = item))

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
        table.meta = deepcopy(self.meta)
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
        #make newcol of type float
        newcol = astropy.table.Column(name = name, length = len(self), dtype = np.float)
        newcol[:] = np.nan
        if (len(splitname) == 2 and splitname[0] in valfuncdict):
            func = valfuncdict[splitname[0]]
            for i in np.arange(0,len(self)):
                if 'm'+splitname[1] in self.lclist[i]:
                    newcol[i] = func(self.lclist[i]['m'+splitname[1]])
        elif (len(splitname) == 2) and (splitname[0] in valerrfuncdict):
            func = valerrfuncdict[splitname[0]]
            for i in np.arange(0, len(self)):
                if ('m'+splitname[1] in self.lclist[i]) and ('m'+splitname[1]+'_error' in self.lclist[i]):
                     newcol[i] = func(self.lclist[i]['m'+splitname[1]], self.lclist[i]['m'+splitname[1]+'_error'])
        elif (len(splitname) == 2) and (splitname[0] == 'n'):
            #want newcol of type int
            newcol = astropy.table.Column(name = name, data = np.zeros(len(self), dtype = np.int))
            for i in np.arange(0,len(self)):
                if 'm'+splitname[1] in self.lclist[i]:
                    newcol[i] = len(self.lclist[i]['m'+splitname[1]])
        else:
            raise ValueError('Column '+ name + ' not found and cannot be autogenerated.')
        return newcol

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
            for each elements in self, `cross_ids` says which row in `data`
            should be included for this object
        band : list of strings
            [name of mag, name or error, name of time]
        channel : string
            name of this channel in the lightcurve. Should be short and unique.

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
        for f in valfuncdict.keys():
            temp = self[f+'_'+band]
        for f in valerrfuncdict.keys():
            temp = self[f+'_'+band]



    def calc_stetson(self, band1, band2, t_simul = None):
        '''calculates the steson index between two bands for all lightcurves

        A new column is added to the datatable that contains the result.
        (If the column existed before, it is overwritten).
        
        Parameters
        ----------
        band1, band2 : string
            name of the bands to be used for the calculation
        t_simul : float
            max distance in days to accept datapoints in band 1 and 2 as simultaneous
            In L1688 and IRAS 20050+2720 the distance between band 1 and 2 coverage
            is within a few minutes, so a small number is sufficent to catch 
            everything and to avoid false matches.
            If `None` is given, this defaults to `self.t_simul`.
        '''
        t_simul = t_simul or self.t_simul
        name = 'stetson_'+band1+'_'+band2
        if name not in self.colnames:
            self.add_column(astropy.table.Column(name = name, length = len(self), dtype = np.float))
        self[name][:] = np.nan
        for i in np.arange(len(self)):
            data = merge_lc(self.lclist[i],[band1, band2], t_simul = t_simul)
            self[name][i] = stetson(data['m'+band1], data['m'+band1+'_error'],
                                    data['m'+band2], data['m'+band2+'_error'])

    def cmd_slope_simple(self, band1='36', band2='45', redvec=redvec_36_45, t_simul = None):
        '''Fit straight line to color-magnitude diagram

        A new column is added to the datatable that contains the result.
        (If the column existed before, it is overwritten).
        
        Parameters
        ----------
        band1, band2 : string
            name of the bands to be used for the calculation
        t_simul : float
            max distance in days to accept datapoints in band 1 and 2 as simultaneous
            In L1688 and IRAS 20050+2720 the distance between band 1 and 2 coverage
            is within a few minutes, so a small number is sufficent to catch 
            everything and to avoid false matches.
            If `None` is given, this defauls to `self.t_simul`.
        '''
        t_simul = t_simul or self.t_simul
        names = ['cmd_m_plain', 'cmd_b_plain', 'cmd_m_redvec', 'cmd_b_redvec']
        for name in names:
            if name not in self.colnames:
                self.add_column(astropy.table.Column(name = name, length = len(self), dtype = np.float))
            self[name][:] = np.nan
        for i in np.arange(len(self)):
            data = merge_lc(self.lclist[i],[band1, band2], t_simul = t_simul)
            if len(data) > 1:
                m,b,m2,b2,chi2,ch2_2 = fit_cmdslope_simple(
                    data['m'+band1], data['m'+band1+'_error'],
                    data['m'+band2], data['m'+band2+'_error'],
                    redvec)
                self['cmd_m_plain'][i] = m
                self['cmd_b_plain'][i] = b
                self['cmd_m_redvec'][i] = m2
                self['cmd_b_redvec'][i] = b2

    def cmd_slope_odr(self, outroot = None, n_bootstrap = None, band1='36', band2='45', redvec=redvec_36_45, t_simul = None):
        '''Performs straight line fit to CMD for all sources.

        Adds fitted parameters to info structure.
    
        Parameters
        ----------
        outroot : string or None
            dictionary where to save the plot, set to `None` for no plotting
        n_bootstrap : integer or None
            how many bootstrap trials, set to `None` for no bootstrapping
        band1, band2 : string
            name of the bands to be used for the calculation
        t_simul : float
            max distance in days to accept datapoints in band 1 and 2 as simultaneous
            In L1688 and IRAS 20050+2720 the distance between band 1 and 2 coverage
            is within a few minutes, so a small number is sufficent to catch 
            everything and to avoid false matches.
            If `None` is given, this defaults to `self.t_simul`.
    
        '''
        t_simul = t_simul or self.t_simul
        names = ['cmd_alpha2', 'cmd_alpha2_error', 'cmd_m2', 'cmd_b2', 'cmd_m2_error', 'cmd_b2_error', 'cmd_x_spread', 'cmd_alpha1', 'cmd_alpha1_error', 'cmd_m', 'cmd_b', 'cmd_m_error', 'cmd_b_error', 'cmd_x_spread']

        for name in names:
            if name not in self.colnames:
                self.add_column(astropy.table.Column(name = name, length = len(self), dtype = np.float))
            self[name][:] = np.nan
        if ('cmd_m_plain' not in self.colnames) or ('cmd_b_plain' not in self.colnames):
            self.cmd_slope_simple( band1, band2, redvec, t_simul)
        for i in np.arange(len(self)):
            data = merge_lc(self.lclist[i],[band1, band2], t_simul = t_simul)
            if len(data) > 1:
                # use the result of the plain least squares fitting as a first parameter guess.
                p_guess = (self['cmd_m_plain'][i], self['cmd_b_plain'][i])
            
                (fit_output, bootstrap_output, bootstrap_raw, alpha, alpha_error, x_spread) = fit_twocolor_odr(data, i, p_guess, outroot, n_bootstrap, True)
                self['cmd_alpha2'][i] = alpha
                self['cmd_alpha2_error'][i] = alpha_error
                self['cmd_m2'][i] = fit_output.beta[0]
                self['cmd_b2'][i] = fit_output.beta[1]
                self['cmd_m2_error'][i] = fit_output.sd_beta[0]
                self['cmd_b2_error'][i] = fit_output.sd_beta[1]
                self['cmd_x_spread'][i] = x_spread

                (fit_output, bootstrap_output, bootstrap_raw, alpha, alpha_error, x_spread) = fit_twocolor_odr(data, i, p_guess, outroot, n_bootstrap, False)
                self['cmd_alpha1'][i] = alpha
                self['cmd_alpha1_error'][i] = alpha_error
                self['cmd_m'][i] = fit_output.beta[0]
                self['cmd_b'][i] = fit_output.beta[1]
                self['cmd_m_error'][i] = fit_output.sd_beta[0]
                self['cmd_b_error'][i] = fit_output.sd_beta[1]
                self['cmd_x_spread'][i] = x_spread

    def good_slope_angle(self):
        '''Checks if ODR fit to slope encountered some pathological case
        
        Checks if the ODR fit with switched X and Y axes yields a more
        constrained fit than the original axes. This basically catches the
        pathological cases with a (nearly) vertical fit with large nominal errors.
        '''
        names = ['cmd_alpha', 'cmd_alpha_error']

        for name in names:
            if name not in self.colnames:
                self.add_column(astropy.table.Column(name = name, length = len(self), dtype = np.float))
            self[name][:] = np.nan

        good = self['cmd_alpha1'] > -99999.
        comp = self['cmd_alpha1_error']/self['cmd_alpha1'] < self['cmd_alpha2_error']/self['cmd_alpha2']
        self['cmd_alpha'][good] = np.where(comp, self['cmd_alpha1'], self['cmd_alpha2'])[good]
        self['cmd_alpha_error'][good] = np.where(comp, self['cmd_alpha1_error'], self['cmd_alpha2_error'])[good]



    def cmd_dominated_by(self, redvec = redvec_36_45):
        '''crude classification of CMD slope

        This is some crude classification of the cmd slope.
        anything that goes up and has a relative slope error of <40% is
        "accretion-dominated", anythin that is within some cone around
        the theoratical reddening and has error <40% is "extinction-dominated",
        anything else is "other".
        If slope is classified as extinction, the spread in the CMD is converted
        to AV and stored.
        '''
        alpha_red = math.asin(redvec[0]/np.sqrt(redvec[0]**2 + 1**2)) # angle of standard reddening
        if 'cmd_dominated' not in self.colnames:
            self.add_column(astropy.table.Column(name = 'cmd_dominated', length = len(self), dtype = 'S10'))
        if 'AV' not in self.colnames:
            self.add_column(astropy.table.Column(name = 'AV', length = len(self), dtype = np.float))
        self['AV'][:] = np.nan
            
        self['cmd_dominated'] = 'no data'
        self['cmd_dominated'][self['cmd_alpha'] > -99999] = 'bad'
        self['cmd_dominated'][(self['cmd_dominated'] == 'bad') & (self['cmd_alpha_error']/self['cmd_alpha'] <=0.3)] = 'extinc.' 
        self['cmd_dominated'][(self['cmd_dominated'] == 'extinc.') & (self['cmd_alpha'] < 0.)] = 'accr.'
        ind = (self['cmd_dominated'] == 'extinc.') 
        self['AV'][ind] = self['cmd_x_spread'][ind]/redvec[1]

    def describe_autocorr(self, band, scale = 0.1):
        '''Describe the autocorrelation for all lightcurves

        Three new columns are added to the datatable that contain the result.
        (If the columns existed before, they are overwritten).
    
        Parameters
        ----------
        band : string
            Band identifier
        scale : float
            Since the lightcurves are unevenly sampeled, the resulting
            autocorrelation function needs to be binned. `scale` sets
            the width of those bins.
        '''
        colnames = ['tcorr_', 'tauto_', 'valauto_']
        for cname in colnames:
            if cname+band not in self.colnames:
                self.add_column(astropy.table.Column(name = cname+band, dtype=np.float, length = len(self)))
                self[cname+band][:] = np.nan

        for i in np.arange(0,len(self)):
            if 't'+band in self.lclist[i].keys():
                t1 = self.lclist[i]['t'+band]
                m1 = self.lclist[i]['m'+band]
                if len(t1) > 5:
                    (tcorr, tauto, valauto) = lightcurves.describe_autocorr(t1, m1, scale = scale)
                    self['tcorr_'+band][i] = tcorr
                    self['tauto_'+band][i] = tauto
                    self['valauto_'+band][i] = valauto

        
 
    def calc_ls(self, band, maxper, oversamp = 4, maxfreq = 1.):
        '''calculate Lomb-Scagle periodograms for all sources

        A new column is added to the datatable that contains the result.
        (If the column exists before, it is overwritten).
    
        Parameters
        ----------
        band : string
            Band identifier
        maxper : float
            periods above this value will be ignored
        oversamp : integer
            oversampling factor
        maxfreq : float
            max freq of LS periodogram is maxfeq * "average" Nyquist frequency
             For very inhomogenously sampled data, values > 1 can be useful
        '''
        colnames = ['period_', 'peak_', 'FAP_']
        for cname in colnames:
            if cname+band not in self.colnames:
                self.add_column(astropy.table.Column(name = cname+band, dtype=np.float, length = len(self)))
                self[cname+band][:] = np.nan

        for i in np.arange(0,len(self)):
            if 't'+band in self.lclist[i].keys():
                t1 = self.lclist[i]['t'+band]
                m1 = self.lclist[i]['m'+band]
                if len(t1) > 2:
                    test1 = ysovar_lombscargle.fasper(t1,m1,oversamp,maxfreq)
                    good = np.where(1/test1[0] < maxper)[0]
                    # be sensitive only to periods shorter than maxper
                    if len(good) > 0:
                        max1 = np.argmax(test1[1][good]) # find peak
                        sig1 = test1[1][good][max1]
                        period1 = 1./test1[0][good][max1]
                        self['period_'+band][i] = period1
                        self['peak_'+band][i] = sig1
                        self['FAP_'+band][i] = ysovar_lombscargle.getSignificance(
                            test1[0][good], test1[1][good], good.sum(), oversamp)[max1]
  

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




def fit_twocolor_odr(dataset, index, p_guess, outroot = None,  n_bootstrap = None, xyswitch = False):
    '''Fits a straight line to a single CMD, using a weighted orthogonal least squares algorithm (ODR).
    
    Parameters
    ----------
    dataset : np.ndarray
        data collection for one detected source
    index : integer
        the index of the dataset within the data structure
    p_guess : tuple
        initial fit parameters derived from fit_twocolor
    outroot : string or None
        dictionary where to save the plot, set to `None` for no plotting
    n_bootstrap : integer or None
        how many bootstrap trials, set to `None` for no bootstrapping
    xyswitch : boolean
        if the X and Y axis will be switched for the fit or not. This has nothing to do with bisector fitting! The fitting algorithm used here takes care of errors in x and y simultaneously; the xyswitch is only for taking care of pathological cases where a vertical fitted line would occur without coordinate switching.
    
    Returns
    -------
    result : tuple
        contains output = fit parameters, bootstrap_output = results from the bootstrap, bootstrap_raw = the actual bootstrapped data, alpha = the fitted slope angle, sd_alpha = the error on the fitted slope angle, x_spread = the spread of the data along the fitted line (0.5*(90th percentile - 10th percentile)))
    '''
    
    #define the fitting function (in this case a straight line)
    def fitfunc(p, x):
        return p[0]*x + p[1]
    
    # define what the x and y data is:
    if not(xyswitch):
        x_data = dataset['m36'] - dataset['m45']
        y_data = dataset['m36']
        x_error = np.sqrt( (dataset['m36_error'])**2 + (dataset['m45_error'])**2 )
        y_error = dataset['m36_error']
    else:
        y_data = dataset['m36'] - dataset['m45']
        x_data = dataset['m36']
        y_error = np.sqrt( (dataset['m36_error'])**2 + (dataset['m45_error'])**2 )
        x_error = dataset['m36_error']
    
    # load data into ODR
    data = scipy.odr.RealData(x=x_data, y=y_data, sx=x_error, sy=y_error)
    # tell ODR what the fitting function is:
    model = scipy.odr.Model(fitfunc)
    # now do the fit:
    fit = scipy.odr.ODR(data, model, p_guess, maxit=1000) 
    output = fit.run()
    
    p = output.beta # the fitted function parameters
    delta = output.delta # array of estimated errors in input variables
    eps   = output.eps # array of estimated errors in response variables
    #print output.stopreason[0]
    bootstrap_output = np.array([np.NaN, np.NaN, np.NaN, np.NaN])
    bootstrap_raw = (np.NaN, np.NaN, np.NaN)
    # calculate slope angle. This is vs. horizontal axis.
    hyp = np.sqrt(output.beta[0]**2 + 1**2)
    alpha = math.asin(output.beta[0]/hyp)
    # calculate error on slope angle by taking the mean difference of the angles derived from m+m_error and m-m_error.
    alpha_plus  = math.asin((output.beta[0]+output.sd_beta[0])/np.sqrt((output.beta[0]+output.sd_beta[0])**2 + 1**2))
    alpha_minus = math.asin((output.beta[0]-output.sd_beta[0])/np.sqrt((output.beta[0]-output.sd_beta[0])**2 + 1**2))
    sd_alpha = 0.5*( np.abs(alpha - alpha_plus) + np.abs(alpha - alpha_minus) ) 
    # define the spread along the fitted line. Use 90th and 10th quantile.
    # output.xplus and output.y are the x and y values of the projection of the original data onto the fit.
    # okay, first transform coordinate system so that x axis is along fit. To do this, first shift everything by -p[1] (this is -b), then rotate by -alpha. New x and y coordinates are then:
    #
    # |x'|   |cos(-alpha) -sin(-alpha)| | x |
    # |  | = |                        | |   |
    # |y'|   |sin(-alpha)  cos(-alpha)| |y-b|
    #
    x_new = math.cos(-alpha) * output.xplus - math.sin(-alpha)*(output.y - p[1])
    y_new = math.sin(-alpha) * output.xplus + math.cos(-alpha)*(output.y - p[1])
    # The y_new values are now essentially zero. (As they should.)
    # Now sort x_new and get 90th and 10th quantile:
    x_new.sort()
    x_spread = scipy.stats.mstats.mquantiles(x_new, prob=0.9)[0] - scipy.stats.mstats.mquantiles(x_new, prob=0.1)[0]
    #print x_spread
    
    
    if outroot is not None:
        # I got the following from a python script from http://www.physics.utoronto.ca/~phy326/python/odr_fit_to_data.py, I have to check this properly.
        # This does a residual plot, and some bootstrapping if desired.
        # error ellipses:
        xstar = x_error*np.sqrt( ((y_error*delta)**2) / ( (y_error*delta)**2 + (x_error*eps)**2 ) )
        ystar = y_error*np.sqrt( ((x_error*eps)**2) / ( (y_error*delta)**2 + (x_error*eps)**2 ) )
        adjusted_err = np.sqrt(xstar**2 + ystar**2)
        residual = np.sign(y_data - fitfunc(p,x_data))*np.sqrt(delta**2 + eps**2)
        fig = plt.figure()
        fit = fig.add_subplot(211)
        fit.set_xticklabels( () ) 
        plt.ylabel("[3.6]")
        plt.title("Orthogonal Distance Regression Fit to Data")
        # plot data as circles and model as line
        x_model = np.arange(min(x_data),max(x_data),(max(x_data)-min(x_data))/1000.)
        fit.plot(x_data,y_data,'ro', x_model, fitfunc(p,x_model))
        fit.errorbar(x_data, y_data, xerr=x_error, yerr=y_error, fmt='r+')
        fit.set_yscale('linear')
        
        a = np.array([output.xplus,x_data])   # output.xplus: x-values of datapoints projected onto fit
        b = np.array([output.y,y_data])  # output.y: y-values of datapoints projected onto fit
        fit.plot(np.array([a[0][0],a[1][0]]), np.array([b[0][0],b[1][0]]), 'k-', label = 'Residuals')
        print np.array([a[0][0],a[1][0]])
        print np.array([b[0][0],b[1][0]])
        for i in range(1,len(y_data)):
            fit.plot(np.array([a[0][i],a[1][i]]), np.array([b[0][i],b[1][i]]),'k-')
        
        fit.set_ylim([min(y_data)-0.05, max(y_data)+0.05])
        fit.set_ylim(fit.get_ylim()[::-1])
        fit.legend(loc='lower left')
        # separate plot to show residuals
        residuals = fig.add_subplot(212) # 3 rows, 1 column, subplot 2
        residuals.errorbar(x=a[0][:],y=residual,yerr=adjusted_err, fmt="r+", label = "Residuals")
        # make sure residual plot has same x axis as fit plot
        residuals.set_xlim(fit.get_xlim())
        residuals.set_ylim(residuals.get_ylim()[::-1])
        # Draw a horizontal line at zero on residuals plot
        plt.axhline(y=0, color='b')
        # Label axes
        plt.xlabel("[3.6] - [4.5]")
        plt.ylabel("Residuals")
        plt.savefig(outroot + str(index) + '_odrfit.eps')
    
    if n_bootstrap is not None:
        print 'bootstrapping...'
        # take a random half of the data and do the fit (choosing without replacement, standard bootstrap). Do this a lot of times and construct a cumulative distribution function for the slope and the intercept of the fitted line.
        # now what I actually want is the slope angle a, not m.
        m = np.array([])
        b = np.array([])
        for i in np.arange(0, n_bootstrap):
            indices = np.arange(0,len(x_data))
            np.random.shuffle(indices)
            ind = indices[0:len(x_data)/2] # dividing by integer on purpose.
            dat = scipy.odr.RealData(x=x_data[ind], y=y_data[ind], sx=x_error[ind], sy=y_error[ind])
            fit = scipy.odr.ODR(dat, model, p_guess, maxit=5000,job=10) 
            out = fit.run()
            m = np.append(m, out.beta[0])
            b = np.append(b, out.beta[1])
        
        a = np.arctan(m) # in radian
        # plot histograms for m and b:
        plt.clf()
        n_m, bins_m, patches_m = plt.hist(m, 100, normed=True )
        plt.savefig('m_hist.eps')
        plt.clf()
        n_b, bins_b, patches_b = plt.hist(b, 100, normed=True)
        plt.savefig('b_hist.eps')
        plt.clf()
        n_a, bins_a, patches_a = plt.hist(a, 100, normed=True)
        plt.savefig('a_hist.eps')
        plt.clf()
        # get median and symmetric 68% interval for m, b and alpha:
        m_median = np.median(m)
        m_down = np.sort(m)[ int(round(0.16*len(m))) ]
        m_up   = np.sort(m)[ int(round(0.84*len(m))) ]
        m_error = np.mean([abs(m_down-m_median), abs(m_up-m_median)])
        #print (m_median, m_up, m_down, m_error)
        b_median = np.median(b)
        b_down = np.sort(b)[ int(round(0.16*len(b))) ]
        b_up   = np.sort(b)[ int(round(0.84*len(b))) ]
        b_error = np.mean([abs(b_down-b_median), abs(b_up-b_median)])
        #print (b_median, b_up, b_down, b_error)
        a_median = np.median(a)
        a_down = np.sort(a)[ int(round(0.16*len(a))) ]
        a_up   = np.sort(a)[ int(round(0.84*len(a))) ]
        a_error = np.mean([abs(a_down-a_median), abs(a_up-a_median)])
        #print (b_median, b_up, b_down, b_error)
        
        bootstrap_output = np.array([m_median, m_error, b_median, b_error, a_median, a_error])
        bootstrap_raw = (m, b, a)
    
    if xyswitch:
        # re-transform slope and intercept to original xy system
        # x = m * y + b
        # y = 1/m * x - b/m
        output.beta[1] = -output.beta[1]/output.beta[0] # b
        output.beta[0] = 1./output.beta[0] # m
        alpha = np.pi/2 - alpha
    
    result = (output, bootstrap_output, bootstrap_raw, alpha, sd_alpha, x_spread)
    return result







