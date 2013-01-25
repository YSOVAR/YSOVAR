# -*- coding: utf-8 -*-
import os.path

import numpy as np
import scipy
import scipy.io
import scipy.odr
from collections import defaultdict
import math
from copy import deepcopy
import string

import scipy.stats
import matplotlib.pyplot as plt
import pyfits
import pylab

import asciitable

import ysovar_lombscargle
from great_circle_dist import dist_radec, dist_radec_fast

def readdata(filename1, filename2):  
    # outdated routine, not used anymore
    print 'reading ysovar data...'
    ysovar1raw = scipy.io.readsav(filename1) # this is the 3.6 mu data
    ysovar1 = ysovar1raw['ch1lightcurves']
    ysovar2raw = scipy.io.readsav(filename2) # this is the 4.5 mu data
    ysovar2 = ysovar2raw['ch2lightcurves']
    return (ysovar1, ysovar2)

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


def test_crossmatch_irac(yso1, yso2):
    # outdated routine, not used anymore
    crossid_test = np.ones(len(yso1), int)*-99999
    id_dist = 1./3600. # = 1 arcsec
    for i in np.arange(0,len(yso1)):
        distance = np.sqrt((yso1['ra'][i] - yso2['ra'])**2 + (yso1['dec'][i] - yso2['dec'])**2)
        min_ind = np.where(distance == min(distance))[0]
        if min(distance) <= id_dist:
            crossid_test[i] = min_ind
    return crossid_test


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
    # outdated function, no longer used
    # make cross-ids (using the coordinates) between Spiter lcs and Guenther2012.
    # use 1 arcsec as radius.
    cross_ids = np.ones(len(data1),int) * -99999
    
    for i in np.arange(0,len(data1)):
        distance = dist_radec(data1[ra1][i], data1[dec1][i], data2[ra2], data2[dec2], unit ='deg') 
        if min(distance) <= radius:
            cross_ids[i] = np.argmin(distance)
        
    print len(np.where(cross_ids != -99999)[0])
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



def add_ysovar_mags(data, ysovar, channel, match_dist = 0.5 /3600.):
    '''Add YSOVAR lightcurves form Louisa's IDL structures to our python
    
    
    Parameters
    ----------
    data : empty list or list of dictionaries
        structure to hold all the information
    ysovar : recarray
        obtained from reading in ysovar .idlsav files with readsav
    channel : string
        usually "1" or "2", as part of the label for keywords in dictionaries
    match_dist : float
        maximum distance to match two positions as one sorce 
    '''
    channel = str(channel).strip()  # just in case someone puts in a number
    for k, yso in enumerate(ysovar):
        radec = radec_from_dict(data, RA = 'ra', DEC = 'dec')
        distance = dist_radec(radec['RA'], radec['DEC'], yso['RA'], yso['DEC'], unit ='deg') 
        if len(data) > 0 and min(distance) <= match_dist:
            dict_temp = data[np.argmin(distance)]
        else:
            dict_temp = defaultdict(list)
            data = np.append(data, dict_temp)
        dict_temp['id_ysovar'+channel].append(k)
        dict_temp['ra'].append(yso['RA'])
        dict_temp['dec'].append(yso['DEC'])
        dict_temp['ISOY_NAME'].append(yso['ISOY_NAME'])
        good1 = np.where(yso['HMJD'+channel] >= 0.)[0]
        dict_temp['t'+channel].extend((yso['HMJD'+channel][good1]).tolist())
        dict_temp['m'+channel].extend((yso['MAG'+channel][good1]).tolist())
        dict_temp['m'+channel+'_error'].extend((yso['EMAG'+channel][good1]).tolist())
    return data

def dict_cleanup(data, t_simul = 0.002, min_number_of_times = 0, floor_error = [0.01, 0.008]):
    '''Clean up dictionaries after add_ysovar_mags
    
    Each object in the `data` list can be constructed from multiple sources per
    band in multiple bands. This function averages the coordinates over all
    contributing sources, converts and sorts lists of times and magnitudes
    and makes multiband lightcurves.
    
    Parameters
    ----------
    data : list of dictionaries
        as obtained from :func:`add_ysovar_mags`
    t_simul : float
        max distance in days to accept datapoints in band 1 and 2 as simultaneous
        In L1688 and IRAS 20050+2720 the distance between band 1 an d 2 coverage
        is within a few minutes, so a small number is sufficent to catch 
        everything and to avoid false matches.
    min_number_of_times : integer
        Remove all sources with less than min_number_of_times datapoints from the list
    floor_error : list of 2 floats
        Floor error for IRAC1 and IRAC2. Will be added in quadrature to all error
        values.
        
    Returns
    -------
    data : list of dictionaries
        individual dictionaries are cleaned up as decribed above
    '''
    n_datapoints = np.zeros(len(data), dtype = np.int)
    for i, d in enumerate(data):
        if 't1' in d.keys():
            if 't2' in d.keys():
                n_datapoints[i] = max(len(d['t1']), len(d['t2']))
            else:     # only t1 present 
                n_datapoints[i] = len(d['t1'])
        else:         # if 't1' is not present 't2' must be
            n_datapoints[i] = len(d['t2'])
    # go in reverse direction, otherwise each removal would change index numbers of
    # later entries
    data = data[n_datapoints >= min_number_of_times]
    if np.max(np.abs(floor_error)) > 0:
        print 'Adding floor error in quadrature to all error values'
        print 'Floor value in IRAC1: ', floor_error[0], '   IRAC2: ', floor_error[1]
    for j, d in enumerate(data):
        d['id'] = j
        # set RA, DEC to the mean values of all sources, which make up an entry
        d['ra'] = np.mean(d['ra'])
        d['dec'] = np.mean(d['dec'])
        d['ISOY_NAME'] = set(d['ISOY_NAME'])
        for n, i in enumerate(['1', '2']):
            if 't'+i in d.keys():
                # sort lightcurves by time.
                # required if multiple original sources contribute to this entry
                ind = np.array(d['t'+i]).argsort()
                d['t'+i] = np.array(d['t'+i])[ind]
                d['m'+i] = np.array(d['m'+i])[ind]
                d['m'+i+'_error'] = np.array(d['m'+i+'_error'])[ind]
                d['m'+i+'_error'] = np.sqrt(d['m'+i+'_error']**2 + floor_error[n]**2)
        if 't1' in d.keys() and 't2' in d.keys():
            for i, t in enumerate(d['t1']):
                diff_t = np.abs(d['t2'] - t)
                min2 = np.argmin(diff_t)  #index of matching time in band 2
                if min(diff_t) < t_simul:
                    d['t'].append(np.mean([t, d['t2'][min2]]))
                    d['m36'].append(d['m1'][i])
                    d['m36_error'].append(d['m1_error'][i])
                    d['m45'].append(d['m2'][min2])
                    d['m45_error'].append(d['m2_error'][min2])
            # if at least one point is in common, t was added
            # in this case change lists into numpy arrays
            if 't' in d.keys():
                for i in ['t', 'm36', 'm36_error', 'm45', 'm45_error']:
                    d[i] = np.array(d[i])
    return data

def make_dict(ysovar1, ysovar2, match_dist = 0.5 /3600., min_number_of_times = 0):
    '''Build YSOVAR lightcurves form Louisa's IDL structures
    
    Parameters
    ----------
    ysovar1 : recarray
        IRAC 1 data obtained from reading in ysovar .idlsav files with readsav
    ysovar2 : recarray
        IRAC 2 data obtained from reading in ysovar .idlsav files with readsav
    match_dist : float
        maximum distance to match two positions as one sorce
    min_number_of_times : integer
        Remove all sources with less than min_number_of_times datapoints from the list
        
    Returns
    -------
    data : empty list or list of dictionaries
        structure to hold all the information
    '''
    data = np.array([])
    data = add_ysovar_mags(data, ysovar1, '1', match_dist = match_dist)
    data = add_ysovar_mags(data, ysovar2, '2', match_dist = match_dist)
    # some dictionaries contain multiple original entries
    # so some clean up is required
    data = dict_cleanup(data, min_number_of_times = min_number_of_times)
    return data

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

def dict_from_csv(csvfile, match_dist = 1.0/3600., min_number_of_times = 5, verbose = True):
    '''Build YSOVAR lightcurves form Louisa's IDL structures
    
    Parameters
    ----------
    cvsfile : sting or file object
        input csv file 
    match_dist : float
        maximum distance to match two positions as one sorce
    min_number_of_times : integer
        Remove all sources with less than min_number_of_times datapoints from the list
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
    data = []
    radec = {'RA':[], 'DEC': []}
    for i, n in enumerate(set(tab['sname'])):
        if verbose and (np.mod(i,100)==0):
            print 'making dict for source ' + str(i) + ' of ' + str(len(set(tab['sname'])))
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
            dict_temp['id_ysovar'].append(len(data))
        dict_temp['ra'].extend([ra] * ind.sum())
        dict_temp['dec'].extend([dec] * ind.sum())
        dict_temp['ISOY_NAME'].append(n)
        dict_temp['YSOVAR2_id'].append(tab['ysovarid'][ind][0])
        for channel in ['1', '2']:
            good = ind & (tab['hmjd'] >= 0.) & (tab['fname'] == 'IRAC'+channel)
            if np.sum(good) > 0:
                dict_temp['t'+channel].extend((tab['hmjd'][good]).tolist())
                dict_temp['m'+channel].extend((tab['mag1'][good]).tolist())
                dict_temp['m'+channel+'_error'].extend((tab['emag1'][good]).tolist())

    if verbose: print 'Cleaning up dictionaries'
    data = dict_cleanup(data, min_number_of_times = min_number_of_times)
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
        if (not ('t1' in d.keys()) or (len(d['t1']) < min_number_of_times)) and  (not ('t2' in d.keys()) or (len(d['t2']) < min_number_of_times)): print i, list(d['ISOY_NAME'])
    print '----------------------------------------------------------------------------'
    # IRAC1==IRAC2 ?
    for i, d in enumerate(data):
        if 't1' in d.keys() and 't2' in d.keys():
            ind1 = np.argsort(d['t1'])
            ind2 = np.argsort(d['t2'])
            if (len(d['t1']) == len(d['t2'])) and np.all(d['m1'][ind1] == d['m2'][ind2]):
                print 'IRAC1 and IRAC2 are identical in source', i
                print 'Probably error in SQL query made to retrieve this data'
    ### Find entries with two mags in same lightcurve
    print 'The following lightcurves have two or more entries with almost identical time stamps'
    for band in ['1', '2']:
        for i, d in enumerate(data):
            if 't'+band in d.keys():
                dt = np.diff(d['t'+band])
                ind = np.where(dt < 0.00001)[0]
                if len(ind) > 0:
                    print 'IRAC'+band+ ', source ', i, 'at times: ', d['t'+band][ind]
    print '----------------------------------------------------------------------------'
    print 'The following entries are combined from multiple sources:'
    for i, d in enumerate(data):
        if len(d['ISOY_NAME']) > 1: print i, d['ISOY_NAME']
    print '----------------------------------------------------------------------------'
    print 'The following sources are less than ', match_dist, ' deg apart'
    radec = radec_from_dict(data)
    for i in range(len(data)-1):
        dist = dist_radec_fast(radec['RA'][i], radec['DEC'][i], radec['RA'][i+1:], radec['DEC'][i+1:], scale = match_dist, unit = 'deg')
        if np.min(dist) < match_dist:
            print 'distance', i, i+np.argmin(dist), ' is only ', np.min(dist)
    print '----------------------------------------------------------------------------'

def make_onecolor_stats(datalist, datalist_error):
    '''Calculates some basic statistical values for a single one-band light curve
    
    Parameters
    ----------
    datalist : np.array
        single light curve in magnitudes
    datalist_error : np.array
        error on data points in magnitudes        
        
    Returns
    -------
    stats : np.array
        array of statistical values (median, MAD, mean, stddev, chi**2 with respect to mean, maximum value, minimum value, "delta" (0.5 * (90th percentile - 1-th percentile)) )
        
    '''
    median = np.median(datalist)
    mad = np.median(abs(datalist - median))
    mean = np.mean(datalist)
    stddev = np.std(datalist)
    chisq_to_mean = np.sum( (datalist - mean)**2/(datalist_error**2) )/(len(datalist)-1)
    delta = (scipy.stats.mstats.mquantiles(datalist, prob=0.9) - scipy.stats.mstats.mquantiles(datalist, prob=0.1))/2.
    stats = np.array([median, mad, mean, stddev, chisq_to_mean, np.max(datalist), np.min(datalist), delta])
    return stats

def make_twocolor_stats(datalist1, datalist1_error,datalist2, datalist2_error, verbose = True):
    '''Calculates the Stetson index for a two-band light curve. Uses only near-simultaneous data points as defined in dict_cleanup.
    
    Parameters
    ----------
    datalist1 : np.array
        single light curve of band 1 in magnitudes
    datalist1_error : np.array
        error on data points of band 1 in magnitudes    
    datalist2 : np.array
        single light curve of band 2 in magnitudes
    datalist2_error : np.array
        error on data points of band 2 in magnitudes          
        
    Returns
    -------
    stetson : float
        Stetson value for the provided two-band light curve
        
    '''
    
    # number of datapoints:
    N = float(len(datalist1))
    if N > 1:
        # weighted mean magnitudes in each passband:
        wmean_36 = np.sum(datalist1/(datalist1_error**2))/np.sum(1/(datalist1_error**2))
        wmean_45 = np.sum(datalist2/(datalist2_error**2))/np.sum(1/(datalist2_error**2))
        # normalized residual from the weighted mean for each datapoint:
        res_36 = (datalist1 - wmean_36) / datalist1_error
        res_45 = (datalist2 - wmean_45) / datalist2_error
        
        stetson = np.sqrt(1./(N*(N-1))) * np.sum( res_36 * res_45 )
        if verbose:
            print stetson
            print str( ("%.3f" % stetson) )
            return stetson


def make_stats(data, info, verbose = True):
    '''Adds appropriate statistical values for all objects to info array.
    
    Parameters
    ----------
    data : np.ndarray
        structure with all the raw object information
    info : np.rec.array
        structure with the refined object properties         
        
    Returns
    -------
    info : np.rec.array
        structure with the updated object properties 
        
    '''
    for i in np.arange(0,len(data)):
        if 'm1' in data[i].keys():
            (info.median_36[i], info.mad_36[i], info.mean_36[i], info.stddev_36[i], info.chisq_36[i], info.max_36[i], info.min_36[i], info.delta_36[i])    = make_onecolor_stats(data[i]['m1'], data[i]['m1_error'])
            info.n_data_36[i] = len(data[i]['m1'])
        else:
            info.n_data_36[i] = 0
        if 'm2' in data[i].keys():
            (info.median_45[i], info.mad_45[i], info.mean_45[i], info.stddev_45[i], info.chisq_45[i], info.max_45[i], info.min_45[i], info.delta_45[i])    = make_onecolor_stats(data[i]['m2'], data[i]['m2_error'])
            info.n_data_45[i] = len(data[i]['m2'])
        else:
            info.n_data_45[i] = 0

        if 'm36' in data[i].keys():
            info.n_data_3645[i] = len(data[i]['m36'])
            if info.n_data_45[i] > 1:
                print i
                # make two-color stats 
                info.stetson[i] = make_twocolor_stats(data[i]['m36'], data[i]['m36_error'], data[i]['m45'], data[i]['m45_error'], verbose  = verbose)
                
                # fit straight line to color-magnitude diagram
                (info.cmd_m_plain[i], info.cmd_b_plain[i], info.cmd_m_redvec[i], info.cmd_b_redvec[i]) = fit_twocolor(data[i])[0:4]
        else:
            info.n_data_3645[i] = 0
        
    return info


def calc_reddening():
    # this is basically from Rieke & Lebofsky 1984.
    # I take the extinctions from the L and M band (3.5, 5.0).
    A36 = 0.058
    A45 = 0.023
    R36 = - A36/(A45 - A36)
    return np.array([R36, A36])




def fit_twocolor(data):
    # measures the slope of the data points in the color-magnitude diagram.
    # this is just fitted with ordinary least squares, using the analytic formula.
    # this is then used as a first guess for an orthogonal least squares fit with simultaneous treatment of errors in x and y (see fit_twocolor_odr)
    x = data['m36'] - data['m45']
    y = data['m36']
    x_error = np.sqrt( data['m36_error']**2 + data['m45_error']**2 )
    y_error = data['m36_error']
    N = float(len(data['m36']))
    # calculate the different sums:
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x**2)
    sum_xy = np.sum(x*y)
    # now get b and m from analytic formula:
    m = (-sum_x*sum_y + N*sum_xy) / (N*sum_xx - sum_x*sum_x)
    b = (-sum_x*sum_xy + sum_xx*sum_y) / (N*sum_xx - sum_x*sum_x)
    # now calculate chisquared for this line:
    reduced_chisq = sum( (y - (m*x+b))**2/ y_error**2)/N
    
    # now fit theoretical reddening vector to data, for plotting purposes (i.e. just shifting it in y:)
    m2 = calc_reddening()[0] # the sign is okay, because the y axis is inverted in the plots
    b2 = 1/N * ( sum_y - m2 * sum_x )
    reduced_chisq2 = sum( (y - (m2*x+b2))**2/y_error**2 )/N
    
    return np.array([m,b,m2,b2,reduced_chisq,reduced_chisq2])






def initialize_info_array(data, guenther_data, guenther_class):
    # creates numpy record array for all refined data of all objects.
    # fills in the id fields from the data dictionary.
    Ndata = 71 # number of data entries to be collected for each source
    oneset = []
    for i in np.arange(0,Ndata):
        oneset.append(-99999)
    oneset = tuple(oneset)
    print len(oneset)
    allsets = []
    for i in np.arange(0,len(data)):
        allsets.append(oneset)
    print len(allsets)
    infos = np.rec.array(allsets, dtype=[
        ('id', '|f4'), # ID number
        ('YSOVAR2_ID', '|f4'), # ID in YSOVAR2 database
        ('id_guenther', '|f4'), # ID as in Guenther+ 2012
        ('index_guenther', '|f4'), # index where to find this in the Guenther data array ( = id_guenther - 1)
        ('ysoclass', '|f4'), # class from Guenther+ 2012 (0='XYSO', 1='I+I*', 2='II+II*', 3='III', 4='star')
        ('ra_spitzer', '|f8'), 
        ('dec_spitzer', '|f8'), 
        ('ra_guenther', '|f8'), 
        ('dec_guenther', '|f8'), 
        ('lambdas',  object), 
        ('fluxes',  object), 
        ('mags',  object), 
        ('mags_error',  object), 
        ('good_period', '|f4'), # is |= -99999 if significant period exists
        ('good_peak', '|f4'),   # is |= -99999 if significant period exists
        ('hectospec_night1', '|O40'), # filename for hectospec spectrum
        ('hectospec_night2', '|O40'), # filename for hectospec spectrum
        ('period_36', '|f4'), 
        ('peak_36', '|f4'), 
        ('period_45', '|f4'), 
        ('peak_45', '|f4'), 
        ('min_36', '|f4'), 
        ('max_36', '|f4'), 
        ('min_45', '|f4'), 
        ('max_45', '|f4'), 
        ('median_36', '|f4'), 
        ('mad_36', '|f4'), 
        ('mean_36', '|f4'), 
        ('stddev_36', '|f4'), 
        ('chisq_36', '|f4'),
        ('delta_36', '|f4'), # the 90% quantile range of the 3.6 mu data
        ('median_45', '|f4'), 
        ('mad_45', '|f4'), 
        ('mean_45', '|f4'),
        ('n_data_36', '|f4'), # number of data points
        ('n_data_45', '|f4'),
        ('n_data_3645', '|f4'),
        ('stddev_45', '|f4'), 
        ('chisq_45', '|f4'), 
        ('delta_45', '|f4'), # the 90% quantile range of the 4.5 mu data
        ('stetson', '|f4'),
        ('chi2poly1_36', '|f4'),  # chi2 after fitting a polynom of degree 1 to the 3_6 lightcurve
        ('chi2poly2_36', '|f4'),
        ('chi2poly3_36', '|f4'),
        ('chi2poly4_36', '|f4'),
        ('chi2poly5_36', '|f4'),
        ('chi2poly1_45', '|f4'),
        ('chi2poly2_45', '|f4'),
        ('chi2poly3_45', '|f4'),
        ('chi2poly4_45', '|f4'),
        ('chi2poly5_45', '|f4'),
        ('AV', '|f4'), # AV extinction for sources which show extinction-like behaviour in the CMD
        ('cmd_m', '|f4'), # slope of the color-magnitude diagram, from ODR
        ('cmd_b', '|f4'), # intercept of the color-magnitude diagram, from ODR
        ('cmd_m_error', '|f4'), # error of the slope of the color-magnitude diagram, from ODR
        ('cmd_b_error', '|f4'), # error of the intercept of the color-magnitude diagram, from ODR
        ('cmd_m2', '|f4'), # slope of the color-magnitude diagram, from ODR, with switched coordinates
        ('cmd_b2', '|f4'), 
        ('cmd_m2_error', '|f4'),
        ('cmd_b2_error', '|f4'),    
        ('cmd_alpha1', '|f4'), # slope angle of the color-magnitude diagram, from ODR
        ('cmd_alpha1_error', '|f4'), # slope angle error of the color-magnitude diagram, from ODR
        ('cmd_alpha2', '|f4'), # slope angle derived from switched ccordinates (and re-transformed to real coordinares)
        ('cmd_alpha2_error', '|f4'),
        ('cmd_alpha', '|f4'), # good slope angle
        ('cmd_alpha_error', '|f4'),
        ('cmd_x_spread', '|f4'), # spread of data along fitted line, 90th-10th quantile
        ('cmd_m_plain', '|f4'), # CMD slope from plain LSq fitting
        ('cmd_b_plain', '|f4'), # CMD intercept from plain LSq fitting
        ('cmd_m_redvec', '|f4'), # theoretical reddening vector slope (constant for all objects)
        ('cmd_b_redvec', '|f4'),   # best-fitting intercept if using reddening vector, from plain LSq fitting
        ('cmd_dominated', '|O10') ] ) # string which identifies accretion, extinction, or other slope in cmd
    
    for i in np.arange(0,len(data)):
        print 'collecting info: ' + str(i)
        infos.id[i] = data[i]['id']
        infos.YSOVAR2_ID[i] = data[i]['YSOVAR2_id']
        infos.id_guenther[i] = data[i]['id_guenther']
        infos.index_guenther[i] = data[i]['index_guenther']
        infos.ra_spitzer[i] = data[i]['ra']
        infos.dec_spitzer[i] = data[i]['dec']
        infos.ysoclass[i] = guenther_class[infos.index_guenther[i]]
        infos.ra_guenther[i] = guenther_data.RAdeg[infos.index_guenther[i]]
        infos.dec_guenther[i] = guenther_data.DEdeg[infos.index_guenther[i]]
        infos.lambdas[i] = get_sed(guenther_data[infos.index_guenther[i]])[0]
        infos.mags[i] = get_sed(guenther_data[infos.index_guenther[i]])[1]
        infos.mags_error[i] = get_sed(guenther_data[infos.index_guenther[i]])[2]
        infos.fluxes[i] = get_sed(guenther_data[infos.index_guenther[i]])[3]
    
    return infos


def calc_ls(data, infos, maxper, oversamp = 4, maxfreq = 1.):
    '''calculate Lomb-Scagle periodograms for all sources
    
    Parameters
    ----------
    data : dict of dicts
        which contains the input lightcurves
    infos : np.ndarray
        output is placed the fields `period_36`, `peak_36`, `period_45`
        and `peak_45`
    maxper : float
        periods above this value will be ignored
    oversamp : integer
        oversampling factor
    maxfreq : float
        maximum frequency for the LS periodogram
    '''
    for i in np.arange(0,len(data)):
        #print i
        if 't1' in data[i].keys():
            t1 = data[i]['t1']
            m1 = data[i]['m1']
            if len(t1) > 2:
                test1 = ysovar_lombscargle.fasper(t1,m1,oversamp,maxfreq)
                good = np.where(1/test1[0] < maxper)[0] # be sensitive only to periods shorter than maxper
                if len(good) > 0:
                    max1 = np.argmax(test1[1][good]) # find peak
                    sig1 = test1[1][good][max1]
                    period1 = 1./test1[0][good][max1]
                else:
                    period1 = np.nan
                    sig1 = np.nan
                infos.period_36[i] = period1
                infos.peak_36[i] = sig1
            #if i == 96:
            #    print period1
        
        if 't2' in data[i].keys():
            t2 = data[i]['t2']
            m2 = data[i]['m2']
            if len(t2) > 2:
                test2 = ysovar_lombscargle.fasper(t2,m2,oversamp,maxfreq)
                good = np.where(1/test2[0] < maxper)[0] # be sensitive only to periods shorter than maxper
                if len(good) > 0:
                    max2 = np.argmax(test2[1][good]) # find peak
                    sig2 = test2[1][good][max2]
                    period2 = 1/test2[0][good][max2]
                else:
                    period2 = np.nan
                    sig2 = np.nan
                
                infos.period_45[i] = period2
                infos.peak_45[i] = sig2
    
    return infos
    



def is_there_a_good_period(data,infos, power, minper, maxper):
    '''check if a strong periodogram peak is found; if yes, this is saved to the info structure.
    
    Parameters
    ----------
    data : np.ndarray
        which contains the input lightcurves
    infos : np.rec.array
        info structure with refined object information
    power : float
        required power threshold for "good" period
    minper : float
        lowest period which is considered
    maxper : float
        maximum period which is considered
    
    Returns
    -------
    infos : np.rec.array
        structure with the updated object properties ("good" period and power of peak)
   '''
    
    for i in np.arange(0, len(data)):
        per = -99999.
        peak = -99999.
        peak1 = infos.peak_36[i]
        peak2 = infos.peak_45[i]
        per1 = infos.period_36[i]
        per2 = infos.period_45[i]
        A = ( (peak1 > power) & (per1 > minper) & (per1 < maxper) ) # take periods between 2 and 20 d with power > 10
        B = ( (peak2 > power) & (per2 > minper) & (per2 < maxper) )
        if (A & B): # take higher peak period.
            if peak1 > peak2:
                per = per1
                peak = peak1
            if peak <= peak2:
                per = per2
                peak = peak2
        else:
            if A:
                per = per1
                peak = peak1
            if B:
                per = per2
                peak = peak2
        infos.good_period[i] = per
        infos.good_peak[i] = peak
    return infos


def phase_fold_data(data,infos):
    # take data sets for which a strong periodogram peak is found and fold these data by period.
    for i in np.arange(0,len(data)):
        period = infos.good_period[i]
        if period > 0: # if there's a good period
            if 't1' in data[i].keys():
                data[i]['p1'] = np.mod(data[i]['t1'],period)/period
            if 't2' in data[i].keys():
                data[i]['p2'] = np.mod(data[i]['t2'],period)/period
            if 't' in data[i].keys():
                data[i]['p'] = np.mod(data[i]['t'],period)/period
    return(data)



def spectra_coordinates(filename):
    # only for IRAS 20050: read ascii file with info about hectospec data.
    a = asciitable.read(filename)
    ra_string = a['RA']
    dec_string = a['DEC']
    ra = np.array([])
    dec = np.array([])
    for i in np.arange(0,len(ra_string)):
        ra_split = string.split(ra_string[i],':')
        print ra_split
        ra_new = float(ra_split[0])*15. + float(ra_split[1])*15./60. + float(ra_split[2])*15./3600.
        print ra_new
        ra = np.append(ra,ra_new)
        dec_split = string.split(dec_string[i],':')
        #print dec_split
        dec_new = float(dec_split[0]) + float(dec_split[1])/60. + float(dec_split[2])/3600.
        #print dec_new
        dec = np.append(dec,dec_new)
    filenames = a['FILENAME']
    
    return (ra, dec, filenames)



def spectra_check(infos, ra, dec, files, night_id, radius):
    # only for IRAS 20050: check if there's a hectospec spectrum available for the sources, and if yes, add the filename of the spectrum to the info array.
    for i in np.arange(0,len(infos)):
        distance = np.sqrt((infos.ra_guenther[i] - ra)**2 + (infos.dec_guenther[i] - dec)**2)
        min_ind = np.where(distance == min(distance))[0]
        if min(distance) <= radius:
            if night_id == 1:
                infos.hectospec_night1[i] = str(files[min_ind][0])
            if night_id == 2:
                infos.hectospec_night2[i] = str(files[min_ind][0])
        
    return infos



def make_latexfile(data, infos, outroot, name, ind, pdflatex = True):
    # write selected data and figures for sources into latex file
    filename = outroot + name + '.tex'
    f = open(filename, 'wb')
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
    plotwidth = '0.45'
    
    for i in ind:
        f.write('\\newpage \n')
        
        f.write('\\begin{minipage}[l]{6.5in} \n')
        
        # write lc (always exists) and cmd (if cmd exists):
        #f.write('\\begin{figure*}[h!]\n')
        filename = outroot + str(i) + '_lc'
        line = '\includegraphics[width=' + plotwidth  + '\\textwidth]{' + filename + '}' + '\n'
        f.write(line)
        
        
        try:
            if len(data[i]['t']) > 5:
                filename = outroot + str(i) + '_color'
                line = '\includegraphics[width=' + plotwidth  + '\\textwidth]{' + filename + '}' + '\n'
                f.write(line)
        except KeyError:
                pass
        
        f.write('\n')
        #f.write('\\end{figure*}\n')
        
        
        # write periodogram. Always exists.
        # And write SED. Always exists.
        #f.write('\\begin{figure}[h!]\n')
        filename = outroot + str(i) + '_ls'
        line = '\includegraphics[width=' + plotwidth  + '\\textwidth]{' + filename + '}' + '\n'
        f.write(line)
        filename = outroot + str(i) + '_sed'
        line = '\includegraphics[width=' + plotwidth  + '\\textwidth]{' + filename + '}' + '\n'
        f.write(line)
        f.write('\n')
        #f.write('\\end{figure}\n')
        
        # write lc_phased and cmd_phased, if both exist:
        try:
            if ( (len(data[i]['t']) > 1) & (infos[i]['good_period'] > 0) ):
                #f.write('\\begin{figure*}[h!]\n')
                filename = outroot + str(i) + '_lc_phased'
                line = '\includegraphics[width=' + plotwidth  + '\\textwidth]{' + filename + '}' + '\n'
                f.write(line)
                filename = outroot + str(i) + '_color_phased'
                line = '\includegraphics[width=' + plotwidth  + '\\textwidth]{' + filename + '}' + '\n'
                f.write(line)
                #f.write('\\end{figure*}\n')
                f.write('\n')
        except KeyError:
            pass
        
        # write stamp - always exists
        #f.write('\\begin{figure*}[h!]\n')
        filename = outroot + str(i) + '_stamp'
        line = '\includegraphics[width=' + plotwidth  + '\\textwidth]{' + filename + '}' + '\n'
        f.write(line)
        #write polyfit to lc if is exists
        filename = outroot + str(i) + '_lcpoly'
        if (('t1' in data[i].keys()) and (len(data[i]['t1']) > 15)) or (('t2' in data[i].keys()) and (len(data[i]['t2']) > 15)):
            line = '\includegraphics[width=' + plotwidth  + '\\textwidth]{' + filename + '}' + '\n'
            f.write(line)
            #f.write('\\end{figure*}\n')
            f.write('\n')

        
        f.write('ID: ' + str(i) + '\\\ \n')
        if 'YSOVAR2_id' in data[i].keys():
            f.write('ID in YSOVAR 2 database: ' + str(data[i]['YSOVAR2_id']) + '\\\ \n')
        if 'ID' in infos[i].dtype.names:
            f.write('ID Guenther+ 2012: ' + str(infos[i]['id_guenther']) + '\\\ \n')
        if 'simbad_MAIN_ID' in infos[i].dtype.names:
            f.write('ID Simbad: ' + infos[i]['simbad_MAIN_ID'] + '\\\ \n')
        if 'simbad_SP_TYPE' in infos[i].dtype.names:
            f.write('Simbad Sp type: ' + infos[i]['simbad_SP_TYPE'] + '\\\ \n')
        if 'wil08_ID' in infos[i].dtype.names:
            if data[i]['wil08_ID'] > 0:
                f.write('Wilking+ 08 ID: '+ str(infos[i]['wil08_ID']) +'\\\ \n')
        if 'bar12_Excess' in infos[i].dtype.names:
            f.write('Barsony+ 12: excess: ' + infos[i]['bar12_Excess'] + '\\\ \n')
        if 'bar12_Teff' in infos[i].dtype.names:
            f.write('Barsony+ 12: Teff: ' + str(infos[i]['bar12_Teff']) + ' (' + infos[i]['bar12_Model']+ ')\\\ \n')
            
        if infos.ysoclass[i] == 0:
            line = 'class (from Guenther+ 2012): ' + 'XYSO' + '\\\ \n'
        else:
            line = "class (from Rob's pipeline): " + str(infos.ysoclass[i]) + '\\\ \n'
        if 'IRclass' in infos[i].dtype.names:
            f.write('Rob class ' + infos[i]['IRclass'] + '\\\ \n')
        if 'wil08_SED' in infos[i].dtype.names:
            f.write('Wilking+ 08 SED: ' + infos[i]['wil08_SED'] + '\\\ \n')
        
        f.write(line)
        f.write('\n')
        
        try:
            line1 = 'median (3.6):   ' + str( ("%.2f" % infos.median_36[i]) ) + '\\\ \n'
            line2 = 'mad (3.6):   ' + str( ("%.2f" % infos.mad_36[i]) ) + '\\\ \n'
            line3 = 'stddev (3.6):   ' + str( ("%.2f" % infos.stddev_36[i]) ) + '\\\ \n'
            f.write(line1)
            f.write(line2)
            f.write(line3)
            f.write('\n')
        except KeyError:
            pass
        
        try:
            line1 = 'median (4.5):   ' + str( ("%.2f" % infos.median_45[i]) ) + '\\\ \n'
            line2 = 'mad (4.5):   ' + str( ("%.2f" % infos.mad_45[i]) ) + '\\\ \n'
            line3 = 'stddev (4.5):   ' + str( ("%.2f" % infos.stddev_45[i]) ) + '\\\ \n'
            f.write(line1)
            f.write(line2)
            f.write(line3)
            f.write('\n')
        except KeyError:
            pass
        
        try:
            print i
            line1 = 'Stetson index:   ' + str( ("%.2f" % infos.stetson[i]) ) + '\\\ \n'
            f.write(line1)
            f.write('\n')
            
        except KeyError:
            pass
        
        f.write('\\end{minipage} \n')
    
    f.write('\\end{document}')
    f.close()



def fit_twocolor_odr(dataset, index, p_guess, outroot, n_bootstrap, ifplot, ifbootstrap, xyswitch):
    '''Fits a straight line to a single CMD, using a weighted orthogonal least squares algorithm (ODR).
    
    Parameters
    ----------
    dataset : np.ndarray
        data collection for one detected source
    index : integer
        the index of the dataset within the data structure
    p_guess : tuple
        initial fit parameters derived from fit_twocolor
    outroot : string
        dictionary where to save the plot
    n_bootstrap : integer
        how many bootstrap trials
    ifplot : boolean
        if you want a residual plot or not
    ifbootstrap : boolean
        if you want to bootstrap or not
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
    
    
    if ifplot:
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
    
    if ifbootstrap:
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


def add_twocolor_fits_to_infos(data, infos, outroot, n_bootstrap, ifplot, ifbootstrap, xyswitch):
    '''Performs straight line fit to CMD for all sources. Adds fitted parameters to info structure.
    
    Parameters
    ----------
    data : np.ndarray
        data collection for all sources
    infos : np.rec.array
        info structure for all sources
    outroot : string
        directory for plots
    n_bootstrap : integer
        how many bootstrap trials
    ifplot : boolean
        if you want a residual plot or not
    ifbootstrap : boolean
        if you want to bootstrap or not
    xyswitch : boolean
        if the X and Y axis will be switched for the fit or not. This has nothing to do with bisector fitting! The fitting algorithm used here takes care of errors in x and y simultaneously; the xyswitch is only for taking care of pathological cases where a vertical fitted line would occur without coordinate switching.
    
    Returns
    -------
    infos : np.rec.array
        updated info structure
    '''
    for i in np.arange(0, len(data)):
        if 't' in data[i].keys() : 
            # use the result of the plain least squares fitting as a first parameter guess.
            p_guess = (infos[i].cmd_m_plain, infos[i].cmd_b_plain)
            
            (fit_output, bootstrap_output, bootstrap_raw, alpha, alpha_error, x_spread) = fit_twocolor_odr(data[i], i, p_guess, outroot, n_bootstrap, ifplot, ifbootstrap, xyswitch)

            if xyswitch:
                infos[i].cmd_alpha2 = alpha
                infos[i].cmd_alpha2_error = alpha_error
                infos[i].cmd_m2 = fit_output.beta[0]
                infos[i].cmd_b2 = fit_output.beta[1]
                infos[i].cmd_m2_error = fit_output.sd_beta[0]
                infos[i].cmd_b2_error = fit_output.sd_beta[1]
                infos[i].cmd_x_spread = x_spread
            else:
                infos[i].cmd_alpha1 = alpha
                infos[i].cmd_alpha1_error = alpha_error
                infos[i].cmd_m = fit_output.beta[0]
                infos[i].cmd_b = fit_output.beta[1]
                infos[i].cmd_m_error = fit_output.sd_beta[0]
                infos[i].cmd_b_error = fit_output.sd_beta[1]
                infos[i].cmd_x_spread = x_spread
    
    return infos

def good_slope_angle(infos):
    '''Checks if the ODR fit with switched X and Y axes yields a more constrained fit than the original axes. This basically catches the pathological cases with a (nearly) vertical fit with large nominal errors.
    
    Parameters
    ----------
    infos : np.rec.array
        info structure for all sources

    Returns
    -------
    infos : np.rec.array
        updated info structure
    '''
    for i in np.arange(0, len(infos)):
        if infos[i]['cmd_alpha1'] > -99999.:
            if infos[i]['cmd_alpha1_error']/infos[i]['cmd_alpha1'] < infos[i]['cmd_alpha2_error']/infos[i]['cmd_alpha2']:
                infos[i]['cmd_alpha'] = infos[i]['cmd_alpha1']
                infos[i]['cmd_alpha_error'] = infos[i]['cmd_alpha1_error']
            else:
                infos[i]['cmd_alpha'] = infos[i]['cmd_alpha2']
                infos[i]['cmd_alpha_error'] = infos[i]['cmd_alpha2_error']
    return infos


def cmd_dominated_by(infos):
    # this is some crude classification of the cmd slope.
    # anything that goes up and has a relative slope error of <40% is "accretion-dominated",
    # anythin that is within some cone around the theoratical reddening and has error <40% is "extinction-dominated",
    # anything else is "other".
    # if slope is classified as extinction, the spread in the CMD is converted to AV and stored in infos.
    alpha_red = math.asin(calc_reddening()[0]/np.sqrt(calc_reddening()[0]**2 + 1**2)) # angle of standard reddening
    for i in np.arange(0, len(infos)):
        if ((infos[i]['cmd_alpha'] > -99999.) & (infos[i]['cmd_alpha'] < 0.) & (infos[i]['cmd_alpha_error']/infos[i]['cmd_alpha'] <= 0.3) ):
            infos[i]['cmd_dominated'] = 'accr.'
        elif ((infos[i]['cmd_alpha'] > -99999.) & (np.abs(infos[i]['cmd_alpha'] - alpha_red) <= np.pi/18.) & (infos[i]['cmd_alpha_error']/infos[i]['cmd_alpha'] <= 0.3) ):
            infos[i]['cmd_dominated'] = 'extinc.'
            infos[i]['AV'] = infos[i]['cmd_x_spread']/calc_reddening()[1]
        elif ((infos[i]['cmd_alpha'] > -99999.) & (infos[i]['cmd_alpha_error']/infos[i]['cmd_alpha'] <= 0.3)):
            infos[i]['cmd_dominated'] = 'other'
        elif (infos[i]['cmd_alpha'] > -99999.):
            infos[i]['cmd_dominated'] = 'bad'
        else:
            infos[i]['cmd_dominated'] = 'no data'
    
    return infos





