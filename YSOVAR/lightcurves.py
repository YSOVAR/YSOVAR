# -*- coding: utf-8 -*-
# Copyright (C) 2013 H.M.Guenther & K.Poppenhaeger. See Licence.rst for details.
'''The routines here are for non-periodic timing, see :mod:`YSOVAR.lombscargle`
for periodograms.
'''

import itertools

import numpy as np
import scipy
import scipy.odr as odr
import scipy.signal
import matplotlib.pylab as plt
import astropy.table

# Run even if statsmodels is not available
# This package is needed for few task only
try:
    import statsmodels.tsa.stattools as tsatools
    import statsmodels.tsa.ar_model as ar
    _has_statsmodels = True
except ImportError:
    try:
        # In some python installations statsmodels is part of scikits
        import scikits.statsmodels.tsa.stattools as tsatools
        import scikits.statsmodels.tsa.ar_model as ar
        _has_statsmodels = True
    except ImportError:
        _has_statsmodels = False

from .registry import register

def combinations_with_replacement(iterable, r):
    '''defined here for backwards compatibility
    From python 2.7 on its included in itertools
    '''
    # combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC
    pool = tuple(iterable)
    n = len(pool)
    if not n and r:
        return
    indices = [0] * r
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != n - 1:
                break
        else:
            return
        indices[i:] = [indices[i] + 1] * (r - i)
        yield tuple(pool[i] for i in indices)


def delta_delta_points(data1, data2):
    '''make a list of scatter delta_data1 vs delta_data2 for all combinations of

    E.g. this can be used to calculate delta_T vs. delta mag

    Parameters
    ----------
    data1 : np.ndarray
        independend variable (x-axis), e.g. time of a lightcurve
    data2 : np.ndarray
        dependent variable (y-axis), e.g. flux for a lightcurve

    Returns
    -------
    diff_1 : np.ndarray
        all possible intervals of the independent variable
    diff_2 : np.ndarray
        corresponding differences in the depended variable

    ..note::
        Essentially, this is an autocorrelation for irregularly sampled data
    '''
    #if not np.all(np.diff(data1) >= 0):
        #raise ValueError('Independent inputarray must by monotonically increasing.')
    if len(data1) != len(data2):
        raise ValueError('Both input arrays must have the same number of elements.')
    comblist = list(itertools.combinations(range(len(data1)), 2))
    diff_1 = np.zeros(len(comblist))
    diff_2 = np.zeros_like(diff_1)
    for i,j in enumerate(comblist):
        diff_1[i] = data1[j[1]] - data1[j[0]]
        diff_2[i] = data2[j[1]] - data2[j[0]]
    ind = np.argsort(diff_1)
    return diff_1[ind], diff_2[ind]

def corr_points(x, data1, data2):
    '''Make all combinations of  two variables at times ``x``

    Parameters
    ----------
    x : np.ndarray
        independend variable (x-axis), e.g. time of a lightcurve
    data1, data2 : np.ndarray
        dependent variables (y-axis), e.g. flux for a lightcurve

    Returns
    -------
    diff_x : np.ndarray
        all possible intervals of the independent variable
    d_2 : 2-d np.ndarray
        corresponding values of dependent variables.
        Array as shape (N, 2), where N is the number of combinations.
    '''
    if (len(x) != len(data2)) or (len(data1) != len(data2)):
        raise ValueError('All input arrays must have the same number of elements.')
    comblist = list(combinations_with_replacement(range(len(data1)), 2))
    diff_x = np.zeros(len(comblist))
    d_2 = np.zeros((len(comblist), 2))
    for i,j in enumerate(comblist):
        diff_x[i] = x[j[1]] - x[j[0]]
        d_2[i,:] = [data1[j[1]], data2[j[0]]]
    ind = np.argsort(diff_x)
    return diff_x[ind], d_2[ind]


def delta_corr_points(x, data1, data2):
    '''correlate two variables sampled at the same (possible irregular) time points

    Parameters
    ----------
    x : np.ndarray
        independend variable (x-axis), e.g. time of a lightcurve
    data1, data2 : np.ndarray
        dependent variables (y-axis), e.g. flux for a lightcurve

    Returns
    -------
    diff_x : np.ndarray
        all possible intervals of the independent variable
    d_2 : np.ndarray
        corresponding correlation in the dependent variables

    ..note::
        Essentially, this is a correltation function for irregularly sampled data

    '''
    diff_x, d_2 = corr_points(x, data1, data2)
    return diff_x, d_2[:,0] * d_2[:, 1]

def slotting(xbins, x, y, kernel = None, normalize = True):
    '''Add up all the y values in each x bin

    `xbins` defines a (possible non-uniform) bin grid. For each bin, find all
    (x,y) pairs that belong in the x bin and add up all the y values in that bin.
    Optionally, the x values can be convolved with a kernel before, so that
    each y can contribute to more than one bin.

    Parameters
    ----------
    xbins : np.ndarray
        edges of the x bins. There are `len(xbins)-1` bins.
    x, y : np.ndarry
        x and y value to be binned
    kernel : function
        Kernel input is binedges, kernel output bin values:
        Thus, len(kernelout) must be len(kernelin)-1!
        The kernal output should be normalized to 1.
    normalize : bool
        If false, get the usual correlation function. For a regularly sampled
        time series, this is the same as zero-padding on the edges.
        For `normalize = true` divide by the number of entries in a time bin.
        This avoids zero-padding, but leads to an irregular "noise" distribution
        over the bins.

    Returns
    -------
    out : np.ndarray
        resulting array of added y values
    n : np.ndarray
        number of entries in wach bin. If `kernel` is used, this can be non-integer.
    '''
    if kernel == None:
        dig = np.digitize(x, xbins) -1 # counts start with bin 1
        n = np.bincount(dig, minlength = len(xbins)-1)
        out = np.bincount(dig, weights = y, minlength = len(xbins)-1)
        if normalize:
            out = out / n
    else:
        out = np.zeros(len(xbins)-1)
        for valx, valy in zip(x, y):
            k = kernel(valx, xbins)
            if len(k) != len(out):
                raise ValueError('Kernel input is binedges, kernel output bin values: Thus, len(kernelout) must be len(kernelin)-1!')
            out += valy * k
    return out, n

def gauss_kernel(scale = 1):
    '''return a Gauss kernel

    Parameters
    ----------
    scale : float
        width (sigma) of the Gauss function

    Returns
    -------
    kernel : function
        `kernel(x, loc)`, where `loc` is the center of the Gauss and `x` are
        the bin boundaries.
    '''
    def kernel(x, loc):
        temp = scipy.stats.norm.cdf(x, loc = loc, scale = scale)
        return temp[1:] - temp[:-1]
    return kernel

def discrete_struc_func(t, val, order  = 2, scale = 0.1):
    '''discrete structure function

    Parameters
    ----------
    t : np.ndarray
        times of time series
    val : np.ndarray
        values of time series
    order : float
        the exponent of the structure function
    scale : float
        In order to accept irregular time series, the calculated autocorrelation
        needs to be binned in time. ``scale`` sets the width of those bins.

    Returns
    -------
    timebins : np.ndarray
        time bins corresponding to the values in ``dsf``
    dsf : np.ndarray
        binned and averaged discrete structure function
    '''
    dt, dm = corr_points(t, val, val)
    dm = np.abs(dm[:,0]-dm[:,1])**order
    timebins = np.arange(0, np.max(dt), scale)
    dsf, n = slotting(timebins, dt, dm)
    return timebins, dsf

def normalize(data):
    '''normalize data to mean = 1 and stddev = 1

    Parameters
    ----------
    data : np.array
        input data

    Returns
    -------
    data : np.array
        normalized set of data
    '''
    return (data - data.mean()) / np.std(data)


def describe_autocorr(t, val, scale = 0.1, autocorr_scale = 0.5, autosum_limit = 1.75):
    '''describe the timescales of time series using an autocorrelation function

    #This procedure takes an unevenly sampled time series and computes
    #the autocorrelation function from that. The result is binned in time bins
    #of width `scale` and three numbers are derived from the shape of the
    #autocorrelation function.

    This is based on the definitions used by Maria for the Orion paper.
    A visual definition is given `on the YSOVAR wiki (restriced access)
    <http://ysovar.ipac.caltech.edu/private/wiki/images/3/3c/Acfdefn.jpg>`_.

    Parameters
    ----------
    t : np.ndarray
        times of time series
    val : np.ndarray
        values of time series
    scale : float
        In order to accept irregular time series, the calculated autocorrelation
        needs to be binned in time. ``scale`` sets the width of those bins.
    autocorr_scale : float
        ``coherence_time`` is the time when the autocorrelation falls below
        ``autocorr_scale``. ``0.5`` is a common value, but for sparse sampling
        ``0.2`` might give better results.
    autosum_limit : float
        The autocorrelation function is also calculated with a time binning
        of ``scale``. To get a robust measure of this, the function
        calculate the timescale for the cumularitve sum of the autocorrelation
        function to exceed ``autosum_limit``.

    Returns
    -------
    cumsumtime : float
        time when the cumulative sum of a finely binned autocorrelation function
        exceeds ``autosum_limit`` for the first time; ``np.inf`` is returned
        if the autocorrelation function never reaches this value.
    coherence_time : float
        time when the autocorrelation function falls below ``autocorr_scale``
    autocorr_time : float
        position of first positive peak
    autocorr_val : float
        value of first positive peak
    '''
    if len(t) != len(val):
        raise ValueError('Time t and vector val must have same length.')

    if len(t) < 5:
        return np.nan, np.nan, np.nan, np.nan

    trebin = np.arange(np.min(t), np.max(t), scale)
    valrebin = np.interp(trebin, t, val)
    valnorm = normalize(valrebin)
    trebin = trebin - trebin[0]
    if not _has_statsmodels:
        raise ImportError('stats models not found')
    acf = tsatools.acf(valnorm, nlags = 100./scale)

    index = np.where(acf < autocorr_scale)
    if len(index) > 0:
        coherence_time = trebin[np.min(index)]
    else:
        coherence_time = np.inf
    ind1max = scipy.signal.argrelmax(acf)[0]

    #set defaults, in case first peak is not found
    autocorr_time = np.inf
    autocorr_val = np.inf

    if len(ind1max) > 0:
        indsubzero = np.min(np.where(acf < 0))
        # autocorr is positive and after the first negative value
        ind = (acf[ind1max] > 0.) & (ind1max > indsubzero)
        if ind.sum() > 0:
            ind1max = np.min(ind1max[ind])
            autocorr_time = trebin[ind1max]
            autocorr_val = acf[ind1max]

    normy = normalize(val)
    dt, dm = delta_corr_points(t, normy, normy)
    autotime  = np.arange(np.min(dt), np.max(dt), scale/10.)
    autocorr, n_autobin = slotting(autotime, dt, dm)
    autocorr[n_autobin == 0] = 0.  #np.nan because of devision
    autosum = scale * autocorr.cumsum()
    #autosum.cumsum() is value at end of bin, so add one bin width to autotime
    ind = np.where(autosum > autosum_limit)
    if len(ind[0]) > 0:
        cumsumtime = scale + autotime[np.min(ind)]
    else:
        cumsumtime = np.inf

    return cumsumtime, coherence_time, autocorr_time, autocorr_val

register(describe_autocorr, n_bands = 1, error= False, time = True, default_colnames = ['cumsumtime', 'coherence_time'], name='ACF', force = True, default_colunits=['d','d'], default_coldescriptions=['Time until cumulative sum of ACf reaches threshold value', 'decay time of ACF'])

def ARmodel(t, val, degree = 2, scale = 0.5):
    '''Fit an auto-regressive (AR) model to data and retrn some parameters

    The inout data can be irregularly binned, it will be resampled on a
    regular grid with bin-width ``scale``.

    Parameters
    -----------
    t : np.ndarray
        input times
    val : np.ndarray
        input values
    degree : int
        degree of AR model
    scale : float
        binning ofthe resampled lightcurve

    Returns
    -------
    params : list of ``(degree + 1)`` floats
        parameters of the model
    sigma2 : float
        sigma of the Gaussian component of the model
    aic : float
       value of the Akaike information criterion
    '''
    if len(t) != len(val):
        raise ValueError('Time t and vector val must have same length.')
    if not _has_statsmodels:
        raise ImportError('statsmodels not found')
    trebin = np.arange(np.min(t), np.max(t), scale)
    valrebin = np.interp(trebin, t, val)
    valrebin = normalize(valrebin)
    modar = ar.AR(valrebin)
    resar = modar.fit(degree)
    return  resar.params, resar.sigma2, resar.aic

def fit_poly(x, y, yerr, degree = 2):
    ''' Fit a polynom to a dataset

    ..note::
        For numerical stability the ``x`` values will be shifted, such that
        x[0] = 0!

    Thus, the parameters describe a fit to this shifted dataset!

    Parameters
    ----------
    x : np.ndarray
        array of independend variable
    y : np.ndarray
        array of dependend variable
    yerr: np.ndarray
        uncertainty of y values
    degree : integer
        degree of polynomial

    Returns
    -------
    res_var : float
        residual of the fit
    shift : float
        shift applied to x value for numerical stability.
    beta : list
        fit parameters
    '''
    if len(x) < degree + 1:
        return np.nan, np.nan, np.nan
    guess = [1.] * degree
    model = odr.Model(np.polyval)
    mydata = odr.RealData(x-x[0], y, sy = yerr)
    myodr = odr.ODR(mydata, model, beta0 = guess)
    myoutput = myodr.run()
    return myoutput.res_var, x[0], myoutput.beta

register(fit_poly, n_bands = 1, time = True, error = True, name = 'fitpoly', force = True)

def plot_all_polys(x, y, yerr, title = ''):
    '''plot polynomial fit of degree 1-6 for a dataset

    Parameters
    ----------
    x : np.ndarray
        array of independend variable
    y : np.ndarray
        array of dependend variable
    yerr : np.ndarray
        uncertainty of y values
    title : string
        title of plot

    Returns
    -------
    fig : matplotlib.figure instance
    '''
    fig = plt.figure(figsize = (8,6))
    ax = fig.add_subplot(111)
    temp = ax.errorbar(x, y,yerr = yerr, fmt='o')
    temp = ax.scatter(x, y, c = x)
    xlong = np.arange(np.min(x), np.max(x))
    for i in np.arange(1,6):
        chi, shift, param = fit_poly(x, y, yerr, i)
        temp = ax.plot(xlong, np.polyval(param, xlong-shift), label = 'n={0}: {1:5.1f}'.format(i,chi))
    temp = ax.set_title(str(title))
    temp = ax.legend(loc = 'best', title='Rank of poly: $\\chi^2$')
    # invert y axis!
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[1], ylim[0])
    return fig

def calc_poly_chi(data, bands=['36','45']):
    '''Fits polynoms of degree 1..6 to all lightcurves in data

    One way to adress if a lightcurve is "smooth" is to fit a low-order
    polynomial. This routine fits polynomial of degree 1 to 6  to each IRAC1 and
    IRAC 2 lightcurve and calculates the chi^2 value for each fit.

    Parameters
    ----------
    data : astropy.table.Table
        structure with the defined object properties.
    bands : list of strings
        Band identifiers, e.g. ['36', '45'], can also be a list with one
        entry, e.g. ['36']

    '''
    for deg in np.arange(1,6):
        for band in bands:
            data.calc('fitpoly', band, degree = deg, colnames = ['chi2poly_'+str(deg)])
