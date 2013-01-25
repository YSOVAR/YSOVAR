import itertools

import numpy as np
import scipy
import scipy.odr as odr
import matplotlib.pylab as plt

import ysovar_atlas as atlas


def combinations_with_replacement(iterable, r):
    '''defined here for backwards compatibility
    From python 2.7 on it's included in itertools
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
    '''make a list of scatter dealta_data1 vs delta_data2 for all combinations of
    
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

def delta_corr_points(x, data1, data2):
    '''correlate two variables sample at the same (possible irregular) time points
    
    Essentially, this is a correlation function for irregularly gridded data.
    
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
    diff_2 : np.ndarray
        corresponding correlation in the dependent variables
    
    ..note::
        Essentially, this is a correltation function for irregularly sampled data
    
    '''
    #if not np.all(np.diff(data1) >= 0):
        #raise ValueError('Independent inputarray must by monotonically increasing.')
    if (len(x) != len(data2)) or (len(data1) != len(data2)):
        raise ValueError('All input arrays must have the same number of elements.')
    comblist = list(combinations_with_replacement(range(len(data1)), 2))
    diff_x = np.zeros(len(comblist))
    diff_2 = np.zeros_like(diff_x)
    for i,j in enumerate(comblist):
        diff_x[i] = x[j[1]] - x[j[0]]
        diff_2[i] = data1[j[1]] * data2[j[0]]
    ind = np.argsort(diff_x)
    return diff_x[ind], diff_2[ind]

def slotting(xbins, x, y, kernel = None, normalize = True):
    '''Add up all the y values in each x bin
    
    `xbins` defines a (possible non-uniform) bin grid. For each bin, find all
    (x,y) pairs that belong in the x bin and add up all the y values in that bin.
    Optiaonally, the y values can be convolved with a kernel before, so that
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
        This avoids zero-padding, but leads to a irragular "noise" distribution
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


def fit_poly(x, y, yerr, degree):
    ''' Fit a polynom to a dataset
    
    ..note:: 
        For numerical stability the `x` values will be shifted, such that
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
    shift : float
        shift applied to x value for numerical stability.
    beta : list
        fit parameters
    res_var : float
        residual of the fit
    '''
    guess = [1.] * degree
    model = odr.Model(np.polyval)
    mydata = odr.RealData(x-x[0], y, sy = yerr)
    myodr = odr.ODR(mydata, model, beta0 = guess)
    myoutput = myodr.run()
    return x[0], myoutput.beta, myoutput.res_var

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
        shift, param, chi = fit_poly(x, y, yerr, i)
        temp = ax.plot(xlong, np.polyval(param, xlong-shift), label = '{0:5.1f}'.format(chi))
    temp = ax.set_title(str(title))
    temp = ax.legend(loc = 'best')
    return fig

def calc_poly_chi(data, infos, verbose = True):
    '''Fits polynoms of degree 1..6 to all lightcurves in data
    
    One way to adress if a lightcurve is "smooth" is to fit a low-order
    polynomial. This routine fits polynomial of degree 1 to 6  to each IRAC1 and
    IRAC 2 lightcurve and calculates the chi^2 value for each fit.
    
    Parameters
    ----------
    data : np.ndarray
        structure with all the raw object information
    info : np.rec.array
        structure with the refined object properties. Must have fields
        `chi2poly1_36`, `chi2poly2_36` etc.
    verbose : bool
        Switch for extra verbosity.
        
    Returns
    -------
    info : np.rec.array
        structure with the refined object properties, modified from input
    
    '''
    for i in range(len(data)):
        if verbose and (np.mod(i, 100) == 0):
            print 'Fitting polynomials to dataset: ', i, ' of ', len(data)
        for deg in np.arange(1,6):
            for band, bandi in zip(['1','2'], ['_36', '_45']):
                if 't' + band in data[i].keys():
                    shift, coeff, chi2 = fit_poly(data[i]['t'+band], data[i]['m'+band], data[i]['m'+band+'_error'], deg)
                    infos['chi2poly'+str(deg) + bandi][i] = chi2
                else:
                    infos['chi2poly'+str(deg) + bandi][i] = np.nan
    return infos



# plots to make for a standard ysovar paper (result of November workweek):
#SED
#3.6 vs. t light curve
#4.5 vs. t light curve
#delta 3.6 vs delta (3.6-4.5)
#delta 3.6 vs. delta t (points and/or some sort of density map)
#delta 4.5 vs. delta t (points and/or some sort of density map)
#delta (3.6-4.5) vs. delta t (points and/or some sort of density map)
#derivative (deltaM/deltaT) vs t
#periodogram
#RMS
##maybe: zero crossings/40d - via smoothed or unsmoothed
#stetson (i1 and i2)
#chisq
