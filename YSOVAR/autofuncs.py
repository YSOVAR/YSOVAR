# -*- coding: utf-8 -*-
# Copyright (C) 2013 H.M.Guenther & K.Poppenhaeger. See Licence.rst for details.
'''Define and register some common functions for the autogeneration of columns.

Module level variables
----------------------
This module contains a dictionary of default reddening vectors.

The form of the vectors is a following:

    redvec[0] : slope of reddening law in CMD
    redvec[1] : reddening value in first band (the $\delta y$ in CMD)

The following command will list all implemented reddening vectors::

    print YSOVAR.plot.redvecs

The reddening vector with the key '36_45' is used as default in most plots if
no specific reddening vector is specified.

Functions
----------
This module defines some commonly used functions (e.g. stetson) for the analysis
of our lightcurves. It registers all the functions defined here and a bunch
of simple numpy function with :mod:`registry`, so that they are availvalbe
for the autogeneration of lightcurves in :class:`atlas.YSOVAR_atlas` obejcts.

Specifically, this module collects function that do *not* depend on the
times of the observations (see :mod:`lightcuves` for that).

'''
import math

import numpy as np
import scipy.stats
import scipy.stats.mstats
import scipy.odr

from astropy.utils.compat.odict import OrderedDict

from .registry import register

#def redvec_36_45():
#    ''' Rieke & Lebofsky 1985 (bibcode 1985ApJ...288..618R)
#  I take the extinctions from the L and M band (3.5, 5.0).'''
#    A36 = 0.058
#    A45 = 0.023
#    R36 = - A36/(A45 - A36)
#    return np.array([R36, A36])
redvecs = {'36_45_rieke_Lebofsky85_vsAV': np.array([1.66, 0.058]),
           '36_45_Flaherty07_vsAK':np.array([-0.632/(0.53-0.632),0.632]),
           '36_45_Indebetouw05_vsAK': np.array([-0.56/(0.43-0.56), .56]),
           '36_45': np.array([-0.56/(0.43-0.56), .56])
           }
'''Dictionary of default reddening vectors

The form of the vectors is a following:
    redvec[0] : slope of reddening law in CMD
    redvec[1] : reddening value in first band (the delta_y in CMD)
'''


### simple function for one band ###
def mad(data):
    '''Median absolute deviation.'''
    return np.median(np.abs(data - np.median(data)))

def redchi2tomean(data, error):
    '''reduced chi^2 to mean'''
    return np.sum( (data - np.mean(data))**2/(error**2) )/(len(data)-1)

def delta(data):
    '''width of distribution from 10% to 90%'''
    return (scipy.stats.mstats.mquantiles(data, prob=0.9) - scipy.stats.mstats.mquantiles(data, prob=0.1))

def AMCM(data):
    '''So-called M value from Ann-Marie Cody's 2014 paper. Light curve asymmetry across the median; specifically, average of top and bottom 10% minus median, divided by rms  scatter.'''
    return (np.mean([scipy.stats.mstats.mquantiles(data, prob=0.9), scipy.stats.mstats.mquantiles(data, prob=0.1)]) - np.median(data))/np.sqrt( ( (data - data.mean())**2).sum()   / len(data) )

def wmean(data, error):
    '''error weighted mean'''
    return np.average(data, weights=1./error**2.)

def isnormal(data):
    'p-value for a 2-sided chi^2 probability that the distribution is normal'
    if len(data) >=20:
        return scipy.stats.normaltest(data)[1]
    else:
        return np.nan

register(np.mean, n_bands = 1, error = False, time = False, force = True, default_colunits=['mag'], 
         default_coldescriptions=['mean magnitude'])
register(np.median, n_bands = 1, error = False, time = False, force = True, default_colunits=['mag'],
         default_coldescriptions=['median magnitude'])
register(mad, n_bands = 1, error = False, time = False, force = True, default_colunits=['mag'])
register(delta, n_bands = 1, error = False, time = False, force = True, default_colunits=['mag'])
register(AMCM, n_bands = 1, error = False, time = False, force = True, default_colunits=['mag'])
register(len, n_bands = 1, error = False, time = False, name = 'n', 
         other_cols = OrderedDict([('n', int)]), force = True, 
         default_coldescriptions=['Number of datapoints'], default_colunits=['ct'])
register(np.min, n_bands = 1, error = False, time = False, name = 'min', force = True, 
         default_colunits=['mag'], default_coldescriptions=['minimum magnitude in lightcurve'])
register(np.max, n_bands = 1, error = False, time = False, name = 'max', force = True, 
         default_colunits=['mag'], default_coldescriptions=['maximum magnitude in lightcurve'])
register(np.std, n_bands = 1, time = False, error = False, name = 'stddev', 
         description = 'standard deviation calculated from non-biased variance', 
         kwargs = {'ddof': 1}, force = True, default_colunits=['mag'])
register(scipy.stats.skew, n_bands = 1, error = False, time = False, 
         description = 'biased (no correction for dof) skew', force = True, default_colunits=['mag'])
register(scipy.stats.kurtosis, n_bands = 1, error = False, time = False, 
         description = 'biased (no correction for dof) kurtosis', force = True, default_colunits=['mag'])
register(isnormal, n_bands = 1, error = False, time = False, force = True)

for func in [redchi2tomean, wmean]:
    register(func, n_bands = 1, time = False, error = True, force = True)


### functions for two bands ###

def stetson(data1, data2, data1_error, data2_error):
    '''Stetson index for a two-band lightcurve.

    According to eqn (1) in Stetson 1996, PSAP, 108, 851.
    This procedure uses on the matched lightcurves
    (not frames with one band only) and assignes a weight (g_i) in
    Stetson (1996) of 1 to each datapoint.

    Parameters
    ----------
    data1 : np.array
        single lightcurve of band 1 in magnitudes
    data2 : np.array
        single lightcurve of band 2 in magnitudes
    data1_error : np.array
        error on data points of band 1 in magnitudes    
    data2_error : np.array
        error on data points of band 2 in magnitudes          
        
    Returns
    -------
    stetson : float
        Stetson value for the provided two-band lightcurve
        
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

register(stetson, n_bands = 2, error = True, time = False, force = True)



def cmd_slope_simple(data1, data2, data1_error, data2_error, redvec = redvecs['36_45']):
    '''Slope of the data points in the color-magnitude diagram
    
    This is just fitted with ordinary least squares, using the analytic formula.
    This is then used as a first guess for an orthogonal least squares fit with simultaneous treatment of errors in x and y (see fit_twocolor_odr)

    Parameters
    ----------
    data1 : np.array
        single lightcurve of band 1 in magnitudes
    data2 : np.array
        single lightcurve of band 2 in magnitudes      
    data1_error : np.array
        error on data points of band 1 in magnitudes    
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
    if N < 3:
        return np.nan
    
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

register(cmd_slope_simple, n_bands = 2, error = True, time = False, default_colnames = ['cmd_m_plain', 'cmd_b_plain', 'cmd_m_redvec', 'cmd_b_redvec'], name = 'cmdslopesimple', force = True)

def fit_twocolor_odr(band1, band2, band1_err, band2_err, outroot = None,  n_bootstrap = None, xyswitch = False, p_guess = None, redvec = redvecs['36_45']):
    '''Fits a straight line to a single CMD, using a weighted orthogonal least squares algorithm (ODR).
    
    Parameters
    ----------
    data1 : np.array
        single lightcurve of band 1 in magnitudes
    data2 : np.array
        single lightcurve of band 2 in magnitudes      
    data1_error : np.array
        error on data points of band 1 in magnitudes    
    data2_error : np.array
        error on data points of band 2 in magnitudes

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
    redvec : np.array with two elements
        theoretical reddening vector for the two bands chosen

    
    Returns
    -------
    result : tuple
        contains output = fit parameters, bootstrap_output = results from the bootstrap, bootstrap_raw = the actual bootstrapped data, alpha = the fitted slope angle, sd_alpha = the error on the fitted slope angle, x_spread = the spread of the data along the fitted line (0.5*(90th percentile - 10th percentile)))
    '''
    #define the fitting function (in this case a straight line)
    def fitfunc(p, x):
        return p[0]*x + p[1]

    if p_guess is None:
        p_guess = list(cmd_slope_simple(band1, band2, band1_err, band2_err))[0:2]
        if ~np.isfinite(p_guess[0]): # pathological case
            p_guess[0] = 0
        if ~np.isfinite(p_guess[1]): # pathological case
            p_guess[1] = np.mean(band1-band2)
    
    # define what the x and y data is:
    x_data = band1 - band2
    y_data = band1
    x_error = np.sqrt( band1_err**2 + band2_err**2 )
    y_error = band1_err
    if xyswitch:
        y_data, x_data = (x_data, y_data)
        y_error, x_error = (x_error, y_error)
    
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
    alpha = math.atan(output.beta[0])
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
    
    result = (output, bootstrap_output, bootstrap_raw, alpha, sd_alpha, x_spread)
    return result

def cmdslope_odr(band1, band2, band1_err, band2_err, p_guess = None, redvec = redvecs['36_45']):
    '''Fits a straight line to a single CMD, using a weighted orthogonal least squares algorithm (ODR).
    
    Parameters
    ----------
    data1 : np.array
        single lightcurve of band 1 in magnitudes
    data2 : np.array
        single lightcurve of band 2 in magnitudes      
    data1_error : np.array
        error on data points of band 1 in magnitudes    
    data2_error : np.array
        error on data points of band 2 in magnitudes
    p_guess : tuple
        initial fit parameters derived from fit_twocolor
    redvec : np.array with two elements
        theoretical reddening vector for the two bands chosen
    
    
    Returns
    -------
    result : tuple
        contains output = fit parameters, bootstrap_output = results from the bootstrap, bootstrap_raw = the actual bootstrapped data, alpha = the fitted slope angle, sd_alpha = the error on the fitted slope angle, x_spread = the spread of the data along the fitted line (0.5*(90th percentile - 10th percentile)))
    '''
    if len(band1) < 10:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, ''
    
    if p_guess is None:
        p_guess = cmd_slope_simple(band1, band2, band1_err, band2_err, redvec = redvec)

    (fit_output2, bootstrap_output2, bootstrap_raw2, alpha2, alpha_error2, spread2) = fit_twocolor_odr(band1, band2, band1_err, band2_err, xyswitch = True, p_guess = p_guess, redvec = redvec)
    (fit_output, bootstrap_output, bootstrap_raw, alpha, alpha_error, spread) = fit_twocolor_odr(band1, band2, band1_err, band2_err, xyswitch = False, p_guess = p_guess, redvec = redvec)

    # Checks if the ODR fit with switched X and Y axes yields a more
    # constrained fit than the original axes. This basically catches the
    # pathological cases with a (nearly) vertical fit with large nominal errors.
    if alpha_error/alpha > alpha_error2/alpha2:
        alpha, alpha_error = (alpha2, alpha_error2)
        cmd_m = 1./fit_output2.beta[0]
        cmd_b = -fit_output2.beta[1] / fit_output2.beta[0]
        cmd_m_error = fit_output2.sd_beta[0] / cmd_m**2
        cmd_b_error = np.sqrt((fit_output2.sd_beta[1]/cmd_m)**2 +
                              (cmd_b**2*cmd_m_error**2)**2)
        spread = spread2
    else:
        cmd_m = fit_output.beta[0]
        cmd_b = fit_output.beta[1]
        cmd_m_error = fit_output.sd_beta[0]
        cmd_b_error = fit_output.sd_beta[1]
 
    # Make new alpha to avoid confusion in case of x/y switch
    alpha = math.atan(cmd_m)

    '''crude classification of CMD slope

    This is some crude classification of the cmd slope.
    anything that goes up and has a relative slope error of <40% is
    "accretion-dominated", anything that is within some cone around
    the theoretical reddening and has error <40% is "extinction-dominated",
    anything else is "other".
    If slope is classified as extinction, the spread in the CMD is converted
    to AV and stored.
    '''
    # angle of standard reddening
    alpha_red = math.atan(redvec[0])

    cmd_dominated = 'bad'
    AV = np.nan
    if alpha_error/alpha <=0.4:
        cmd_dominated = 'other'
        if np.abs(alpha - alpha_red) < 0.3:
            cmd_dominated = 'extinc.'
            AV = spread/redvec[1]
        if alpha < 0.:
            cmd_dominated = 'accr.'
            

    return alpha, alpha_error, cmd_m, cmd_b, cmd_m_error, cmd_b_error, AV, cmd_dominated, spread

register(cmdslope_odr, n_bands= 2, error = True, time = False, default_colnames = ['cmd_alpha', 'cmd_alpha_error', 'cmd_m', 'cmd_b', 'cmd_m_error', 'cmd_b_error', 'AV'], other_cols = OrderedDict([['cmd_dominated', 'S10'], ['CMD_length', 'float']]), name = 'cmdslopeodr', force = True, default_colunits=['rad','rad',None, None, None, None, 'mag',None, None, 'mag'], default_coldescriptions=['angle of best-fit line in CMD', 'uncertainty on angle', 'slope in CMD', 'offset of best-fits line', 'uncertainty on slope', 'uncertainty on angle', 'length of reddening vector', 'classification of slope in CMD', '90% spread in slope in CMD'])
