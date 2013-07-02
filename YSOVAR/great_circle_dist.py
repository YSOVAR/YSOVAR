# Copyright (C) 2013 H.M.Guenther & K.Poppenhaeger. See Licence.rst for details.
'''Calculate distances on the celestial sphere

Methods in here are required as helper functions for cross-matching sources by position.
Different algorithms with different numerical stability are implemented, but usually
you should not need to call this directly. See the source code of 
:func:`YSOVAR.atlas.dict_from_csv` for an example how how this works.
'''
import numpy as np

def simple_decorator(decorator):
    """This decorator can be used to turn simple functions
    into well-behaved decorators, so long as the decorators
    are fairly simple. If a decorator expects a function and
    returns a function (no descriptors), and if it doesn't
    modify function attributes or docstring, then it is
    eligible to use this. Simply apply @simple_decorator to
    your decorator and it will automatically preserve the
    docstring and function attributes of functions to which
    it is applied.
    """
    def new_decorator(f):
        g = decorator(f)
        g.__name__ = f.__name__
        g.__doc__ = f.__doc__
        g.__dict__.update(f.__dict__)
        return g
    # Now a few lines needed to make simple_decorator itself
    # be a well-behaved decorator.
    new_decorator.__name__ = decorator.__name__
    new_decorator.__doc__ = decorator.__doc__
    new_decorator.__dict__.update(decorator.__dict__)
    return new_decorator


# Be careful: here order Lat, Lon, but in astronomy often RA (=Lon), DEC(=Lat)
# TBD: add check that RA, DEC have same length

@simple_decorator
def unitchecked(func):
    '''Decorator to transfrom units of angles
    
    This decorator transforms units of angle, before they are fed into 
    any a function to calculate the angular distance. It expects the 
    unit as a keyword and transforms two sets of angular coordinates
    (phi_0, lam_0, phi_1, lam_1)
    to radian, calls the function and converts the output (in radian)
    into the unit of choice.
    '''
    def unit_conversion(phi_0, lam_0, phi_1, lam_1, unit = None):
        if (unit == None) or (unit in ['rad', 'radians', 'radian']):
            return func(phi_0, lam_0, phi_1, lam_1)
        elif unit in ['deg', 'degree']:
            ret = func(np.radians(phi_0), np.radians(lam_0), np.radians(phi_1), np.radians(lam_1))
            return np.degrees(ret)
        elif unit in ['arcmin', 'amin']:
            ret = func(np.radians(phi_0*60.), np.radians(lam_0*60.), np.radians(phi_1*60.), np.radians(lam_1*60.))
            return 60. * np.degrees(ret)
        elif unit in ['arcsec','asec']:
            ret = func(np.radians(phi_0*3600.), np.radians(lam_0*3600.), np.radians(phi_1*3600.), np.radians(lam_1*3600.))
            return 3600. * np.degrees(ret)
        else:
            raise ValueError('unit not recognized')
    return unit_conversion


def simple(phi_0, lam_0, phi_1, lam_1):
    '''Calculates the angular distance between point 0 and point 1
    
    uses a very simple formula, prone to numeric inaccuracies
    see http://en.wikipedia.org/wiki/Great-circle_distance
    
    Parameters
    ----------
    phi_0 : float or numpy array
        lattitude phi of point 0
    lam_0 : float or numpy array
        longitude lambda of point 0
    phi_1 : float or numpy array
        lattitude phi of point 1
    lam_1 : float or numpy array
        longitude lambda of point 1

    Returns
    -------
    dist : float or numpy array
        angluar distance on great circle
    '''
    return np.arccos(np.sin(phi_0)*np.sin(phi_1) + np.cos(phi_0)*np.cos(phi_1)*np.cos(lam_1-lam_0) )

def Haversine(phi_0, lam_0, phi_1, lam_1):
    '''Calculates the angular distance between point 0 and point 1
    
    uses the Haversine function, is numerically stable, except
    for antipodal points http://en.wikipedia.org/wiki/Great-circle_distance

    Parameters
    ----------
    phi_0 : float or numpy array
        lattitude phi of point 0
    lam_0 : float or numpy array
        longitude lambda of point 0
    phi_1 : float or numpy array
        lattitude phi of point 1
    lam_1 : float or numpy array
        longitude lambda of point 1

    Returns
    -------
    dist : float or numpy array
        angular distance on great circle
    '''

    return 2. * np.arcsin(np.sqrt((np.sin((phi_1-phi_0)/2.))**2. + (np.cos(phi_0)*np.cos(phi_1)*(np.sin((lam_1-lam_0)/2.))**2.)))

def Vincenty(phi_0, lam_0, phi_1, lam_1):
    '''Calculates the angular distance between point 0 and point 1
    
    uses a special case of the Vincenty formula (which is for ellipsoides)
    numerically accurate, but computationally intensive
    see http://en.wikipedia.org/wiki/Great-circle_distance

    Parameters
    ----------
    phi_0 : float or numpy array
        lattitude phi of point 0
    lam_0 : float or numpy array
        longitude lambda of point 0
    phi_1 : float or numpy array
        lattitude phi of point 1
    lam_1 : float or numpy array
        longitude lambda of point 1

    Returns
    -------
    dist : float or numpy array
        angular distance on great circle
    '''

    d_lam = lam_1-lam_0

    sp0 = np.sin(phi_0)
    cp0 = np.cos(phi_0)
    sp1 = np.sin(phi_1)
    cp1 = np.cos(phi_1)
    sl = np.sin(d_lam)
    cl = np.cos(d_lam)

    nom = np.sqrt((cp1*sl)**2.+(cp0*sp1-sp0*cp1*cl)**2.)
    denom = sp0*sp1 + cp0*cp1*cl

    return np.arctan2(nom,denom)

# Several aliases for special cases

# Define that as new function, so I can attach the decorator
@unitchecked
def dist(phi_0, lam_0, phi_1, lam_1):
    '''Calculates the angular distance between point 0 and point 1
    
    see http://en.wikipedia.org/wiki/Great-circle_distance

    Parameters
    ----------
    phi_0 : float or numpy array
        lattitude phi of point 0
    lam_0 : float or numpy array
        longitude lambda of point 0
    phi_1 : float or numpy array
        lattitude phi of point 1
    lam_1 : float or numpy array
        longitude lambda of point 1

    Returns
    -------
    dist : float or numpy array
        angular distance on great circle
    '''
    return Vincenty(phi_0, lam_0, phi_1, lam_1)

@unitchecked
def dist_radec(ra0,dec0,ra1,dec1):
    '''Calculates the angular distance between point 0 and point 1
    
    see http://en.wikipedia.org/wiki/Great-circle_distance
    
    Parameters
    ----------
    ra0 : float or numpy array
        RA of point 0
    dec0 : float or numpy array
        DEC of point 0
    ra1 : float or numpy array
        RA of point 1
    dec1 : float or numpy array
        DEC of point 1

    Returns
    -------
    dist : float or numpy array
        angluar distance on great circle
    '''
    return Vincenty(dec0,ra0,dec1,ra1)

def dist_radec_fast(ra0,dec0,ra,dec, scale = np.inf, *arg, **kwargs):
    '''Calculates the angular distance between point 0 and point 1
    
    Only if delta_dec  is < scale, the full trigonometric
    calculation is done, otherwise return np.inf
    
    see http://en.wikipedia.org/wiki/Great-circle_distance
    
    Parameters
    ----------
    ra0 : float or numpy array
        RA of point 0
    dec0 : float or numpy array
        DEC of point 0
    ra1 : float or numpy array
        RA of point 1
    dec1 : float or numpy array
        DEC of point 1

    Returns
    -------
    dist : float or numpy array
        angluar distance on great circle
    
    ..note::
        To be done:
            - cut on RA as well, but that requires knowledge of scale and
              the decorator transforms ra, dec only
            - merge this with ra_dec_dist and do fast verion if scale != None
    '''
    if len(ra) != len(dec):
        raise ValueError('RA and DEC arrays must have same length')
    out = np.zeros_like(ra)
    out[:] = np.inf
    ind = (np.abs(dec0-dec) < scale)
    out[ind] = dist_radec(ra0, dec0,ra[ind],dec[ind], *arg, **kwargs)
    return out
