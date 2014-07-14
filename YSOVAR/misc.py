'''This module holds helper routines that are not part of the main package.

Examples for this are::

    - Routines that were written for a specific cluster and are not 
      generalized yet.
    - I/O routines that are not directly related to the YSOVAR database
      (e.g. reading the IDL .sav files that Rob Gutermuth uses).
'''
import os

import numpy as np
from scipy.io.idl import readsav

from astropy.table import Table, Column

def read_cluster_grinder(filepath):
    ''' Import Robs Spitzer data

    read Rob's IDL format and make it into a a catalog, 
    deleting multiple columns and adding identifiers
    
    Parameters
    ----------
    filepath : string
        Path to a directory that holds the output of the ClusterGrinder
        pipeline. All files need to have standard names.
        Specifically, this routine reads:
        
            - ``cg_merged_srclist_mips.sav``
            - ``cg_classified.sav``

    Returns
    -------
    cat : astropy.table.Table
        Table with 2MASS ans Spitzer magnitudes and the clustergrinder 
        classification.
    '''
    s = readsav(os.path.join(filepath, 'cg_merged_srclist_mips.sav'))
    coo=np.ma.array(s.out[:,0:20],mask=(s.out[:,0:20] == 0.))
    s.out[:,20:30][np.where(s.out[:,20:30] < -99)] = np.nan
    s.out[:,30:40][np.where(s.out[:,30:40]==10)] = np.nan
    
    dat = Table()
    dat.add_column(Column(name='RA', data=np.ma.mean(coo[:,[0,2,4,12,14,16,18]],axis=1), unit = 'deg', format = '9.6g'))   
    #RA is avarage of all valid (non-zero) Ra values in 2MASS JHK, IRAC 1234
    dat.add_column(Column(name='DEC', data=np.ma.mean(coo[:,[1,3,5,13,15,17,19]],axis=1), unit='deg', format='+9.6g'))

    robsyukyformat={'J_MAG': 20,'H_MAG': 21, 'K_MAG': 22,'J_ERR': 30,
                    'H_ERR': 31,'K_ERR': 32,'IRAC_1': 26,'IRAC_2': 27, 
                    'IRAC_3': 28, 'IRAC_4': 29,'IRAC_1_ERR':36,'IRAC_2_ERR':37,
                    'IRAC_3_ERR':38, 'IRAC_4_ERR':39}
    for col in robsyukyformat:
        dat.add_column(Column(name=col, data=s.out[:, robsyukyformat[col]], unit='mag', format='4.2g'))

    s.mips[:,2][np.where(s.mips[:,2] == -100)] = np.nan
    s.mips[:,3][np.where(s.mips[:,3] == 10)] = np.nan
    dat.add_column(Column(name='MIPS', data=s.mips[:,2], unit='mag', format='4.2g'))
    dat.add_column(Column(name='MIPS_ERR',data=s.mips[:,3], unit='mag', format='4.2g'))

    IRclass = readsav(os.path.join(filepath, 'cg_classified.sav'))
    dat.add_column(Column(name='IRclass', dtype='|S5', length=len(dat)))
    for n1, n2 in zip(['wdeep', 'w1','w2','wtd','w3'], ['I*', 'I', 'II', 'II*', 'III']):
        if n1 in IRclass:
            dat['IRclass'][IRclass[n1]] = n2
    dat.add_column(Column(name='AK', data=IRclass.ak, unit='mag', format='4.2g'))
   
    return dat


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

def format_or_string(format_str):
        def func(val):
            if isinstance(val, basestring):
                return val
            else:
                return format_str % val
        return func

def makeclassinteger(guenther_data_yso):
    # assigns an integer to the Guenther+ 2012 classes. 0=XYSO, 1=I+I*, 2=II+II*, 3=III, 4=star
    guenther_class = np.ones(len(guenther_data_yso),int)*-9
    guenther_class[np.where(guenther_data_yso['Class'] == 'XYSO' )[0]] = 0
    guenther_class[np.where((guenther_data_yso['Class'] == 'I*') | (guenther_data_yso['Class'] == 'I'))[0]] = 1
    guenther_class[np.where((guenther_data_yso['Class'] == 'II*') | (guenther_data_yso['Class'] == 'II'))[0]] = 2
    guenther_class[np.where((guenther_data_yso['Class'] == 'III') | (guenther_data_yso['Class'] == 'III*'))[0]] = 3
    guenther_class[np.where(guenther_data_yso['Class'] == 'star')[0]] = 4
    return guenther_class


