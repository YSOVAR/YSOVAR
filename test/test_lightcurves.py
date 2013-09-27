import numpy as np

import pytest
from YSOVAR import lightcurves as lc


def test_norm():
    x = np.arange(.2,5.6,.1)
    normx = lc.normalize(x)
    assert np.abs(normx.mean()) < 0.00001
    assert np.abs(np.std(normx)-1.) < 0.00001

class Test_kernels:
    
    kernels = [
    {'func': lc.gauss_kernel,
        'args': [],
        'kwargs': {'scale': 2}
    },
        {'func': lc.gauss_kernel,
        'args': [],
        'kwargs': {}
    },
    ]
    
    ##def test_returntype(self):
        ##for k in self.kernels:
            ##assert type(k['func']()) == type(np.sin)
    
    def test_norm(self):
        x = np.arange(-200.,200.,.001)
        for k in self.kernels:
            kernel = k['func'](**k['kwargs'])
            y = kernel(x, 2.6) 
            assert np.abs(y.sum() - 1) < 0.001
    
    def test_center(self):
        x = np.arange(-200.,200.,.001)
        for k in self.kernels:
            kernel = k['func'](**k['kwargs'])
            y = kernel(x, 2.6) 
            assert np.abs(np.sum(0.5*(x[1:]+x[:-1])*y) - 2.6) < 0.001


def test_delta_delta_points():
    t = [1.,2.,4., 5.1]
    result = np.array([1.,1.1,2.,3.,3.1,4.1])
    d1, d2 = lc.delta_delta_points(t, t)
    assert np.all(np.diff(d1) >= 0)
    assert np.all(np.abs(d1 - result) < 1e-10)
    assert np.all(np.abs(d2 - result) < 1e-10)

#def test_input_checking():
    #with pytest.raises(ValueError):
        #delta_delta_points([1.,5.,3.], np.arange(3))

def test_input_checking2():
    with pytest.raises(ValueError):
       lc.delta_delta_points([1.,5.], np.arange(3))
       
def test_delta_corr_points():
    # Test for regular time series where a numpy function exist.
    t = np.arange(100)
    y = np.sin(t)
    x = np.correlate(y,y,mode='full')[len(y)-1:]
    txlc, xlc = lc.delta_corr_points(t, y, y)
    sxlc, n = lc.slotting(np.arange(-0.5,100,1.), txlc, xlc, normalize = False)
    
    assert np.all(np.abs(x - sxlc) < 1e-10)
    
def test_delta_corr_points2():
    # Test for regular time series where a numpy function exist.
    t = np.arange(100)
    y = np.sin(t)
    x = np.correlate(y,y,mode='full')[len(y)-1:] / np.arange(100,0,-1)
    txlc, xlc = lc.delta_corr_points(t, y, y)
    sxlc, n = lc.slotting(np.arange(-0.5,100,1.), txlc, xlc)
    
    assert np.all(np.abs(x - sxlc) < 1e-10)
    assert np.all(n == np.arange(100,0,-1))

def test_ACF_little_data():
    t = np.array([0.,1.,2.5,8.])
    val = t
    out = lc.describe_autocorr(t, val)
    assert len(out) == 4
    assert np.all(np.isnan(out))

def test_ACF_uneven_data():
    with pytest.raises(ValueError):
        lc.describe_autocorr(np.arange(49), np.arange(50))

#def test_ACF_sparse_data():
#    # Even for sparse data all results should be finite
#    t = np.array([0.,1.,2.5,8., 10.5, 12.,15.])
#    val = t
#    out = lc.describe_autocorr(t, val)
#    assert(np.all(np.isfinite(out)))

