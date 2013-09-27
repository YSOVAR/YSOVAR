import numpy as np
import pytest
from YSOVAR.great_circle_dist import *

def test_dist_comparison():
    phi_0 = .1
    lam_0 = .1
    phi_1 = -.4
    lam_1 = .2
    val1 = simple(phi_0, lam_0, phi_1, lam_1)
    val2 = Haversine(phi_0, lam_0, phi_1, lam_1)
    val3 = Vincenty(phi_0, lam_0, phi_1, lam_1)
    val4 = dist(phi_0, lam_0, phi_1, lam_1)
    val5 = dist_radec(lam_0, phi_0, lam_1, phi_1)
    
    assert (np.abs(val3 - val2)) < 0.001
    assert (np.abs(val3 - val1)) < 0.0001
    # The last three ar different interfaces to the same formulae
    assert val3 == val4
    assert val3 == val5

def test_units():
    val = dist_radec(150., 20.,150., 30., unit = 'deg')
    assert np.abs(val - 10.) < 0.00001

def test_fast():
    rain = np.array([150., 160.])
    decin = np.array([30., -10.])
    val = dist_radec(150., 20., rain, decin, unit = 'deg')
    val1 = dist_radec_fast(150., 20., rain, decin, unit = 'deg')
    assert np.all(val == val1)

def test_fast_number_in():
    with pytest.raises(TypeError):
        val = dist_radec_fast(150., 20.,150., 30., unit = 'deg')
