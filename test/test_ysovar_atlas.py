import numpy as np

import ysovar_atlas as atlas

def test_isoy2radec():
    ra, dec = atlas.Isoy2radec('ISOY_J162722.02-242114.5')
    assert np.abs(ra - 246.84175) < 0.00001
    assert np.abs(dec + 24.354028) < 0.00001
    ra, dec = atlas.Isoy2radec('ISOY_J162722.02+242114.5')
    assert np.abs(dec - 24.354028) < 0.00001