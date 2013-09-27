import os

import pytest

from YSOVAR import atlas
from YSOVAR import lightcurves

@pytest.fixture(scope = 'module')
def data():
    datadict = atlas.dict_from_csv(os.path.join(os.path.dirname(__file__), 'data','expected_results.csv'), floor_error = {'IRAC1': 0., 'IRAC2': 0.})
    data = atlas.YSOVAR_atlas(lclist = datadict)
    data.calc('lombscargle', '36', maxper = 100)
    data.calc('lombscargle', '45', maxper = 100)
    data.is_there_a_good_period(20, 1,100)
    data.calc('cmdslopesimple', ['36', '45'])
    data.calc('cmdslopeodr', ['36', '45'])
    data.calc('stetson', ['36','45'])
    lightcurves.calc_poly_chi(data, ['36', '45'])
    return data
