import numpy as np

import ysovar_atlas as atlas

import pytest

def test_isoy2radec():
    ra, dec = atlas.Isoy2radec('ISOY_J162722.02-242114.5')
    assert np.abs(ra - 246.84175) < 0.00001
    assert np.abs(dec + 24.354028) < 0.00001
    ra, dec = atlas.Isoy2radec('ISOY_J162722.02+242114.5')
    assert np.abs(dec - 24.354028) < 0.00001

@pytest.fixture(scope = "module")
def data():
    datadict = atlas.dict_from_csv('test/data/expected_results.csv', floor_error = {'IRAC1': 0., 'IRAC2': 0.})
    dat = atlas.YSOVAR_atlas(lclist = datadict)
    dat.calc_ls('36', 100)
    dat.calc_ls('45', 100)
    dat.is_there_a_good_period(20, 1,100)
    dat.cmd_slope_odr()
    dat.good_slope_angle()
    dat.cmd_dominated_by()
    return dat

@pytest.mark.usefixtures("data")
class Test_expected_results():
    def test_const_lc_36(self, data):
        ind1 = np.where(data['YSOVAR2_id'] == '-1000')[0][0]
        assert data['n_36'][ind1] == 5
        assert abs(data['max_36'][ind1] - 12.) < 1e-6
        assert abs(data['min_36'][ind1] - 12.) < 1e-6
        assert abs(data['median_36'][ind1] - 12.) < 1e-6
        assert abs(data['wmean_36'][ind1] - 12.) < 1e-6
        assert abs(data['stddev_36'][ind1]) < 1e-6
        assert abs(data['mad_36'][ind1]) < 1e-6
        assert abs(data['delta_36'][ind1]) < 1e-6
        assert abs(data['redchi2tomean_36'][ind1]) < 1e-6
        assert data['cmd_dominated'][ind1] == 'no data'

    def test_const_lc_45(self, data):
        ind1 = np.where(data['YSOVAR2_id'] == '-1001')[0][0]
        assert data['n_45'][ind1] == 5
        assert abs(data['max_45'][ind1] - 12.) < 1e-6
        assert abs(data['min_45'][ind1] - 12.) < 1e-6
        assert abs(data['median_45'][ind1] - 12.) < 1e-6
        assert abs(data['wmean_45'][ind1] - 12.) < 1e-6
        assert abs(data['stddev_45'][ind1]) < 1e-6
        assert abs(data['mad_45'][ind1]) < 1e-6
        assert abs(data['delta_45'][ind1]) < 1e-6
        assert abs(data['redchi2tomean_45'][ind1]) < 1e-6

    def test_lin_lc_36(self, data):
        ind1 = np.where(data['YSOVAR2_id'] == '-1002')[0][0]
        assert data['n_36'][ind1] == 5
        assert abs(data['max_36'][ind1] - 13.) < 1e-6
        assert abs(data['min_36'][ind1] - 11.) < 1e-6
        assert abs(data['median_36'][ind1] - 12.) < 1e-6
        assert abs(data['wmean_36'][ind1] - 12.) < 1e-6
        assert abs(data['stddev_36'][ind1]- 1./np.sqrt(2.)) < 1e-6
        assert abs(data['mad_36'][ind1] - 0.5) < 1e-6 
        assert abs(data['redchi2tomean_36'][ind1] - 62.5 ) < 1e-6
        # too short for testing delta reliably. Add test for that.

    def test_lin_lc_45(self, data):
        ind1 = np.where(data['YSOVAR2_id'] == '-1003')[0][0]
        assert data['n_45'][ind1] == 5
        assert abs(data['max_45'][ind1] - 13.) < 1e-6
        assert abs(data['min_45'][ind1] - 11.) < 1e-6
        assert abs(data['median_45'][ind1] - 12.) < 1e-6
        assert abs(data['wmean_45'][ind1] - 12.) < 1e-6
        assert abs(data['stddev_45'][ind1]- 1./np.sqrt(2.)) < 1e-6
        assert abs(data['mad_45'][ind1] - 0.5) < 1e-6 
        assert abs(data['redchi2tomean_45'][ind1] - 62.5 ) < 1e-6
        # too short for testing delta reliably. Add test for that.

    def test_one_period(self, data):
        ind1 = np.where(data['YSOVAR2_id'] == '-1500')[0][0]
        ind2 = np.where(data['YSOVAR2_id'] == '-1501')[0][0]
        assert abs(data['period_36'][ind1] -3.) < 0.1
        assert abs(data['period_36'][ind2] - 50.) < 1.
        
    def test_good_period(self, data):
        ind1 = np.where(data['YSOVAR2_id'] == '-1502')[0][0]
        ind2 = np.where(data['YSOVAR2_id'] == '-1503')[0][0]
        assert not np.isfinite(data['good_period'][ind1])
        assert abs(data['good_period'][ind2] - 3.) < .1

    def test_multi_lc(self, data):
        for ind, n01, n05 in zip(['-2000','-2001', '-2002'],
                                 [0,50,10], [0,50,50]):
            print 'Testing lc: ', ind
            i = np.where(data['YSOVAR2_id'] == ind)[0][0]
            lc01 = atlas.merge_lc(data.lclist[i], ['36','45'], t_simul=0.01)
            lc05 = atlas.merge_lc(data.lclist[i], ['36','45'], t_simul=0.05)
            assert len(lc01) == n01
            assert len(lc05) == n05

    def test_cmd_flat(self, data):
        ind = np.where(data['YSOVAR2_id'] == '-2500')[0][0]
        assert abs(data['cmd_m'][ind] - .5) < 0.1
        assert data['cmd_b_error'][ind] < 0.001
        assert data['cmd_m_error'][ind] < 0.001
        assert abs(data['cmd_b'][ind] - 11.5) < 0.001
        assert data['cmd_alpha_error'][ind] < 0.001

    def test_cmd_random(self, data):
        ind = np.where(data['YSOVAR2_id'] == '-2501')[0][0]
        assert data['cmd_b_error'][ind] > 0.01
        assert data['cmd_m_error'][ind] > 0.01
        assert data['cmd_alpha_error'][ind] > 0.01

    def test_cmd_slope(self, data):
        ind = np.where(data['YSOVAR2_id'] == '-2502')[0][0]
        assert abs(data['cmd_m'][ind] - 6.12) < 0.1
        assert data['cmd_b_error'][ind] < 0.001
        assert data['cmd_m_error'][ind] < 0.001
        assert abs(data['cmd_b'][ind] - 5.8) < 0.5
        assert data['cmd_alpha_error'][ind] < 0.001
        assert data['cmd_dominated'][ind] == 'extinc.'

    def test_stetson(self, data):
        ind1 = np.where(data['YSOVAR2_id'] == '-2700')[0][0]
        ind2 = np.where(data['YSOVAR2_id'] == '-2701')[0][0]
        data.calc_stetson('36','45')
        assert data['stetson_36_45'][ind1] < 0.01
        assert data['stetson_36_45'][ind2] - 101.02 < 0.1
        
        
