import numpy as np
import pytest
from astropy.table import Table

from YSOVAR import atlas

def test_coord_strhmsdmx2RADEC():
    dat = Table({'r':['10:10:10.234', '20:20:20','0:30:30'],
                    'd':['+10:10:10.234', '+0:10:30','-10:10:10']})
    atlas.coord_strhmsdms2RADEC(dat, ra = 'r', dec = 'd')
    assert np.all((dat['RAdeg']-np.array([152.54264167, 305.08333333,7.625]))<1e-5)
    assert np.all((dat['DEdeg']-np.array([10.16950944, 0.175, -10.16944444]))<1e-5)

def test_coord_hmsdmx2RADEC():
    dat = Table({'r1': [10,20,0.], 'r2':[10, 20., 30], 'r3': [10.234, 20,30.],
                 'd1': [10,0,-10.], 'd2':[10,10.,10], 'd3':[10.234, 30,10.]})
    atlas.coord_hmsdms2RADEC(dat, ra = ['r1','r2','r3'], dec = ['d1','d2','d3'])
    assert np.all((dat['RAdeg']-np.array([152.54264167, 305.08333333,7.625]))<1e-5)
    assert np.all((dat['DEdeg']-np.array([10.16950944, 0.175, -10.16944444]))<1e-5)

def test_coord_strhmsdmx2RADEC():
    dat = Table({'r':['10:10:10.234', '20:20:20','0:30:30'],
                    'd':['+10:10:10.234', '+0:10:30','-10:10:10']})
    atlas.coord_strhmsdms2RADEC(dat, ra = 'r', dec = 'd')
    assert np.all((dat['RAdeg']-np.array([152.54264167, 305.08333333,7.625]))<1e-5)
    assert np.all((dat['DEdeg']-np.array([10.16950944, 0.175, -10.16944444]))<1e-5)


def test_makecrossids():
    d1 = np.rec.array([[0,0.1],[23,45],[45,89],[230,-50], [255,-66], [0,0]], dtype=[('ra', np.float),('dec', np.float)])
    d2 = np.rec.array([[.01,0.9],[23,44.1],[55,89.4],[229.8,-50.1], [23, 44.05], [23.01,44.05], [100,70]], dtype=[('ra', np.float),('dec', np.float)])
    cid = atlas.makecrossids(d1,d2,1.0,'ra','dec','ra','dec')
    assert cid[0] == 0
    assert cid[1] == 1
    assert cid[2] == 2
    assert cid[3] == 3
    assert cid[4] == -99999
    assert cid[5] == -99999
    assert len(cid) == len(d1)

def test_makecrossids_double_match():
    d1 = np.rec.array([[0,0.1],[23,45],[45,89],[230,-50], [255,-66], [0,0]], dtype=[('ra', np.float),('dec', np.float)])
    d2 = np.rec.array([[.01,0.9],[23,44.1],[55,89.4],[229.8,-50.1], [23, 44.05], [23.01,44.05], [100,70]], dtype=[('ra', np.float),('dec', np.float)])
    cid = atlas.makecrossids(d1,d2,1.0,'ra','dec','ra','dec', double_match=True)
    assert cid[0] == 0
    assert cid[1] == 1
    assert cid[2] == 2
    assert cid[3] == 3
    assert cid[4] == -99999
    assert cid[5] == 0
    assert len(cid) == len(d1)
 
    
def test_isoy2radec():
    ra, dec = atlas.IAU2radec('SSTYSV_J162722.02-242114.5')
    assert np.abs(ra - 246.84175) < 0.00001
    assert np.abs(dec + 24.354028) < 0.00001
    ra, dec = atlas.IAU2radec('SSTYSV_J162722.02+242114.5')
    assert np.abs(dec - 24.354028) < 0.00001

def test_makecrossids_all():
    data1 = Table({'ra':np.array([0.,10.,15.]), 'dec':np.array([0.,0.,0.])})
    data2 = Table({'ra':np.array([0.,0,0,0,0]), 'dec':np.array([0.,.1,.5,1,5])})
    ids = atlas.makecrossids_all(data1, data2, 1., ra1 = 'ra', dec1 = 'dec', ra2 = 'ra', dec2 = 'dec')
    assert len(ids) == len(data1)
    assert np.all(ids[0] == [0, 1, 2, 3])
    for i in np.arange(1, len(data1)):
        assert len(ids[i]) == 0

@pytest.mark.usefixtures("data")
class Test_atlas_functions():
    def test_get_row(self, data):
        t4 = data[4]
        assert t4.lclist[0] == data.lclist[4]

    def test_get_slice(self, data):
        t4step = data[0:10:4]
        assert t4step.lclist[1]  == data.lclist[4]


@pytest.mark.usefixtures("data")
class Test_expected_results():
    def test_ignore_short_lcs(self, data):
        assert '-998' not in set(data['YSOVAR2_id'])
        
    def test_const_lc_36(self, data):
        ind1 = np.where(data['YSOVAR2_id'] == '-1000')[0][0]
        assert data['n_36'][ind1] == 6
        assert abs(data['max_36'][ind1] - 12.) < 1e-6
        assert abs(data['min_36'][ind1] - 12.) < 1e-6
        assert abs(data['median_36'][ind1] - 12.) < 1e-6
        assert abs(data['wmean_36'][ind1] - 12.) < 1e-6
        assert abs(data['stddev_36'][ind1]) < 1e-6
        assert abs(data['mad_36'][ind1]) < 1e-6
        assert abs(data['delta_36'][ind1]) < 1e-6
        assert abs(data['redchi2tomean_36'][ind1]) < 1e-6
        assert data['cmd_dominated_36_45'][ind1] == ''

    def test_const_lc_45(self, data):
        ind1 = np.where(data['YSOVAR2_id'] == '-1001')[0][0]
        assert data['n_45'][ind1] == 6
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
        assert data['n_36'][ind1] == 6
        assert abs(data['max_36'][ind1] - 13.) < 1e-6
        assert abs(data['min_36'][ind1] - 11.) < 1e-6
        assert abs(data['median_36'][ind1] - 12.) < 1e-6
        assert abs(data['wmean_36'][ind1] - 12.) < 1e-6
        assert abs(data['stddev_36'][ind1]- 1./np.sqrt(2.)) < 1e-3
        assert abs(data['mad_36'][ind1] - 0.5) < 1e-6 
        assert abs(data['redchi2tomean_36'][ind1] - 50 ) < 1e-6

    def test_lin_lc_45(self, data):
        ind1 = np.where(data['YSOVAR2_id'] == '-1003')[0][0]
        assert data['n_45'][ind1] == 6
        assert abs(data['max_45'][ind1] - 13.) < 1e-6
        assert abs(data['min_45'][ind1] - 11.) < 1e-6
        assert abs(data['median_45'][ind1] - 12.) < 1e-6
        assert abs(data['wmean_45'][ind1] - 12.) < 1e-6
        assert abs(data['stddev_45'][ind1]- 1./np.sqrt(2.)) < 1e-3
        assert abs(data['mad_45'][ind1] - 0.5) < 1e-6 
        assert abs(data['redchi2tomean_45'][ind1] - 50 ) < 1e-6

    def test_lin_lc_36_long(self, data):
        ind1 = np.where(data['YSOVAR2_id'] == '-1004')[0][0]
        assert data['n_36'][ind1] == 100
        assert abs(data['max_36'][ind1] - 13.) < 1e-6
        assert abs(data['min_36'][ind1] - 11.) < 1e-6
        assert abs(data['wmean_36'][ind1] - 12.) < 1e-6
        assert abs(data['redchi2tomean_36'][ind1] - 50 ) < 1e-2
        assert abs(data['delta_36'][ind1] - 1.6) < 0.03

    def test_mag_and_err_not_reordered(self, data):
        ind = np.where(data['YSOVAR2_id'] == '-1010')[0][0]
        assert abs(data['redchi2tomean_36'][ind] - 0.966514) < 1e-5
        assert abs(data['wmean_36'][ind] - 12.3179614) < 1e-5

    def test_one_period(self, data):
        ind1 = np.where(data['YSOVAR2_id'] == '-1500')[0][0]
        ind2 = np.where(data['YSOVAR2_id'] == '-1501')[0][0]
        assert abs(data['period_36'][ind1] -3.) < 0.1
        assert abs(data['period_36'][ind2] - 50.) < 1.
        assert abs(data['good_period'][ind1] -3.) < 0.1
        assert abs(data['good_period'][ind2] - 50.) < 1.

        
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
        assert abs(data['cmd_m_36_45'][ind]) < 0.1
        assert data['cmd_b_error_36_45'][ind] < 0.001
        assert data['cmd_m_error_36_45'][ind] < 0.001
        assert abs(data['cmd_b_36_45'][ind] - 12.) < 0.001
        assert data['cmd_alpha_error_36_45'][ind] < 0.001

    def test_cmd_random(self, data):
        ind = np.where(data['YSOVAR2_id'] == '-2501')[0][0]
        assert data['cmd_b_error_36_45'][ind] > 0.01
        assert data['cmd_m_error_36_45'][ind] > 0.01
        assert data['cmd_alpha_error_36_45'][ind] > 0.01
        assert data['isnormal_36'][ind] < 0.5

    def test_cmd_slope(self, data):
        ind = np.where(data['YSOVAR2_id'] == '-2502')[0][0]
        assert abs(data['cmd_m_36_45'][ind] - 6.12) < 0.1
        assert data['cmd_b_error_36_45'][ind] < 0.001
        assert data['cmd_m_error_36_45'][ind] < 0.001
        assert abs(data['cmd_b_36_45'][ind] - 5.8) < 0.5
        assert data['cmd_alpha_error_36_45'][ind] < 0.001
        assert data['cmd_dominated_36_45'][ind] == 'extinc.'

    def test_stetson(self, data):
        data.calc('stetson', ['36', '45'])
        for ind, res in zip(['-2700', '-2701', '-2702', '-2703', '-2704'], [0.0,0.0, 10.1012, 10.1012, -10.1012]):
            print 'number:', ind, ' -- expected: ', res
            i = np.where(data['YSOVAR2_id'] == ind)[0][0]
            assert np.abs(data['stetson_36_45'][i] - res) < 0.01

    def test_data_preprocessor(self, data):
        def smooth(source):
            lc = source.lclist[0]
            w = np.ones(5)
            if 'm36' in lc and len(lc['m36'] > 10):
                lc['m36'] = np.convolve(w/w.sum(),lc['m36'],mode='valid')
                # need to keep m36 and t36 same length
                lc['t36'] = lc['t36'][2:-2]
                lc['m36_error'] = lc['m36_error'][2:-2]
            return source

        data.calc('stddev', '36', data_preprocessor = smooth, colnames = 'stdsmooth')
        ind = np.where(data['YSOVAR2_id'] == '-2501')[0][0]
        assert data['stdsmooth_36'][ind] < data['stddev_36'][ind]

    def test_timefiltering(self, data):
        data.calc('n','36', timefilter = lambda x : x < 55303.,
                  colnames = ['filtern'])
        ind = np.where(data['YSOVAR2_id'] == '-1000')[0][0]
        assert data['filtern_36'][ind] == 3
        
def test_add_catalog():
    twosources = [{'ra':1.0, 'dec':0.0, 'YSOVAR2_id': -1,'IAU_NAME':'Test1'}, 
                       {'ra':1e-6, 'dec':0.0, 'YSOVAR2_id': -1,'IAU_NAME':'Test2'}]
    addsource = Table({'RA':[0.0,5.], 'DE':[0.0,5.], 'DAT': [1.5, 2.5]})
    cat = atlas.YSOVAR_atlas(lclist = twosources)
    cat.add_catalog_data(addsource, ra1='ra', dec1='dec', ra2='RA', dec2='DE')
    assert np.isnan(cat['DAT'][0])
    assert cat['DAT'][1] == 1.5
    assert len(cat) == 2

def test_add_catalog_warning(recwarn):
    twoclosesources = [{'ra':0.0, 'dec':0.0, 'YSOVAR2_id': -1,'IAU_NAME':'Test1'}, 
                       {'ra':1e-6, 'dec':0.0, 'YSOVAR2_id': -1,'IAU_NAME':'Test2'}]
    addsource = Table({'RA':[0.0], 'DE':[0.0], 'DAT': [42]})
    cat = atlas.YSOVAR_atlas(lclist = twoclosesources)
    cat.add_catalog_data(addsource, ra1='ra', dec1='dec', ra2='RA', dec2='DE', double_match=True)
    w = recwarn.pop(UserWarning)
    assert issubclass(w.category, UserWarning)
    assert 'add_catalog_data: The following sources in the input catalog are matched to more than one source in this atlas:' in str(w.message)
    assert '[0]' in str(w.message)
    assert np.all(cat['DAT'] == 42)
    assert len(cat) == 2
