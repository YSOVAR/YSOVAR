import os
import numpy as np
import ysovar_atlas as atlas
import plot_atlas
import pytest

outroot = 'test/testout'


@pytest.fixture(scope = 'module')
def dat():
    plot_atlas.filetype = ['.pdf']
    
    if not os.path.exists(outroot):
        os.makedirs(outroot)
    datadict = atlas.dict_from_csv('test/data/expected_results.csv', floor_error = {'IRAC1': 0., 'IRAC2': 0.})
    dat = atlas.YSOVAR_atlas(lclist = datadict)
    dat.calc_ls('36', 100)
    dat.calc_ls('45', 100)
    dat.is_there_a_good_period(20, 1,100)
    dat.cmd_slope_odr()
    dat.good_slope_angle()
    dat.cmd_dominated_by()
    dat.calc_stetson('36','45')
    return dat

@pytest.mark.usefixtures("dat")
class Test_plots():
    def test_lc_plots(self, dat):
        plot_atlas.make_lc_plots(dat, outroot, twinx = True)

    def test_cmd_plots(self, dat):
        plot_atlas.make_cmd_plots(dat, outroot)

    def test_ls_plots(self, dat):
        plot_atlas.make_ls_plots(dat, outroot, 100, 4, 1)
        
    def test_phased_plots(self, dat):
        plot_atlas.make_phased_lc_cmd_plots(dat, outroot)

    def test_latex(self, dat):
        output_cols = {'YSOVAR2_id': 'ID in YSOVAR 2 database',
                                  'median_36': 'median [3.6]',
                                  'mad_36': 'medium abs dev [3.6]',
                                  'stddev_36': 'stddev [3.6]',
                                  'median_45': 'median [4.5]',
                                  'mad_45': 'medium abs dev [4.5]',
                                  'stddev_45': 'stddev [4.5]',
                                  'stetson_36_45': 'Stetson [3.6] vs. [4.5]'}
        plot_atlas.make_latexfile(dat, outroot, 'testatlas', output_cols = output_cols)

