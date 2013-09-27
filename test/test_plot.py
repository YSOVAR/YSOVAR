import os

import pytest

from YSOVAR import plot
from . import outroot

@pytest.mark.usefixtures("data")
class Test_plots():
    def test_lc_plots(self, data):
        plot.make_lc_plots(data, outroot, twinx = True)

    def test_cmd_plots(self, data):
        plot.make_cmd_plots(data, outroot)

    def test_ls_plots(self, data):
        plot.make_ls_plots(data, outroot, 100, 4, 1)
        
    def test_phased_plots(self, data):
        plot.make_phased_lc_cmd_plots(data, outroot)

    def test_latex(self, data):
        output_cols = {'YSOVAR2_id': 'ID in YSOVAR 2 database',
                                  'median_36': 'median [3.6]',
                                  'mad_36': 'medium abs dev [3.6]',
                                  'stddev_36': 'stddev [3.6]',
                                  'median_45': 'median [4.5]',
                                  'mad_45': 'medium abs dev [4.5]',
                                  'stddev_45': 'stddev [4.5]',
                                  'stetson_36_45': 'Stetson [3.6] vs. [4.5]'}
        plot.make_latexfile(data, outroot, 'testatlas', output_cols = output_cols)

