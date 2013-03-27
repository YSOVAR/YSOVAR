import sys
sys.path.append('/data/hguenther/Dropbox/code/python/YSOVAR')
sys.path.append('/data/hguenther/Dropbox/code/python/cluster')
sys.path.append('/data/hguenther/Dropbox/code/python/')

#import astropy.io.ascii as ascii
import ysovar_atlas as atlas
import plot_atlas
import lightcurves as lc

inroot = '/data/hguenther/misc/YSOVAR/RhoOph/'
outroot_ysos = '/data/hguenther/misc/YSOVAR/RhoOph/ysos_plots/'
outroot_stars = '/data/hguenther/misc/YSOVAR/RhoOph/stars_plots/'
outroot_overview = '/data/hguenther/misc/YSOVAR/RhoOph/overview_plots/'

# read basic data
data = atlas.dict_from_csv(inroot + 'irac.csv', match_dist = 0.)
atlas.check_dataset(data)

L1688 = atlas.YSOVAR_atlas(lclist = data)

t4 = L1688[4]
assert t4.lclist['m36'][5] == data[4]['m36'][5]

t4 = L1688[0:100:4]
assert t4.lclist[1]['m36'][5]  == data[4]['m36'][5]


