'''
Check if sources are correctly cross-matched.
check how much data is added, if the full database is used  compared with a filter

I recommend to run this as copy & paste and look at the results interactively.
'''
import ysovar_atlas as atlas

def min_delta_t(data):
    '''Return the minimum delta t for any source in data
    
    If the matching circle is made too big, then at some point to seperate
    source will be joined. One hint that this happens would be to find near
    simulteneous entries
    
    Parameters
    ----------
    data : list of dicts
        entry data in YSOVAR form with 't1' and 't2' tags
    
    Returns
    -------
    min1, min2 : tuple of np.arrays
        minimum delta t in each source
    '''
    min1 = np.zeros(len(data))
    min2 = zeros_like(min1)
    for i, d in enumerate(data):
        if 't1' in d.keys():
            if len(d['t1']) > 1:
                min1[i] = np.min(np.diff(d['t1']))
            else:
                min1[i] = np.inf
        else:
             min1[i] = np.nan
        if 't2' in d.keys():
            if len(d['t2']) > 1:
                min2[i] = np.min(np.diff(d['t2']))
            else:
                min2[i] = np.inf
        else:
             min2[i] = np.nan
    return min1, min2

inroot = '/data/hguenther/misc/YSOVAR/RhoOph/lightcurves/'
(ysovar1, ysovar2) = atlas.readdata(inroot + 'fulltimeseriesi1.idlsave', inroot + 'fulltimeseriesi2_filter.idlsave')
(y1, y2) = atlas.readdata(inroot + 'fulltimeseriesi1.idlsave', inroot + 'fulltimeseriesi2.idlsave')

### compare number of datapoints with and without Louisas > 5 filtering
data = atlas.make_dict(ysovar1, ysovar2)  # ysovar1 and ysovar2 are the sets from Louisa
dat = atlas.make_dict(y1, y2)  # y1 and y2 are the unfiltered sets
test1 = np.zeros(len(dat))
test2 = np.zeros(len(dat))
for i,d in enumerate(dat):
    if 't1' in d.keys():
        test1[i] = len(d['t1'])
    if 't2' in d.keys():
        test2[i] = len(d['t2'])

test1a = np.zeros(len(data))
test2a = np.zeros(len(data))
for i,d in enumerate(data):
    if 't1' in d.keys():
        test1a[i] = len(d['t1'])
    if 't2' in d.keys():
        test2a[i] = len(d['t2'])


# compare number of entries and sources
print '# entries old ',' new ','# number of ditinct sources old',' new '
len(ysovar1), len(y1), np.sum(test1a >=5), np.sum(test1 >=5)
len(ysovar2), len(y2), np.sum(test2a >=5), np.sum(test2 >=5)

#compare individual sources
radecold = atlas.radec_from_dict(data, RA = 'ra', DEC = 'dec')
radecnew = atlas.radec_from_dict(dat, RA = 'ra', DEC = 'dec')

oldinnew = atlas.makecrossids(radecnew, radecold, 0.5/3600, ra1='RA', dec1 = 'DEC', ra2='RA', dec2='DEC')

checkallthere = oldinnew.copy()
len(data), len(dat), len(oldinnew),  np.sum(checkallthere >=0)
checkallthere.sort()
# see if there is a double entry.
# The first lines contain unmatched sources -99999, but towards the end are
# the index values of sources with more than one match.
np.where(np.diff(checkallthere)==0)
# check out those cases manually to see if it's all OK.
np.where(oldinnew == checkallthere[1506])
# >>> (array([ 637, 1302]),)
dat[637]
dat[1302]
dist_radec(dat[637]['ra'],dat[637]['dec'], [dat[1302]['ra']], [dat[1302]['dec']], unit = 'deg') * 3600.
''' -> dat[637] is compased of two sources.
 in 1302 is too far away to be matched to 637 in atlas.add_ysovar_mags
 but then later in atlas.dict_cleanup we take the average of both RA, DEC values
 and that brings the mean RA, DEC a little closer to 1302, so that this get matched
 now in atlas.makecrossids
 In my cluster these cases are few, so I can choose to ignore them'''

# How much extra information did I gain?

count = np.zeros((len(oldinnew), 4))

for i, ind in enumerate(oldinnew):
    if 't1' in dat[i].keys(): count[i,0] = len(dat[i]['t1'])
    if 't2' in dat[i].keys(): count[i,2] = len(dat[i]['t2'])
    oldi = oldinnew[i]
    if oldi >=0:
        if 't1' in data[oldi].keys(): count[i,1] = len(data[oldi]['t1'])
        if 't2' in data[oldi].keys(): count[i,3] = len(data[oldi]['t2'])

print 'How does the number of datapoints change?'
print 'IRAC 1          IRAC 2'
print 'new -- old    new --- old'
print 'Sources > 5 datapoints which change number of datapoint in IRAC 1'
ind = np.where((count[:,0] > count[:,1]) & (count[:,0] > 5))
count[ind,:]
print 'Sources > 5 datapoints which change number of datapoint in IRAC 2'
ind = np.where((count[:,2] > count[:,3]) & (count[:,2] > 5))
count[ind,:]

# Check for double entries
# This could be an indication that the matchign radius is too big
# However, there are already double entries in the original data before any merging happens.
m1, m2 = min_delta_t(dat)

print np.nanmin(m1), np.nanmin(m2)
print np.where(m1 == 0)
print np.where(m2 == 0)
