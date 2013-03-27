def spectra_check(infos, ra, dec, files, night_id, radius):
    # only for IRAS 20050: check if there's a hectospec spectrum available for the sources, and if yes, add the filename of the spectrum to the info array.
    for i in np.arange(0,len(infos)):
        distance = np.sqrt((infos.ra_guenther[i] - ra)**2 + (infos.dec_guenther[i] - dec)**2)
        min_ind = np.where(distance == min(distance))[0]
        if min(distance) <= radius:
            if night_id == 1:
                infos.hectospec_night1[i] = str(files[min_ind][0])
            if night_id == 2:
                infos.hectospec_night2[i] = str(files[min_ind][0])
        
    return infos

def format_or_string(format_str):
        def func(val):
            if isinstance(val, basestring):
                return val
            else:
                return format_str % val
        return func
