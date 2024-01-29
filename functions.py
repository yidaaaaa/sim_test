import numpy as np






def get_otf_val(otf, band, cycl, use_att):
    pos = cycl / otf['cyclesPerMicron']
    cpos = pos + 1

    lpos = np.floor(cpos).astype(int)
    hpos = np.ceil(cpos).astype(int)
    f = cpos - lpos

    if use_att:
        retl = otf['valsAtt'][lpos] * (1 - f)
        reth = otf['valsAtt'][hpos] * f
    else:
        retl = otf['vals'][lpos] * (1 - f)
        reth = otf['vals'][hpos] * f
    
    val = retl + reth
    mask = np.ceil(cpos) > otf['sampleLateral']
    val[mask] = 0

    return val

def otf_to_vector(vec, otf, band, kx, ky, use_att, write):
    siz = vec.shape
    w, h = siz[1], siz[0]
    cnt = [s // 2 + 1 for s in siz]
    kx += cnt[1]
    ky += cnt[0]

    x, y = np.meshgrid(np.arange(1, w + 1), np.arange(1, h + 1))
    rad = np.hypot(y - ky, x - kx)
    cycl = rad * otf['cyclesPerMicron']

    mask = cycl > otf['cutoff']
    cycl[mask] = 0

    val = get_otf_val(otf, band, cycl, use_att)
    if write == 0:
        vec = vec * val
    else:
        vec = val

    vec[mask] = 0
    return vec


def write_otf_vector(vec, otf, band, kx, ky):
    return otf_to_vector(vec, otf, band, kx, ky, 0, 1)

