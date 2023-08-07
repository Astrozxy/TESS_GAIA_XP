#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import astropy.io.fits as fits
from astropy.table import Table, vstack as table_vstack
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as units
from astropy.time import Time

import h5py

import os
from glob import glob
from tqdm import tqdm

from time import perf_counter


import crossmatch

# Loading Gaia XP parameters from params/
data = {}
for i in range(10):
    name = f'params/stellar_params_catalog_0{i}.h5'
    with h5py.File(name, 'r') as f:
        for key in tqdm(f):
            if key not in data:
                data[key] = []
            data[key].append(f[key][:])
for key in data:
    data[key] = np.concatenate(data[key], axis=0)

tess_id, tess_ra, tess_dec = np.loadtxt('TESS_pc_infor.txt').T


def main():
    match_radius = 0.5*units.arcsec
    tess_nside = 1024
    gaia_nside = 512

    print(f'Partitioning TESS into HEALPix pixels (nside={tess_nside})...')
    t0 = perf_counter()
    ra = tess_ra
    dec = tess_dec
    
    tess_hpxcat = crossmatch.HEALPixCatalog(
        tess_ra*units.deg,
        tess_dec*units.deg,
        tess_nside,
        show_progress=True
    )
    t1 = perf_counter()
    print(f'  --> {t1-t0:.5f} s')

    # Load metadata on Gaia BP/RP spectra
    # Gaia coordinates
    print(f'Partitioning Gaia into HEALPix pixels (nside={gaia_nside})...')
    t0 = perf_counter()
    gaia_hpxcat = crossmatch.HEALPixCatalog(
        data['ra']*units.deg,
        data['dec']*units.deg,
        gaia_nside
    )
    
    t1 = perf_counter()
    print(f'  --> {t1-t0:.5f} s')

    # Match to unWISE
    print('Calculating crossmatch ...')
    t0 = perf_counter()
    idx_tess, idx_gaia, sep2d = crossmatch.match_catalogs(
        tess_hpxcat,
        gaia_hpxcat,
        match_radius
    )
    t1 = perf_counter()
    print(f'  --> {t1-t0:.5f} s')
    source_id = data['gdr3_source_id'][idx_gaia]

    print(
        f'{len(idx_gaia)} of {len(source_id)} '
        'Gaia sources have TESS match.'
    )

    tess_matches = tess_id[idx_tess]

    # Save matches
    kw = dict(compression='lzf', chunks=True)
    with h5py.File('matched_xp_TESS.h5', 'w') as f:
        f.create_dataset('gdr3_source_id', data=source_id, **kw)
        f.create_dataset('gaia_index', data=idx_gaia, **kw)
        f.create_dataset('sep_arcsec', data=sep2d.to('arcsec').value, **kw)
        tess_matches.write(
            f, path='tess_id',
            append=True,
            compression='lzf'
        )

    return 0


if __name__ == '__main__':
    main()


