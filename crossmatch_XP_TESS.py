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


def main():
    match_radius = 0.5*units.arcsec
    TESS_nside = 1024
    gaia_nside = 512

    TESS = Table.read('./TESS_pc_infor.txt', format='ascii')

    print(f'Partitioning TESS into HEALPix pixels (nside={TESS_nside})...')
    t0 = perf_counter()
    c_sch = SkyCoord(frame="icrs", ra=TESS['RA Degrees']*units.deg, dec=TESS['DEC Degrees']*units.deg)
    ra = c_sch.ra
    dec = c_sch.dec

    tess_hpxcat = crossmatch.HEALPixCatalog(
        ra,
        dec,
        TESS_nside,
        show_progress=True
    )
    t1 = perf_counter()
    print(f'  --> {t1-t0:.5f} s')

    # Loop over Gaia BP/RP spectra metadata files, and match each to TESS
    out_dir = 'data/XP_TESS_match/'
    fnames = glob(os.path.join('data/xp_continuous_metadata/xp_metadata_*-*.h5'))
    fnames.sort()

    for fn in tqdm(fnames):
        # Skip file if matches have already been stored
        fid = fn.split('_')[-1].split('.')[0]
        out_fname = os.path.join(
            out_dir,
            f'xp_tess_match_{fid}.h5'
        )
        if os.path.exists(out_fname):
            continue

        print(f'Finding TESS matches for {fn} ...')

        # Load metadata on Gaia BP/RP spectra
        gaia_meta = Table.read(fn)
        
        # Replace NaN proper motions with 0
        pmra_cosdec = gaia_meta['pmra']
        pmdec = gaia_meta['pmdec']
        idx_no_pm = ~np.isfinite(pmra_cosdec) | ~np.isfinite(pmdec)
        pmra_cosdec[idx_no_pm] = 0.
        pmdec[idx_no_pm] = 0.

        gaia_coords = SkyCoord(
            gaia_meta['ra'],
            gaia_meta['dec'],
            pm_ra_cosdec=pmra_cosdec,
            pm_dec=pmdec,
            obstime=Time('2016-01-01 12:00:00'),
            frame='icrs'
        )

        # Convert to LAMOST epoch (J2000)
        gaia_coords = gaia_coords.apply_space_motion(
            new_obstime=Time('2000-01-01 12:00:00')
        )
        pm_err = np.sqrt(
            gaia_meta['pmra_error']**2
          + gaia_meta['pmdec_error']**2
        )
        pm_err_small = (pm_err < 0.5*(match_radius/(16*units.yr)))
        pm_err_small &= ~idx_no_pm # There has to be a proper motion!
        

        # Gaia coordinates
        print(f'Partitioning Gaia into HEALPix pixels (nside={gaia_nside})...')
        t0 = perf_counter()             
        
             
        gaia_hpxcat = crossmatch.HEALPixCatalog(
            gaia_coords.ra,
            gaia_coords.dec,
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

        has_match = pm_err_small[idx_gaia]
        if len(has_match)<1:
            continue
        idx_gaia = idx_gaia[has_match]
        idx_tess = idx_tess[has_match]
        sep2d = sep2d[has_match]
        source_id = gaia_meta['source_id'][idx_gaia]      
        
        print(
            f'{len(idx_gaia)} of {len(gaia_meta)} '
            'Gaia sources have TESS match.'
        )

        tess_matches = TESS[idx_tess]

        # Save matches
        kw = dict(compression='lzf', chunks=True)
        with h5py.File(out_fname, 'w') as f:
            f.create_dataset('gdr3_source_id', data=source_id, **kw)
            f.create_dataset('gaia_index', data=idx_gaia, **kw)
            f.create_dataset('sep_arcsec', data=sep2d.to('arcsec').value, **kw)
            tess_matches.write(
                f, path='TESS_data',
                append=True,
                compression='lzf'
            )

    return 0


if __name__ == '__main__':
    main()
