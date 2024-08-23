import numpy as np
from astropy.io import fits
import healpy as hp
import pickle

import pandas as pd
from collections import OrderedDict

import sys
sys.path.insert(0, '/global/homes/q/qhang/desc/notebooks_for_analysis/')
import spatial_var_functions as svf
import measure_properties_with_systematics as mp

import tables_io

# for (de-)reddening:
import dustmaps
from dustmaps.sfd import SFDQuery
from astropy.coordinates import SkyCoord
from dustmaps.config import config
config['data_dir'] = '/global/cfs/cdirs/lsst/groups/PZ/PhotoZDC2/run2.2i_dr6_test/TESTDUST/mapdata' 
#update this path when dustmaps are copied to a more stable location!

import desc_bpz


# here put together a function to assign the error and save, then loop 10 times

def assign_pixels_to_gals(Ngal, pixels, random_seed=10):
    # set random seed
    np.random.seed(random_seed)
    
    npix = len(pixels)
    print('Average no. of gal per pix: ', Ngal/npix)
    
    # generate a uniform distribution between 0 to 1 for each galaxy
    rand = np.random.uniform(size=Ngal)
    
    pixel_index = np.digitize(rand, np.linspace(0,1, npix + 1))
    pixel_index -= 1

    assigned_pixels = pixels[pixel_index]
    return assigned_pixels


def assign_obs_cond_to_gals(assigned_pixels, obs_cond, mask, sys, bands):
    
    assigned_obs_cond = {}
    
    for key in sys:
        assigned_obs_cond[key] = {}
        for b in bands:
            temp = np.zeros(len(mask))
            temp[mask.astype(bool)] = obs_cond[key][b]
            assigned_obs_cond[key][b] = temp[assigned_pixels]
    
    return assigned_obs_cond
   

def apply_galactic_extinction(data, assigned_pixels, bands, nside=128):
    
    # set the A_lamba/E(B-V) values for the six LSST filters 
    band_a_ebv = np.array([4.81,3.64,2.70,2.06,1.58,1.31])
    
    # turn assigned pixels to ra, dec:
    ra, dec = hp.pix2ang(nside, assigned_pixels, lonlat=True)
    coords = SkyCoord(ra, dec, unit = 'deg', frame='fk5')
    # compute EBV
    sfd = SFDQuery()
    ebvvec = sfd(coords)

    for ii, b in enumerate(bands):
        data[b] = (data[b].copy())+ebvvec*band_a_ebv[ii]

    return data
    
    
def compute_Nsigma_limit(assigned_obs_cond, obs_cond, m5key='coaddm5_per_visit'):
    assigned_obs_cond['sigLim'] = {}
    for ii, b in enumerate(bands):
        NSR = 1/obs_cond['sigLim']
        Nvis = assigned_obs_cond['nvis'][b]
        nsrRandSingleExp = np.sqrt((NSR**2-obs_cond['sigmasys']**2)*Nvis)
        
        gamma = obs_cond['gamma'][b]
        # then solve for the quadratic equation:
        x = (
            (gamma - 0.04)
            + np.sqrt((gamma - 0.04) ** 2 + 4 * gamma * nsrRandSingleExp**2)
        ) / (2 * gamma)
        
        m5 = assigned_obs_cond[m5key][b]
        assigned_obs_cond['sigLim'][b] = m5 + 2.5 * np.log10(x)
    
    return assigned_obs_cond
    
    
def get_semi_major_minor(data, scale=1):
    
    # check data columns:
    if "ellipticity" in data:
        q = (1 - data['ellipticity'])/(1 + data['ellipticity'])
        ai = data['size']
        bi = ai*q
    elif "major" in data:
        ai = data["major"]
        bi = data["minor"]

    ai = ai.to_numpy()*scale
    bi = bi.to_numpy()*scale
    
    return ai, bi


def get_area_ratio_auto(majors, minors, theta, airmass, meanApsf=False, theta_eff=False):
    """Get the ratio between PSF area and galaxy aperture area for "auto" model.
    Parameters
    ----------
    majors : np.ndarray
        The semi-major axes of the galaxies in arcseconds
    minors : np.ndarray
        The semi-minor axes of the galaxies in arcseconds
    theta : np.ndarray
        Seeing in arcseconds
    airmass : np.ndarray
        Airmass
    meanApsf : bool
        Tag for whether to compute the mean PSF size for A_psf, 
        A_ap will still be computed using PSF size for each pixel.
        This is added to remove the 1/A_psf scaling that does not
        correlate perfectly with the depth.
    theta_eff: bool
        Tag for whether the seeing is already accounting for
        the dependence on airmass, i.e. if the given theta is
        theta_effective. If set to True, will not include the
        X^0.6 factor. The input airmass values are ignored.
    Returns
    -------
    np.ndarray
        The ratio of aperture size to PSF size for each band and galaxy.
    """
    
    # get the psf size for each band
    #psf_size = np.array([theta[band] for band in bands])
    #theta will be available for each galaxy
    if theta_eff == False:
        psf_size = theta*airmass**0.6
    elif theta_eff == True:
        psf_size = theta
    
    # convert PSF FWHM to Gaussian sigma
    psf_sig = psf_size / 2.355
    
    if meanApsf==True:
        # add a mean psf size computed by the mean of the i-band quantile
        # in case some pixels are nan !(happens to Y1 u-band, not sure why)
        ind = ~np.isnan(psf_size)
        mean_psf_size = np.mean(psf_size[ind])
        mean_psf_sig = mean_psf_size / 2.355
        # calculate the area of the psf in each band
        A_psf = np.pi * mean_psf_sig**2
    if meanApsf==False:
        # calculate the area of the psf in each band
        A_psf = np.pi * psf_sig**2
    
    # calculate the area of the galaxy aperture in each band
    a_ap = np.sqrt(psf_sig ** 2 + (2.5 * majors) ** 2)
    b_ap = np.sqrt(psf_sig ** 2 + (2.5 * minors) ** 2)

    A_ap = np.pi * a_ap * b_ap

    # return their ratio
    return A_ap / A_psf


def compute_mag_err(mag, ai, bi, assigned_obs_cond, obs_cond, band='u', m5key='coaddm5_per_visit',
                   highSNR=False, meanApsf=False, theta_eff=False):
    
    A_ratio = get_area_ratio_auto(ai, bi, assigned_obs_cond['theta'][band], 
                                  assigned_obs_cond['airmass'][band],
                                  meanApsf=meanApsf, theta_eff=theta_eff)
    nsr = svf.compute_sigma_rand_sq(
        assigned_obs_cond[m5key][band], 
        mag, 
        obs_cond['gamma'][band],
        assigned_obs_cond['nvis'][band],  
        A_ratio=A_ratio
    )
    err = svf.compute_magerr_lsstmodel(
        nsr, 
        obs_cond['sigmasys'], 
        highSNR=highSNR
    )
    magerr = err*obs_cond['calibration']['magerrscale'][band]
        
    return magerr


def assign_new_mag_magerr(data, magerr, ai, bi, assigned_obs_cond, 
                          obs_cond, bands, rng, 
                          m5key='coaddm5_per_visit', 
                          meanApsf=False, theta_eff=False):

    """default is highSNR=False"""
    
    #totObsIndex = 1
    ObsMags = np.zeros((len(data), len(bands)))
    ObsMagsErr = np.zeros((len(data), len(bands)))
     
    for ii, b in enumerate(bands):

        nsr = magerr[b]
        mags = data[b].to_numpy()

        # calculate observed magnitudes
        fluxes = 10 ** (mags / -2.5)
        obsFluxes = fluxes * (1 + rng.normal(scale=nsr))
        
        with np.errstate(divide="ignore"):
            newmags = -2.5 * np.log10(np.clip(obsFluxes, 0, None))

        #index for selecting samples within the sigLim
        #idx = newmags>assigned_obs_cond['sigLim'][b]
        
        # new magnitudes:
        ObsMags[:,ii] = np.copy(newmags)
        #ObsMags[idx,ii] = np.nan
        
        # new errors:
        ObsMagsErr[:,ii] = compute_mag_err(newmags, ai, bi, assigned_obs_cond, 
                             obs_cond, band=b, m5key=m5key, 
                             meanApsf=meanApsf, theta_eff=theta_eff)
        #ObsMagsErr[idx,ii] = 2.5 * np.log10(1 +  1/obs_cond['sigLim'])
        
        # flag all non-detections with the ndFlag
        # calculate SNR
        #if self.params.highSNR:
        #snr = 1 / obsMagErrs
        #else:
        snr = 1 / (10 ** (ObsMagsErr[:,ii] / 2.5) - 1)

        # flag non-finite mags and where SNR is below sigLim
        idx = (~np.isfinite(ObsMags[:,ii])) | (snr < obs_cond['sigLim'])
        ObsMags[idx,ii] = np.nan
        
        if obs_cond['sigLim']>0:
            ObsMagsErr[idx,ii] = 2.5*np.log10(1/obs_cond['sigLim']+1)
        else:
            ObsMagsErr[idx,ii] = np.nan  
        if b=='i': 
            # only count i-band non-detection
            totObsIndex = ~idx

    return totObsIndex, ObsMags, ObsMagsErr


def join_tables_and_save(data, ObsMags, ObsMagsErr, pixels, totObsIndex, outfile, bands):

    df = data.copy()
    magDf = pd.DataFrame(
                ObsMags, columns=[f"ObsMag_{band}" for band in bands], index=data.index
            )
    errDf = pd.DataFrame(
                ObsMagsErr, columns=[f"ObsMagErr_{band}" for band in bands], index=data.index
            )
    pixDf = pd.DataFrame(
                pixels, columns = ['pixels'], index=data.index,
    )

    # let's not save the undegraded bands; save true redshifts, pixels, and degraded magnitudes etc.
    obsCatalog = pd.concat([df["redshift"], magDf], axis=1)
    obsCatalog = pd.concat([obsCatalog, errDf], axis=1)
    obsCatalog = pd.concat([obsCatalog, pixDf],axis=1)

    #finally select the indices to use:
    print("Total detection in i-band: ", sum(totObsIndex))
    #savecat = obsCatalog.loc[totObsIndex, :]

    #Let's not cut any objects!
    svf.dump_save(obsCatalog, outfile)
    
    print(f"Saved: {outfile}.")
    
    
def obs_cond_pipeline(data, usepixels, random_seed, rng, obs_cond, mask, sys, bands, outfile,
                      m5key='coaddm5_per_visit', meanApsf=False, theta_eff=False):
    """default is highSNR=False"""
    Ngal = len(data)

    assigned_pixels = assign_pixels_to_gals(Ngal, usepixels, random_seed=random_seed)
    #print('flag1')

    assigned_obs_cond = assign_obs_cond_to_gals(assigned_pixels, obs_cond, mask, sys, bands)
    #print('flag2')
    
    # update the limiting magnitudes:
    #assigned_obs_cond = compute_Nsigma_limit(assigned_obs_cond, obs_cond, m5key=m5key)
    
    # apply galactic extinction:
    data = apply_galactic_extinction(data, assigned_pixels, bands)
    
    # assign photo-z errors:
    scale = obs_cond['calibration']['abscale']
    ai, bi = get_semi_major_minor(data, scale=scale)
    
    nsr = {}
    for b in bands:
        mag = data[b].to_numpy() # this step we want snr rather than sigma to be passed onto the next
        nsr[b] = compute_mag_err(mag, ai, bi, assigned_obs_cond, 
                                 obs_cond, band=b, 
                                 m5key=m5key, highSNR=True,
                                 meanApsf=meanApsf, theta_eff=theta_eff)# returning nsr
    #print('flag3')

    # apply error to get new magnitudes, compute new magnitudes, 
    # and cut objects beyond magnitude limits:
    rng = np.random.default_rng(10)
    totObsIndex, ObsMags, ObsMagsErr = assign_new_mag_magerr(data, nsr, ai, bi, 
                                                             assigned_obs_cond, 
                                                             obs_cond, bands, rng, 
                                                             m5key = m5key,
                                                             meanApsf=meanApsf, 
                                                             theta_eff=theta_eff)
    #print('flag4')
    join_tables_and_save(data, ObsMags, ObsMagsErr, assigned_pixels, totObsIndex, outfile, bands)
    
    
# de-redden:
def deredden_galaxy(data, bands, nside=128):
    
    # set the A_lamba/E(B-V) values for the six LSST filters 
    band_a_ebv = np.array([4.81,3.64,2.70,2.06,1.58,1.31])
    
    # turn assigned pixels to ra, dec:
    assigned_pixels = data["pixels"]
    ra, dec = hp.pix2ang(nside, assigned_pixels, lonlat=True)
    coords = SkyCoord(ra, dec, unit = 'deg', frame='fk5')
    # compute EBV
    sfd = SFDQuery()
    ebvvec = sfd(coords)

    mag_dered = pd.DataFrame()
    for ii, b in enumerate(bands):
        mag_dered[b] = data[f"ObsMag_{b}"]-ebvvec*band_a_ebv[ii]
    
    return mag_dered


def convert_catalog_to_test_data(data, DS, TableHandle, bands, nside=128, dered=False):

    data2 = OrderedDict()
    
    if dered == True:
        mag_dered = deredden_galaxy(data, bands, nside=nside)
    
    for bb in bands:
        data2['mag_err_%s_lsst'%bb] = data['ObsMagErr_%s'%bb].to_numpy()
        if dered == False:
            data2['mag_%s_lsst'%bb] = data['ObsMag_%s'%bb].to_numpy()
        elif dered == True:
            data2['mag_%s_lsst'%bb] = mag_dered[bb].to_numpy()
        
    data2['redshift'] = data['redshift'].to_numpy()

    xtest_data = tables_io.convert(data2, tables_io.types.NUMPY_DICT)
    test_data = DS.add_data("test_data", xtest_data, TableHandle)
    
    return test_data


def compute_bpz_odds(chunks, zgrid=np.linspace(0, 3., 301)):
    gal_pdf=chunks.pdf(zgrid)
    zmode = chunks.ancil['zmode']
    dz=zgrid[1]-zgrid[0]
    
    vmin=zmode-0.06*(1+zmode)
    vmax=zmode+0.06*(1+zmode)
    ind=(zgrid[None,:]>=vmin[:,None])&(zgrid[None,:]<vmax[:,None])
    
    gal_pdf[~ind]=0
    odds=np.sum(gal_pdf,axis=1)*dz
    # need to remove rubbish ones with very big odds:
    odds=np.round(odds,decimals=5)
    odds[odds>1]=0
    
    return odds
