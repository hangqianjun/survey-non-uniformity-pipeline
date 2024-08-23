# requires condarail environment to run.
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
import numpy as np

import healpy as hp
import pickle

import pzflow
from pzflow import Flow
from pzflow.bijectors import Chain, ShiftBounds, RollingSplineCoupling
from pzflow.examples import get_galaxy_data

import yaml
import pandas as pd


# load pickle:
def dump_load(filename):
    with open(filename,'rb') as fin:
        stuff=pickle.load(fin)
        #self.impute = json.load(fin)
    #print('loaded impute ditionary:',filename)
    return stuff

def dump_save(stuff,filename):
    '''This saves the dictionary and loads it from appropriate files'''
    with open(filename,'wb') as fout:
        pickle.dump(stuff,fout,pickle.HIGHEST_PROTOCOL)
        #json.dump(self.impute, fout, sort_keys=True, indent=3)
    #print('written impute ditionary:',filename)
    return 0


def apply_magnitude_limit(fin, nyr=5):
    if nyr == 10:
        ind = fin[1].data['mag_u_cModel']<26.1
        ind = ind*(fin[1].data['mag_g_cModel']<27.4)
        ind = ind*(fin[1].data['mag_r_cModel']<27.5)
        ind = ind*(fin[1].data['mag_i_cModel']<25.3)
        ind = ind*(fin[1].data['mag_z_cModel']<26.1)
        ind = ind*(fin[1].data['mag_y_cModel']<24.9)
    elif nyr == 5:
        ind = fin[1].data['mag_u_cModel']<25.7
        ind = ind*(fin[1].data['mag_g_cModel']<27.0)
        ind = ind*(fin[1].data['mag_r_cModel']<27.1)
        ind = ind*(fin[1].data['mag_i_cModel']<25.0)
        ind = ind*(fin[1].data['mag_z_cModel']<25.7)
        ind = ind*(fin[1].data['mag_y_cModel']<24.5)
    elif nyr == 1:
        ind = fin[1].data['mag_u_cModel']<24.9
        ind = ind*(fin[1].data['mag_g_cModel']<26.2)
        ind = ind*(fin[1].data['mag_r_cModel']<26.3)
        ind = ind*(fin[1].data['mag_i_cModel']<24.1)
        ind = ind*(fin[1].data['mag_z_cModel']<24.9)
        ind = ind*(fin[1].data['mag_y_cModel']<23.7)
    # convert ind from boolean to number
    numind = np.arange(len(fin[1].data['mag_y_cModel']))
    numind = numind[ind]
    return numind


def compute_sigma_rand_sq(m5, mag, gamma, nvis, A_ratio=1):
    x = 10**(0.4*(mag-m5))
    sigma_rand2 = (0.04-gamma)*x + gamma*x**2
    sigma_rand2 = sigma_rand2/nvis
    
    sigma_rand2 *= A_ratio
    # also this is nsr
    return sigma_rand2

def compute_magerr_lsstmodel(sigma_rand_sq, sigmasys, highSNR=False):
    if highSNR==True:
        sigmasys = 10**(sigmasys/2.5)-1  
    magerr_lsst = np.sqrt(sigma_rand_sq + sigmasys**2) ### this step actually returns nsr
    if highSNR==False:
        magerr_lsst = 2.5*np.log10(1+magerr_lsst) ### convert this to sigma=MagErr
    return magerr_lsst


# from photerr package:


def get_semi_major_minor(data, scale=1):

    q = (1 - data['ellipticity'])/(1 + data['ellipticity'])
    ai = data['size']
    bi = ai*q

    ai = ai.to_numpy()*scale
    bi = bi.to_numpy()*scale
    
    return ai, bi


def get_area_ratio_auto(majors, minors, theta, airmass):
    """Get the ratio between PSF area and galaxy aperture area for "auto" model.
    Parameters
    ----------
    majors : np.ndarray
        The semi-major axes of the galaxies in arcseconds
    minors : np.ndarray
        The semi-minor axes of the galaxies in arcseconds
    bands : list
        The list of bands to calculate ratios for
    Returns
    -------
    np.ndarray
        The ratio of aperture size to PSF size for each band and galaxy.
        
    Change this to each band!!
        
    """
    
    
    # get the psf size for each band
    #psf_size = np.array([theta[band] for band in bands])
    #theta will be available for each galaxy
    psf_size = np.copy(theta)
    psf_size *= airmass**0.6

    # convert PSF FWHM to Gaussian sigma
    psf_sig = psf_size / 2.355

    # calculate the area of the psf in each band
    A_psf = np.pi * psf_sig**2

    # calculate the area of the galaxy aperture in each band
    a_ap = np.sqrt(psf_sig ** 2 + (2.5 * majors) ** 2)
    b_ap = np.sqrt(psf_sig ** 2 + (2.5 * minors) ** 2)

    A_ap = np.pi * a_ap * b_ap

    # return their ratio
    return A_ap / A_psf

    
#def compute_magerr():
    
#    return magerr



#def assign_magerror():
#    return new_mag


#def append_to_pandas_table():
    
#    return table


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


def assign_obs_cond_to_gals(assigned_pixels, obs_cond, mask, sys=[], bands=[]):
    
    print('sys: ', sys)
    print('bands: ', bands)
    
    assigned_obs_cond = {}
    
    for key in obs_cond.keys():
        if (len(sys)==0) or (key in sys): 
            assigned_obs_cond[key] = {}
            for b in obs_cond[key].keys():
                if (len(bands)==0) or (b in bands):
                    temp = np.zeros(len(mask))
                    temp[mask.astype(bool)] = obs_cond[key][b]
                    assigned_obs_cond[key][b] = temp[assigned_pixels]
    
    return assigned_obs_cond
   

### for functions in run_nzstat_summary:

def compute_nzstats(usecat, z_col, zgrid=np.linspace(0,1,31), nbootstrap=None):
    """
    Computes some summary statistics of the n_z for input catalog, specifically:
    - nz: given zgrid, output will be [zcentre, Nz]
    - meanz: given zgrid, weighted mean redshift with bootstrap error [meanz, meanz_err]
    - sigmaz: given zgrid, weighted first moment of z (sigmaz) with bootstrap error [sigmaz, sigmaz_err]
    if nbootstrap == None, just use Poisson errors
    """
    
    Nz,ig = np.histogram(usecat[z_col], bins=zgrid)
    zcentre = (zgrid[1:] + zgrid[:-1])*0.5
    nz = np.c_[zcentre, Nz]
    
    meanz = np.sum(Nz*zcentre)/np.sum(Nz)
    
    sigmaz = np.sqrt(np.sum(Nz*(zcentre-meanz)**2)/np.sum(Nz))
    
    # compute the bootstrap error:
    sampholder_meanz = np.zeros(nbootstrap)
    sampholder_sigmaz = np.zeros(nbootstrap)
        
    for kk in range(nbootstrap):
        samp = np.random.choice(usecat[z_col], 
                        size=len(usecat[z_col]),
                        replace=True)
        # repeat the operation 
        cc = np.histogram(samp,bins=zgrid)
        sampholder_meanz[kk] = np.sum(cc[0] * zcentre)/np.sum(cc[0])
        sampholder_sigmaz[kk] = np.sqrt(np.sum(cc[0]*(zcentre-meanz)**2)/np.sum(cc[0]))

    meanz_err = np.std(sampholder_meanz)
    sigmaz_err = np.std(sampholder_sigmaz)
    
    meanz = np.array([meanz, meanz_err])
    sigmaz = np.array([sigmaz, sigmaz_err])
    
    return nz, meanz, sigmaz


def compute_nzsq(usecat, z_col, zgrid=np.linspace(0,1,31), nbootstrap=None):
    """
    Computes nz_sq with bootstrap error [int_nz_sq, int_nz_sq_err]
    """
    
    Nz,ig = np.histogram(usecat[z_col], bins=zgrid)
    zcentre = (zgrid[1:] + zgrid[:-1])*0.5
    nz = np.c_[zcentre, Nz]
    
    # normalise nz
    nz_norm=Nz/np.sum(Nz)/(zgrid[1:]-zgrid[:-1])
    
    int_nz_sq = np.sum(nz_norm**2*(zgrid[1:]-zgrid[:-1]))
    
    # compute the bootstrap error:
    sampholder_nzsq = np.zeros(nbootstrap)
    
    for kk in range(nbootstrap):
        samp = np.random.choice(usecat[z_col], 
                        size=len(usecat[z_col]),
                        replace=True)
        # repeat the operation 
        cc = np.histogram(samp,bins=zgrid)
        nz_norm = cc[0]/np.sum(cc[0])/(zgrid[1:]-zgrid[:-1])
        sampholder_nzsq[kk] = np.sum(nz_norm**2*(zgrid[1:]-zgrid[:-1]))
        
    int_nz_sq_err = np.std(sampholder_nzsq)
    
    int_nz_sq = np.array([int_nz_sq, int_nz_sq_err])
    
    return nz, int_nz_sq

def write_evaluation_results(outroot, meanv, nzstat_summary_split, nzstat_summary_tot, verbose=False):
    
    ntomo = list(nzstat_summary_split["q0"].keys())
    
    out = {}
    
    out["nquantile"] = len(meanv)
    out["mean_systematic"] = meanv
    
    for jj in ntomo:
        out[jj]={
            "nz": [],
            "meanz": np.zeros((len(meanv),2)),
            "sigmaz": np.zeros((len(meanv),2)),
        }
        for kk in range(len(meanv)):
            nz=nzstat_summary_split["q%d"%kk][jj][0]
            meanz=nzstat_summary_split["q%d"%kk][jj][1]
            sigmaz =nzstat_summary_split["q%d"%kk][jj][2]
            
            out[jj]["nz"].append(nz) 
            out[jj]["meanz"][kk,:] = meanz
            out[jj]["sigmaz"][kk,:] = sigmaz
            
        # add unbinned stats:
        nz, meanz, sigmaz = nzstat_summary_tot[jj]
        out[jj]["nztot"]=nz 
        out[jj]["meanztot"] = meanz
        out[jj]["sigmaztot"] = sigmaz
            
    # save to yaml file
    file=open(outroot,"w")
    yaml.dump(out,file)
    file.close()
    
    if verbose==True:
        print(f"Written: {outroot}.")
    
    
def get_wfd_DESI_overlap(scratch):
    fname = scratch + "rubin_baseline_v2/"
    fname += "wfd_footprint_nvisitcut_500_nside_128.fits"
    wfd_mask = hp.read_map(fname)
    
    fname = scratch + "rubin_baseline_v2/"
    fname += "DESI_footprint_completeness_mask_128.fits"

    desi_mask = hp.read_map(fname)

    desi_mask[desi_mask<=0]=0
    desi_mask[desi_mask>0]=1
    
    pix = np.arange(len(wfd_mask))
    overlap_mask = wfd_mask*desi_mask
    overlap_pix = pix[overlap_mask.astype(bool)]
    return wfd_mask, desi_mask, overlap_pix




# first we load data and assign tomographic bins according to SRD

# this is the simplist binning method using the pz point estimate directly:
def assign_lens_bins(catalog, nYrObs=1, pzkey='pz_point'):
    """
    nYrObs: lens binning requirement to adopt.
    """
    pz = catalog[pzkey]
    if nYrObs==1:
        # adopt 1 year criteria: 5 bins between 0.2<z<1.2
        bin_edges = np.linspace(0.2,1.2,6)
        bin_index = np.digitize(pz, bin_edges) 
    if nYrObs==10:
        # adopt 10 year criteria: 10 bins between 0.2<z<1.2
        bin_edges = np.linspace(0.2,1.2,11)
        bin_index = np.digitize(pz, bin_edges) 

    # append the tomographic binning
    tomo = pd.DataFrame(data={"tomo": bin_index})
    catalog_tomo = pd.concat([catalog, tomo], axis=1)
    # trim the catalogue for objects not in the bin:
    sel = np.where((bin_index>=1)&(bin_index<len(bin_edges)))[0]
    catalog_tomo = catalog_tomo.loc[sel,:]
    
    return catalog_tomo


def assign_source_bins(catalog, pzkey='pz_point'):
    # requirement is 5 bins with equal number of objects:
    # here we just use simple quantile from the estimated photo-z
    pz = catalog[pzkey].to_numpy()
    nbins=5
    sortind = np.argsort(pz)
    N_perbin = int(len(pz)/nbins)
    bin_index = np.zeros(len(pz))
    for ii in range(nbins):
        useind = sortind[ii*N_perbin:(ii+1)*N_perbin]
        bin_index[useind] = int(ii+1)
    
    # append the tomographic binning
    tomo = pd.DataFrame(data={"tomo": bin_index})
    catalog_tomo = pd.concat([catalog, tomo], axis=1)
    return catalog_tomo


# for now only i limit but can improve later
def select_data_with_cuts(cat, i_lim=24.1, snr_lim=10, odds_lim=0):
    
    # select i-band first
    ind1=cat["ObsMag_i"]<i_lim
    
    # also select SNR in i-band > 10
    #ind2=(cat["ObsMag_i"]/cat["ObsMagErr_i"])>=snr_lim
    snr=1/(10**(cat["ObsMagErr_i"]/2.5)-1)
    ind2=snr>=snr_lim
    
    sel = ind1*ind2
    
    # select photoz odds if odds_lim>0
    if odds_lim>0:
        sel = sel*(cat["odds"]>=odds_lim)
    
    cat_out = cat.loc[sel,:]
    cat_out = cat_out.reset_index(drop=True)
    
    return cat_out


def select_deslike_lens(cat, snr_lim=10, odds_lim=0):
    ### see https://arxiv.org/pdf/2209.05853.pdf sec. 2.2
    ### following DES Y3 maglim selection cuts
    ### keeping snr and odds selection as before
    
    # des y3 like selection:
    ind1=cat["ObsMag_i"]>17.5
    ind2=cat["ObsMag_i"]<(4*cat["z_mode"]+18)
    
    # also select SNR in i-band > 10
    #ind2=(cat["ObsMag_i"]/cat["ObsMagErr_i"])>=snr_lim
    snr=1/(10**(cat["ObsMagErr_i"]/2.5)-1)
    ind3=snr>=snr_lim
    
    sel = ind1*ind2*ind3
    
    # select photoz odds if odds_lim>0
    if odds_lim>0:
        sel = sel*(cat["odds"]>=odds_lim)
    
    cat_out = cat.loc[sel,:]
    cat_out = cat_out.reset_index(drop=True)
    
    return cat_out