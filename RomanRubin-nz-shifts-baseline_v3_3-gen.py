# python dependences:
import numpy as np
from astropy.io import fits
import numpy as np
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
desc_bpz.__version__
import PZflow_nz_shifts_functions as pnsf
#import RailStage stuff
from rail.core.data import TableHandle
from rail.core.stage import RailStage
DS = RailStage.data_store
DS.__class__.allow_overwrite = True
from rail.estimation.algos.bpz_lite import BPZ_lite
from rail.estimation.algos.flexzboost import FZBoost

import argparse

parser = argparse.ArgumentParser(description='Generate input files required for TXpipe measurement stages.')
parser.add_argument('-Year', type=int, default=1, help="Years of observation")
#parser.add_argument('-outroot', default="", help="Where to save things")
parser.add_argument('-load_sys', type=int, default=1, help="1=load, 0=overwrite")
parser.add_argument('-do_degrade', type=int, default=0, help="0=skip degradation, 1=do degradation")
parser.add_argument('-do_bpz', type=int, default=0, help="0=don't run pz, 1=run bpz")
parser.add_argument('-train_fzb', type=int, default=0, help="0=don't train fzboost, 1=train fzboost")
parser.add_argument('-do_fzb', type=int, default=0, help="0=don't run pz, 1=run fzboost")
parser.add_argument('-meanApsf', type=int, default=0, help="sets the meanApsf flag, 0 or 1")
parser.add_argument('-theta_eff', type=int, default=0, help="sets the theta_eff flag, 0 or 1")
args = parser.parse_args()


### === Basic stuff goes here === ###
print("Initializing...")

savedir = f"/pscratch/sd/q/qhang/roman-rubin-sims/baselinev3.3/y{args.Year}/"
bands = ['u','g','r','i','z','y']
fname="/pscratch/sd/q/qhang/glass_mock/catalog_info/mask/wfd_footprint_nvisitcut_500_nside_128-ebv-0.2.fits"
mask = hp.read_map(fname)
# here load the observing cnoditions (Y1/Y5):
# load the Y1 baseline maps
# opsim directory
Opsimdir = f'/pscratch/sd/q/qhang/rubin_baseline_v3.3/MAF-{args.Year}year/'

# Here load the median 5sigma depth map in each band:
metric_dict = {'theta':'Median_seeingFwhmEff',
               'coaddm5':'ExgalM5',
               'airmass': 'Median_airmass',}

print("Collecting obs_cond...")
obs_cond = {}
nights=int(365*args.Year)
for key in metric_dict.keys():
    print(f'Loading {key}...')
    name = metric_dict[key]
    obs_cond[key] = {}
    for b in bands:
        fname = Opsimdir+f'baseline_v3_3_10yrs_{name}_{b}_and_nightlt{nights}_HEAL.fits'
        fin1=hp.read_map(fname)
        obs_cond[key][b] = fin1[mask.astype(bool)]
        
obs_cond['gamma'] = {
    'u':0.038,
    'g':0.039,
    'r':0.039,
    'i':0.039,
    'z':0.039,
    'y':0.039,
}
obs_cond['sigmasys'] = 0.005
# depricated
obs_cond['calibration'] = {}
obs_cond['calibration']['abscale']=1
obs_cond['calibration']['magerrscale']={
    'u': 1,
    'g': 1,
    'r': 1, 
    'i': 1,
    'z': 1,
    'y': 1,
}
obs_cond['sigLim']=0 # not doing any snr cuts
# if input is given by coadd depth map, this is set to one
obs_cond['nvis']={}
obs_cond['nvis']['u']=1
obs_cond['nvis']['g']=1
obs_cond['nvis']['r']=1
obs_cond['nvis']['i']=1
obs_cond['nvis']['z']=1
obs_cond['nvis']['y']=1


print("Collecting quantiles of the systematic maps...")
sysmap = np.zeros(len(mask))
sysmap[mask.astype(bool)] = obs_cond['coaddm5']['i']

fname=savedir + "ExgalM5-i-qtl-mean-weights.txt"

if args.load_sys==1:
    print("Loading: ", fname)
    fin=np.loadtxt(fname)
    qtl=fin[:,0]
    mean_sys=fin[:-1,1]
    qweights=fin[:-1,2]
    nquantiles=len(qtl)-1
    
    # select pixels
    selected_pix = mp.select_pixels_from_sysmap(sysmap, mask, qtl, added_range=False)

else:
    # here let's use quantiles
    nquantiles = 10
    # define the bins
    sort_sys = np.sort(obs_cond['coaddm5']['i'])
    L = int(len(obs_cond['coaddm5']['i'])/nquantiles)
    qtl = np.zeros(nquantiles + 1)
    for ii in range(1,nquantiles):
        qtl[ii] = (sort_sys[L*ii] + sort_sys[L*ii+1])/2.
    qtl[0] = sort_sys[0] - (sort_sys[1] - sort_sys[0])
    qtl[-1] = sort_sys[-1] + (sort_sys[-1] - sort_sys[-2])
    selected_pix = mp.select_pixels_from_sysmap(sysmap, mask, qtl, added_range=False)
    
    # calculate the mean value of sysmap in each quantile
    mean_sys = np.zeros(nquantiles)
    qweights=np.zeros(nquantiles)
    totpix=0
    for ii in range(nquantiles):
        pix = selected_pix[ii]
        mean_sys[ii] = np.mean(sysmap[pix])
        qweights[ii] = len(pix)
        totpix+=len(pix)
    qweights=qweights/totpix

    # save these info:
    out=np.c_[qtl, np.append(mean_sys,-99), np.append(qweights,-99)]
    fname=savedir + "ExgalM5-i-qtl-mean-weights.txt"
    np.savetxt(fname, out)
    print("saved: ", fname)
    
print("Quantiles: ", qtl)
print("mean_sys: ", mean_sys)
print("qweights: ", qweights)
print("Pixels in each quantile: ")
for ii in range(nquantiles):
    print(len(selected_pix[ii]))

# registering meanApsf tag:
if args.meanApsf==0:
    meanApsftag=""
elif args.meanApsf==1:
    meanApsftag="-meanApsf"
    
# registering theta_eff tag:
if args.theta_eff==0:
    theta_efftag=""
elif args.theta_eff==1:
    theta_efftag="-theta_eff"


if args.do_degrade==1:
    # here load the PZflow testing sample:
    print("Loading sample...")
    fname="/pscratch/sd/q/qhang/roman-rubin-sims/nonuniform-maf/roman_rubin_2023_v1.1.3_elais-subset.pkl"
    print("Loading: ", fname)
    mock_tract = svf.dump_load(fname)
    mock_tract = mock_tract.reset_index()
    print("Number of obj in this sample: ", len(mock_tract))

    # degradation:
    print("Begin degradation...")
    random_seed=10
    rng = np.random.default_rng(10)
    sys = ['coaddm5', 'theta' , 'airmass', 'nvis']
    m5key='coaddm5'

    for q in range(nquantiles):  
        data = mock_tract.copy()
        usepixels = selected_pix[q]
        outfile = savedir + f'roman-rubin_sample_obs_cal-coaddm5-i-qtl-{q}{meanApsftag}{theta_efftag}.pkl'
        pnsf.obs_cond_pipeline(data, usepixels, random_seed, rng, obs_cond, mask, sys, bands, outfile, m5key=m5key, meanApsf=bool(args.meanApsf), theta_eff=bool(args.theta_eff))

if args.do_bpz==1:
    # use mpi for this process:
    from orphics import mpi,stats
    
    print("Begin assigning redshifts...")

    band_names = [
        'mag_u_lsst','mag_g_lsst','mag_r_lsst',
        'mag_i_lsst','mag_z_lsst','mag_y_lsst'  
    ]

    band_err_names = [
        'mag_err_u_lsst','mag_err_g_lsst','mag_err_r_lsst',
        'mag_err_i_lsst','mag_err_z_lsst','mag_err_y_lsst'
    ]
    prior_band='mag_i_lsst'

    comm,rank,my_tasks = mpi.distribute(nquantiles)
    s = stats.Stats(comm)
    
    for q in my_tasks:
        print(f"Working on qtl {q}...")
        output = savedir + f"bpz/BPZ-estimator-output-{q}{meanApsftag}{theta_efftag}.hdf5"
        estimate_bpz = BPZ_lite.make_stage(name='estimate_bpz', hdf5_groupname='', 
                                       #columns_file=inroot+'test_bpz.columns',
                                       #prior_file='CWW_HDFN_prior.pkl',
                                       nondetect_val=np.nan, #spectra_file='SED/CWWSB4.list',
                                       band_names=band_names,
                                       band_err_names=band_err_names,
                                       prior_band=prior_band,
                                       mag_limits = dict(mag_u_lsst=27.79,
                                                    mag_g_lsst=29.04,
                                                    mag_r_lsst=29.06,
                                                    mag_i_lsst=28.62,
                                                    mag_z_lsst=27.98,
                                                    mag_y_lsst=27.05),
                                       output=output)

        fname = savedir + f'roman-rubin_sample_obs_cal-coaddm5-i-qtl-{q}{meanApsftag}{theta_efftag}.pkl'
        data = svf.dump_load(fname)

        test_data = pnsf.convert_catalog_to_test_data(data, DS, TableHandle, bands, dered=True)
        bpz_estimated = estimate_bpz.estimate(test_data)

        # obtain the mode:
        zmode = bpz_estimated().ancil['zmode']
        # read in the bpz file in chunks, and compute the odds, save it 
        # with the mode:

        odds = pnsf.compute_bpz_odds(bpz_estimated(), zgrid=np.linspace(0, 3., 301))
        # save:
        fname = savedir + f'bpz/roman-rubin_sample_obs_cal-coaddm5-i-qtl-{q}{meanApsftag}{theta_efftag}-zmode.pkl'
        svf.dump_save(np.c_[zmode, odds], fname)
        
              
def nan_to_faint_mag(cat, bands, faint_mag=30):
    for b in bands:
        # replace that by 30
        cat[f"ObsMag_{b}"].loc[np.isnan(cat[f"ObsMag_{b}"])]=faint_mag   
    return cat

def save_training_file(cat, DS, TableHandle):
    save_obj = OrderedDict()

    save_obj["id"] = np.arange(len(cat["redshift"]))
    save_obj["mag_err_u_lsst"] = cat["ObsMagErr_u"]
    save_obj["mag_err_g_lsst"] = cat["ObsMagErr_g"]
    save_obj["mag_err_r_lsst"] = cat["ObsMagErr_r"]
    save_obj["mag_err_i_lsst"] = cat["ObsMagErr_i"]
    save_obj["mag_err_z_lsst"] = cat["ObsMagErr_z"]
    save_obj["mag_err_y_lsst"] = cat["ObsMagErr_y"]
    save_obj["mag_u_lsst"] = cat["ObsMag_u"]
    save_obj["mag_g_lsst"] = cat["ObsMag_g"]
    save_obj["mag_r_lsst"] = cat["ObsMag_r"]
    save_obj["mag_i_lsst"] = cat["ObsMag_i"]
    save_obj["mag_z_lsst"] = cat["ObsMag_z"]
    save_obj["mag_y_lsst"] = cat["ObsMag_y"]
    save_obj["redshift"] = cat["redshift"]

    save_obj_highlevel = OrderedDict()
    save_obj_highlevel["photometry"] = save_obj
    test_data = DS.add_data("test_data", save_obj_highlevel, TableHandle)
    return test_data
    
if args.train_fzb==1:
    print("Training FZBoost...")
    from rail.estimation.algos.flexzboost import Inform_FZBoost

    # Construct the training sample:
    use_keys=[
    "ObsMagErr_u", "ObsMagErr_g", "ObsMagErr_r", "ObsMagErr_i", "ObsMagErr_z", "ObsMagErr_y",
    "ObsMag_u", "ObsMag_g", "ObsMag_r", "ObsMag_i", "ObsMag_z", "ObsMag_y",
    "redshift",
        ]
    save_cat={}
    for q in range(nquantiles):
        fcat = savedir + f'roman-rubin_sample_obs_cal-coaddm5-i-qtl-{q}{meanApsftag}{theta_efftag}.pkl'
        cat = svf.dump_load(fcat)
        #cat = cat.drop(columns="index").reset_index(drop=True)
        
        # swap nan to mag=30
        cat=nan_to_faint_mag(cat, bands, faint_mag=30)
        ind_comb=np.arange(len(cat["ObsMag_i"]))
        n=int(len(ind_comb)*qweights[q]*0.1)

        # select random index from each quantile:
        ind_rand = np.random.choice(ind_comb, size=n, replace=False)
        for key in use_keys:
            if q==0:
                save_cat[key]=cat[key][ind_rand].to_numpy()
            else:
                save_cat[key]=np.append(save_cat[key], cat[key][ind_rand].to_numpy())
    print("Total sample for training: ", len(save_cat[key]))

    training_data_ds = save_training_file(save_cat, DS, TableHandle)
    fz_dict = dict(zmin=0.0, zmax=3.0, nzbins=301,
                   trainfrac=0.75, bumpmin=0.02, bumpmax=0.35,
                   nbump=20, sharpmin=0.7, sharpmax=2.1, nsharp=15,
                   max_basis=35, basis_system='cosine',
                   hdf5_groupname='photometry',
                   regression_params={'max_depth': 8,'objective':'reg:squarederror'})
    fz_modelfile = savedir + f'fzb/FZB_model_nan_to_mag30{meanApsftag}{theta_efftag}.pkl'
    inform_pzflex = Inform_FZBoost.make_stage(name='inform_fzboost', model=fz_modelfile, **fz_dict)
    inform_pzflex.inform(training_data_ds)

if args.do_fzb==1:
    print("Special run!")
    # use mpi for this process:
    from orphics import mpi,stats

    #load BPZ model:
    fz_modelfile = savedir + f'fzb/FZB_model_nan_to_mag30{meanApsftag}{theta_efftag}.pkl'

    comm,rank,my_tasks = mpi.distribute(nquantiles)
    #s = stats.Stats(comm)

    for q in my_tasks:
        print("Working on qtl %d"%q)
        output = savedir + f"fzb/FZBoost-estimator-output-{q}{meanApsftag}{theta_efftag}.hdf5"
        pzflex = FZBoost.make_stage(name='fzboost', hdf5_groupname='',
                                    model=fz_modelfile, output=output)
        
        fname = savedir + f'roman-rubin_sample_obs_cal-coaddm5-i-qtl-{q}{meanApsftag}{theta_efftag}.pkl'
        data = svf.dump_load(fname)
        data=nan_to_faint_mag(data, bands, faint_mag=30)
        test_data = save_training_file(data, DS, TableHandle)

        fzb_estimated = pzflex.estimate(test_data.data["photometry"])

        # obtain the mode:
        zgrid = zgrid = np.linspace(0, 3., 301)
        zmode = fzb_estimated.data.mode(grid=zgrid)
        print(zmode.shape)

        # save:
        fname = savedir + f'fzb/roman-rubin_sample_obs_cal-coaddm5-i-qtl-{q}{meanApsftag}{theta_efftag}-zmode.pkl'
        svf.dump_save(zmode, fname)
