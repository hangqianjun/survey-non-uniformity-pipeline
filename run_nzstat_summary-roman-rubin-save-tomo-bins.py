"""
Save tomographic bin files.
"""
import spatial_var_functions as svf
import sys
sys.path.insert(0, '/global/homes/q/qhang/desc/notebooks_for_analysis/')
import measure_properties_with_systematics as mp
from astropy.io import fits
import numpy as np
import healpy as hp
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='Generate input files required for TXpipe measurement stages.')
parser.add_argument('-Year', type=int, default=1, help="Years of observation")
parser.add_argument('-pztype', default="bpz", help="Choose between bpz and fzb")
parser.add_argument('-odds_lim', type=float, default=-1, help="Whether to select photo-z odds, only works if using bpz, -1=no selection, >0 means select odds>=odds_lim")
parser.add_argument('-snr_lim', type=float, default=10, help="Apply snr cut where snr>=snr_lim, default is snr>=10")
parser.add_argument('-tracer_type', default="lens", help="choose between lens and source, all 5 bins according to SRD")
parser.add_argument('-meanApsf', type=int, default=0, help="sets the meanApsf flag, 0 or 1")
parser.add_argument('-theta_eff', type=int, default=0, help="sets the theta_eff flag, 0 or 1")
parser.add_argument('-maglim', type=int, default=0, help="To be applied on lens sample only. 0=no maglim cut, 1=apply maglim cut")
args = parser.parse_args()


def assign_lens_bins_local(catalog, nYrObs=1, pzkey='pz_point'):
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
    #tomo = pd.DataFrame(data={"tomo": bin_index})
    #catalog_tomo = pd.concat([catalog, tomo], axis=1)
    # trim the catalogue for objects not in the bin:
    sel = np.where(bin_index==0)[0]
    bin_index[sel]=-99
    sel = np.where(bin_index==len(bin_edges))[0]
    bin_index[sel]=-99
    return bin_index


###===These parameters can be changed===###
print("Initializing...")

savedir = f"/pscratch/sd/q/qhang/roman-rubin-sims/baselinev3.3/y{args.Year}/"
bands = ['u','g','r','i','z','y']

sys_info = savedir + "ExgalM5-i-qtl-mean-weights.txt"
print("Loading systematic map info...")
fin=np.loadtxt(sys_info)
qtl=fin[:,0]
mean_sys=fin[:-1,1]
qweights=fin[:-1,2]
print("mean_sys: ", mean_sys)
print("qweights: ", qweights)

print(f"Computing gold cuts for i-band in Y{args.Year}...")
gold_i_10yr=25.3
gold_i = gold_i_10yr - 2.5*np.log10(np.sqrt(10/args.Year))
gold_i = round(gold_i,1)
print(f"Gold limit is i={gold_i}.")

# flag for using BPZ redshifts or FZBoost
print(f"Using pztype: {args.pztype}.")

# flag for selecting photoz odds >= odds_sel, if set to <0, no selection will be done
if args.pztype=="bpz":
    if args.odds_lim<0:
        print("No odds limit.")
    elif args.odds_lim>0:
        print(f"Applying odds selection: odds >= {args.odds_lim}.")
elif args.pztype=="fzb":
    print("Ignoring odds limit.")
    
print(f"Applying SNR cut with SNR>={args.snr_lim}.")


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
    
# registering maglim tag:
if args.tracer_type=='lens':
    # only applied to lens, this is ignored for source
    if args.maglim==0:
        maglimtag=""
    elif args.maglim==1:
        maglimtag="-maglim"



zgrid = np.linspace(0,3,101)
nbootstrap=1000
z_col="redshift"

# concatenate source samples together:
if args.tracer_type=="source":
    print("Determining source bins by quantiles...")
    for q in range(len(qtl)-1):
        fpz = savedir + f'{args.pztype}/roman-rubin_sample_obs_cal-coaddm5-i-qtl-{q}{meanApsftag}{theta_efftag}-zmode.pkl'
        fcat = savedir + f'roman-rubin_sample_obs_cal-coaddm5-i-qtl-{q}{meanApsftag}{theta_efftag}.pkl'
        cat = svf.dump_load(fcat)
        #cat = cat.drop(columns="index").reset_index(drop=True)
        pz = svf.dump_load(fpz)
        if args.pztype=="bpz":
            pz = pd.DataFrame(data={"z_mode": pz[:,0], "odds": pz[:,1]}, index=np.arange(len(cat)))
        elif args.pztype=="fzb":
            pz = pd.DataFrame(data={"z_mode": pz.flatten(), "odds": np.ones(len(pz))}, index=np.arange(len(cat)))
        cat = pd.concat([cat, pz], axis=1)

        # for source sample:
        cat = svf.select_data_with_cuts(cat, i_lim=gold_i, snr_lim=args.snr_lim, odds_lim=args.odds_lim)

        # extract pz:
        pz=cat["z_mode"].to_numpy()

        # attach them
        L=len(pz)
        if q==0:
            pz_all=pz
            quantile_all=np.ones(L)*q
        elif q>0:
            pz_all=np.append(pz_all, pz)
            # attach index from the quantile:
            quantile_all=np.append(quantile_all,np.ones(L)*q)

    # assign bins:
    # add a very small random number to the pz so the 
    #sort function is not weird
    np.random.seed(50)
    grid_size=zgrid[1]-zgrid[0]
    small_rand = np.random.normal(loc=0,scale=grid_size/5.,size=len(pz_all))
    sortind = np.argsort(pz_all+small_rand)
    nbins=5
    bin_index = np.zeros(len(pz_all))
    N=int(len(pz_all)/5)
    for ii in range(nbins):
        ind1=ii*N
        ind2=(ii+1)*N
        useind=sortind[ind1:ind2]
        bin_index[useind] = int(ii+1)
    print("Done assigning source bins.")

# for both lens and source
print("Working on each quantile...")
for q in range(len(qtl)-1):

    print(f"Working on q={q}...")
    fpz = savedir + f'{args.pztype}/roman-rubin_sample_obs_cal-coaddm5-i-qtl-{q}{meanApsftag}{theta_efftag}-zmode.pkl'
    fcat = savedir + f'roman-rubin_sample_obs_cal-coaddm5-i-qtl-{q}{meanApsftag}{theta_efftag}.pkl'
    cat = svf.dump_load(fcat)
    #cat = cat.drop(columns="index").reset_index(drop=True)
    pz = svf.dump_load(fpz)
    if args.pztype=="bpz":
        pz = pd.DataFrame(data={"z_mode": pz[:,0], "odds": pz[:,1]}, index=np.arange(len(cat)))
    elif args.pztype=="fzb":
        pz = pd.DataFrame(data={"z_mode": pz.flatten(), "odds": np.ones(len(pz))}, index=np.arange(len(cat)))
    cat = pd.concat([cat, pz], axis=1)

    if args.tracer_type=='lens':
        if args.maglim==0:
            cat = svf.select_data_with_cuts(cat, i_lim=gold_i, snr_lim=args.snr_lim, odds_lim=args.odds_lim)
        elif args.maglim==1:
            cat = svf.select_deslike_lens(cat, snr_lim=args.snr_lim, odds_lim=args.odds_lim)
    elif args.tracer_type=='source':
        cat = svf.select_data_with_cuts(cat, i_lim=gold_i, snr_lim=args.snr_lim, odds_lim=args.odds_lim)
    
    if args.tracer_type=="lens":
        save_bin_index=assign_lens_bins_local(cat, nYrObs=1, pzkey='z_mode')
    elif args.tracer_type=="source":
        save_bin_index = bin_index[quantile_all==q].astype(int)
        
    # write to file:
    outroot = savedir + f"{args.pztype}/"
    fname = outroot + f"tomo-bins-{args.tracer_type}-qtl-{q}{meanApsftag}{theta_efftag}-Y{args.Year}"
    fname += f"-snr-{args.snr_lim}"
    if args.pztype=="bpz" and args.odds_lim>0:
        fname += f"-odds-{args.odds_lim}"
    if args.tracer_type=='lens':
        fname += f"{maglimtag}"
    fname = fname + ".txt"
    np.savetxt(fname, save_bin_index)
    print("Saved: ", fname)
