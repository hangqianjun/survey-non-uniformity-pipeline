"""
Run the nz summary stats for each quantile/bin in depth for each tomo bin:
Specify following things:

1. sysmap and quantiles/bins file
2. inroot for the files (Y2)
3. Tomo binning (lens, source)
4. Cuts for gold selection / SNR
5. (Optional) rolling effects

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

# compute simple summary statistic: No_obj, meanz, stdz, nzsq
print("Computing stats...")
zgrid = np.linspace(0,3,101)
nbootstrap=1000
z_col="redshift"

nzstat_summary_split={}
nzstat_summary_nzsq={}

# since we are working with actual quantiles it's probably okay to just put things together
# load binning files here:
print("Loading tomographic binning files...")

bin_index = {}
for q in range(len(qtl)-1):
    fname = savedir + f'{args.pztype}/tomo-bins-{args.tracer_type}-qtl-{q}{meanApsftag}{theta_efftag}-Y{args.Year}-snr-{args.snr_lim}'
    if args.pztype=="bpz" and args.odds_lim>0:
        fname += f"-odds-{args.odds_lim}"
    if args.tracer_type=='lens':
        fname += f"{maglimtag}"
    fname+=".txt"
    bin_index[q] = np.loadtxt(fname) # this is matched to the selection of the cat in each qtl

    
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
    
    tomo = pd.DataFrame(data={"tomo": bin_index[q]})
    cat_tomo = pd.concat([cat, tomo], axis=1)
    
    npzbins = int(cat_tomo["tomo"].max())

    # now run summary statistics
    nzstat_summary_split["q%d"%q] = {}
    nzstat_summary_nzsq["q%d"%q] = {}
    
    for jj in range(npzbins):
        nzstat_summary_split["q%d"%q]["tomo-%d"%(jj+1)]={}
        nzstat_summary_nzsq["q%d"%q]["tomo-%d"%(jj+1)]={}
        
        ind = cat_tomo["tomo"] == (jj+1)
        usecat = cat_tomo.loc[ind, :]
        
        # now for each tomographic bin, return redshift distribution:
        nzstat_summary_split["q%d"%q]["tomo-%d"%(jj+1)] = svf.compute_nzstats(usecat, z_col, zgrid=zgrid, nbootstrap=nbootstrap)
        nzstat_summary_nzsq["q%d"%q]["tomo-%d"%(jj+1)] = svf.compute_nzsq(usecat, z_col, zgrid=zgrid, nbootstrap=nbootstrap)
        

# Now compute the average nz, meanz, sigmaz from each quantile:
nzstat_summary_tot = {}
nzstat_summary_tot_nzsq = {}

for jj in range(npzbins):
    nzstat_summary_tot["tomo-%d"%(jj+1)] = []
    nzstat_summary_tot_nzsq["tomo-%d"%(jj+1)] = []
    totnz = 0
    for q in range(len(qtl)-1):
        usestat = nzstat_summary_split["q%d"%q]["tomo-%d"%(jj+1)]
        totnz += usestat[0][:,1]*qweights[q]
        
    nzstat_summary_tot["tomo-%d"%(jj+1)].append(np.c_[usestat[0][:,0], totnz])
    # mean:
    meanz = np.sum(totnz * usestat[0][:,0])/np.sum(totnz)
    nzstat_summary_tot["tomo-%d"%(jj+1)].append(meanz)
    # sigma
    sigmaz = np.sqrt(np.sum(totnz*(usestat[0][:,0]-meanz)**2)/np.sum(totnz))
    nzstat_summary_tot["tomo-%d"%(jj+1)].append(sigmaz)
    
    # nzsq
    nz_norm=totnz/np.sum(totnz)/(zgrid[1:]-zgrid[:-1])
    int_nz_sq = np.sum(nz_norm**2*(zgrid[1:]-zgrid[:-1]))
    nzstat_summary_tot_nzsq["tomo-%d"%(jj+1)].append(int_nz_sq)
      
    
# write to file:
outroot = savedir + f"{args.pztype}/"
fname = outroot + f"test-pz-with-i-band-coadd-Y{args.Year}"
fname += f"-snr-{args.snr_lim}"
if args.pztype=="bpz" and args.odds_lim>0:
    fname += f"-odds-{args.odds_lim}"
fname += f"-{args.tracer_type}{meanApsftag}{theta_efftag}"

if args.tracer_type=='lens':
    fname += f"{maglimtag}"

fname_stat = fname + ".yml"
svf.write_evaluation_results(fname_stat, mean_sys, nzstat_summary_split, nzstat_summary_tot, verbose=True)

fname_nzsq = fname + "-nzsq.pkl"
out=[nzstat_summary_nzsq, nzstat_summary_tot_nzsq]
svf.dump_save(out, fname_nzsq)