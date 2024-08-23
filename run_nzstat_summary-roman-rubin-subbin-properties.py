"""
Run the nz summary stats in a few i-band depth bins (0, 5, 9), split by other properties:
e.g. u-band and seeing into ~5 bins
Keep:
Option for both BPZ and FZBoost
Option for each year
Option for source and lens
"""
import spatial_var_functions as svf
import sys
sys.path.insert(0, '/global/homes/q/qhang/desc/notebooks_for_analysis/')
import measure_properties_with_systematics as mp
from astropy.io import fits
import numpy as np
import healpy as hp
import pandas as pd
import yaml

import argparse

parser = argparse.ArgumentParser(description='Generate input files required for TXpipe measurement stages.')
parser.add_argument('-Year', type=int, default=1, help="Years of observation")
parser.add_argument('-pztype', default="bpz", help="Choose between bpz and fzb")
parser.add_argument('-odds_lim', type=float, default=-1, help="Whether to select photo-z odds, only works if using bpz, -1=no selection, >0 means select odds>=odds_lim")
parser.add_argument('-snr_lim', type=float, default=10, help="Apply snr cut where snr>=snr_lim, default is snr>=10")
parser.add_argument('-tracer_type', default="lens", help="choose between lens and source, all 5 bins according to SRD")
parser.add_argument('-sub_prop', default="ExgalM5_u", help="Other properties to subdivide the bins into")
parser.add_argument('-sub_prop_nbin', type=int, default=5, help="Number of bins to split the sub properties, will take N quantiles")
parser.add_argument('-qtls', nargs='+', default=[0, 5, 9], help="List indicating which i-band quantile to run this on")
parser.add_argument('-meanApsf', type=int, default=0, help="sets the meanApsf flag, 0 or 1")
args = parser.parse_args()


def write_evaluation_results_local(outroot, meanv, nzstat_summary_split, verbose=False):
    
    ntomo = list(nzstat_summary_split["sub0"].keys())
    
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
            nz=nzstat_summary_split[f"sub{kk}"][jj][0]
            meanz=nzstat_summary_split[f"sub{kk}"][jj][1]
            sigmaz =nzstat_summary_split[f"sub{kk}"][jj][2]
            
            out[jj]["nz"].append(nz) 
            out[jj]["meanz"][kk,:] = meanz
            out[jj]["sigmaz"][kk,:] = sigmaz
            
    # save to yaml file
    file=open(outroot,"w")
    yaml.dump(out,file)
    file.close()
    
    if verbose==True:
        print(f"Written: {outroot}.")


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


#####======These functions are added for sub-binning selection======#####
print("Loading sub-property maps...")
# need to map the pixels of the certain i-band qtl to the submaps
maf_root=f"/pscratch/sd/q/qhang/rubin_baseline_v3.3/MAF-{args.Year}year/"
# get the pixels from the i-band qtl
fname = maf_root + f"baseline_v3_3_10yrs_ExgalM5_i_and_nightlt{int(args.Year*365)}_HEAL.fits"
sysmap = hp.read_map(fname)
fname="/pscratch/sd/q/qhang/glass_mock/catalog_info/mask/wfd_footprint_nvisitcut_500_nside_128-ebv-0.2.fits"
mask = hp.read_map(fname)
selected_pix = mp.select_pixels_from_sysmap(sysmap, mask, qtl, added_range=False)

# load the sub-property map here
fname = maf_root + f"baseline_v3_3_10yrs_{args.sub_prop}_and_nightlt{int(args.Year*365)}_HEAL.fits"
subprop_map = hp.read_map(fname)

subprop_list={}
subprop_info_list={}

for use_qtl in args.qtls:
    # find the pixels
    pixels = selected_pix[int(use_qtl)]
    usemask = np.zeros(len(subprop_map))
    usemask[pixels]=1

    # define the bins
    sort_sys = np.sort(subprop_map[pixels])
    L = int(len(sort_sys)/args.sub_prop_nbin)
    ranges = np.zeros(args.sub_prop_nbin + 1)
    for ii in range(1,args.sub_prop_nbin):
        ranges[ii] = (sort_sys[L*ii] + sort_sys[L*ii+1])/2.
    ranges[0] = sort_sys[0] - (sort_sys[1] - sort_sys[0])
    ranges[-1] = sort_sys[-1] + (sort_sys[-1] - sort_sys[-2])
    
    subprop_list[int(use_qtl)]=mp.select_pixels_from_sysmap(subprop_map, usemask, ranges, added_range=False)
    
    # calculate the mean value of sysmap in each quantile
    mean_sys_sub = np.zeros(args.sub_prop_nbin)
    qweights_sub=np.zeros(args.sub_prop_nbin)
    totpix=0
    for ii in range(args.sub_prop_nbin):
        pix = subprop_list[int(use_qtl)][ii]
        mean_sys_sub[ii] = np.mean(subprop_map[pix])
        qweights_sub[ii] = len(pix)
        totpix+=len(pix)
    qweights_sub=qweights_sub/totpix
    # save the stats:
    out=np.c_[ranges, np.append(mean_sys_sub,-99), np.append(qweights_sub,-99)]
    fname=savedir + f"sub-{args.sub_prop}-qtl-{use_qtl}-mean-weights.txt"
    np.savetxt(fname, out)
    print("saved: ", fname)
    
    subprop_info_list[int(use_qtl)]={
        "mean_sys_sub":mean_sys_sub,
        "weights_sub":qweights_sub,  
    }
    
print("Done sub prop binning.")
    
#####======Above are functions added for sub-binning selection======#####
    
# registering meanApsf tag:
if args.meanApsf==0:
    meanApsftag=""
elif args.meanApsf==1:
    meanApsftag="-meanApsf"

# load the tomographic binnning:
print("Loading tomographic binning files...")
bin_index = {}
for use_qtl in args.qtls:
    fname = savedir + f'{args.pztype}/tomo-bins-{args.tracer_type}-qtl-{use_qtl}{meanApsftag}-Y{args.Year}-snr-{args.snr_lim}'
    if args.pztype=="bpz" and args.odds_lim>0:
        fname += f"-odds-{args.odds_lim}"
    fname+=".txt"
    bin_index[int(use_qtl)] = np.loadtxt(fname) # this is matched to the selection of the cat in each qtl
    
# compute simple summary statistic: No_obj, meanz, stdz, nzsq
print("Computing stats...")
zgrid = np.linspace(0,3,101)
nbootstrap=1000
z_col="redshift"


print("Working on each quantile...")
for q in args.qtls:
    
    q=int(q)
    print(f"Working on i-band depth q={q}...")

    fpz = savedir + f'{args.pztype}/roman-rubin_sample_obs_cal-coaddm5-i-qtl-{q}{meanApsftag}-zmode.pkl'
    fcat = savedir + f'roman-rubin_sample_obs_cal-coaddm5-i-qtl-{q}{meanApsftag}.pkl'
    cat = svf.dump_load(fcat)
    #cat = cat.drop(columns="index").reset_index(drop=True)
    pz = svf.dump_load(fpz)
    if args.pztype=="bpz":
        pz = pd.DataFrame(data={"z_mode": pz[:,0], "odds": pz[:,1]}, index=np.arange(len(cat)))
    elif args.pztype=="fzb":
        pz = pd.DataFrame(data={"z_mode": pz.flatten(), "odds": np.ones(len(pz))}, index=np.arange(len(cat)))
    cat = pd.concat([cat, pz], axis=1)
    
    cat = svf.select_data_with_cuts(cat, i_lim=gold_i, snr_lim=args.snr_lim, odds_lim=args.odds_lim)

    tomo = pd.DataFrame(data={"tomo": bin_index[q]})
    cat_tomo = pd.concat([cat, tomo], axis=1)
    
    npzbins = int(cat_tomo["tomo"].max())

    # now run summary statistics, save every qtl separately
    nzstat_summary_split = {}
    nzstat_summary_nzsq = {}
    
    # now also select objects from particular pixels:
    # now select tomo bins, also other properties bins:
    for kk in range(args.sub_prop_nbin):
        print(f"Working on {args.sub_prop} bin {kk}...")
        
        pix =  subprop_list[q][kk]
        print("m5_i: ", np.mean(sysmap[pix]), "sub_prop: ", np.mean(subprop_map[pix]))
        
        nzstat_summary_split[f"sub{kk}"] = {}
        nzstat_summary_nzsq[f"sub{kk}"] = {}

        for jj in range(npzbins):
        #for jj in range(1):
            nzstat_summary_split[f"sub{kk}"]["tomo-%d"%(jj+1)]={}
            nzstat_summary_nzsq[f"sub{kk}"]["tomo-%d"%(jj+1)]={}
        
            ind = cat_tomo["tomo"] == (jj+1)
            ind *= np.in1d(cat_tomo["pixels"],pix)
            #ind = np.in1d(cat_tomo["pixels"],pix)
            usecat = cat_tomo.loc[ind, :]
            
            #print(len(usecat))
            #print(np.mean(usecat["redshift"]))
        
            # now for each tomographic bin, return redshift distribution:
            nzstat_summary_split[f"sub{kk}"]["tomo-%d"%(jj+1)] = svf.compute_nzstats(usecat, z_col, zgrid=zgrid, nbootstrap=nbootstrap)
            nzstat_summary_nzsq[f"sub{kk}"]["tomo-%d"%(jj+1)] = svf.compute_nzsq(usecat, z_col, zgrid=zgrid, nbootstrap=nbootstrap)
    
    # write to file:
    outroot = savedir + f"{args.pztype}/sub-prop/"
    fname = outroot + f"test-pz-with-i-band-qtl-{q}-{args.sub_prop}-Y{args.Year}"
    fname += f"-snr-{args.snr_lim}"
    if args.pztype=="bpz" and args.odds_lim>0:
        fname += f"-odds-{args.odds_lim}"
    fname += f"-{args.tracer_type}{meanApsftag}"
    fname_stat = fname + ".yml"
    mean_sys_sub = subprop_info_list[q]["mean_sys_sub"]
    out = write_evaluation_results_local(fname_stat, mean_sys_sub, nzstat_summary_split, verbose=True)

    fname_nzsq = fname + "-nzsq.pkl"
    svf.dump_save(nzstat_summary_nzsq, fname_nzsq)