# run with pymaster or other DESC kernels
"""
Compute the coupling matrix
"""
import pymaster as nmt
import numpy as np
import h5py
import healpy as hp

def from_binning_info(ell_min, ell_max, n_ell, ell_spacing):
    # Creating the ell binning from the edges using this Namaster constructor.
    if ell_spacing == "log":
        edges = np.unique(np.geomspace(ell_min, ell_max, n_ell).astype(int))
    else:
        edges = np.unique(np.linspace(ell_min, ell_max, n_ell).astype(int))

    ell_bins = nmt.NmtBin.from_edges(edges[:-1], edges[1:], is_Dell=False)

    return ell_bins

def read_map_from_hdf5(fname, name, nside):
    """Return the map stored in the hdf5 TXPipe-like file.

    Args:
        fname (str): Path to the hdf5 file
        name (str): Name of the map in th hdf5 file
        nside (int): Map's HEALPix nside.

    Returns:
        array: HEALPix map
    """
    with h5py.File(fname, "r") as f:
        pixel = f[f"maps/{name}/pixel"]
        value = f[f"maps/{name}/value"]

        m = np.zeros(hp.nside2npix(nside))
        m[pixel] = value

        return m

def compute_master(f_a, f_b, wsp, clb):
    # Compute the power spectrum (a la anafast) of the masked fields
    # Note that we only use n_iter=0 here to speed up the computation,
    # but the default value of 3 is recommended in general.
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    # Decouple power spectrum into bandpowers inverting the coupling matrix
    cl_decoupled = wsp.decouple_cell(cl_coupled, cl_bias=clb)

    return cl_decoupled


######
vd = 0
nside=512
savedir="/pscratch/sd/q/qhang/dirac_mock/desc-project285/"
######

if vd == 1:
    vd_tag = "-vd"
elif vd == 0:
    vd_tag = ""

# binning
ell_min = 20
ell_max = 1000
n_ell = 15
ell_spacing = 'log'
ell_bins = from_binning_info(ell_min, ell_max, n_ell, ell_spacing)
ell_arr = ell_bins.get_effective_ells()
n_ell = len(ell_arr)
#savedir="/pscratch/sd/q/qhang/dirac_mock/desc-project285/"
#np.savetxt(savedir + "effective_ells.txt", ell_arr)

# mask
root="/pscratch/sd/q/qhang/glass_mock/catalog_info/mask/"
mask=hp.read_map(root+"wfd_footprint_nvisitcut_500_nside_128-ebv-0.2-1024.fits")
mask=hp.ud_grade(mask,nside)

# field # dummy
delta={}
shear={}
for ii in range(5):
    fname=savedir+f"galmap{vd_tag}-lens-tomo-{ii}-nside-512.fits"
    galmap = hp.read_map(fname)
    delta[ii]=(galmap/(sum(galmap)/sum(mask))-1)*mask
    
    fname=savedir+f"kappa-true{vd_tag}-tomo-{ii}-nside-512.fits"
    shear[ii] = hp.read_map(fname,field=[1,2])
    shear[ii][0]*=mask
    shear[ii][1]*=mask
    

# load cls:
cl_00=np.loadtxt(savedir+"clgg-theory-z_bin-nside-512.txt")
cl_02=np.loadtxt(savedir+"clgk-theory-z_bin-nside-512.txt")
cl_22=np.loadtxt(savedir+"clkk-theory-nside-512.txt")

# make cls in form of data
# add shotnoise!
pshot = np.array([2.14608250e-08, 1.39388165e-08, 1.49310243e-08, 1.48179262e-08, 2.79369541e-08])
for ii in range(5):
    cl_00[:,ii] += pshot[ii]

pshape = np.array([3.7982262757815385e-09, 3.691708135570749e-09, 3.5761122650727328e-09, 3.532698922231069e-09 ,3.406168937618163e-09])
for ii, kk in enumerate([0,2,5,9,14]):
    cl_22[:,kk] += pshape[ii]  
    
# spin-0
g_field = nmt.NmtField(mask, [delta[0]], masked_on_input=True)
# spin-2
s_field = nmt.NmtField(mask, shear[0], masked_on_input=True)

# save the coupling matrix
w00 = nmt.NmtWorkspace()
w00.compute_coupling_matrix(g_field, g_field, ell_bins)
w02 = nmt.NmtWorkspace()
w02.compute_coupling_matrix(g_field, s_field, ell_bins)
w22 = nmt.NmtWorkspace()
w22.compute_coupling_matrix(s_field, s_field, ell_bins)

print("Covariance")
cw = nmt.NmtCovarianceWorkspace()
cw.compute_coupling_coefficients(g_field, g_field, g_field, g_field)
print("Done computing coupling coefficients")

# get source - lensing ordering:
tomo_bin1 = np.zeros(15)
tomo_bin2 = np.zeros(15)
kk=0
for zz in range(5):
    for ll in range(5):
        if ll<=zz:
            tomo_bin1[kk]=zz
            tomo_bin2[kk]=ll
            kk+=1

"""
# clgg
for ii in range(5):
    covar_00_00 = nmt.gaussian_covariance(cw,
                                      0, 0, 0, 0,  # Spins of the 4 fields
                                      [cl_00[:,ii]],  # TT
                                      [cl_00[:,ii]],  # TT
                                      [cl_00[:,ii]],  # TT
                                      [cl_00[:,ii]],  # TT
                                      w00, wb=w00).reshape([n_ell, 1,
                                                            n_ell, 1])
    covar_TT_TT = covar_00_00[:, 0, :, 0]
    fname = savedir+f"cov-nmt-clgg-tomo-{ii}{vd_tag}-{nside}-binning-20-2000.txt"
    np.savetxt(fname, covar_TT_TT)
    print("Written: ", fname)


# clgk

kk=0
for zz in range(5):
    mm=np.where((tomo_bin1==zz)&(tomo_bin2==zz))[0][0]
    for ll in range(5):
        if ll<=zz:
            covar_02_02 = nmt.gaussian_covariance(cw, 0, 2, 0, 2,  # Spins of the 4 fields
                                      [cl_00[:,ll]],  # TT
                                      [cl_02[:,kk], np.zeros(3*nside)],  # TE, TB
                                      [cl_02[:,kk], np.zeros(3*nside)],  # ET, BT
                                      [cl_22[:,mm], np.zeros(3*nside),
                                       np.zeros(3*nside), np.ones(3*nside)*pshape[zz]],  # EE, EB, BE, BB
                                      w02, wb=w02).reshape([n_ell, 2,
                                                            n_ell, 2])
            covar_TE_TE = covar_02_02[:, 0, :, 0]
            kk+=1
            
            fname=savedir+f'cov-noisy-nmt-clgk-tomo-{zz}{ll}{vd_tag}-{nside}-binning-20-2000.txt'
            np.savetxt(fname,covar_TE_TE)
            print("Written: ", fname)

"""
# clkk
kk=0
for zz in range(5):
    m1=np.where((tomo_bin1==zz)&(tomo_bin2==zz))[0][0]
    for ll in range(5):
        m2=np.where((tomo_bin1==ll)&(tomo_bin2==ll))[0][0]
        if ll<zz:
            covar_22_22 = nmt.gaussian_covariance(cw, 2, 2, 2, 2,  # Spins of the 4 fields
                                      [cl_22[:,m1], np.zeros(3*nside),
                                       np.zeros(3*nside), np.ones(3*nside)*pshape[zz]],  # EE, EB, BE, BB
                                      [cl_22[:,kk], np.zeros(3*nside),
                                       np.zeros(3*nside), np.zeros(3*nside)],  # EE, EB, BE, BB
                                      [cl_22[:,kk], np.zeros(3*nside),
                                       np.zeros(3*nside), np.zeros(3*nside)],  # EE, EB, BE, BB
                                      [cl_22[:,m2], np.zeros(3*nside),
                                       np.zeros(3*nside), np.ones(3*nside)*pshape[ll]],  # EE, EB, BE, BB
                                      w22, wb=w22).reshape([n_ell, 4,
                                                            n_ell, 4])
            covar_EE_EE = covar_22_22[:, 0, :, 0]
            kk+1
                      
            fname=savedir+f'cov-noisy-nmt-clkk-tomo-{zz}{ll}{vd_tag}-{nside}-binning-20-2000.txt'
            np.savetxt(fname,covar_EE_EE)
            print("Written: ", fname)    
        
        if ll==zz:
            covar_22_22 = nmt.gaussian_covariance(cw, 2, 2, 2, 2,  # Spins of the 4 fields
                                      [cl_22[:,m1], np.zeros(3*nside),
                                       np.zeros(3*nside), np.ones(3*nside)*pshape[zz]],  # EE, EB, BE, BB
                                      [cl_22[:,kk], np.zeros(3*nside),
                                       np.zeros(3*nside), np.ones(3*nside)*pshape[zz]],  # EE, EB, BE, BB
                                      [cl_22[:,kk], np.zeros(3*nside),
                                       np.zeros(3*nside), np.ones(3*nside)*pshape[zz]],  # EE, EB, BE, BB
                                      [cl_22[:,m2], np.zeros(3*nside),
                                       np.zeros(3*nside), np.ones(3*nside)*pshape[ll]],  # EE, EB, BE, BB
                                      w22, wb=w22).reshape([n_ell, 4,
                                                            n_ell, 4])
            covar_EE_EE = covar_22_22[:, 0, :, 0]
            kk+1
                      
            fname=savedir+f'cov-noisy-nmt-clkk-tomo-{zz}{ll}{vd_tag}-{nside}-binning-20-2000.txt'
            np.savetxt(fname,covar_EE_EE)
            print("Written: ", fname)