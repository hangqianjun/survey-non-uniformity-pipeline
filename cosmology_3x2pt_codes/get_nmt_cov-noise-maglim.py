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

for ii in range(5):
    fname=savedir+f"galmap{vd_tag}-lens-maglim-tomo-{ii}-nside-512.fits"
    galmap = hp.read_map(fname)
    delta[ii]=(galmap/(sum(galmap)/sum(mask))-1)*mask
    
# load cls:
cl_00=np.loadtxt(savedir+"clgg-theory-z_bin-nside-512.txt")

# make cls in form of data
# add shotnoise!
pshot = np.array([6.451400144212375e-07, 7.381277724137032e-07, 5.58668132340528e-07, 3.943298655590948e-07, 3.249200879550556e-07])
for ii in range(5):
    cl_00[:,ii] += pshot[ii]
    
# spin-0
g_field = nmt.NmtField(mask, [delta[0]], masked_on_input=True)

# save the coupling matrix
w00 = nmt.NmtWorkspace()
w00.compute_coupling_matrix(g_field, g_field, ell_bins)

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
    fname = savedir+f"cov-nmt-clgg-maglim-tomo-{ii}{vd_tag}-{nside}-binning-20-2000.txt"
    np.savetxt(fname, covar_TT_TT)
    print("Written: ", fname)
