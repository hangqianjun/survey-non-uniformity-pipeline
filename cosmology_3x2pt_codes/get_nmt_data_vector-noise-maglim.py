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
savedir="/pscratch/sd/q/qhang/dirac_mock/desc-project285/"
#np.savetxt(savedir + "effective_ells.txt", ell_arr)

# mask
root="/pscratch/sd/q/qhang/glass_mock/catalog_info/mask/"
mask=hp.read_map(root+"wfd_footprint_nvisitcut_500_nside_128-ebv-0.2-1024.fits")
mask=hp.ud_grade(mask,nside)

# field # dummy
delta={}
shear={}
shape_noise={}
for ii in range(5):
    fname=savedir+f"galmap{vd_tag}-lens-maglim-tomo-{ii}-nside-512.fits"
    galmap = hp.read_map(fname)
    delta[ii]=(galmap/(sum(galmap)/sum(mask))-1)*mask
    
    
# spin-0
g_field = nmt.NmtField(mask, [delta[0]], masked_on_input=True)

# now measure 3x2pt for each bin:
# save the coupling matrix
w = nmt.NmtWorkspace()
w.compute_coupling_matrix(g_field, g_field, ell_bins)
window=w.get_bandpower_windows()
np.save(savedir+'wb-clgg-binning-20-2000-nell-20-log.npy', window)

# clgg
xcl=[]
for zz in range(5):
    g_field = nmt.NmtField(mask, [delta[zz]], masked_on_input=True)
    cl_master = compute_master(g_field, g_field, w, None)
    xcl.append(cl_master[0])#TT
fname=savedir+f'data-nmt-clgg{vd_tag}-maglim-{nside}-binning-20-2000.txt'
np.savetxt(fname,xcl)
print("Written: ", fname)
