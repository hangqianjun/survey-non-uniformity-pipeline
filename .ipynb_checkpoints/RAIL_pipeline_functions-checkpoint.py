# some of these are included in RAIL degrader!!

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
    

    
def get_semi_major_minor(data, scale=1):

    q = (1 - data['ellipticity'])/(1 + data['ellipticity'])
    ai = data['size']
    bi = ai*q

    ai = ai.to_numpy()*scale
    bi = bi.to_numpy()*scale
    
    return ai, bi


def compute_mag_err(mag, ai, bi, assigned_obs_cond, obs_cond, band='u', m5key='coaddm5_per_visit'):
    
    A_ratio = svf.get_area_ratio_auto(ai, bi, assigned_obs_cond['theta'][band])
    err = svf.compute_sigma_rand_sq(
        assigned_obs_cond[m5key][band], 
        mag, 
        obs_cond['gamma'][band],
        assigned_obs_cond['nvis'][band],  
        A_ratio=A_ratio
    )
    err = svf.compute_magerr_lsstmodel(
        err, 
        obs_cond['sigmasys'], 
        highSNR=True
    )
    magerr = err*obs_cond['calibration']['magerrscale'][band]
        
    return magerr


def assign_new_mag_magerr(data, magerr, ai, bi, assigned_obs_cond, obs_cond, bands, rng):

    totObsIndex = 1
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

        #index for selecting samples within the 
        ind = newmags<obs_cond['sigLim'][b]
        
        totObsIndex *= ind
        
        # new magnitudes:
        newmags[~ind] = np.nan
        ObsMags[:,ii] = np.copy(newmags)
        
        # new errors:
        
        mag = ObsMags[:,ii]
        ObsMagsErr[:,ii] = compute_mag_err(mag, ai, bi, assigned_obs_cond, 
                             obs_cond, band=b, m5key='coaddm5_per_visit')
    totObsIndex = totObsIndex.astype(bool)
    
    return totObsIndex, ObsMags, ObsMagsErr


def join_tables_and_save(data, ObsMags, ObsMagsErr, pixels, totObsIndex, outfile):

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

    obsCatalog = pd.concat([df, magDf], axis=1)
    obsCatalog = pd.concat([obsCatalog, errDf], axis=1)
    obsCatalog = pd.concat([obsCatalog, pixDf],axis=1)

    #finally select the indices to use:
    savecat = obsCatalog.loc[totObsIndex, :]

    svf.dump_save(savecat, outfile)
    
    print(f"Saved: {outfile}.")
    
    
def obs_cond_pipeline(data, usepixels, random_seed, rng, obs_cond, mask, sys, bands, outfile,
                      m5key='coaddm5_per_visit'):
    
    Ngal = len(mock_tract)

    assigned_pixels = assign_pixels_to_gals(Ngal, usepixels, random_seed=random_seed)
    #print('flag1')

    assigned_obs_cond = assign_obs_cond_to_gals(assigned_pixels, obs_cond, mask, sys, bands)
    #print('flag2')
    
    # assign photo-z errors:
    scale = obs_cond['calibration']['abscale']
    ai, bi = get_semi_major_minor(data, scale=scale)
    
    magerr = {}
    for b in bands:
        mag = data[b].to_numpy()
        magerr[b] = compute_mag_err(mag, ai, bi, assigned_obs_cond, 
                                 obs_cond, band=b, m5key='coaddm5_per_visit')
    #print('flag3')

    # apply error to get new magnitudes, compute new magnitudes, 
    # and cut objects beyond magnitude limits:
    rng = np.random.default_rng(10)
    totObsIndex, ObsMags, ObsMagsErr = assign_new_mag_magerr(data, magerr, ai, bi, 
                                                             assigned_obs_cond, 
                                                             obs_cond, bands, rng)
    #print('flag4')

    join_tables_and_save(data, ObsMags, ObsMagsErr, assigned_pixels, totObsIndex, outfile)
    
    
def convert_catalog_to_test_data(data, DS, bands):

    data2 = OrderedDict()

    key1 = 'ObsMagErr_'
    key2 = 'ObsMag_'

    for bb in bands:
        data2['mag_err_%s_lsst'%bb] = data[key1 + bb].to_numpy()
        data2['mag_%s_lsst'%bb] = data[key2 + bb].to_numpy()

    data2['redshift'] = data['redshift'].to_numpy()

    xtest_data = tables_io.convert(data2, tables_io.types.NUMPY_DICT)
    test_data = DS.add_data("test_data", xtest_data, TableHandle)
    
    return test_data


# this can be a summariser module
def get_nz_meanz(pz, truez, pzbins, nbootstrap, zlim = [0,2.0], bins=100):
    
    nztrue = {}
    meanztrue = np.zeros(len(pzbins)-1)
    stdmeanz = np.zeros(len(pzbins)-1)
    
    for ii in range(len(pzbins)-1):
        ind = (pz>= pzbins[ii])&(pz < pzbins[ii+1])
        ind = ind.flatten()
        cc = np.histogram(truez[ind], range=zlim, bins=bins)
        
        zz = (cc[1][1:] + cc[1][:-1])*0.5
        nztrue[ii] = np.c_[zz,cc[0]]
        
        # calculate true mean z
        #meanztrue[ii] = np.mean(truez[ind])
        meanztrue[ii] = np.sum(cc[0] * zz)/np.sum(cc[0])
        
        
        # stdmeanz using bootstrap method:
        sampholder = np.zeros(nbootstrap)
    
        data = truez[ind]
        for kk in range(nbootstrap):
            samp = np.random.choice(data, 
                            size=len(data),
                            replace=True)
            # repeat the operation 
            cc = np.histogram(samp, range=zlim, bins=bins)
            sampholder[kk] = np.sum(cc[0] * zz)/np.sum(cc[0])
            #sampholder[kk] = np.mean(samp)
        stdmeanz[ii] = np.std(sampholder)
        
    return nztrue, meanztrue, stdmeanz 


