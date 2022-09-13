import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.special import erf
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import StrMethodFormatter, NullFormatter
import matplotlib.patheffects as path_effects
from scipy.ndimage import gaussian_filter1d
import illustris_python as il
import pickle

# Universal constants
c = 2.99792458e10          # Speed of light [cm/s]
kB = 1.380648813e-16       # Boltzmann's constant [g cm^2/s^2/K]
h = 6.626069573e-27        # Planck's constant [erg/s]
mH = 1.6735327e-24         # Mass of hydrogen atom [g]
me = 9.109382917e-28       # Electron mass [g]
ee = 4.80320451e-10        # Electron charge [g^(1/2) cm^(3/2) / s]

# Emperical unit definitions
Msun = 1.988435e33         # Solar mass [g]
Lsun = 3.839e33            # Solar luminosity [erg/s]
Zsun = 0.0134              # Solar metallicity (mass fraction)
arcsec = 648000. / np.pi   # arseconds per radian
pc = 3.085677581467192e18  # Units: 1 pc  = 3e18 cm
kpc = 1e3 * pc             # Units: 1 kpc = 3e21 cm
Mpc = 1e6 * pc             # Units: 1 Mpc = 3e24 cm
km = 1e5                   # Units: 1 km  = 1e5  cm
angstrom = 1e-8            # Units: 1 angstrom = 1e-8 cm
day = 86400.               # Units: 1 day = 24 * 3600 seconds
yr = 365.24 * day          # Units: 1 year = 365.24 days
kyr = 1e3 * yr             # Units: 1 Myr = 10^6 yr
Myr = 1e6 * yr             # Units: 1 Myr = 10^6 yr
lambda_1216 = 1215.67 * angstrom # Lyman-alpha wavelength [cm]
lambda_1500 = 1500. * angstrom # Continuum wavelength [cm]
lambda_2500 = 2500. * angstrom # Continuum wavelength [cm]
R_10pc = 10. * pc              # Reference distance for continuum [cm]
fnu_1216_fac = lambda_1216**2 / (4. * np.pi * c * R_10pc**2 * angstrom)
fnu_1500_fac = lambda_1500**2 / (4. * np.pi * c * R_10pc**2 * angstrom)
fnu_2500_fac = lambda_2500**2 / (4. * np.pi * c * R_10pc**2 * angstrom)
E_AGN = 5.29e-11 # Mean photon energy [erg]
E_Lya = h * c / lambda_1216 # Lyman-alpha energy [erg]

params_combined_maxmean = {'log_Mhalo_min': 10.924026577681639, 
                           'log_Mhalo_max': 11.413897827775873, 
                           'fesc_min': 0.18520081754976464, 
                           'a_mu': 0.31800355494781263, 
                           'b_mu': 1.418108791867382}

def read_Lya(snap=70, sim='Thesan-1'):
    print(f'Reading snapshot {snap} ...')
    basePath = f'/nfs/mvogelsblab001/Thesan/{sim}/output'
    fields = ['SubhaloMass','SubhaloSFRinRad', 'SubhaloLenType', 'SubhaloMassType', 'SubhaloVelDisp', 'SubhaloPos', 'SubhaloGrNr']
#     fields = ['SubhaloLenType', 'SubhaloMass', 'SubhaloMassInRadType', 'SubhaloStellarPhotometrics', 'SubhaloStarMetallicity']
#   'SubhaloSFRinRad', 'SubhaloCM', 'SubhaloPos', 'SubhaloVel', 'SubhaloVelDisp']
    s = il.groupcat.loadSubhalos(basePath, snap, fields=fields)

    s['snap'] = snap
    s['basePath'] = basePath
    Lya_filename = basePath + f'/../postprocessing/Lya/Lya_{snap:03d}.hdf5'
    with h5py.File(Lya_filename, 'r') as f:
        g = f['Header']
        for key in ['BoxSize', 'EscapeFraction', 'Omega0', 'OmegaBaryon', 'OmegaLambda', 'HubbleParam', 'Time', 'Redshift']:
            s[key] = g.attrs[key]
#         BoxSize = g.attrs['BoxSize']
#         EscapeFraction = g.attrs['EscapeFraction']
#         Omega0 = g.attrs['Omega0']
#         OmegaBaryon = g.attrs['OmegaBaryon']
#         OmegaLambda = g.attrs['OmegaLambda']
        h = g.attrs['HubbleParam']
        a = g.attrs['Time']
        z = g.attrs['Redshift']
        UnitLength_in_cm = g.attrs['UnitLength_in_cm']
        UnitMass_in_g = g.attrs['UnitMass_in_g']
        UnitVelocity_in_cm_per_s = g.attrs['UnitVelocity_in_cm_per_s']
        UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s
        UnitEnergy_in_cgs = UnitMass_in_g * UnitVelocity_in_cm_per_s * UnitVelocity_in_cm_per_s
        UnitLum_in_cgs = UnitEnergy_in_cgs / UnitTime_in_s
        length_to_cgs = a * UnitLength_in_cm / h
        length_to_kpc = length_to_cgs / kpc
        volume_to_cgs = length_to_cgs * length_to_cgs * length_to_cgs
        s['BoxSize_Mpc'] = s['BoxSize'] * length_to_cgs / Mpc # Box size in Mpc (physical)
        s['V_box_Mpc3'] = s['BoxSize_Mpc']**3 # Box volume in Mpc^3 (physical)
        s['V_box_cMpc3'] = s['V_box_Mpc3'] / s['Time']**3
        mass_to_cgs = UnitMass_in_g / h
        mass_to_Msun = mass_to_cgs / Msun

        s['SubhaloMass'] = mass_to_Msun * s['SubhaloMass'] # Msun
        s['Lya'] = UnitLum_in_cgs * f['Subhalo']['LyaLum'][:].astype(np.float64) # Lya = LyaCol + LyaRec + LyaStars
#         s['LyaCol'] = UnitLum_in_cgs * f['Subhalo']['LyaLumCol'][:].astype(np.float64) # erg/s
#         s['LyaRec'] = UnitLum_in_cgs * f['Subhalo']['LyaLumRec'][:].astype(np.float64) # erg/s
#         s['LyaStars'] = UnitLum_in_cgs * f['Subhalo']['LyaLumStars'][:].astype(np.float64) # erg/s
#         s['L1216'] = UnitLum_in_cgs * f['Subhalo']['1216LumStars'][:].astype(np.float64) # erg/s/Angstrom
#         s['L1500'] = UnitLum_in_cgs * f['Subhalo']['1500LumStars'][:].astype(np.float64) # erg/s/Angstrom
#         s['L2500'] = UnitLum_in_cgs * f['Subhalo']['2500LumStars'][:].astype(np.float64) # erg/s/Angstrom
#         s['IonAGN'] = UnitLum_in_cgs * f['Subhalo']['IonLumAGN'][:].astype(np.float64) # erg/s
#         mask = (s['IonAGN'] > 0.)
#         y_AGN = s['IonAGN'][mask] / E_AGN # AGN ionizing photon rate [photons/s]
#         y_stars = s['LyaStars'][mask] / (0.68 * E_Lya * (1. - s['EscapeFraction'])) # Star ionizing photon rate [photons/s]
#         s['f_AGN'] = y_AGN / (y_AGN + y_stars)
#         s['M1216'] = -2.5 * np.log10(fnu_1216_fac * s['L1216']) - 48.6 # Continuum absolute magnitude
#         s['M1500'] = -2.5 * np.log10(fnu_1500_fac * s['L1500']) - 48.6 # Continuum absolute magnitude
#         s['M2500'] = -2.5 * np.log10(fnu_2500_fac * s['L2500']) - 48.6 # Continuum absolute magnitude
#         s['LyaVelDisp'] = f['Subhalo']['LyaVelDisp'][:][mask].astype(np.float64) # km/s
        
        # remove low LyaLum values
        mask = (s['Lya'] > 10**41.5)
        s['Lya'] = s['Lya'][mask]
        for field in fields:
            s[field] = s[field][mask]
        # remove high LyaLum values
        mask = (s['Lya'] < 10**45)
        s['Lya'] = s['Lya'][mask]
        for field in fields:
            s[field] = s[field][mask]
        
    return s

def read_spectra(snap, sim): #
    if sim=='Thesan-WC-2' or sim=='Thesan-sDAO-2':
        tau_dir = f'/nfs/mvogelsblab001/Lab/thesan-lya/{sim}/tau'
    elif sim=='Thesan-1':
        tau_dir = '/nfs/mvogelsblab001/Thesan/Thesan-1/postprocessing/tau'
    else:
        tau_dir=f'/pool001/users/claraxu/Thesan/{sim}/tau'
    s = {'snap': snap}
    filename = tau_dir + f'/full_stats/tau_{snap:03d}.hdf5'
    
    with h5py.File(filename, 'r') as f:
#         print([key for key in f.keys()])
#         print([item for item in f.attrs.items()])
        Dv_min = f.attrs['Dv_min']
        Dv_max = f.attrs['Dv_max']
        n_freq = f.attrs['NumFreq']
        s['n_bands'] = f.attrs['NumBands']
        s['freq'] = np.linspace(Dv_min, Dv_max, n_freq)
        
        for key in ['T_IGM_avg', 'TauIGM_16', 'TauIGM_50', 'TauIGM_84', 'TauIGM_avg']: # {freqs}
            s[key] = f[key][:]
        s['T_IGM_50'] = np.exp(-s['TauIGM_50'])
        s['T_IGM_16'] = np.exp(-s['TauIGM_16'])
        s['T_IGM_84'] = np.exp(-s['TauIGM_84'])
    
    return s

def read_transmission(snap, sim): # change this if you want to change the sightline
    if sim=='Thesan-WC-2' or sim=='Thesan-sDAO-2':
        tau_dir = f'/nfs/mvogelsblab001/Lab/thesan-lya/{sim}/tau'
    elif sim=='Thesan-1':
        tau_dir = '/nfs/mvogelsblab001/Thesan/Thesan-1/postprocessing/tau'
    else:
        tau_dir=f'/pool001/users/claraxu/Thesan/{sim}/tau'
        
    s_galaxies = {'snap': snap} # gotta stop calling everything s
    
    filename = tau_dir + f'/tau/tau_{snap:03d}.hdf5'
    
    with h5py.File(filename, 'r') as f:
#         print([key for key in f.keys()])
#         print([item for item in f.attrs.items()])
        Dv_min = f.attrs['Dv_min']
        Dv_max = f.attrs['Dv_max']
        n_freq = f.attrs['NumFreq']
        s_galaxies['freq'] = np.linspace(Dv_min, Dv_max, n_freq)
        
        for key in ['GroupIDs', 'SubhaloIDs']: 
            s_galaxies[key] = f[key][:] 
        s_galaxies['TauIGMs'] = f['TauIGMs'][:,0,:].astype(np.float64) # {galaxies, sightlines, freqs}. 0th sightline
        s_galaxies['T_IGM'] = np.exp(-s_galaxies['TauIGMs'])
    
    return s_galaxies

def calculate_x_dust(snap, sim, mp): # from make_x_files + dust
    log_Mhalo_min = mp['log_Mhalo_min']
    log_Mhalo_max = mp['log_Mhalo_max']
    fesc_min = mp['fesc_min']
    a_mu = mp['a_mu']
    b_mu = mp['b_mu']
    
    s_lya = read_Lya(snap, sim)
    s = read_spectra(snap, sim)
    s_galaxies = read_transmission(snap, sim)

    v = s['freq']
    v_c = s_lya['SubhaloVelDisp']
    halomasses = s_lya['SubhaloMass']
    
    V_box_cMpc3 = s_lya['V_box_cMpc3'] # solely for the purpose of calculating volume normalization later

    T_IGM_avg = s['T_IGM_50'] # an array over frequency
    T_IGM_galaxies = s_galaxies['T_IGM'] # a 2d array: galaxy, frequency at sightline 0
    n_centrals = len(T_IGM_galaxies)

    LyaLum = s_lya['Lya']
    n_galaxies = len(LyaLum)
    print(n_galaxies)
    Xs = np.zeros(n_galaxies) 

    transmission_ids = s_galaxies['SubhaloIDs']

    group_ids_centrals = s_galaxies['GroupIDs']
    group_ids_all = s_lya['SubhaloGrNr']

    T_IGM_groups = {} # dictionary of the transmission value of each group, keys are the group ids

    for i in range(n_centrals): # making the dict of centrals, transmission values for each group
        group_id = group_ids_centrals[i]
        T_IGM_groups[group_id] = T_IGM_galaxies[i][:]

    # intrinsic X
    for i in range(n_galaxies): # calculating observed lum of each galaxy
        sigma = 200. # km/s
        mu = a_mu*v_c[i] + 10**b_mu
        spectrum = np.exp(-0.5*((v-mu)/sigma)**2) / sigma

        group_id = group_ids_all[i]
        if group_id in T_IGM_groups: # if this group has a centrals. aka if this group id is a key in the dict of keys of the centrals. 
            T_IGM = T_IGM_groups[group_id]
        else:
            T_IGM = T_IGM_avg

        idk = spectrum * T_IGM # thing to integrate. not sure what to call it
        numerator = np.sum(idk) 
        denominator = np.sum(spectrum)
        Xs[i] = numerator/denominator # intrinsic X
    
    # adding dust model
    fesc = np.zeros(n_galaxies)
    
    for i in range(n_galaxies):
        halomass = halomasses[i]
        loghalo = np.log10(halomass)
        if loghalo < log_Mhalo_min:
            fesc[i] = 1.
        elif loghalo > log_Mhalo_max:
            fesc[i] = fesc_min
        else:
            fesc[i] = 1. + (loghalo - log_Mhalo_min) * (fesc_min - 1.)/(log_Mhalo_max - log_Mhalo_min) # point slope form
    
    X_dust = fesc * Xs

    return X_dust, LyaLum, V_box_cMpc3

def calc_hist(data, n_0=7, n_max=50000, reverse=True, use_median=True):
    y = np.sort(data)
    if reverse:
        y = y[::-1] # Start from high mass end
    n_y = len(y)
    n_c = n_i = n_0
    n_bins = 1
    while n_c < n_y:
        n_i = min(2 * n_i, n_max, n_y - n_c)
        n_c += n_i
        n_bins += 1
    y_edges = np.zeros(n_bins+1)
    y_avg = np.zeros(n_bins)
    y_num = np.zeros(n_bins)
    y_edges[1] = 0.5 * (y[n_0-1] + y[n_0])
    y_edges[0] = y[0] + (y[n_0-1] - y_edges[1])
    y_avg[0] = np.median(y[0:n_0]) if use_median else np.mean(y[0:n_0])
    y_num[0] = n_0
    if reverse:
        assert y[n_0-1] > y[n_0]
        assert y_edges[0] > y[0]
    else:
        assert y[n_0-1] < y[n_0]
        assert y_edges[0] < y[0]
    n_c = n_i = n_0
    i_bin = 1
    while n_c < n_y:
        n_i = min(2 * n_i, n_max, n_y - n_c)
        y_edges[i_bin] = 0.5 * (y[n_c-1] + y[n_c])
        y_avg[i_bin] = np.median(y[n_c:n_c+n_i]) if use_median else np.mean(y[n_c:n_c+n_i])
        y_num[i_bin] = n_i
        n_c += n_i
        i_bin += 1
    y_edges[-1] = y[-1]
#     if False:
#         y_edges[-2] = y_edges[-1]
#         y_avg[-2] = (y_avg[-2]*y_num[-2] + y_avg[-1]*y_num[-1]) / (y_num[-2] + y_num[-1])
#         y_num[-2] = y_num[-2] + y_num[-1]
#         y_edges = y_edges[:-1]
#         y_avg = y_avg[:-1]
#         y_num = y_num[:-1]
    dy = y_edges[1:] - y_edges[:-1]
    if reverse:
        dy = -dy
#     print(n_bins)
#     print(np.sum(y_num)-n_y)
    return y_avg, y_num/dy, np.sqrt(y_num)/dy

def write_lf_test_emulators(snap, sim, mp, label):
    if sim=='Thesan-WC-2' or sim=='Thesan-sDAO-2':
        tau_dir = f'/nfs/mvogelsblab001/Lab/thesan-lya/{sim}/tau'
    elif sim=='Thesan-1':
        tau_dir = '/nfs/mvogelsblab001/Thesan/Thesan-1/postprocessing/tau'
    else:
        tau_dir=f'/pool001/users/claraxu/Thesan/{sim}/tau'
    
    X_dust, LyaLum, V_box_cMpc3 = calculate_x_dust(snap, sim, mp)

    LyaLumObserved = X_dust * LyaLum
    LyaLumObserved = LyaLumObserved[~np.isnan(LyaLumObserved)]
    
#     print(LyaLumObserved)
    
    y_avg_obs, y_phi_obs, y_err_obs = calc_hist(np.log10(LyaLumObserved), n_0=6)
    y_avg_obs = y_avg_obs[:-1]; y_phi_obs = y_phi_obs[:-1]; y_err_obs = y_err_obs[:-1]; # Remove tail
    y_avg_obs = 10.**y_avg_obs; y_phi_obs /= V_box_cMpc3; y_err_obs /= V_box_cMpc3; # Volume normalization
    
    with h5py.File(f'{tau_dir}/LF_bestfits/LF_5d_s0_{snap:03d}_{label}.hdf5', 'w') as f: 
        f.create_dataset('y_avg_obs', data=y_avg_obs, dtype=np.float64)
        f.create_dataset('y_phi_obs', data=y_phi_obs, dtype=np.float64)
        f.create_dataset('y_err_obs', data=y_err_obs, dtype=np.float64)

write_lf_test_emulators(snap=80, sim='Thesan-1', mp=params_combined_maxmean, label='limittest')
