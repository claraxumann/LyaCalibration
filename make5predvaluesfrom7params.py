import swiftemulator as se
from swiftemulator.design import latin
from swiftemulator.emulators import gaussian_process
from swiftemulator.emulators import multi_gaussian_process
from swiftemulator.mean_models.linear import LinearMeanModel
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.special import erf
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import StrMethodFormatter, NullFormatter
import matplotlib.patheffects as path_effects
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d
import illustris_python as il
import pickle
from scipy.optimize import curve_fit
from swiftemulator.comparison.penalty import L1PenaltyCalculator

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

model_specification = se.ModelSpecification(
    number_of_parameters=5,
    parameter_names=["log_Mhalo_min","log_Mhalo_max", "fesc_min", "a_mu", "b_mu"],
    parameter_limits=[[9.5,11.5],[10.5,12.],[0.05,0.4],[0.,2.],[1.,2.]],
    parameter_printable_names=["$M_{\\rm H, min}$", "$M_{\\rm H, max}$", "$f_E$", "$\mu_c$", "$\mu_o$"],
)

modelparameters_file = open('/nfs/mvogelsblab001/Thesan/Thesan-1/postprocessing/tau/model_parameters_5d_5.5.obj', 'rb')
model_parameters = pickle.load(modelparameters_file)
modelparameters_file.close()
n_models = len(model_parameters)

modelvalues = {}

for unique_identifier in model_parameters.model_parameters.keys():
    filename = f'/nfs/mvogelsblab001/Thesan/Thesan-1/postprocessing/tau/5d_LF_test_emulators_5.5_new/LF_5d_s0_080_{unique_identifier}.hdf5'
    
    with h5py.File(filename, 'r') as f: 
        y_avg_obs = f['y_avg_obs'][:].astype(np.float64)
        y_phi_obs = f['y_phi_obs'][:].astype(np.float64)
        y_err_obs = f['y_err_obs'][:].astype(np.float64)

        mask = (y_avg_obs > 10**41.5) # Ignore faint objects
        y_avg_obs = y_avg_obs[mask]
        y_phi_obs = y_phi_obs[mask]
        y_err_obs = y_err_obs[mask]
        
        log_yavg = np.log10(y_avg_obs)
        log_yphi = np.log10(y_phi_obs)
        log_yerr = 0.5 * (np.log10(y_phi_obs + y_err_obs) - np.log10(y_phi_obs - y_err_obs))

    modelvalues[unique_identifier] = {"independent": log_yavg, "dependent": log_yphi, "dependent_error": log_yerr}

model_values = se.ModelValues(model_values=modelvalues)

LF_emulator_obj = open('LF_emulator_5d_5.5_new.obj', 'rb')
LF_emulator = pickle.load(LF_emulator_obj)
LF_emulator_obj.close()

pred_params_obj = open('pred_params_5d.obj', 'rb')
pred_params = pickle.load(pred_params_obj)
pred_params_obj.close()

independent_variables = np.unique(
    [
        item
        for uid in LF_emulator.model_values.model_values.keys()
        for item in LF_emulator.model_values.model_values[uid]["independent"]
    ]
)

emulated_models = {}

for uid, pars in pred_params.model_parameters.items():
    dep, dep_err = LF_emulator.predict_values(
        independent=independent_variables,
        model_parameters=pars,
    )

    emulated_models[uid] = {
        "independent": independent_variables,
        "dependent": dep,
        "dependent_error": dep_err,
    }

pred_values = se.ModelValues(model_values=emulated_models)

pred_values_obj = open('pred_values_5d_5.5_new.obj', 'wb')
pickle.dump(pred_values, pred_values_obj)
pred_values_obj.close()
