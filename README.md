# LyaCalibration

Other files that are necessary, but aren't in the repo because they're in the Thesan-1 tau directory:
- model_parameters_5d_{5.5, 6.6}.obj
- make_LF_models_5d_new.py
- X_emulated.py

Order to run files:
* Run make_LF_models_5d_new.py first, which creates the model LFs to calibrate to. Change parameter_file depending on the redshift, and the snap at the end when calling write_lf_test_emulators, and also the directory name to write files to in that function
* Run train_emulator_5d-{redshift}.ipynb up til the “Penalty functions” header (can probably change directory or file names)
    * These use the model_parameters files, which are the same as before
* Run make{5, 6}predvaluesfrom7params.py, which uses pred_params_5d.obj (doesn’t need to change) and the pickled trained LF emulator to make pred values for all the pred params
* Keep running train_emulator_5d-{redshift}.ipynb until the last plot before the “idk hist stuff aka sweeps” header. This plot has the real calculated best fit from the best fit parameters
* Run best_fit_5d.py to calculate the best fit. Change the params variable to the best fit parameters, the snap when calling write_lf_test_emulators, and maybe the label (which I mostly used to record the way I calculated the penalty)
* Run the last plot before the sweeps in train_emulator_5d-{redshift}.ipynb
* Run combiningpenalties5-6.ipynb up to the “Old models, no dust” header (after finishing everything above for both redshifts)
* Run the sweeps (last part) of train_emulator_5d-{redshift}.ipynb, changed combined_key to the combined best fit key in pred_params 
* Run X_emulated.py several times to calculated X for all galaxies and sightlines at different redshifts and with different best fit parameters. Change best_fit_params to be the combined best fit parameters, params_6_max to the best fit parameters from only the 6.6 fitting, and similarly for params_5_max. Run the file with params = params_6_max for all snaps, and with params = params_{5-6}_max for calculating X with the independent redshift best fits (when calculating these I also changed the file name to include “ind” at the end, and that’s how the script is right now - probably get rid of that when running the combined best fit. That’s also the case with the calc_lf_sightlines function in combiningpenalties5-6.ipynb, where the final LF file also has “ind” in the name) (this also takes quite a while and took several job submissions to finish)
    * I also ran this file for all other simulations at all snaps (except 60) with the combined best fit params
    * If calculating X without dust, I just changed the file manually (made calculate_x_dust return Xs instead X_dust, changed the filename to include “nodust”)
* Run the rest of combiningpenalties5-6.ipynb
