import os, sys
from argparse import ArgumentParser
import numpy as np
import json

parser = ArgumentParser(description="Generate parameter files for simulating model OSN data for the 2 pulse condition with  multiple amplitudes.")

opt_args = [
    ("outputdir",   "Output directory",                  str,  "./model_osn_data_two_pulse_multiple_amps"),
    ("dt",          "Time step size",                    float, 1e-3),
    ("t_trial",     "Trial length in seconds",           float, 2.5),
    ("t_pre",       "Pre-trial length in seconds",       float, 0.1),
    ("t_on",        "Length of each pulse in seconds",   float, 0.01),
    ("n_trials",    "Number of trials",                  int,   25),
    ("n_osn",       "Number of OSNs per glomerulus",     int,   5000),
    ("n_glom",      "Number of glomeruli to generate",   int,   100),
    ("glom_sd",     "S.D. of the membrane noise.",       float, 0.25),
    ("glom_sd_var", "Amplitude of the variability of the glom S.D.",       float, 0.25),        
    ("glom_amp",    "Mean of the unit glomerular response", float, 15),
    ("var_scale",   "Scale of the variability of glomerular parameters around their mean.", float, 0.25),
    ("input_amps",  "Input amplitudes to use as a comma-separated string", str, "1,2,3,4,5,6,7,8,9,10"),
    ("input_scale", "How much to scale the input amplitudes", float, 0.5),
]

for name, descr, tp, default in opt_args:
    parser.add_argument(f"--{name}", type=tp, default=default, help=f"{descr}. (default: {default})")

args = parser.parse_args()
print(f"Running with {args=}.")
# The pulse patterns are going to be pSgS!, and pShS!
# g is the short gap, h is the long gap.
defaults = {
    "dt":args.dt,   # Integration time constant
    "t_on":args.t_on,   # Length of each pulse in seconds. (S)
    "t_pre":args.t_pre,    # Time before first pulse in each trial. (p)
    "t_trial":args.t_trial,   # Length of each trial in seconds. 
    "gaps":[0.01,0.025],    # The size of the two interpulse gaps. (g, h)
    "n_trials":args.n_trials,
    "n_osn":args.n_osn, # Number of OSNs per glomerulus
    "input_amps":[float(a)*args.input_scale for a in args.input_amps.split(",")] # The input amplitudes we'll try.
}
print("Default parameters:")
print(defaults)

# These are the glomerular parameters.
# They're specified as a center plus the width of the uniform interval around the center
# from which to generate variation.
glom_params = { 
    "tau_chem":(0.075,          args.var_scale),
    "tau_osn": (0.075,          args.var_scale),
    "sd":      (args.glom_sd,   args.glom_sd_var),        
    "th":      (2,              args.var_scale),
    "amp":     (args.glom_amp,  args.var_scale),
}
print("Glomerular parameters:")
print(glom_params)

if os.path.isdir(args.outputdir):
    resp = input(f"\nDirectory {args.outputdir} exists. Parameter files may be overwritten. Type 'yes' to continue: ")
    if resp != "yes":
        print("Aborted.")
        exit(0)
else:
    print(f"Creating target directory {args.outputdir}")
    os.makedirs(args.outputdir, exist_ok = True)    

# Each one of these will be a glomerulus, with its parameters randomly generated from above.
for seed in range(args.n_glom):
    np.random.seed(seed)
    p = {fld:cntr * (1 + amp*(2*np.random.rand() - 1)) for fld, (cntr, amp) in glom_params.items()}
    p["seed"] = seed        
    p.update(defaults)
    file_name = f"params{seed}.json"
    full_path = os.path.join(args.outputdir, file_name)
    with open(full_path, "w") as f:
        json.dump(p, f)
        print(f"Wrote {full_path}.")

# Also write the parameters that that were used to generate the run parameters.
with open(input_args_file:=os.path.join(args.outputdir, "input_args.json"), "w") as f:
    input_args = {name:args.__getattribute__(name) for (name, *rest) in opt_args}
    json.dump(input_args, f)
    print(f"Wrote {input_args_file}.")

with open(base_params_file:=os.path.join(args.outputdir, "base_params.json"), "w") as f:
    json.dump(defaults, f)
    print(f"Wrote {base_params_file}.")
        
with open(glom_params_file:=os.path.join(args.outputdir, "glom_params.json"), "w") as f:
    json.dump(glom_params, f)
    print(f"Wrote {glom_params_file}.")

    
        
