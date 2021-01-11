import os, sys
from argparse import ArgumentParser
import json
import numpy as np
import pickle

import osn_model as osn

parser = ArgumentParser()
parser.add_argument("params_file", type=str, help="Path to the json parameter file to run.")
parser.add_argument("--save",  type=str, default="0", help="Save the voltages and spikes for for every N OSNs > 0, a comma separated subset, or 0 for none.")
args = parser.parse_args()


dir_name, file_name = os.path.split(args.params_file)
param_name = file_name.split(".json")[0]

with open(args.params_file, "r") as f:
    params = json.load(f)

n_osn = params["n_osn"]
ind_save = []
if args.save != "0":
    if "," in args.save:
        ind_save = np.array([int(i) for i in args.save.split(",")])
        print(f"Saving voltage and spiking activity for {list(ind_save)}.")
    else:
        every = int(args.save)
        ind_save = np.arange(0, n_osn, every)
        print(f"Saving voltage and spiking activity for {every=} for a total of {len(ind_save)}.")
else:
    print("Saving only spike counts.")

    
print(f"Running with parameters:")
for fld, val in params.items():
    print(f"{fld:>12s}: {val}")

stim, t, V, S, counts = {}, {}, {}, {}, {}
base_seed = params["seed"]
widths = {"p":params["t_pre"], "S":params["t_on"], "g":params["gaps"][0], "h":params["gaps"][1]}
for i, pat in enumerate(["pSgS!", "pShS!"]):
    stim[pat], t[pat]  = osn.str2stim(pat, widths, t_trial=params["t_trial"], n_trials=params["n_trials"], dt=params["dt"])
    for j, input_amp in enumerate(params["input_amps"]):
        this_params = dict(params)
        this_params["seed"] = base_seed + j*100000 + i*10000 # A different seed for each gap
        amp = this_params["amp"] * input_amp
        del this_params["amp"]
        print(f'Running {pat=}, {input_amp=:1.3f}, {amp=:1.3f}, with seed {this_params["seed"]}')        
        VV,SS, *_ = osn.run(stim[pat], amp, **this_params)
        counts[pat,input_amp] = np.sum(SS,axis=1)
        if len(ind_save):
            V[pat,input_amp], S[pat,input_amp] = np.copy(VV[ind_save]), np.copy(SS[ind_save])

files = {"t":t, "V":V, "S":S, "stim":stim, "counts":counts, "ind_save":ind_save}
output_dir = os.path.join(dir_name, param_name)
print(f"Writing output to {output_dir}.")
os.system(f"mkdir -p {output_dir}")
for f, data in files.items():
    if f in ["V", "S", "ind_save"] and not len(ind_save):
        print(f"Not saving {f}.")
        continue
    full_file = os.path.join(output_dir, f + ".p")
    with open(full_file, "wb") as fp:
        pickle.dump(data, fp)
    print(f"Wrote {full_file}")
print("ALLDONE")    
