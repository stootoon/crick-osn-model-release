import os, sys
from argparse import ArgumentParser
import pickle
import numpy as np

base_dir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]

sys.path.append(base_dir)
from classify import generate_dataset, classify

data_dir = os.path.join(base_dir, "data")

get_dataset_name         = lambda dataset_path: os.path.normpath(dataset_path).split("/")[-1]
get_conc_dir             = lambda dataset_path, which_conc: os.path.join(data_dir, f"decoding-ppi/{get_dataset_name(dataset_path)}_conc{which_conc}")
get_input_file_name      = lambda dataset_path, which_conc: os.path.join(get_conc_dir(dataset_path, which_conc), "input.p")
get_output_file_name     = lambda dataset_path, which_conc, which_perm: os.path.join(get_conc_dir(dataset_path, which_conc), f"output.{which_perm:03d}.p")
get_collection_file_name = lambda dataset_path, which_conc: os.path.join(get_conc_dir(dataset_path, which_conc), f"output.collected.p")

acc_from_cm              = lambda cm: np.trace(cm)/np.sum(cm) # Accuracy from confusion matrix

if __name__ == "__main__":
    parser = ArgumentParser("Generate the PPI decoding results.") 
    parser.add_argument("dataset",       type=str, default="model_osn_data_two_pulse_multiple_amps_gamp5", help="Folder containing the glomerular spike counts.")   
    parser.add_argument("which_conc",    type=str,                   help= "Which concentration to use, as a zero padded string e.g. '050', '100', etc.")
    parser.add_argument("--which_sizes", type=str, default="1,-1,1", help="Which sizes to run, in 'First,Last,Step' format. -1 means last.")
    parser.add_argument("--which_perm",  type=int, default=0,        help="Which permutation to run.")
    parser.add_argument("--C_vals",      type=str, default="-4,4,1", help="Comma-separated list of the minimum,maximum and step in powers of 10 to use for C.")
    parser.add_argument("--collect",     action="store_true",        help="Collect results from output folder.")    
    args = parser.parse_args()
    print(f"Running with {args=}")
    
    conc_dir = get_conc_dir(args.dataset, args.which_conc)
    print(f"{conc_dir=}")
    if not os.path.isdir(conc_dir):
        print(f"Concentration directory {conc_dir} does not exist. Creating.")
        os.makedirs(conc_dir, exist_ok = True)
        
    input_file = get_input_file_name(args.dataset, args.which_conc)
    if not os.path.isfile(input_file):
        print(f"Classifier input file {input_file} not found.")
        
        print(f"GENERATING CLASSIFER INPUTS.")
        print(f"Determining labels for conc={args.which_conc}.")
        _,y,_ = generate_dataset(args.dataset, "ppi.conc", labels_only = True)
        labels = sorted(list(set(y)))
        concs_avail = sorted(list(set([lab.split(".")[1] for lab in labels])))
        print(f"{concs_avail=}")
        if args.which_conc not in concs_avail:
            raise ValueError(f"Desired concentration {args.which_conc} is not available.")
    
        print(f"Generating dataset for conc={args.which_conc}.")        
        X, y, params = generate_dataset(args.dataset, "ppi.conc", labels_only = False)
        
        ind_conc = [i for i,lab in enumerate(y) if lab.endswith(args.which_conc)]
        X_conc = X[ind_conc]
        y_conc = [yi for yi in y if yi.endswith(args.which_conc)]
        print(f"Found {len(y_conc)} trials for {args.which_conc=}")
    
        with open(input_file, "wb") as f:
            pickle.dump({"X":X_conc, "y":y_conc, "source_dataset":args.dataset, "params":params}, f)
        print(f"Wrote {input_file}.")
    
    if args.which_perm >= 0 and not args.collect:
        print(f"RUNNING CLASSIFICATION.")
        file_name = get_input_file_name(args.dataset, args.which_conc)
    
        print(f"Loading classification inputs from {file_name}.")
        with open(file_name, "rb") as f:
            data = pickle.load(f)
        X_conc, y_conc = data["X"], data["y"]
        
        C_min, C_max, C_step = [int(val) for val in args.C_vals.split(",")]
        C_vals = [10**i for i in range(C_min, C_max+1, C_step)]
        print(f"Classifying using {C_vals=}")
    
        print(f"Running  permutation {args.which_perm}")
        n_glom = X_conc.shape[1]
        perf   = np.zeros((n_glom, 2))
        seeds  = []
    
        first_size, last_size, step_size = [int(i) for i in args.which_sizes.split(",")]
        last_size = n_glom if last_size == -1 else last_size
        sizes  = [i for i in range(first_size, last_size, step_size)]
        if n_glom not in sizes:
            sizes.append(n_glom)
        print(f"Running for {len(sizes)} starting at {sizes[0]}, ending at {sizes[-1]}, which a stepsize of {step_size}.")
        
        for isz, sz in enumerate(sizes):
            seeds.append(sz*1000 + args.which_perm)
            np.random.seed(seeds[-1])
            isub = np.random.permutation(n_glom)[:sz]
            X1   = np.copy(X_conc[:, isub])        
            perf[isz, :]  = [acc_from_cm(classify(X1, y_conc, C_vals = C_vals, seed = seeds[-1], shuf=shuf)[0]) for shuf in [0,1]]
            print(f"{sz:>3d}: {perf[isz,0]:1.3f}, {perf[isz,1]:1.3f}")
    
        output_file = get_output_file_name(args.dataset, args.which_conc, args.which_perm)
        print(f"Writing results to {output_file}.")
        with open(output_file, "wb") as f:
            pickle.dump({"perf":perf, "sizes":sizes, "seeds":seeds, "which_perm":args.which_perm, "C_vals":C_vals}, f)
    
    if args.collect:
        print(f"COLLECTING RESULTS FROM {conc_dir}")
        perfs = []
        which_perm = 0
        sizes = []
        while os.path.isfile(output_file := os.path.join(conc_dir, get_output_file_name(args.dataset, args.which_conc, which_perm))):
            print(f"Loading {output_file}")
            with open(output_file, "rb") as f:
                data = pickle.load(f)
            if sizes == []:
                sizes = data["sizes"]
            else:
                assert sizes == data["sizes"], "Size vectors don't match."
            perfs.append(data["perf"])
    
            which_perm += 1
        perfs = np.array(perfs)                    
        print(f"Found data for {len(perfs)} permutations.")

        collection_file = get_collection_file_name(args.dataset, args.which_conc)
        with open(collection_file, "wb") as f:
            pickle.dump({"perfs":perfs, "sizes":sizes}, f)
        print(f"Wrote collected results to {collection_file}.")
                           
    print("ALLDONE")
        
        
        
                
        
        
        
                    

        
