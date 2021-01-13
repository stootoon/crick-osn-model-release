import os, sys, json
import pandas as pd
import numpy as np
from builtins import sum as bsum
from tqdm import tqdm
import pickle
import logging

logging.basicConfig()
logger = logging.getLogger("classify")
logger.setLevel(logging.INFO)

base_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(base_dir, "data")

get_output_root = lambda osn_data_dir: os.path.join(data_dir, "decoding-delay-conc", os.path.normpath(osn_data_dir).split("/")[-1])


from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
    
def generate_dataset(osn_data_dir, which_label, first_trial=5, start_time = 0.1, window_size = 2,
                     ca2exp=2, ca2tau=0.15, labels_only = False):
    logger.info(f"Loading OSN data from {osn_data_dir}.")
    records = []
    for f in [g for g in os.scandir(osn_data_dir) if g.name.startswith("params") and g.name.endswith(".json")]:
        file_path = os.path.join(osn_data_dir, f.name)
        with open(file_path, "rb") as fp:
            record = json.load(fp)
        record.update({"file":f.name, "folder":f.name.strip(".json")})
        records.append(record)
    df_params = pd.DataFrame(records)
    logger.info(f"Read {len(df_params)} parameters.")
    logger.info(df_params.head())
    
    n_trials = set(df_params.n_trials.values)
    assert len(n_trials) == 1, f"Expected exactly one value for n_trials, found {len(n_trials)}." 
    n_trials = list(n_trials)[0]    
    logger.info(f"{n_trials=}")
    if n_trials< first_trial:
        raise ValueError("First trial {first_trial} > {n_trials=}")

    dt = set(df_params.dt.values)
    assert len(dt) == 1, f"Expected exactly one value for dt, found {len(dt)}."
    dt = list(dt)[0]
    
    t_trial = set(df_params.t_trial.values)
    assert len(t_trial) == 1, f"Expected exactly one value for t_trial, found {len(t_trial)}."
    t_trial = list(t_trial)[0]
    
    load_data = lambda which_params: {fld:np.load(os.path.join(osn_data_dir, which_params, fld+".p"), allow_pickle=True) for fld in ["t", "stim", "counts"]}
    
    params = sorted(list(df_params.folder.values))
    
    # Load one of the folders to get the timing information
    data  = load_data(params[0])
    
    t      = data["t"]
    pats   = list(t.keys())
    for p in pats:
        assert np.allclose(t[pats[0]], t[p]), f"Times for pattern {p} did not match those for {pats[0]}."
    t = t[pats[0]]
    logger.info(f"{n_trials} consecutive trials span t={t[0]}-{t[-1]} seconds.")
    
    end_time = start_time + window_size
    logger.info(f"Using interval {start_time} - {end_time} seconds in each trial.")
    
    summary_interval = slice(np.where(t>=start_time)[0][0], np.where(t<=end_time)[0][-1])
    logger.info(f"Corresponding slice {summary_interval.start}-{summary_interval.stop}")
    
    label_fun   = {"conc":       lambda pat, amp: f"{int(100*amp)}", 
                   "delay.conc": lambda pat, amp: "{}.{:03d}".format("10" if "g" in pat else "25", int(100*amp))}[which_label]

    logger.info(f"Raising counts to the power of {ca2exp:1.3f} before applying Ca2 filter.")        
    logger.info(f"Ca2 filter time constant = {ca2tau:1.3f} seconds.")
    
    t_ca2       = np.arange(0,t_trial,dt)
    ca2_filter  = t_ca2*np.exp(-t_ca2/ca2tau)
    ca2_fun     = lambda counts: np.convolve(counts**ca2exp, ca2_filter, 'full')[:len(counts)]    
    summary_fun = lambda counts: np.sum(ca2_fun(counts).reshape(n_trials,-1)[first_trial:][:,summary_interval],axis=1)

    X , y = [], []
    
    keys = sorted(list(load_data(params[0])["counts"].keys()))
    logger.info("Assembling predictors and labels." if not labels_only else "Assembling labels only.")
    for p in tqdm(params):
        counts_pat_amp = load_data(p)["counts"]
        Xp, yp = [], []
        for (pat, amp) in keys:
            yy = label_fun(pat, amp)
            if not labels_only:
                s = summary_fun(counts_pat_amp[pat,amp])
                Xp.append(s) # Append all the trials for this pat, amp
                yp.append([yy] * len(s))
            else:
                yp.append([yy])
        if not labels_only:
            X.append(np.array(Xp).flatten()) # Append all the trials for ths glomerulus
    if not labels_only:
        X = np.array(X).T
        X /= np.std(X)
    y = bsum(yp, [])

    return X,y,params

def classify(X, y, C_vals = [10**i for i in range(-4,5,1)], seed=0, n_cv = 10, shuf=False,
             gridcv_verbosity = 0, penalty = "l2", scaling = "none",
             verbosity=0):
    
    logger.info(f"Setting random seed to {seed}.")
    np.random.seed(seed)
    
    sss = StratifiedShuffleSplit(n_splits = n_cv, random_state = seed)

    labels   = sorted(np.unique(y), key=float)
    n_labels = len(labels)
    
    confusion_matrix = np.zeros((n_labels, n_labels)).astype(int)

    search = GridSearchCV(LinearSVC(penalty=penalty, dual=False, max_iter = 100000),
                          {"C":C_vals, "fit_intercept":[True,False]},
                          verbose = gridcv_verbosity,
    )

    if scaling == "none":
        logger.info(f"{scaling=} so leaving data unchanged.")
    elif scaling == "rows":
        logger.info(f"{scaling=} so applying scaling to rows.")
        X = StandardScaler().fit_transform(X.T).T
    elif scaling == "cols":
        logger.info(f"{scaling=} so applying scaling to cols.")
        X = StandardScaler().fit_transform(X)
    else:
        raise ValueError(f"Don't know what to do for {scaling=}.")
    
    
    format_val = lambda val: str(val) if type(val) not in [float] else f"{val:>7.3f}"

    logger.info(f"Starting {n_cv} cross-validation runs.")
    acc = []    
    for icv, (train_index, test_index) in enumerate(sss.split(X,y)):
        X_trn, y_trn = X[train_index], [y[i] for i in train_index]
        X_tst, y_tst = X[test_index],  [y[i] for i in test_index]
        if shuf:
            y_trn = np.random.permutation(y_trn)
            y_tst = np.random.permutation(y_tst)
    
        search.fit(X_trn, y_trn)
        yp = search.predict(X_tst)
        acc.append(np.mean(yp==y_tst))
        best_params_str = " ".join([f"{fld:>12s}={format_val(val):<5s}" for fld,val  in search.best_params_.items()])
        logger.info(f"{best_params_str:>40s}: {acc[-1]:>1.3f}")

        cm1 = 0* confusion_matrix
        for y_true, y_pred in zip(y_tst, yp):
            cm1[labels.index(y_true), labels.index(y_pred)] += 1
        print(cm1)
        confusion_matrix+= cm1

    print("Confusion matrix:")
    print(confusion_matrix)
    return confusion_matrix, labels
    
if __name__ == "__main__":
    from collections import namedtuple
    ParserOpt = namedtuple('ParserOpt', 'name alias type default help')
    parser_opts = [
        ParserOpt(name="seed",             alias=None,    type=int,   default=0,        help="Random seed to use."),
        ParserOpt(name="first_trial",      alias="1tr",   type=int,   default=5,        help="The first trial to use for decoding"),
        ParserOpt(name="start_time",       alias="t0",    type=float, default=0.1,      help="Time within each trial to start the count"),
        ParserOpt(name="window_size",      alias="wnd",   type=float, default=2,        help="Window size in seconds to sum over"),
        ParserOpt(name="ca2tau",           alias=None,    type=float, default=0.15,     help="Ca2+ filter time constant in seconds"),
        ParserOpt(name="ca2exp",           alias=None,    type=float, default=2,        help="Exponent applied to the counts before convolving with Ca2+ filter"),
        ParserOpt(name="n_cv",             alias="ncv",   type=int,   default=10,       help="Number of cross-validation runs"),
        ParserOpt(name="C_vals",           alias="C",     type=str,   default="-4,4,1", help="Minimum,maximum, and step of powers of 10 to try for C"),
        ParserOpt(name="scaling",          alias="scal",  type=str,   default="none",   help="Whether to apply standard scaling to the 'rows', 'columns', or 'none' for not at all")
    ]
    
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("osn_data_dir",       type=str,                     help="Root folder containing the data for each glomerulus")
    parser.add_argument("--delay",            action="store_true",          help="Whether to decode delay and concentration instead of just concentration")
    parser.add_argument("--shuf",             action="store_true",          help="Whether to compute the shuffled performance")
    parser.add_argument("--gridcv_verbosity", type=int,   default=0,        help="Verbosity of GridSearchCV.")
    for popt in parser_opts:
        parser.add_argument(f"--{popt.name}", type=popt.type, default=popt.default, help=popt.help + f". (Default: {popt.default})")
    args = parser.parse_args()
    print(f"Running with {args=}")
        
    non_defaults = []
    for popt in parser_opts:
        val = args.__getattribute__(popt.name)
        if val != popt.default:
            alias = popt.alias if popt.alias else popt.name
            non_defaults.append(f"{alias}{val}")
    subdir="_".join(non_defaults) if len(non_defaults) else "default"
    print(f"Using {subdir=}")

    name = f"{'delay.' if args.delay else ''}conc_{'shuf' if args.shuf else 'orig'}"
    print(f"{name=}")
    
    which_label = "conc" if not args.delay else "delay.conc"

    output_root = get_output_root(args.osn_data_dir)
    print(f"{output_root=}")

    output_dir = os.path.join(output_root, subdir)
    print(f"{output_dir=}")
    if not os.path.isdir(output_dir):
        print(f"{output_dir} not found, creating.")
        os.makedirs(output_dir, exist_ok = True)

    input_file = os.path.join(output_dir, f"{'delay.' if args.delay else ''}conc.input.p")
    print(f"{input_file=}")
    
    if not os.path.isfile(input_file):
        print(f"{input_file} not found, so generating it.")
        X, y, params = generate_dataset(args.osn_data_dir, which_label,
                                        first_trial = args.first_trial,
                                        start_time  = args.start_time,
                                        window_size = args.window_size,
                                        ca2exp      = args.ca2exp,
                                        ca2tau      = args.ca2tau)
        with open(input_file, "wb") as f:
            pickle.dump({"X":X, "y":y, "params":params}, f)
    else:
        print(f"{input_file} found, loading classifier inputs from it.")
        with open(input_file, "rb") as f:
            data = pickle.load(f)
        X,y,params = data["X"], data["y"], data["params"]
            
    print(f"{X.shape=}")
    n_labels = len(np.unique(y))
    print(f"{len(y)=} total trials with {n_labels=} ")

    min_C, max_C, C_step = [float(x) for x in args.C_vals.split(",")]
    C_vals = [10**val for val in np.arange(min_C, max_C+1, C_step)]
    print(f"Using {C_vals=}")

    confusion_matrix, labels = classify(X, y, C_vals, n_cv = args.n_cv, shuf=args.shuf, seed = args.seed,
                                        scaling = args.scaling, gridcv_verbosity=args.gridcv_verbosity)
    
    mean_acc = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    print("{}: {:1.3f}".format("Mean accuracy", mean_acc + 1e-8))
    
    to_save = {"X":X, "y":y, "labels":labels, "params":params, "args":args, "confusion_matrix":confusion_matrix}

    output_file = os.path.join(output_dir, f"{name}.p")
    with open(output_file, "wb") as out_file:
        pickle.dump(to_save, out_file)
    print(f"Saved results to {output_file}.")
        
    print("ALLDONE")
    
    


