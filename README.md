# Model for olfactory sensory neurons 
This repository contains the code to generate the data and produce the plots for Extended Data Figure 1 of Ackels et al. (under review). 
## Loading precomputed data
The steps below describe how the data for each analysis can be generated from scratch. Alternatively, precomputed data can be downloaded and installed as follows:
1. Download the code repository and unpack at your desired **installation root**.
2. Download the data from [this](https://www.dropbox.com/s/pncq56d4evnx7v4/crick-osn-model-release-data.tar.gz?dl=0) link (~250 MB).
3. Unpack the data file to yield a `data` folder.
4. Move this folder into the **installation root**. It should now sit at the same level as the `README.md` file.
## Generating data for multiple glomeruli
1. Create a `data` folder in the **installation root**.
1. Generate the parameters by running ```python gen_osn_model_params.py --outputdir data/model-osn-data/gamp5 --glom_amp 5```
2. Run the model for each of the parameter settings by ```cd data/model-osn-data/gamp5; python run_for_params.py params0.json```

The folder `data/model-osn-data/gamp5` now contains one folder for each parameter json file, containing the spike counts.
## Decoding PPI using different numbers of glomeruli
### Generate the data
1. Change to the `decoding-ppi` directory.
2. Generate the classifer inputs by `python run_decoding_ppi.py ../data/model-osn-data/gamp5/ 100 --which_perm -1`
- This will first generate the concentration directory, `data/decoding_ppi/gamp5_conc100`
- It then generates the inputs using the subset of the `gamp5` data with concentration 100. 
- The results are in `input.p` in that folder.
3. Run the classifiers for all sizes, for a given permutation `python run_decoding_ppi.py ../data/model-osn-data/gamp5/ 100 --which_perm=0`
4. Collect the classification results using `python run_decoding_ppi.py ../data/model-osn-data/gamp5/ 100 --collect` 
- This produces `output.collected.p` in the directory `data/decoding_ppi/gamp5_conc100`
### Plot the data
Run the notebook `make_figure_decoding_ppi.ipynb` to produce the figure showing decoding accuracy vs # glomeruli used.
## Decoding PPI and concentration using all glomeruli
### Generate the data
Classifying PPI and concentration is run by calling the code in `classify.py` directly.
- To run concentration decoding on a dataset, say `gamp5`, run `python classify.py data/model-osn-data/gamp5`
- To run PPI and concentration decoding, use the `--ppi` flag: `python classify.py data/model-osn-data/gamp5 --ppi`
- To run the shuffled setting, add `--shuf`, e.g. flag: `python classify.py data/model-osn-data/gamp5 --ppi --shuf`
- The script `classify.all.sh` is provided a convenient way to run all of the above combinations.
- The input and output of these runs will be written to `data/decoding-ppi-conc/gamp5/default`
- A number of parameters can be set to custom values. To see a list, run `python classify.py --help`.
- For example, to classify concentrations with Ca2+ exponent set to 1.0, use `python classify.py data/model-osn-data/gamp5 --ca2exp 1`
- The results will be written to folders whose names indicate the parameters to `classify.py` that were changed from their default values.
- For example, the results of runs with `--ca2exp 1`  will be written to `data/decoding-ppi-conc/gamp5/ca2exp1.0`
### Plot the data
To plot the figures showing PPI and concentration decoding performance, run the notebook `make_figure_decoding_ppi_conc.ipynb`.


