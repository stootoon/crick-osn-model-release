# crick-osn-tracking-release
The code for the OSN model to be publicly released.
## Generating data for multiple glomeruli
For example, to generate data for a
1. Generate the parameters by running ```python gen_osn_model_params.py --outputdir model-osn-data/gamp5 --glom_amp 5```
2. Run the model for each of the parameter settings by ```cd model-osn-data/gamp5; python run_for_params.py params0.json```
The folder `model-osn-data/gamp5` now contains one folder for each parameter json file, containing the spike counts.

## Decoding a single concentration
### Generating classifier inputs
1. Change to the `multiple-glomeruli-single-amp` directory.
2. Generate the classifer inputs by `python run_decoding_delay.py ../data/model-osn-data/gamp5/ 100 --which_perm -1`
This will first generate the concentration directory, `data/decoding_delay_gamp5_conc100`
It then generates the inputs using the subset of the `gamp5` data with concentration 100. 
The results are in `input.p` in that folder.
3. Run the classifiers for all sizes, for a given permutation `python run_decoding_delay.py ../data/model-osn-data/gamp5/ 100 --which_perm=0`
4. Collect the classification results using `python run_decoding_delay.py ../data/model-osn-data/gamp5/ 100 --collect .` 
This produces `output.collected.p` in the directory data/decoding_delay_gamp5_conc100

## Decoding at multiple concentrations
- To run concentration decoding on a dataset, say `gamp5`, run `python classify.py data/model-osn-data/gamp5`
- To run delay and concentration decoding, use the `--delay` flag: `python classify.py data/model-osn-data/gamp5 --delay`
- To run the shuffled setting, add `--shuf`, e.g. flag: `python classify.py data/model-osn-data/gamp5 --delay --shuf`
- The input and output of these models will be written to `data/decoding-delay-conc/gamp5/default`
- A number of parameters can be set to custom values. To see a list, run `python classify.py --help`.
- For example, to classify concentrations with Ca2+ exponent set to 1.0, use `python classify.py data/model-osn-data/gamp5 --ca2exp 1`

