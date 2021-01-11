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
2. Generate the classifer inputs by ```python run_decoding_delay.py 100 --dataset ../model-osn-data/gamp5 --which_perm -1```
This generates the inputs using the subset of the `gamp5` data with concentration 100. 
3. Run the classifiers for all sizes, for a given permutation ```python run_decoding_delay.py 100 --which_perm=0```
4. Collect the classification results using ```python run_decoding_delay.py 100 --collect .``` This produces `decoding_delay_conc100.output.collected.p`

## Decoding at multiple concentrations
