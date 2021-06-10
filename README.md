# SELCON

## Dependencies

To run this code fully, you'll need PyTorch (we're using version 1.4.0) and scikit-learn. We've been running our code in Python 3.7.

## To run the experiments

To run the code use `python3 run_methods.py <type> <if_time_series> <datadir>`. **type** can be *Fair*, *Deep* or *Main* to run fair, deep and main experiments respectively. Default is *Main*.
For **if_time_series** pass *true* for NYSE datasets, otherwise *false*. Pass location of datasets for **datadir**.

To generate the results use scripts in `gen_results` folder

Due to limitation of size, we uploaded the data (along with the code ) in https://rebrand.ly/selcon
