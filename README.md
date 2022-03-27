## Environment

To run the codes in this repo, you have to install Gurobi, AIF360, and their dependent libraries. The best way to do this is to create a new envrionment that clones the envrionment we used. You can do this by using the following commands:

conda env create --name miss-fair --file=conda_env.yml
conda activate miss-fair

## Experiments 

Our experiments can be replicated in the Jupyter notebooks in the datasets folder. 

For each dataset, we provide 
1) an example notebook where you can train a new Fair MIP Forest model with different parameters (e.g., `adult_example.ipynb`), 
2) a notebook that opens already-trained Fair MIP Forest models stored in the sub-folder 'forests' (e.g., `open_trained_forests.ipynb`), and
3) notebooks where you can run other benchmarks (Equalized Odds, Disparate Mistreatement, and Exponentiated Gradient) on the dataset with missing values (e.g., `baseline_cal_eqodds.ipynb`). 

For the Adult and COMPAS datasets, we include the full dataset under the 'data' folder, and the example notebooks (adult_example.ipynb and compas_example.ipynb) contain the data loading and missing value generation part. However, for the HSLS dataset, we do not include the original dataset due to its size (~1GB). Instead, we include the pickle file that only has a set of variables we use in our prediction tasks. There is no aritificial missing value generation in the HSLS dataset either.

