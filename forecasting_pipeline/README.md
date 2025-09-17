

# Dam deformation forecasting pipeline

This section holds the code for two publications: 

IGARSS 2024 ("Data-Driven Prediction of Large Infrastructure Movements Through Persistent Scatterer Time Series Modeling")
MDPI 2024 (Enhancing the Prediction of Dam Deformations: A Novel Data-Driven Approach)


Both experiment sets are based on the same code base. 



There are two main scripts. **model_grid.py** and **forecast_points.py**. Both use Hydra configs for arguments. Further, visualizations of the correponding results can be found in the folders "igarss_publication" and "mdpi_publication".

Notably, you can download the entire model search results by running: 

```
./download_search.sh
```




Alternative you can reconstruct the entire search space for the igarss paper via:

```bash
python model_grid.py -m search=4_igarss_search.yaml save_path=../save_and_results/new/igarss/igarss/ model_type=sarimax,var loc=Lister  dir=X 
python model_grid.py -m search=4_igarss_search.yaml save_path=../save_and_results/new/igarss/igarss_remove_T/ model_type=sarimax,var loc=Lister remove_T=True dir=X 
python model_grid.py -m search=4_igarss_search.yaml save_path=../save_and_results/new/igarss/igarss_remove_W/ model_type=sarimax,var loc=Lister  dir=X remove_W=True 
python model_grid.py -m search=5_igarss_no_exogs.yaml save_path=../save_and_results/new/igarss/igarss_univariate/ model_type=sarimax,var loc=Lister  dir=X 
```

And for the MDPI paper via: 

```bash
python model_grid.py -m search=3_no_exogs.yaml save_path=../save_and_results/new/journal/journal_new_univariate  model_type=X dir=desc1,lot
python model_grid.py -m search=0_full_search.yaml save_path=../save_and_results/new/journal/journal_new model_type=X dir=desc1,lot parallel=False
python model_grid.py -m search=0_full_search.yaml save_path=../save_and_results/new/journal/journal_new_remove_W model_type=X dir=desc1,lot parallel=False remove_W=True
python model_grid.py -m search=0_full_search.yaml save_path=../save_and_results/new/journal/journal_new_remove_T model_type=X dir=desc1,lot parallel=False remove_T=True
python model_grid.py -m search=0_full_search.yaml save_path=../save_and_results/new/journal/journal_new model_type=var dir=desc1 parallel=False
python model_grid.py -m search=0_full_search.yaml save_path=../save_and_results/new/journal/journal_new_remove_W model_type=vat dir=desc1 parallel=False remove_W=True
python model_grid.py -m search=0_full_search.yaml save_path=../save_and_results/new/journal/journal_new_remove_T model_type=var dir=desc1 parallel=False remove_T=True

```




To generate the forecasting result from the grid search run: 


For Igarss: 

```bash
python forecast_points.py -m save_path=../save_and_results/old/igarss/igarss_new forecast_path=../save_and_results/new/igarss/fc_igarss criteria=MAPE dir=X model_type=sarimax,var search=4_igarss_search.yaml loc=Lister
python forecast_points.py -m save_path=../save_and_results/old/igarss/igarss_new forecast_path=../save_and_results/new/igarss/fc_igarss_baseline criteria=MAPE dir=X model_type=sarimax search=1_baseline.yaml loc=Lister  remove_everything=True
```

For MDPI paper: 

```bash
python forecast_points.py -m  save_path=../save_and_results/old/journal/journal_new  forecast_path=../save_and_results/new/journal/2 criteria=MAE dir=lot,desc1 model_type=X search=2_no_autoreg.yaml 
python forecast_points.py -m save_path=../save_and_results/old/journal/journal_new  forecast_path=../save_and_results/new/journal/3 criteria=MAE dir=lot,desc1 model_type=X search=3_no_exogs.yaml 
python forecast_points.py -m   save_path=../save_and_results/old/journal/journal_new  forecast_path=../save_and_results/new/journal/1 criteria=MAE dir=lot,desc1 model_type=X search=1_baseline.yaml
python forecast_points.py -m  save_path=../save_and_results/old/journal/journal_new forecast_path=../save_and_results/new/journal/0 criteria=MAE dir=lot,desc1 model_type=X search=0_full_search.yaml
python forecast_points.py -m  save_path=../save_and_results/old/journal/journal_new forecast_path=../save_and_results/new/journal/0 criteria=MAE dir=desc1 model_type=var search=0_full_search.yaml
```

For the foundation model forecasts follow the readme that is provided in the foundational folder.



