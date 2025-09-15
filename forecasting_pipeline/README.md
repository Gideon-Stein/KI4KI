

# Dam deformation forecasting pipeline

This section holds the code for two publications: 

IGARSS 2024 ("Data-Driven Prediction of Large Infrastructure Movements Through Persistent Scatterer Time Series Modeling")
MDPI 2024 (Enhancing the Prediction of Dam Deformations: A Novel Data-Driven Approach)


Both experiment sets are based on the same code base. 



There are two main scripts. **model_grid.py** and **forecast_points.py**. Both use Hydra configs for arguments. Further, visualizations of the correponding results can be found in the folders "igarss_publication" and "mdpi_publication".


Via:

```
python model_grid.py search=4_igarss_search.yaml save_path="../save_and_results/igarss_new/ model_type=sarimax loc=Lister  dir=X all_dirs="["asc0", "ew", "ver", "desc0"]" 
python model_grid.py search=4_igarss_search.yaml save_path="../save_and_results/igarss_new/ model_type=sarimax loc=Lister  dir=X all_dirs="["asc0", "ew", "ver", "desc0"]"  remove_T=True # ignores T completely
python model_grid.py search=4_igarss_search.yaml save_path="../save_and_results/igarss_new/ model_type=sarimax loc=Lister  dir=X all_dirs="["asc0", "ew", "ver", "desc0"]" remove_W=True # ignores W_completely


```

You can reconstruct the entire search space of the IGARSS paper.


Via:

```
python forecast_points.py  save_path=results/res forecast_path=results/fce criteria=MAPE dir=asc0 model_type=sarimax search=4_igarss_search.yaml loc=Lister
```
you can take a previously executed grid result, select the best models for each Time series and forecast 2020.



For the foundational models follow the readme that is provided in the foundational folder.
