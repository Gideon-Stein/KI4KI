

# Dam deformation forecasting pipeline

This repo holds the code for two publications: 

IGARSS 2024 (Insert Link)
Journal 2024 (Insert Link)


Both experiment sets are based on the same code base. 

Install environment via: 

conda env create --n env.yml


There are two main scripts. **model_grid.py** and **forecast_points.py**. Both use Hydra configs for arguments. 




Via:

```
python model_grid.py search=4_igarss_search.yaml save_path=results/res model_type=sarimax loc=Lister  dir=X parallel=True

```

you can search all possible models for a specific config and location and save various performance metrices.


Via:

```
python forecast_points.py  save_path=results/res forecast_path=results/fce criteria=MAPE dir=asc0 model_type=sarimax search=4_igarss_search.yaml loc=Lister
```
you can take a previously executed grid result, select the best models for each Time series and forecast 2020.



For the foundational models follow the readme that is provided in the foundational folder.
