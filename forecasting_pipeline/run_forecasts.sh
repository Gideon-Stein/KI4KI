#!/usr/bin/env bash





python forecast_points.py  save_path=results/journal/journal_new forecast_path=results/journal/0 criteria=MAE dir=desc1 model_type=X search=0_full_search.yaml
python forecast_points.py  save_path=results/journal/journal_new forecast_path=results/journal/1 criteria=MAE dir=desc1 model_type=X search=1_baseline.yaml remove_everything=True
python forecast_points.py  save_path=results/journal/journal_new forecast_path=results/journal/2 criteria=MAE dir=desc1 model_type=X search=2_no_autoreg.yaml
python forecast_points.py  save_path=results/journal/journal_new forecast_path=results/journal/3 criteria=MAE dir=desc1 model_type=X search=3_no_exogs.yaml

python forecast_points.py  save_path=results/journal/journal_new forecast_path=results/journal/0 criteria=MAE dir=lot model_type=X search=0_full_search.yaml
python forecast_points.py  save_path=results/journal/journal_new forecast_path=results/journal/1 criteria=MAE dir=lot model_type=X search=1_baseline.yaml remove_everything=True
python forecast_points.py  save_path=results/journal/journal_new forecast_path=results/journal/2 criteria=MAE dir=lot model_type=X search=2_no_autoreg.yaml
python forecast_points.py  save_path=results/journal/journal_new forecast_path=results/journal/3 criteria=MAE dir=lot model_type=X search=3_no_exogs.yaml

python forecast_points.py  save_path=results/journal/journal_new forecast_path=results/journal/0 criteria=MAE dir=desc1 model_type=var search=0_full_search.yaml
python forecast_points.py  save_path=results/journal/journal_new forecast_path=results/journal/3 criteria=MAE dir=desc1 model_type=var search=3_no_exogs.yaml
