# Foundation Model Forecasting Setup

Follow these steps to generate foundation model forecasts:

1. **Install the TimesFM environment:**  
    [Google TimesFM GitHub](https://github.com/google-research/timesfm)

2. **Install Chronos in the same environment:**  
    [Amazon Chronos GitHub](https://github.com/amazon-science/chronos-forecasting)

3. **Validate the installation:**  
    Use the notebooks `fm_chronos.ipynb` and `fm_tfm.ipynb`.

---

## Running Zero-Shot Forecast

Example command:

```bash
python zero_shot_forecast.py save_path=../../save_and_results/tfm dir=lot model_type=tfm parallel=False
```
