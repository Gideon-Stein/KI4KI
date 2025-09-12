## Experiments for the Paper

**Title:** *Identifying Deformation Drivers in Dam Segments Using Combined X- and Cm-Band PS Time Series*

### Reproducing Results

Use the main environment of the repository.

To reproduce the results, run:

```bash
python glör_paper_hydra.py -m which_ps=tsx,S1,t+s
# Sentinel-1, TerraSAR-X, Sensor Fusion
```

This command evaluates all regression models on all three PS time series.

### Analysis

You can generate the results of this analysis in `glör_paper.ipynb`.

> **Note:**  
> This section provides more functionality than what is used in the paper. The script can run various model specifications (configured via the Hydra config or command line).  
> Only results for the Sensor Fusion data (`t+s`) are used in the paper. The rest is included for an upcoming potential product.
>  In legacy content we include a couple of old scripts. Feel free to ignore them as they are NOT maintained.