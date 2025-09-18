
# KI4KI - Künstliche Intelligenz für klimaresilientes Infrastrukturmonitoring  



      

> Monitoring dam infrastructure via Remote sensing technology

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

This repository contains research contributions and tools for forecasting dam deformations based on environmental drivers using PSI (Persistent Scatterer Interferometry) and pendulum swing time-series data. Note, this is work in process.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Structure](#structure)
- [Usage](#usage)
- [Publications](#publications)
- [Contributing](#contributing)
- [License](#license)
- [Maintainers](#maintainers)
- [Contributors](#contributors)

## Background

The KI4KI project focuses on machine learning solutions for climate-resilient infrastructure monitoring, specifically targeting dam deformation prediction. Using satellite-based Persistent Scatterer Interferometry (PSI) data combined with environmental variables, this repository provides tools for a potential future early warning system for infrastructure management.

### Key Features

- **Time-series forecasting**: A number of forecasting models including linear models and foundational approaches
- **Multi-modal data integration**: Combines PSI, weather, and environmental data
- **Grid search optimization**: Automated hyperparameter tuning for model selection
- **Scalable pipeline**: Supports processing multiple monitoring points simultaneously
- **Research validation**: Validated through peer-reviewed publications

### Publications

This codebase supports the following research publications:

- [Data-Driven Prediction of Large Infrastructure Movements Through Persistent Scatterer Time Series Modeling](https://ieeexplore.ieee.org/document/10642253)
- [Enhancing the Prediction of Dam Deformations: A Novel Data-Driven Approach](https://www.mdpi.com/2072-4292/17/6/1026)
- [Identifying Deformation Drivers in Dam Segments Using Combined X-and C-Band PS Time Series](https://www.mdpi.com/2072-4292/17/15/2629)

## Install

### Prerequisites

- Tested with Python 3.13
- Conda package manager

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/Gideon-Stein/KI4KI.git
cd KI4KI
mkdir save_and_results/
mkdir imgs

```

2. Create and activate the conda environment:
```bash
conda env create -f ki4ki.yml
conda activate ki4ki
```


3. Download Raw data 
```bash
wget https://github.com/Gideon-Stein/KI4KI/releases/download/raw_data/raw_data.zip
unzip raw_data.zip
rm raw_data.zip
```

4. Model search and forecasting results (optional)

We provide the full data stack from our publication as a release. 
It can be used to reproduce the results from our papers without running everything. 
To download it simply run: 

```bash
wget https://github.com/Gideon-Stein/KI4KI/releases/download/experimental_results/experimental_results.zip
unzip experimental_results.zip
rm experimental_results.zip
mv old save_and_results/
```



## Usage

### Basic Forecasting Pipeline

The system provides a complete pipeline for dam deformation forecasting using ARIMAX models.

#### 1. Model Grid Search

To find optimal model parameters for all PCI points:

```bash
python forecasting_analysis/model_grid.py --savePath /path/to/results --loc LOCATION --dir DIRECTION
```

Parameters:
- `--savePath`: Directory to save model results
- `--loc`: Specific location code (optional, uses all locations if not specified)
- `--dir`: Direction code (optional, uses all directions if not specified)

#### 2. Generate Forecasts

After grid search completion, generate forecasts for 2021:

```bash
python forecasting_analysis/forecast_points.py --savePath /path/to/results --loc LOCATION --dir DIRECTION
```

#### 3. Export Results

Process and format the forecasting results:

```bash
jupyter notebook export_results.ipynb
```

### Configuration

Model parameters and data paths can be configured in `forecasting_analysis/conf/config.yaml`. Key settings include:

- Data source paths
- Model hyperparameter ranges
- Output directories
- Processing options

### Example Workflow

```bash
# 1. Set up environment
conda activate dam_test

# 2. Configure data paths in config.yaml
# Edit forecasting_analysis/conf/config.yaml

# 3. Run grid search for all points
python forecasting_analysis/model_grid.py --savePath ./results

# 4. Generate forecasts
python forecasting_analysis/forecast_points.py --savePath ./results

# 5. Analyze results
jupyter notebook check_results.ipynb
```

## Structure

```
KI4KI/
├── README.md                        # Project overview and instructions
├── anomalies/                       # Synthetic anomaly generation and analysis
│   ├── anomaly_baseline.ipynb
│   ├── anomaly_generator.ipynb
│   ├── synthetic_anomalies.py
│   └── conf/
│       └── config.yaml
├── forecasting_pipeline/            # Main forecasting pipeline
│   ├── forecast_points.py           # Forecasting script
│   ├── model_grid.py                # Model grid search
│   ├── conf/
│   │   ├── forecast_config.yaml     # Main config for forecasting
│   │   ├── grid_config.yaml         # Grid search config
│   │   └── search/
│   ├── foundational/                # Foundational models and notebooks
│   ├── igarss_publication/          # IGARSS publication notebooks
│   ├── legacy_analysis/             # Legacy analysis notebooks
│   ├── mdpi_publication/            # MDPI publication notebooks
│   └── utils/                       # Utility scripts for data/model handling
├── glörtal_analysis/                # Location-specific analysis and config
│   ├── glör_paper_hydra.py
│   ├── glör_paper.ipynb
│   └── config/
│       └── reg.yaml
├── imgs/                            # Figures and result visualizations
├── raw_data/                        # Raw input data
├── save_and_results/                # Output, cache, and results
│   ├── cache_endog.p
│   ├── cache_exogs.p
│   ├── new/
│   └── old/
```


## Usage

Sub projects provide further explanations.
In anomalies, we provide code for an early warning strategy based on residual anomalies. Along with this, we provide ways to generate synthetic anomalies


In glörtal_analysis, we investigate the effects of a full dam drainage

In forecasting_pipeline we include a large stack of forecasting tools that can be used to forecast either PS-time series or pendulum swing measurements.

The projects features many Notebooks to explore. Importantly, scripting is build on Hydra to make it easy to run things in parallel and keep an overview. 


## Publications


1. **Data-Driven Prediction of Large Infrastructure Movements Through Persistent Scatterer Time Series Modeling** - Introduces the foundational methodology for PSI-based infrastructure monitoring (cite this as well if you use the repo).

```bibtex
@INPROCEEDINGS{10642253,

  author={Stein, Gideon and Ziemer, Jonas and Wicker, Carolin and Jänichen, Jannik and Demisch, Gabriele and Klöpper, Daniel and Last, Katja and Denzler, Joachim and Schmullius, Christiane and Shadaydeh, Maha and Dubois, Clémence},
  booktitle={IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={Data-Driven Prediction Of Large Infrastructure Movements Through Persistent Scatterer Time Series Modeling}, 
  year={2024},
  volume={},
  number={},
  pages={8669-8673},
  doi={10.1109/IGARSS53475.2024.10642253}}

```


2. **Enhancing the Prediction of Dam Deformations: A Novel Data-Driven Approach** - Published in Remote Sensing, demonstrates improved prediction accuracy using multi-modal environmental data.

```bibtex
@Article{rs17061026,
AUTHOR = {Ziemer, Jonas and Stein, Gideon and Wicker, Carolin and Jänichen, Jannik and Klöpper, Daniel and Last, Katja and Denzler, Joachim and Schmullius, Christiane and Shadaydeh, Maha and Dubois, Clémence},
TITLE = {Enhancing the Prediction of Dam Deformations: A Novel Data-Driven Approach},
JOURNAL = {Remote Sensing},
VOLUME = {17},
YEAR = {2025},
NUMBER = {6},
ARTICLE-NUMBER = {1026},
URL = {https://www.mdpi.com/2072-4292/17/6/1026},
ISSN = {2072-4292},
}
```


3. **Identifying Deformation Drivers in Dam Segments Using Combined X-and C-Band PS Time Series** - Published in Remote Sensing, focuses on identifying key environmental drivers of dam deformations.

```bibtex

@article{ziemer2025identifying,
  title={Identifying Deformation Drivers in Dam Segments Using Combined X-and C-Band PS Time Series},
  author={Ziemer, Jonas and J{\"a}nichen, Jannik and Stein, Gideon and Liedel, Natascha and Wicker, Carolin and Last, Katja and Denzler, Joachim and Schmullius, Christiane and Shadaydeh, Maha and Dubois, Cl{\'e}mence},
  journal={Remote Sensing},
  volume={17},
  number={15},
  pages={2629},
  year={2025},
  publisher={MDPI}
}
```



## Contributing

If you are interested in extending this projects towards and actual service, feel free to contact us.


## License

This project is licensed under the [MIT LICENSE](https://opensource.org/license/mit).



## Maintainers

[@GideonStein](https://github.com/Gideon-Stein) - Primary maintainer and lead developer

For questions regarding the research methodology or technical implementation, please open an issue or contact the maintainer directly.

## Contributors

This project exists thanks to the collaboration and generous data provision by the following German institutions:

### Data Providers
- **[Ruhrverband](https://ruhrverband.de/)** - Dam infrastructure and operational data
- **[DWD (Deutscher Wetterdienst)](https://www.dwd.de/DE/Home/home_node.html)** - Meteorological and climate data
- **[BBD (Bodenbewegungsdienst Deutschland)](https://bodenbewegungsdienst.bgr.de)** - Persistent Scatterer Interferometry data

### Research Collaborators
- Computer vision Group Jena  [CVG JENA](https://inf-cv.uni-jena.de/)
- Institute for Remote Sensing, Friedrich Schiller University Jena
- German Federal Institute for Geosciences and Natural Resources (BGR)
- Ruhr Association for Water Management

---

**Note**: Results may vary slightly between different computing environments due to CPU instruction differences and floating-point precision. As this is early work and not part of any product pipeline yet, we will work on these issue in the future.
