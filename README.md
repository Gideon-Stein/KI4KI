
# KI4KI - Künstliche Intelligenz für klimaresilientes Infrastrukturmonitoring  

> Monitoring dam infrastructure via Remote sensing technology

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

This repository contains research contributions and tools for forecasting dam deformations based on environmental drivers using PSI (Persistent Scatterer Interferometry) and pendulum swing time-series data.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Structure](#structure)
- [API](#api)
- [Publications](#publications)
- [Contributing](#contributing)
- [License](#license)
- [Maintainers](#maintainers)
- [Contributors](#contributors)

## Background

The KI4KI project focuses on developing artificial intelligence solutions for climate-resilient infrastructure monitoring, specifically targeting dam deformation prediction. Using satellite-based Persistent Scatterer Interferometry (PSI) data combined with environmental variables, this repository provides tools for a potential future early warning system for infrastructure management.

### Key Features

- **Time-series forecasting**: ARIMAX-based models for dam deformation prediction
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

- Python 3.10 or higher
- Conda package manager
- Access to proprietary data sources (see [Data Requirements](#data-requirements))

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/Gideon-Stein/KI4KI.git
cd KI4KI/full_baseline
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate dam_test
```

3. For the exploration tools, install additional requirements:
```bash
cd ../explore
pip install -r requirements.txt
```

### Data Requirements

**Note**: This project requires proprietary datasets that are not publicly available:

- **Weather Data ("Daten")**: Downloaded from Draco platform
- **PSI Data ("Datenpaket_BBD")**: Persistent Scatterer Interferometry data from the German Ground Motion Service (BBD)

Contact the maintainers or data providers for access permissions.

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
full_baseline/
├── README.md                    # This file
├── environment.yml              # Conda environment specification
├── forecasting_analysis/       # Main forecasting pipeline
│   ├── conf/                   # Configuration files
│   ├── forecast_points.py      # Generate forecasts with best models
│   ├── model_grid.py          # Grid search for optimal parameters
│   ├── foundational/          # Foundational model implementations
│   └── utils/                 # Utility functions
├── glörtal_analysis/          # Location-specific analysis tools
├── utils/                     # Shared utility functions
│   ├── data_parsing_tools.py  # Data loading and preprocessing
│   ├── general_tools.py       # General helper functions
│   └── regression_tools.py    # Model fitting and evaluation
├── saves_and_results/         # Output directory for results
└── *.png                      # Result visualizations

../explore/                    # Exploratory analysis and tools
├── check_ds.ipynb            # Dataset exploration
├── cluster.ipynb             # Clustering analysis
├── filtering.ipynb           # Data filtering techniques
├── known_anomalies.ipynb     # Anomaly detection
├── tools.py                  # Analysis utilities
├── filters.py                # Filtering functions
└── display_tools.py          # Visualization tools
```

## API

### Core Classes and Functions

#### Model Grid Search
```python
from forecasting_analysis.model_grid import run_grid_search

# Run automated grid search for optimal ARIMAX parameters
results = run_grid_search(
    location="DAM_001",
    direction="UP",
    save_path="./results"
)
```

#### Forecasting
```python
from forecasting_analysis.forecast_points import generate_forecast
from utils.regression_tools import forecast_with_best

# Generate forecasts using best model
forecast = forecast_with_best(
    data=time_series_data,
    exog_data=environmental_data,
    model_params=best_params
)
```

#### Data Loading
```python
from utils.data_parsing_tools import load_data_bases

# Load and preprocess all datasets
data, exogs = load_data_bases(config)
```

### Configuration Options

The system uses Hydra for configuration management. Key configuration parameters:

- `data_path`: Path to PSI and weather data
- `forecast_horizon`: Number of time steps to forecast
- `model_types`: List of model types to evaluate
- `locations`: Monitoring point identifiers
- `directions`: Movement directions to analyze

## Publications


1. **Data-Driven Prediction of Large Infrastructure Movements Through Persistent Scatterer Time Series Modeling** - Introduces the foundational methodology for PSI-based infrastructure monitoring.

2. **Enhancing the Prediction of Dam Deformations: A Novel Data-Driven Approach** - Published in Remote Sensing, demonstrates improved prediction accuracy using multi-modal environmental data.

3. **Identifying Deformation Drivers in Dam Segments Using Combined X-and C-Band PS Time Series** - Published in Remote Sensing, focuses on identifying key environmental drivers of dam deformations.

### Citation

If you use this code in your research, please cite:

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

## Contributing

If you are interested in extending this projects towards and actual service, feel free to contact us.


## License

This project is licensed under the [MIT LICENSE](https://opensource.org/license/mit) file for details.

**Data License**: The datasets used in this project are proprietary and require separate licensing agreements with the respective data providers.

## Maintainers

[@GideonStein](https://github.com/Gideon-Stein) - Primary maintainer and lead developer

For questions regarding the research methodology or technical implementation, please open an issue or contact the maintainer directly.

## Contributors

This project exists thanks to the collaboration and generous data provision by the following German institutions:

### Data Providers
- **[Ruhrverband](https://ruhrverband.de/)** - Dam infrastructure and operational data
- **[DWD (Deutscher Wetterdienst)](https://www.dwd.de/DE/Home/home_node.html)** - Meteorological and climate data
- **[BBD (Bodenbewegungsdienst Deutschland)](https://bodenbewegungsdienst.bgr.de)** - Persistent Scatterer Interferometry data
- **[FSU Jena (Lehrstuhl für Fernerkundung)](https://www.chemgeo.uni-jena.de/29150/fernerkundung)** - Remote sensing expertise and validation

### Research Collaborators
- Institute for Remote Sensing, Friedrich Schiller University Jena
- German Federal Institute for Geosciences and Natural Resources (BGR)
- Ruhr Association for Water Management

---

**Note**: Results may vary slightly between different computing environments due to CPU instruction differences and floating-point precision. As this is early work and not part of any product pipeline yet, we will work on these issue in the future.
