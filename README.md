# PPI-CPI-Inflation-Ghana-Replications-Files
Replication files and dataset for the manuscript examining the relationship between PPI and CPI inflation in Ghana.

# Replication Materials: CPI–PPI Inflation Dynamics in Ghana

## Overview

This repository contains the dataset and Python code used for the empirical analysis in the study examining the relationship between Consumer Price Index (CPI) inflation and Producer Price Index (PPI) inflation in Ghana.

The materials provided allow other researchers to reproduce the statistical analysis conducted in the study.

## Repository Contents

### Dataset

**CPI_PPI_INFLATION.xlsx**
This dataset contains monthly inflation percentage changes for:

* Consumer Price Index (CPI)
* Producer Price Index (PPI)

The data covers the period **January 2014 to December 2025** and forms the basis of the time-series analysis conducted in the study.

### Model Estimation Code

**CPI_PPI VECM Model Estimation.py**
This Python script contains the code used to estimate the econometric model applied in the study.
The script performs the primary analysis used to evaluate the relationship between CPI and PPI inflation.

### Model Validation Code

**CPI_PPI_Model Validation.py**
This Python script contains preliminary model validation procedures used prior to the main estimation.
The validation process informed the selection of the final econometric model applied in the study.

## Replication Instructions

1. Download the dataset **CPI_PPI_INFLATION.xlsx**.
2. Ensure the required Python libraries for time-series econometric analysis are installed.
3. Run the scripts in the following order:

   * `CPI_PPI_Model Validation.py`
   * `CPI_PPI VECM Model Estimation.py`

The validation script identifies the appropriate modelling approach, while the estimation script performs the final model estimation used in the study.

## Notes

Additional files containing figures or extended results may be added to this repository as supplementary materials.

## Author

Jefferson Mensah

## License

This repository is provided for research transparency and academic replication purposes.
