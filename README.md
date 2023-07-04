# CEEMDAN-EWT-LSTM
## Wind Power Forecasting Based on Hybrid CEEMDAN- EWT Deep Learning Method

This is the original source code used for all experiments in the paper ***"Wind Power Forecasting Based on Hybrid CEEMDAN-EWT Deep Learning Method"***

Please cite the paper if you utilize the code in this paper.

## Authors

*Irene Karijadi, Shuo-Yan Chou, Anindhita Dewabharata*


*corresponding author: irenekarijadi92@gmail.com (Irene Karijadi)*

## Background
A precise wind power forecast is required for the renewable energy platform to function effectively. By having a precise wind power forecast, the power system can better manage its supply and ensure grid reliability. However, the nature of wind power generation is intermittent and exhibits high randomness, which poses a challenge to obtain accurate forecasting results. In this study, a hybrid method is proposed based on Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN), Empirical Wavelet Transform (EWT), and deep learning-based Long Short-Term Memory (LSTM) for ultra-short-term wind power forecasting. A combination of CEEMDAN and EWT is used as the preprocessing technique, where CEEMDAN is first employed to decompose the original wind power data into several subseries and EWT denoising technique is used to denoise the highest frequency series generated from CEEMDAN. Then, LSTM is utilized to forecast all the subseries from CEEMDAN-EWT process, and the forecasting results of each subseries are aggregated to achieve the final forecasting results. The proposed method is validated on real-world wind power data in France and Turkey. Our experimental results demonstrate that the proposed method can forecast more accurately than the benchmarking methods.

## Framework
This is the framework of the proposed method      
![Alt text](Figures/Framework.png)

## Prerequisites
The proposed method is coded in Python 3.7.6 and the experiments were performed on Intel Core i3-8130U CPU, 2.20GHz, with a memory size of 4.00 GB.
The Python version is specified in [runtime.txt.](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/runtime.txt)
In order to run the experiments, a number of packages need to be installed. Here is the list of the package  version that we used to run all the experiments

* EMD-signal==0.2.10
* pandas==0.25.3
* keras==2.4.3
* tensorflow>=2.0.0
* sklearn==0.22.1
* numpy==1.18.1
* matplotlib
* ewtpy

The complete list of packages can be found in [requirements.txt.](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/requirements.txt)

In order to run the model, the packages need to be installed first using this line of code:

`pip install -r requirements.txt()`


## Description of File
Non-python file
* Dataset - This folder includes all datasets used in this study
* Figures - This folder includes all generated figures to be used in reporting
* README.md - The README file for using this code 
* License - The License file
* requirement.txt - This file contains a list of packages used in this study
* runtime.txt - This file contains Python version used in this study


Python Files:
* `1.Experiments for University Dormitory Building.ipynb` - This notebook is the main file used to conduct the experiments for university dormitory building using parameter settings specified in Setting.py
* `2.Experiments for University Laboratory Building.ipynb` - This notebook is the main file used to conduct the experiments for university laboratory building using parameter settings specified in Setting.py
* `3.Experiments for University Classroom Building.ipynb` - This notebook is the main file used to conduct the experiments for university classroom building using parameter settings specified in Setting.py
* `4.Experiments for Office Building.ipynb` - This notebook is the main file used to conduct the experiments for office building using parameter settings specified in Setting.py
* `5.Experiments for Primary Classroom Building.ipynb`- This is the main file used to conduct the experiments for primary classroom building using parameter settings specified in Setting.py
* `myfunctions.py` - This python script includes all functions required for building proposed method and other benchmark methods that are used in the experiments



## Dataset
The performance of the proposed method is tested by using wind power datasets in two different countries. The first dataset is from a wind farm with an installed capacity of 2050 kW located in 
[France](https://opendata-renewables.engie.com/explore/?sort=modified), and the second dataset is from a wind farm with an installed capacity of 3600 kW located in [Turkey](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset)


## Experiments
The code that generated results presented in Chapter 4 of the paper can be executed from these notebooks:


`1. Experiments for France Dataset.ipynb` - By running all the cell in this notebook, it will train and test the proposed method and other benchmark methods on France dataset.
`2. Experiments for Turkey Dataset.ipynb` - By running all the cell in this notebook, it will train and test the proposed method and other benchmark methods on France dataset.****
`3. Time Series Cross Validation for France Dataset.ipynb` - By running all the cell in this notebook, it will train and test the proposed method and other benchmark methods on France dataset.
`4. Time Series Cross Validation for Turkey Dataset.ipynb` - By running all the cell in this notebook, it will train and test the proposed method and other benchmark methods on France dataset.
`5. Comparative Experiments for France Dataset.ipynb` - By running all the cell in this notebook, it will train and test the proposed method and other benchmark methods on France dataset.
`6. Comparative Experiments for Turkey Dataset.ipynb`- By running all the cell in this notebook, it will train and test the proposed method and other benchmark methods on France dataset.


