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


![Proposed Method Framework](https://github.com/irenekarijadi/CEEMDAN-EWT-LSTM/assets/28720072/922f6554-ff1c-4acb-b8c0-2ef167fc0d51)


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
* ewtpy==0.2

The complete list of packages can be found in [requirements.txt.](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/requirements.txt)

In order to run the model, the packages need to be installed first using this line of code:

`pip install -r requirements.txt()`


## Dataset
The performance of the proposed method is tested by using wind power datasets in two different countries. The first dataset is from a wind farm with an installed capacity of 2050 kW located in 
[France](https://opendata-renewables.engie.com/explore/?sort=modified), and the second dataset is from a wind farm with an installed capacity of 3600 kW located in [Turkey](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset)

https://drive.google.com/drive/folders/1uwYtYdqzDw4ozum5xnuLAXkyb4pjbApc?usp=sharing 


## Experiments
The code that generated results presented in Chapter 4 of the paper can be executed from these notebooks:


* `1. Experiments for France Dataset.ipynb` - By running all the cells in this notebook, it will train and test the proposed method and other benchmark methods on the France dataset.
* `2. Experiments for Turkey Dataset.ipynb` - By running all the cells in this notebook, it will train and test the proposed method and other benchmark methods on the Turkey dataset.
* `3. Time Series Cross Validation for France Dataset.ipynb` - By running all the cells in this notebook, a time series cross-validation experiment of the France dataset will be conducted.
* `4. Time Series Cross Validation for Turkey Dataset.ipynb` - By running all the cells in this notebook, a time series cross-validation experiment of the Turkey dataset will be conducted.
* `5. Comparative Experiments EEMD_BO_LSTM.ipynb` - By running all the cells in this notebook,  a comparative experiment of the EEMD BO LSTM will be conducted.
* `6. Comparative Experiments EMD_ENN.ipynb`- By running all the cells in this notebook, a comparative experiment of the EMD ENN will be conducted.


