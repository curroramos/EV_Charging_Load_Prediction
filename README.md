# EV_Charging_Load_Prediction

Final Degree Project.
Degree in Industrial Engineering. University of Seville.

This repository hosts the files of the project *Deep Learning Applications to Optimise an EV Charging Station* related with the EV charging demand prediction.

## Project breakdown
- Data processing: Public data processing and preparing in order to feed the time series models.
- Training and benchmarking: Train different neural network models on the gathered data and evaluate and compare their performance on the value prediction task.
- Deployment: Build test functions to make inferences and predictions on new data

## Content
The content is arranged in different folders:
- `Root folder`: contains basic [train.py](train.py) script, also contains a [train_demo.ipynb](train_demo.ipynb) and [test_demo.ipynb](test_demo.ipynb) jupyter notebooks to get started into training models and making inferences.
- `data_preprocessing\`: contains files related to first data processing and visualization. Also contains the datasets in [raw_data](raw_data), [preprocessed_data](preprocessed_data), [processed_data](processed_data) 

```bash
├── data_handler.py
├── data_preprocessing
│   ├── data_preprocessing.py
│   ├── data_processing.ipynb
│   ├── preprocessed_data
│   │   ├── final_2018
│   │   ├── final_2019
│   │   ├── final_2019_2020.pkl
│   │   └── final_2020
│   ├── processed_data
│   │   ├── data2019.csv
│   │   ├── data2019.json
│   │   └── data2019.pkl
│   └── raw_data
│       ├── acndata_sessions_2019.json
│       └── acndata_sessions_2020.json
├── figures
│   └── Figure 2022-03-22 200750.png
├── LICENSE
├── model_utils.py
├── plot_utils.py
├── README.md
├── test_demo.ipynb
├── train_demo.ipynb
└── train.py
```

## Get started
Please refer to installation

## Main outcomes
- Model benchmarking
    - Main results for each model in [Results](https://github.com/curroramos/EV_Charging_Load_Prediction/tree/main/Results) directory

## Results
### Prediction benchmark
![alt text](https://github.com/curroramos/EV_Charging_Load_Prediction/blob/main/figures/benchmarking_table.png)

![alt text](https://github.com/curroramos/EV_Charging_Load_Prediction/blob/main/figures/Figure%202022-03-22%20200750.png)

### Real time showcase 
![alt text](https://github.com/curroramos/EV_Charging_Load_Prediction/blob/main/Results/real_time.gif)