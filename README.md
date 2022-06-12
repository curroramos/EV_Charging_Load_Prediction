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
- Root folder: contains basic train and test scripts, also contains a training and testing demo implemented using jupyter notebooks.


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

## Results
### Prediction benchmarking
![alt text](https://github.com/curroramos/EV_Charging_Load_Prediction/blob/main/figures/benchmarking_table.png)


### Models comparison
![alt text](https://github.com/curroramos/EV_Charging_Load_Prediction/blob/main/figures/Figure%202022-03-22%20200750.png)

### Real time showcase
![alt text](https://github.com/curroramos/EV_Charging_Load_Prediction/blob/main/Results/real_time.gif)