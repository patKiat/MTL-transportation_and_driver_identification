# TMD Dataset

The TMD dataset is used to achieve the objectives of this project.
You can find more information about the dataset at: http://cs.unibo.it/projects/us-tm2017.

## Dependecies
To execute the code in this repository, you will need to install the following dependencies:
* [Python 3.9.18](https://www.python.org/)
* [Scikit-learn](http://scikit-learn.org/stable/)
* [Pandas](http://pandas.pydata.org/)

## Documentation
### Code
In this section we show the functionalities developed in our work and the relative parameters used.

#### download_dataset.py
Handles the downloading of the TMD dataset from the specified online source, ensuring that all necessary data files are retrieved and stored correctly for further processing.

#### TMDatasetRemoveNan.py
* Functions to clean, transform, and prepare the dataset for further analysis.
* Features include deleting specific sensor data, handling incorrect time values, creating balanced datasets, and more detailed sensor analysis.

#### main.ipynb
1. Downloading the dataset by calling download_dataset.py
2. Cleaning the raw data and extracting the feature: TMDatasetRemoveNan.py
3. Data analysis
4. Building neural the Multi-task Neural Network models and individual Neural Network models to make predictions and conducting experiments
5. Getting the results

### Project Structure
Up to now the projects is structured as follows:

├── TransportationData
|   ├── datasetBalanced (This dataset provided the window partitioning data files but I did the window partitioning by myself.)
|         └── dataset_5secondWindow.csv
|         └── dataset_halfSecondWindow.csv
|   └── _RawDataOriginal (Extract all files in sub-folder of users to be 1 folder)
|         └── ...
|   └── _RawDataCorrect
|   └── _RawDataTransform
|   └── _RawDataHeader
|   └── _RawDataFeatures
|   └── _Dataset
├── README.md
├── LICENSE
├── const.py
├── TMDatasetRemoveNan.py
├── util.py
├── download_dataset.py
└── cleanLog.log

## Reference
@article{carpineti18,
  Author = {Claudia Carpineti, Vincenzo Lomonaco, Luca Bedogni, Marco Di Felice, Luciano Bononi},
  Journal = {Proc. of the 14th Workshop on Context and Activity Modeling and Recognition (IEEE COMOREA 2018)},
  Title = {Custom Dual Transportation Mode Detection by Smartphone Devices Exploiting Sensor Diversity},
  Year = {2018},
  DOI = {https://doi.org/10.1109/PERCOMW.2018.8480119}
}
```