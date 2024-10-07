#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################## WORKING ###########################################
# if is set to false build new dataset from directory with original files collected from users
HAVE_DT = False
CREATE_BALANCED_DATASET = False
REMOVE_NAN = True
REMOVE_USER = False

################################## Directories ########################################
if REMOVE_NAN == True & REMOVE_USER == False:
    # working directory
    DIR_TM = './TransportationDataWithoutNan'
    EXCLUDED_USER = {}
elif REMOVE_NAN == False & REMOVE_USER == False:
    DIR_TM = './TransportationData'
    EXCLUDED_USER = {}
elif REMOVE_NAN == True & REMOVE_USER == True:
    DIR_TM = './TransportationDataWithoutNanAndSomeUser'
    EXCLUDED_USER = {'U4', 'U6', 'U8', 'U16'}
else:
    DIR_TM = './TransportationDataWithoutNan'
    EXCLUDED_USER = {}

# directory with original files collected from users
DIR_RAW_DATA_ORIGINAL = DIR_TM + '/_RawDataOriginal'

# directory to fill with correct files
DIR_RAW_DATA_CORRECT = DIR_TM + '/_RawDataCorrect/'
# directory to fill with correct files and sensor data after transformations
DIR_RAW_DATA_TRANSFORM = DIR_TM + '/_RawDataTransform/'
# directory to fill with all file transformed according to dataset header
DIR_RAW_DATA_HEADER = DIR_TM + '/_RawDataHeader'
# directory to fill with all file transformed according to dataset header and features on time windows
DIR_RAW_DATA_FEATURES = DIR_TM + '/_RawDataFeatures'
# directory to fill with dataset file
DIR_DATASET = DIR_TM + '/_Dataset'

# directory to fill with all results
DIR_RESULTS = './TransportationDetectionResults'

##################################### Files ###########################################
CLEAN_LOG = 'cleanLog.log'
# file for save users and sensors support
FILE_SUPPORT = 'sensors_support.csv'

# dataset file name
if CREATE_BALANCED_DATASET == False:
    FILE_DATASET = "dataset.csv"
    FILE_TRAINING = "training.csv"
    FILE_TEST = "test.csv"
    FILE_CV = "cv.csv"
else:
    # balanced files name
    FILE_DATASET_BALANCED = "dataset_balanced.csv"
    FILE_TRAINING = "training_balanced.csv"
    FILE_TEST = "test_balanced.csv"
    FILE_CV = "cv_balanced.csv"

# classification results
FILE_DECISION_TREE_RESULTS = "_decision_tree.csv"
FILE_RANDOM_FOREST_RESULTS = "_random_forest.csv"
FILE_NEURAL_NETWORK_RESULTS = "_neural_network.csv"
FILE_SUPPORT_VECTOR_MACHINE_RESULTS = "_support_vector_machine.csv"


# classification combination of classes results
FILE_TWO_CLASSES_COMBINATION = "_two_classes_combination.csv"

# leave one subject out results
FILE_LEAVE_ONE_SUBJECT_OUT = "_leave_one_subject_out.csv"

# single sensor analysis
FILE_SINGLE_SENSOR_ANALYSIS = "_single_sensors_accuracy.csv"
#######################################################################################

################################## Sensors #############################################
################## list of sensors to exclude for classification #######################
SENSORS_TO_EXCLUDE_FROM_FILES = ['', 'com.samsung.sensor.grip']

if REMOVE_NAN == False:
    SENSORS_TO_EXCLUDE_FROM_DATASET = ['com.google.sensor.internal_temperature', 'com.qti.sensor.amd',
                                   'com.qti.sensor.rmd', 'android.sensor.tilt_detector',
                                   'android.sensor.geomagnetic_rotation_vector', 'android.sensor.step_detector',
                                   'android.sensor.gesture']
else:
    SENSORS_TO_EXCLUDE_FROM_DATASET = ['activityrecognition', 'com.google.sensor.internal_temperature', 'com.qti.sensor.amd',
                                   'com.qti.sensor.rmd', 'android.sensor.tilt_detector',
                                   'android.sensor.geomagnetic_rotation_vector', 'android.sensor.step_detector',
                                   'android.sensor.gesture','speed',
                                       'android.sensor.light', 'android.sensor.step_counter', 'android.sensor.step_counter', 
                                       'android.sensor.proximity', 'android.sensor.pressure']

# compute magnitude
SENSOR_TO_TRANSFORM_MAGNITUDE = ['android.sensor.accelerometer', 'android.sensor.orientation',
                                 'android.sensor.linear_acceleration', 'android.sensor.gyroscope',
                                 'android.sensor.magnetic_field', 'android.sensor.magnetic_field_uncalibrated',
                                 'android.sensor.gyroscope_uncalibrated', 'android.sensor.gravity']

# compute sin(θ/2)
SENSOR_TO_TRANSFROM_4ROTATION = ['android.sensor.rotation_vector',
                                 'android.sensor.game_rotation_vector']

# sensors for which it is relevant only the first data
SENSOR_TO_TAKE_FIRST = ['android.sensor.light', 'speed', 'android.sensor.proximity', 'android.sensor.pressure']

#######################################################################################

################################# Time Division ########################################
# dimension in second
WINDOW_DIMENSION = 5
# in our application every second the application write 10 model
SAMPLE_FOR_SECOND = 1000

#######################################################################################

################################# Split rules ########################################
TRAINING_PERC = 0.6
TEST_PERC = 0.4
CV_PERC = 0.0

#######################################################################################

###################### sensor classification levels ###################################
sensor_to_exclude_first = ['com.google.sensor.internal_temperature',
                           'com.qti.sensor.amd',
                           'com.qti.sensor.rmd',
                           'android.sensor.step_detector',
                           'android.sensor.tilt_detector',
                           'android.sensor.geomagnetic_rotation_vector',
                           'android.sensor.step_counter',
                           'android.sensor.pressure',
                           'android.sensor.magnetic_field_uncalibrated',
                           'android.sensor.proximity',
                           'android.sensor.magnetic_field',
                           'android.sensor.gravity',
                           'android.sensor.light',
                           'android.sensor.gesture',
                           'android.sensor.linear_acceleration',
                           'android.sensor.orientation',
                           'android.sensor.rotation_vector',
                           'android.sensor.game_rotation_vector',
                           'android.sensor.gyroscope_uncalibrated',
                           'activityrecognition',
                           'speed',
                           'time']

sensor_to_exclude_second = ['com.google.sensor.internal_temperature',
                            'com.qti.sensor.amd',
                            'com.qti.sensor.rmd',
                            'android.sensor.step_detector',
                            'android.sensor.tilt_detector',
                            'android.sensor.geomagnetic_rotation_vector',
                            'android.sensor.step_counter',
                            'android.sensor.pressure',
                            'android.sensor.magnetic_field_uncalibrated',
                            'android.sensor.proximity',
                            'android.sensor.magnetic_field',
                            'android.sensor.gravity',
                            'android.sensor.light',
                            'android.sensor.gesture',
                            'speed',
                            'activityrecognition',
                            'time']

sensors_to_exclude_third = ['com.google.sensor.internal_temperature',
                            'com.qti.sensor.amd',
                            'com.qti.sensor.rmd',
                            'android.sensor.step_detector',
                            'android.sensor.tilt_detector',
                            'android.sensor.geomagnetic_rotation_vector',
                            'android.sensor.step_counter',
                            'android.sensor.pressure',
                            'android.sensor.magnetic_field_uncalibrated',
                            'android.sensor.proximity',
                            'android.sensor.magnetic_field',
                            'android.sensor.gravity',
                            'android.sensor.light',
                            'android.sensor.gesture',
                            'activityrecognition',
                            'time']
########################################################################################

################################### data type for dataset ##################################
DATASET_DATA_TYPE = {"user_id": int}

## Reference
# @article{carpineti18,
#   Author = {Claudia Carpineti, Vincenzo Lomonaco, Luca Bedogni, Marco Di Felice, Luciano Bononi},
#   Journal = {Proc. of the 14th Workshop on Context and Activity Modeling and Recognition (IEEE COMOREA 2018)},
#   Title = {Custom Dual Transportation Mode Detection by Smartphone Devices Exploiting Sensor Diversity},
#   Year = {2018},
#   DOI = {https://doi.org/10.1109/PERCOMW.2018.8480119}
# }

