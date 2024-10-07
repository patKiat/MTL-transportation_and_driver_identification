'''
TMDatasetRemoveNan with removing all NaN values
'''
import csv
import logging
import math
import os
import re
import shutil
import json
from os import listdir
import pandas as pd
import const
import util
import sys
import math
from sklearn.impute import SimpleImputer

class TMDatasetRemoveNan:
    tm = []
    users = []
    sensors = []
    n_files = 0
    header = {}
    header_with_features = {}
    balance_time = 0  # in seconds
    train = pd.DataFrame()
    test = pd.DataFrame()
    cv = pd.DataFrame()

    @property
    def get_users(self):
        if len(self.users) == 0:
            self.__fill_data_structure()
        return self.users

    @property
    def get_tm(self):
        if len(self.tm) == 0:
            self.__fill_data_structure()
        return self.tm

    @property
    def get_sensors(self):
        if len(self.sensors) == 0:
            self.__fill_data_structure()
        return self.sensors

    @property
    def get_header(self):
        if len(self.header_with_features) == 0:
            self.__fill_data_structure()
        return self.header_with_features

    # Fix original raw files problems:
    # (1)delete measure from  **sensor_to_exclude**
    # (2)if **sound** or **speed** measure rows have negative time --> use module
    # (3)if **time** have incorrect values ("/", ">", "<", "-", "_"...) --> delete file
    # (4)if file is empty --> delete file
    def clean_files(self):
        # Remove Existing Log File
        if os.path.exists(const.CLEAN_LOG):
            os.remove(const.CLEAN_LOG)

        # Compile Regular Expressions for Validation
        patternNegative = re.compile("-[0-9]+")
        patternNumber = re.compile("[0-9]+")

        # Create directory for correct files
        if not os.path.exists(const.DIR_RAW_DATA_CORRECT):
            os.makedirs(const.DIR_RAW_DATA_CORRECT)
        else:
            shutil.rmtree(const.DIR_RAW_DATA_CORRECT)
            os.makedirs(const.DIR_RAW_DATA_CORRECT)

        # Create log file
        logging.basicConfig(filename=const.CLEAN_LOG, level=logging.INFO)
        logging.info("CLEANING FILES...")
        print("CLEAN FILES...")
        filenames = listdir(const.DIR_RAW_DATA_ORIGINAL)
        # iterate on files in raw data directory - delete files with incorrect rows
        nFiles = 0
        deletedFiles = 0

        # Adjusting file reading and handling
        for file in filenames:
            if file.endswith(".csv"):
                nFiles += 1
                to_delete = 0
                file_path = os.path.join(const.DIR_RAW_DATA_ORIGINAL, file)
                res_file_path = os.path.join(const.DIR_RAW_DATA_CORRECT, file)

                # It reads the file line by line, correcting negative time values and removing excluded sensor rows. If time values are invalid, it marks the file for deletion.
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as current_file:
                        lines = current_file.readlines()
                except UnicodeDecodeError:
                    logging.info(f"Decode Error: {file}, skipped.")
                    continue

                with open(res_file_path, "w", encoding='utf-8') as file_result:
                    first_line = True

                    for line in lines:
                        if first_line:
                            first_line = False
                            # Optionally process header if needed, assuming first line is header
                            continue

                        line_data = line.strip().split(",")
                        # Correct time data
                        if re.match(patternNegative, line_data[0]):
                            line_data[0] = line_data[0][1:]  # Use modulo of negative time

                        # Check if time is a number, else mark file to delete
                        if not re.match(patternNumber, line_data[0]):
                            to_delete = 1

                        # Check if the sensor data should be included
                        if line_data[1] not in const.SENSORS_TO_EXCLUDE_FROM_FILES:
                            line_result = ",".join(line_data) + '\n'
                            file_result.write(line_result)

                # Remove files with incorrect values or marked for deletion
                if to_delete == 1:
                    logging.info(f"Delete: {file} --- Time with incorrect values")
                    deletedFiles += 1
                    os.remove(res_file_path)

        # delete empty files
        file_empty = []
        filenames = listdir(const.DIR_RAW_DATA_CORRECT)
        for file in filenames:
            full_path = os.path.join(const.DIR_RAW_DATA_CORRECT, file)
            # check if file is empty
            if (os.path.getsize(full_path)) == 0:
                deletedFiles += 1
                file_empty.append(file)
                logging.info("  Delete: " + file + " --- is Empty")
                os.remove(full_path)

        # ensures that all rows in the corrected files match a specific regular expression pattern
        pattern = re.compile("^[0-9]+,[a-z,A-Z._]+,[-,0-9a-zA-Z.]+$", re.VERBOSE)
        # pattern = re.compile("^[0-9]+,[a-z,A-Z,\.,_]+,[-,0-9,a-z,A-Z,\.]+$", re.VERBOSE)
        filenames = listdir(const.DIR_RAW_DATA_CORRECT)
        for file in filenames:
            n_error = 0
            full_path = os.path.join(const.DIR_RAW_DATA_CORRECT, file)
            # check if all row respect regular expression
            with open(full_path) as f:
                for line in f:
                    match = re.match(pattern, line)
                    if match is None:
                        n_error += 1
            if n_error > 0:
                deletedFiles += 1
                os.remove(full_path)

        logging.info("  Tot files in Dataset : " + str(nFiles))
        logging.info("  Tot deleted files : " + str(deletedFiles))
        logging.info("  Remaining files : " + str(len(listdir(const.DIR_RAW_DATA_CORRECT))))

        self.n_files = len(listdir(const.DIR_RAW_DATA_CORRECT))
        logging.info("END CLEAN FILES")
        print("END CLEAN.... results on log file")

    # transform sensor raw data in orientation independent data (with magnitude metric)
    def transform_raw_data(self):
        dir_src = const.DIR_RAW_DATA_CORRECT
        dir_dst = const.DIR_RAW_DATA_TRANSFORM

        if not os.path.exists(dir_src):
            self.clean_files()

        if not os.path.exists(dir_dst):
            os.makedirs(dir_dst)
        else:
            shutil.rmtree(dir_dst)
            os.makedirs(dir_dst)

        if os.path.exists(dir_src):
            filenames = listdir(dir_src)
        else:
            shutil.rmtree(dir_dst)
            sys.exit("THERE ARE NO SYNTHETIC DATA TO BE PROCESSED")

        logging.info("TRANSFORMING RAW DATA...")
        print("TRANSFORMING RAW DATA...")
        for file in filenames:
            if file.endswith(".csv"):
                with open(os.path.join(dir_src, file)) as current_file:
                    with open(os.path.join(dir_dst, file), "w") as file_result:
                        for line in current_file:
                            line_data = line.split(",")
                            endLine = ",".join(line_data[2:])
                            current_time = line_data[0]
                            sensor = line_data[1]
                            user = "," + line_data[(len(line_data) - 2)]
                            target = "," + line_data[(len(line_data) - 1)]
                            target = target.replace("\n","")
                            # check sensors
                            if line_data[1] not in const.SENSORS_TO_EXCLUDE_FROM_DATASET:  # the sensor is not to exclude
                                if line_data[1] not in const.SENSOR_TO_TRANSFORM_MAGNITUDE:  # not to transfrom
                                    if line_data[1] not in const.SENSOR_TO_TRANSFROM_4ROTATION:  # not to trasform (4 rotation)
                                        if line_data[1] not in const.SENSOR_TO_TAKE_FIRST:  # not to take only first data
                                            # report the line as it is
                                            current_sensor = line_data[1]
                                            line_result = current_time + "," + current_sensor + "," + endLine
                                        else:
                                            current_sensor = line_data[1]
                                            # vector_data = line_data[2:] if not self.sintetic else line_data[2:(len(line_data) - 2)]
                                            vector_data = line_data[2:]
                                            vector_data = [float(i) for i in vector_data]
                                            line_result = current_time + "," + current_sensor + "," + str(vector_data[0]) + user + target + "\n"
                                    else:  # the sensor is to transform 4 rotation
                                        current_sensor = line_data[1]
                                        # vector_data = line_data[2:] if not self.sintetic else line_data[2:(len(line_data) - 2)]
                                        vector_data = line_data[2:]
                                        vector_data = [float(i) for i in vector_data]
                                        magnitude = math.sin(math.acos(vector_data[3]))
                                        line_result = current_time + "," + current_sensor + "," + str(magnitude) + user + target + "\n"
                                else:  # the sensor is to transform
                                    current_sensor = line_data[1]
                                    # vector_data = line_data[2:] if not self.sintetic else line_data[2:(len(line_data)-2)]
                                    vector_data = line_data[2:]
                                    vector_data = [float(i) for i in vector_data]
                                    magnitude = math.sqrt(sum(((math.pow(vector_data[0], 2)),
                                                               (math.pow(vector_data[1], 2)),
                                                               (math.pow(vector_data[2], 2)))))
                                    line_result = current_time + "," + current_sensor + "," + str(magnitude) + user + target + "\n"
                                file_result.write(line_result)
            elif file.endswith(".json"):
                shutil.copyfile(os.path.join(dir_src,file),os.path.join(dir_dst,file))
        logging.info("END TRANSFORMING RAW DATA...")
        print("END TRANSFORMING RAW DATA...")

    # Fill tm, users, sensors data structures with the relative data from dataset
    def __fill_data_structure(self):
        dir_src = const.DIR_RAW_DATA_TRANSFORM
        if not os.path.exists(dir_src):
            print("You should clean files first!")
            return -1
        
        filenames = listdir(dir_src)

        for file in filenames:
            if file.endswith(".csv"):
                data = file.split("_")
                if data[2] not in self.tm:
                    self.tm.append(data[2])
                if data[1] not in self.users:
                    self.users.append(data[1])

                f = open(os.path.join(dir_src, file))
                reader = csv.reader(f, delimiter=",")
                for row in reader:
                    if row[1] not in self.sensors and not row[1] == "":
                        self.sensors.append(row[1])
                f.close()

        self.header_with_features = {0: "time"}
        header_index = 1
        for s in self.sensors:
            self.header_with_features[header_index] = s + "#mean"
            self.header_with_features[header_index + 1] = s + "#min"
            self.header_with_features[header_index + 2] = s + "#max"
            self.header_with_features[header_index + 3] = s + "#std"
            header_index += 4

        self.header = {0: "time"}
        header_index = 1
        for s in self.sensors:
            if s != "activityrecognition":
                self.header[header_index] = s + "#0"
                header_index += 1

    # return position of input sensor in header without features
    def __range_position_in_header(self, sensor_name):
        if len(self.header) == 0 or len(self.header_with_features) == 0:
            self.__fill_data_structure()
        range_position = []
        start_pos = end_pos = -1
        i = 0
        found = False
        while True and i < len(self.header):
            compare = (str(self.header[i])).split("#")[0]
            if compare == sensor_name:
                found = True
                if start_pos == -1:
                    start_pos = i
                else:
                    end_pos = i
                i += 1
            else:
                i += 1
                if found:
                    if end_pos == -1:
                        end_pos = i - 2
                    break
        if end_pos == -1:
            end_pos = len(self.header) - 1
        range_position.append(start_pos)
        range_position.append(end_pos)
        return range_position

    # Fill directory with all file consistent with the header without features
    def create_header_files(self):
        dir_src = const.DIR_RAW_DATA_TRANSFORM
        dir_dst = const.DIR_RAW_DATA_HEADER

        if not os.path.exists(dir_src):
            self.transform_raw_data()

        if len(self.header) == 0 or len(self.header_with_features) == 0:
            self.__fill_data_structure()

        if not os.path.exists(dir_dst):
            os.makedirs(dir_dst)
        else:
            shutil.rmtree(dir_dst)
            os.makedirs(dir_dst)

        print("CREATE HEADER FILES....")
        filenames = listdir(dir_src)

        for file in filenames:
            if file.endswith(".csv"):
                current_file_data = file.split("_")
                target = current_file_data[2]
                user = current_file_data[1]
                full_current_file_path = os.path.join(dir_src, file)
                with open(full_current_file_path) as current_file:
                    full_current_file_path = os.path.join(dir_dst, file)
                    with open(full_current_file_path, "w") as file_header:
                        # write first line of file
                        header_line = ""
                        for x in range(0, len(self.header)):
                            if x == 0:  # time
                                header_line = self.header[0]
                            else:
                                header_line = header_line + "," + self.header[x]
                        header_line = header_line + ",target,user" + "\n"
                        file_header.write(header_line)
                        # write all other lines
                        j = -1
                        for line in current_file:
                            j += 1
                            line_data = line.split(",")
                            # first element time
                            new_line_data = {0: line_data[0]}
                            sensor_c = line_data[1]
                            pos = self.__range_position_in_header(sensor_c)
                            # others elements all -1 except elements in range between pos[0] and pos[1]
                            curr_line_data = 2
                            for x in range(1, len(self.header)):  # x is the offset in list new_line_data
                                if x in range(pos[0], pos[1] + 1):
                                    if curr_line_data < len(line_data):
                                        if "\n" not in line_data[curr_line_data]:
                                            if "-Infinity" in line_data[curr_line_data]:
                                                new_line_data[x] = ""
                                            else:
                                                new_line_data[x] = line_data[curr_line_data]
                                        else:
                                            if "-Infinity" in line_data[curr_line_data]:
                                                new_line_data[x] = ""
                                            else:
                                                new_line_data[x] = line_data[curr_line_data].split("\n")[0]
                                        curr_line_data += 1
                                    else:
                                        new_line_data[x] = ""
                                else:
                                    new_line_data[x] = ""
                            new_line = ""
                            for x in range(0, len(new_line_data)):
                                if x == 0:
                                    new_line = new_line_data[0]
                                else:
                                    new_line = new_line + "," + new_line_data[x]
                            new_line = new_line + "," + str(target) + "," + str(user) + "\n"
                            file_header.write(new_line)
            elif file.endswith(".json"):
                shutil.copyfile(os.path.join(dir_src, file), os.path.join(dir_dst, file))
        print("END HEADER FILES....")

    # Fill directory with all file consistent with the featured header divided in time window
    def __create_time_files(self):
        dir_src = const.DIR_RAW_DATA_HEADER
        dir_dst = const.DIR_RAW_DATA_FEATURES

        # create files with header if not exist
        if not os.path.exists(dir_src):
            self.create_header_files()
        if len(self.header) == 0 or len(self.header_with_features) == 0:
            self.__fill_data_structure()
        if not os.path.exists(dir_dst):
            os.makedirs(dir_dst)
        else:
            shutil.rmtree(dir_dst)
            os.makedirs(dir_dst)

        print("DIVIDE FILES IN TIME WINDOWS AND COMPUTE FEATURES....")
        # Build string header
        header_string = ",".join(self.header_with_features.values()) + ",target,user\n"

        # Compute window dimension
        window_dim = int(const.SAMPLE_FOR_SECOND * const.WINDOW_DIMENSION)

        # Loop on header files
        filenames = listdir(dir_src)
        for current_file in filenames:
            if current_file.endswith("csv"):
                current_tm = current_file.split("_")[2]
                current_user = current_file.split("_")[1]

                source_file_path = os.path.join(dir_src, current_file)
                df_file = pd.read_csv(source_file_path, dtype=const.DATASET_DATA_TYPE)

                # Handle NaN values before processing
                df_file.fillna(method='ffill', inplace=True)  # Forward fill
                df_file.fillna(method='bfill', inplace=True)  # Backward fill

                # # Remove columns with excessive NaN values
                # threshold = 0.5  # Example threshold
                # for col in df_file.columns:
                #     if df_file[col].isnull().mean() > threshold:
                #         df_file.drop(col, axis=1, inplace=True)

                # # Use interpolation for remaining NaNs
                # df_file.interpolate(method='linear', limit_direction='both', inplace=True)

                featureNames = [col for col in df_file.columns if col not in ['target', 'user', 'time']]
                end_time = df_file['time'].max()
                destination_file_path = os.path.join(dir_dst, current_file)
                with open(destination_file_path, 'w') as destination_file:
                    destination_file.write(header_string)

                    start_current = 0
                    i = 0
                    while start_current < end_time:
                        end_current = start_current + window_dim
                        df_current = df_file[(df_file['time'] >= start_current) & (df_file['time'] < end_current)]

                        currentLine = []
                        for feature in featureNames:
                            currentMean = df_current[feature].mean()
                            currentMin = df_current[feature].min()
                            currentMax = df_current[feature].max()
                            currentStd = df_current[feature].std()
                            currentLine.extend([currentMean, currentMin, currentMax, currentStd])

                        if df_current.shape[0] > 0:
                            line = f"{i}," + ",".join(map(str, currentLine)) + f",{current_tm},{current_user}\n"
                            destination_file.write(line)
                        start_current += window_dim
                        i += 1

        print("END DIVIDE FILES IN TIME WINDOWS AND COMPUTE FEATURES......")

    def handle_missing_data(self, df):
        """Fill NaN values with zero, considering NaNs represent 0%."""
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        return df_filled

    def __create_dataset(self):
        dir_src = const.DIR_RAW_DATA_FEATURES
        dir_dst = const.DIR_DATASET
        file_dst = const.FILE_DATASET

        # Ensure directories are properly set up
        if not os.path.exists(dir_src):
            self.__create_time_files()

        if not os.path.exists(dir_dst):
            os.makedirs(dir_dst)
        else:
            shutil.rmtree(dir_dst)
            os.makedirs(dir_dst)

        filenames = listdir(dir_src)
        result_file_path = os.path.join(dir_dst, file_dst)
        is_header_written = False  # Flag to track header writing

        with open(result_file_path, 'w', newline='') as result_file:
            for file in filenames:
                if file.endswith(".csv"):
                    current_file_path = os.path.join(dir_src, file)
                    with open(current_file_path, 'r') as current_file:
                        for i, line in enumerate(current_file):
                            if i == 0:  # This is the header line
                                if not is_header_written:
                                    result_file.write(line)  # Write header only once
                                    is_header_written = True
                            else:
                                line_data = line.strip().split(',')
                                # Replace empty strings or 'nan' with '0'
                                line_data = [x if x not in ['', 'nan', 'NaN'] else '0' for x in line_data]
                                result_file.write(','.join(line_data) + '\n')
        print("DATASET CREATION COMPLETED....")


    # splid passed dataframe into test, train and cv
    def __split_dataset(self, df):
        dir_src = const.DIR_DATASET
        file_training_dst = const.FILE_TRAINING
        file_test_dst = const.FILE_TEST
        file_cv_dst = const.FILE_CV

        training, cv, test = util.split_data(df, train_perc=const.TRAINING_PERC, cv_perc=const.CV_PERC,
                                        test_perc=const.TEST_PERC)
        training.to_csv(dir_src + '/' + file_training_dst, index=False)
        test.to_csv(dir_src + '/' + file_test_dst, index=False)
        cv.to_csv(dir_src + '/' + file_cv_dst, index=False)

    # clean files and transform in orientation independent
    def preprocessing_files(self):
        print("START PREPROCESSING...")
        self.clean_files()
        self.transform_raw_data()

    # analyze dataset composition in term of class and user contribution fill balance_time
    # with minimum number of window for transportation mode
    def create_balanced_dataset(self):
        # Create dataset from files
        self.__create_dataset()
        if const.CREATE_BALANCED_DATASET == False:
            dir_src = const.DIR_DATASET
            file_src = const.FILE_DATASET
            print(os.path.join(dir_src, file_src))
            df = pd.read_csv(os.path.join(dir_src, file_src))
            print("START SPLIT TRAIN AND TEST DATASETS....")
            self.__split_dataset(df)
            print("END SPLIT TRAIN AND TEST DATASETS....")
        else:
            dir_src = const.DIR_DATASET
            file_src = const.FILE_DATASET
            file_dst = const.FILE_DATASET_BALANCED

            # Study dataset composition to balance
            if not os.path.exists(dir_src):
                self.__create_dataset()
            if len(self.users) == 0 or len(self.sensors) == 0 or len(self.tm) == 0:
                self.__fill_data_structure()

            print("START CREATE BALANCED DATASET....")
            df = pd.read_csv(os.path.join(dir_src, file_src))
            min_windows = df.shape[0]

            for t in self.tm:  # Loop on transportation mode
                df_t = df[df['target'] == t]
                if df_t.shape[0] < min_windows:
                    min_windows = df_t.shape[0]

            target_df = df.groupby(['target', 'user']).size().reset_index(name='count')
            target_df['percent'] = target_df.groupby('target')['count'].transform(lambda x: 100 * x / float(x.sum()))
            target_df['num'] = target_df['count'].apply(lambda x: min(x, min_windows))

            # Create balanced dataset
            dataset_incremental = pd.DataFrame()
            for _, row in target_df.iterrows():
                current_df = df[(df['user'] == row['user']) & (df['target'] == row['target'])]
                dataset_incremental = pd.concat([dataset_incremental, current_df.sample(n=int(row['num']))])

            dataset_incremental.to_csv(os.path.join(dir_src, file_dst), index=False)
            self.__split_dataset(dataset_incremental)
            print("END CREATE BALANCED DATASET....")

    @property
    def get_train(self):
        return pd.read_csv(f"{const.DIR_DATASET}/{const.FILE_TRAININGD}")

    @property
    def get_test(self):
        return pd.read_csv(f"{const.DIR_DATASET}/{const.FILE_TEST}")

    @property
    def get_cv(self):
        return pd.read_csv(f"{const.DIR_DATASET}/{const.FILE_CV}")

    @property
    def get_dataset(self):
        return pd.read_csv(f"{const.DIR_DATASET}/{const.FILE_DATASET}")

if __name__ == "__main__":
    dataset = TMDatasetRemoveNan()
    dataset.preprocessing_files()
    dataset.create_balanced_dataset()

## Reference
# @article{carpineti18,
#   Author = {Claudia Carpineti, Vincenzo Lomonaco, Luca Bedogni, Marco Di Felice, Luciano Bononi},
#   Journal = {Proc. of the 14th Workshop on Context and Activity Modeling and Recognition (IEEE COMOREA 2018)},
#   Title = {Custom Dual Transportation Mode Detection by Smartphone Devices Exploiting Sensor Diversity},
#   Year = {2018},
#   DOI = {https://doi.org/10.1109/PERCOMW.2018.8480119}
# }