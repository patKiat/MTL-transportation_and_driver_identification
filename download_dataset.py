from urllib.request import urlopen 
import os
import tarfile
import shutil
import glob

# Constant
dataset_dir = './TransportationData'
datasetBalanced = dataset_dir + '/datasetBalanced'
rawOriginaldata = dataset_dir + '/_RawDataOriginal'
url_list = ['http://cs.unibo.it/projects/us-tm2017/static/dataset/extension/5second/dataset_5secondWindow.csv',
            'http://cs.unibo.it/projects/us-tm2017/static/dataset/extension/halfsecond/dataset_halfSecondWindow.csv',
            'http://cs.unibo.it/projects/us-tm2017/static/dataset/raw_data/raw_data.tar.gz']
dataset5second = 'dataset_5secondWindow.csv'
datasethalfsecond = 'dataset_halfSecondWindow.csv'
rawdataorig = "raw_data.tar.gz"


if __name__ == "__main__":
    # create folders
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if not os.path.exists(datasetBalanced):
        os.makedirs(datasetBalanced)

    if not os.path.exists(rawOriginaldata):
        os.makedirs(rawOriginaldata)

    print("DOWNLOAD........")
    for url in url_list:
        response = urlopen(url)
        csv = response.read()
        if url == 'http://cs.unibo.it/projects/us-tm2017/static/dataset/extension/5second/dataset_5secondWindow.csv':
            outfile = datasetBalanced + '/' +dataset5second
        elif url == 'http://cs.unibo.it/projects/us-tm2017/static/dataset/raw_data/raw_data.tar.gz':
            outfile = rawOriginaldata + '/' + rawdataorig
        else:
            outfile = datasetBalanced + '/' + datasethalfsecond

        with open(outfile, 'wb') as f:
            f.write(csv)

        if url == "http://cs.unibo.it/projects/us-tm2017/static/dataset/raw_data/raw_data.tar.gz":
            tar = tarfile.open(outfile, "r:gz")
            tar.extractall(path="TransportationData/")
            tar.close()
            for filename in glob.iglob('TransportationData/raw_data/*/*.csv'):
                shutil.move(filename, rawOriginaldata)
            os.remove(outfile)
            shutil.rmtree('TransportationData/raw_data/')

    print("DOWNLOAD ENDED.")

## Reference
# @article{carpineti18,
#   Author = {Claudia Carpineti, Vincenzo Lomonaco, Luca Bedogni, Marco Di Felice, Luciano Bononi},
#   Journal = {Proc. of the 14th Workshop on Context and Activity Modeling and Recognition (IEEE COMOREA 2018)},
#   Title = {Custom Dual Transportation Mode Detection by Smartphone Devices Exploiting Sensor Diversity},
#   Year = {2018},
#   DOI = {https://doi.org/10.1109/PERCOMW.2018.8480119}
# }