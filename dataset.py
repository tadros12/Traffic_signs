import os
import urllib.request
import zipfile

os.makedirs('dataset', exist_ok=True)

train_data_url = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip'
test_data_url = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip'
test_labels_url = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip'



urllib.request.urlretrieve(train_data_url, 'dataset/GTSRB_Final_Training_Images.zip')
urllib.request.urlretrieve(test_data_url, 'dataset/GTSRB_Final_Test_Images.zip')
urllib.request.urlretrieve(test_labels_url, 'dataset/GTSRB_Final_Test_Labels.zip')




with zipfile.ZipFile('dataset/GTSRB_Final_Training_Images.zip', 'r') as zip_ref:
    zip_ref.extractall('dataset')

with zipfile.ZipFile('dataset/GTSRB_Final_Test_Images.zip', 'r') as zip_ref:
    zip_ref.extractall('dataset')

with zipfile.ZipFile('dataset/GTSRB_Final_Test_Labels.zip', 'r') as zip_ref:
    zip_ref.extractall('dataset')

