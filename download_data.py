import urllib.request
import os
import zipfile

print('Beginning file download with urllib2...')

url_train = 'http://images.cocodataset.org/zips/train2014.zip'
url_test = 'http://images.cocodataset.org/zips/test2014.zip'
url_val = 'http://images.cocodataset.org/zips/val2014.zip'
url_resnet_ckpt = 'http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz'

print("Beginning to download checkpoint..")
urllib.request.urlretrieve(url_resnet_ckpt, 'resnet_v1_101.ckpt')

try:
    os.mkdir('train2014')
except:
    # directory already exists
    pass

try:
    os.mkdir('test2014')
except:
    # directory already exists
    pass

try:
    os.mkdir('val2014')
except:
    # directory already exists
    pass

print("Beginning to download train image data..")
urllib.request.urlretrieve(url_train, 'train2014/train2014.zip')
zip = zipfile.ZipFile('1train2014/train2014.zip', 'r')
zip.extractall('train2014/')
zip.close()

print("Beginning to download test image data..")
urllib.request.urlretrieve(url_test, 'test2014/test2014.zip')
zip = zipfile.ZipFile('test2014/test2014.zip', 'r')
zip.extractall('test2014/')
zip.close()

print("Beginning to download val image data..")
urllib.request.urlretrieve(url_val, 'val2014/val2014.zip')
zip = zipfile.ZipFile('val2014/val2014.zip', 'r')
zip.extractall('val2014/')
zip.close()