from skimage import io as skio
from skimage import transform as skitra
from scipy import misc
import glob
import os

# _RGB_MEANS = [123.68, 116.78, 103.94]
_IMG_SCALE = 448
_IMG_REGION_SIZE = 224
_IMG_NR_REGION = 9 # 3 x 3
_SUB_IMG_STEPSIZE = _IMG_REGION_SIZE / 2
_SAVE_PATH = 'sub_img_'
_IMAGE_PATHS = ['train2014/*.jpg', 'test2014/*.jpg', 'val2014/*.jpg']


def split_image(file):
    image = skio.imread(file)
    image = skitra.resize(image, [_IMG_SCALE, _IMG_SCALE])
    #image = image[:,:] - _RGB_MEANS # subtract rgb means from image (we don't normalize, since vgg was trained w/o norm.)
    sub_images = [image for i in range(_IMG_NR_REGION)]
    for i in range(3):
        for j in range(3):
            sub_images[i*3 + j] = image[int(i*_SUB_IMG_STEPSIZE):int(_SUB_IMG_STEPSIZE+(i+1)*_SUB_IMG_STEPSIZE),
									    int(j*_SUB_IMG_STEPSIZE):int(_SUB_IMG_STEPSIZE+(j+1)*_SUB_IMG_STEPSIZE)]

    return sub_images


for path in _IMAGE_PATHS:
    if not os.path.exists(_SAVE_PATH + path[:-5]):
        os.makedirs(_SAVE_PATH + path[:-5])

    for file in glob.glob(path): # iterate over all pictures
        sub_imgs = split_image(file) # split image into _IMG_NR_REGION regions

        ctr = 0
        for img in sub_imgs:
            try:
                misc.toimage(img).save(_SAVE_PATH + path[:-5] + file[9:-4] + str(ctr) +'.jpg') # save sub_imgs in folder sub_img
            except:
                print("Something with", _SAVE_PATH + path[:-5] + file[9:-4] + str(ctr) , " went wrong...")
            ctr += 1


