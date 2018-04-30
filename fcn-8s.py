import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
%matplotlib inline
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat
import cv2
from skimage.io import imshow
from keras.models import Sequential, Input, Model
from keras.layers import Conv2D,Conv2DTranspose, Cropping2D, Dense, Activation
from keras.layers import Dropout, Flatten,MaxPooling2D, Merge, Average,add
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

#fcn-8s architecture

# Number of classes
n_classes = 21

# Input for a tensor in tensorflow
inp = Input(batch_shape=(None, 224, 224, 3))

#Notation : b12 implies block 1 layer 2 
# block 1
b11 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1' )(inp)
b12 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2' )(b11)
b1_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool' )(b12)

# block 2
b21 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1' )(b1_pool)
b22 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(b21)
b2_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(b22)

# block 3
b21 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1' )(b2_pool)
b22 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2' )(b21)
b23 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3' )(b22)
b3_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool' )(b23)

# block 4
b41 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1' )(b3_pool)
b42 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2' )(b41)
b43 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3' )(b42)
b4_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool' )(b43)
# block 5
b51 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(b4_pool)
b52 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(b51)
b53 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(b52)
b5_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool' )(b53)

# FCN
fc1 = (Conv2D(4096, (7, 7), activation='relu', padding="same", name='fc6'))(b5_pool)
fc1 = Dropout(0.5)(fc1)
fc2 = (Conv2D(4096, (1, 1), activation='relu', padding="same", name='fc7'))(fc1)
fc2 = Dropout(0.5)(fc2)
fc3 = (Conv2D(n_classes, (1, 1), activation='relu', padding='same', name='score_fr'))(fc2)

#deconvolution 1
dc1 = Conv2DTranspose(n_classes, (4, 4), strides=(2, 2), activation=None, padding='valid', name='score2')(fc3)
dc1_crop = Cropping2D(cropping = 1)(dc1)

# Skip architecture - layer 1
skip_1 = Conv2D(n_classes, (1, 1), activation=None, padding="same", name="score_pool4")(b4_pool)

# Combibe 1
merge_1 = add(inputs = [dc1_crop, skip_1])

#deconvolution 2
dc2 = Conv2DTranspose(n_classes, (4, 4), strides=(2, 2), activation=None, padding="valid", name="score4")(merge_1)
dc2_crop = Cropping2D(cropping = 1)(dc2)

# Skip architecture - layer 2
skip_2 = Conv2D(n_classes, (1, 1), activation=None, padding="same", name="score_pool3")(b3_pool)

# Combine 2
merge_2 = add(inputs = [dc2_crop, skip_2])

#deconvolution 3
dc3 = Conv2DTranspose(n_classes, (16, 16), strides=(8, 8), activation=None, padding="valid", name="upsample")(merge_2)
dc3_crop = Cropping2D(cropping=4)(dc3)

# softmax activation - probability estimation
op = (Activation('softmax'))(dc3_crop)

output =  Conv2D(1,(3, 3), activation='relu', name = 'output',padding='same')(op)

# Model summary
FCN8 = Model(inp, output)
# FCN8.summary()

#compile
FCN8.compile(loss="kullback_leibler_divergence", optimizer='adam', metrics=['accuracy'])

#transfer learning - VGGnet to FCN8

#pascal-fcn8s-tvg-dag.mat can be found at http://www.vlfeat.org/matconvnet/pretrained/#semantic-segmentation

transfer_weights = loadmat('C://Users/jchin/Desktop/image_segmen/pascal-fcn8s-tvg-dag.mat', struct_as_record=False)
params = transfer_weights['params']

def transfer_learning(input_model):
    layer_names = [l.name for l in input_model.layers]
    for i in range(0, params.shape[1]):
        param_name = params[0,i].name[0][0:-1]
        param_type = params[0,i].name[0][-1] 
        
        if param_name in layer_names:
            index = layer_names.index(param_name)
            assert (len(input_model.layers[index].get_weights()) == 2)
            
            if  param_type in ['f','filter']:
                layer_weights = (params[0,i].value).transpose((0,1,2,3))
                layer_weights = np.flip(layer_weights, 2)
                layer_weights = np.flip(layer_weights, 3)
                assert (layer_weights.shape == input_model.layers[index].get_weights()[0].shape)
                bias = input_model.layers[index].get_weights()[1]
                input_model.layers[index].set_weights([layer_weights, bias])
                
            elif param_type in ['b','bias']:
                layer_bias = params[0,i].value
                assert (layer_bias.shape[1] == 1)
                assert (layer_bias[:,0].shape == input_model.layers[index].get_weights()[1].shape)
                filter_value = input_model.layers[index].get_weights()[0]
                input_model.layers[index].set_weights([filter_value, layer_bias[:,0]])
        else:
            print ('not found : ', str(param_name))
            
transfer_learning(FCN8)

# Image directory
image_directory = 'C://Users/jchin/Desktop/image_segmen/VOC2012/JPEGImages/'
segm_image_directory = 'C://Users/jchin/Desktop/image_segmen/VOC2012/SegmentationClass/'
train_set_list = 'C://Users/jchin/Desktop/image_segmen/VOC2012/ImageSets/Segmentation/train.txt'
validation_set_list = 'C://Users/jchin/Desktop/image_segmen/VOC2012/ImageSets/Segmentation/trainval.txt'

#data preprocessing 

#Extract train and validation sets

# Train set
train_set = open(train_set_list, "r")
train_set_names = []
for l in train_set:
    train_set_names.append(l.strip())
train_set.close()   

#Prepare training images
train_images = []
for i in range(len(train_set_names)):
    train_images.append(image_directory + train_set_names[i] + '.jpg')
train_images.sort()

#segmented images of training data 
segm_set = []
for i in range(len(train_set_names)):
    segm_set.append(segm_image_directory + train_set_names[i] + '.png')
segm_set.sort()

#validation set
valid_set = open(validation_set_list, "r")
valid_set_names = []
for l in valid_set:
    valid_set_names.append(l.strip())
valid_set.close()

# validation set images 
valid_set = []
for i in range(len(valid_set_names)):
    valid_set.append(image_directory + valid_set_names[i] + '.jpg')
valid_set.sort()

#Load images and generate numpy arrays of images to feed into the model
height, width = (224, 224)

def extract_data(path, label=None):
    img = Image.open(path)
    img = img.resize((224,224))
   
    if label:
        y = np.frombuffer(img.tobytes(), dtype=np.uint8).reshape((224,224,1))
        y = y.astype('float64')
        y = y[None,:]
        return y
    else:
        X = np.frombuffer(img.tobytes(), dtype=np.uint8).reshape((224,224,3))
        X = X.astype('float64')
        X = X[None,:]
        return X

def generate_arrays_from_file(image_list, train_directory, test_directory,validate = None):
    while True:
        for image_name in image_list:
            train_path = train_directory + "{}.jpg".format(image_name)
            test_path = test_directory  + "{}.png".format(image_name)

            X = extract_data(train_path, label=False)
            y = extract_data(test_path, label=True)
            
            if validate:
                yield np.array(X)
            else:
                yield np.array(X) , np.array(y)

# Model training 
n_epoch = 100
steps_per_epoch = len(train_set_names)/100

FCN8.fit_generator(generator=generate_arrays_from_file(train_set_names, image_directory, segm_image_directory),
                     steps_per_epoch=steps_per_epoch,
                     epochs=n_epoch)
                     
#validate model

n_steps = len(valid_set_names)
predicted_images = FCN8.predict_generator(generate_arrays_from_file(valid_set_names, image_directory, 
                                                                     segm_image_directory,validate= 1), steps =n_steps)

#Evaluate model accuracy

mean_accuracy = FCN8.evaluate_generator(generate_arrays_from_file(valid_set_names, image_directory, 
                                                                     segm_image_directory), steps =n_steps)
print('Accuracy of FCN8 model: ',mean_accuracy)

#generate segmentation images of validation set

valid_segm_img = []
for i in range(len(valid_set_names)):
    valid_segm_raw = Image.open(segm_image_directory + valid_set_names[i] + '.png')
    valid_segm_raw = valid_segm_raw.resize((224, 224))
    reshaped_img = np.frombuffer(valid_segm_raw.tobytes(), dtype=np.uint8).reshape((224,224,1))
    reshaped_img = reshaped_img.astype('float32')
    valid_segm_img.append(reshaped_img)

#Pixel accuracy
def Acc(y_true, y_pred):
        return np.average(np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1)))
    
rmse = Acc(predicted_images,valid_segm_img)

print('pixel accuracy:',rmse)

# Intersection over Union
def IoU(y_true, y_pred):
    return np.mean(np.asarray([Acc(y_pred[i], y_true[i]) for i in range(len(y_true))])) 

IoU_acc = IoU(predicted_images,valid_segm_img)
print('Intersection over Union accuracy:',IoU_acc )

#Sample segmented image with FCN-8s model

# Read Image
inputImg = Image.open('C://Users/jchin/Desktop/image_segmen/VOC2012/JPEGImages/2007_002227.jpg')
inputImg = inputImg.resize((224, 224))
inputImgP = np.frombuffer(inputImg.tobytes(), dtype=np.uint8).reshape((224,224,3))
inputImgP = inputImgP[None,:]
inputImgP = inputImgP.astype('float32')

# Feed image to trained model and predict segmented image
preds = FCN8.predict(inputImgP)

#Plot input image
plt.subplot(1, 3, 1)
plt.imshow(Image.open('C://Users/jchin/Desktop/image_segmen/VOC2012/JPEGImages/2007_002227.jpg'))

#Plot segmented image
plt.subplot(1, 3, 2)
plt.imshow(Image.open('C://Users/jchin/Desktop/image_segmen/VOC2012/SegmentationClass/2007_002227.png'))

#Plot predicted segmented image
plt.subplot(1, 3, 3)
plt.imshow(preds)

