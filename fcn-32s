import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat
import cv2
from skimage.io import imshow
from keras.models import Sequential
from keras.layers import Conv2D,Conv2DTranspose, Cropping2D, Dense, Activation, Dropout, Flatten,MaxPooling2D, Merge, Average
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

# Number of classes
n_classes = 21
input_shape = (224, 224, 3)

#fcn-32s architecture

#block1
FCN32 = Sequential()
FCN32.add(Conv2D(64,(3, 3), activation='relu', input_shape=input_shape, padding='same',name = 'conv1_1'))
FCN32.add(Conv2D(64,(3, 3), activation='relu', name = 'conv1_2',padding='same'))
FCN32.add(MaxPooling2D(pool_size=(2,2), strides = (2,2), name = 'block1_pool'))

#block2
FCN32.add(Conv2D(128,(3, 3), activation='relu', name = 'conv2_1',padding='same'))
FCN32.add(Conv2D(128,(3, 3), activation='relu', name = 'conv2_2',padding='same'))
FCN32.add(MaxPooling2D(pool_size=(2,2), strides = (2,2), name = 'block2_pool'))

#block3
FCN32.add(Conv2D(256,(3, 3), activation='relu', name = 'conv3_1',padding='same'))
FCN32.add(Conv2D(256,(3, 3), activation='relu', name = 'conv3_2',padding='same'))
FCN32.add(Conv2D(256,(3, 3), activation='relu', name = 'conv3_3',padding='same'))
FCN32.add(MaxPooling2D(pool_size=(2,2), strides = (2,2), name = 'block3_pool'))

#block4
FCN32.add(Conv2D(512,(3, 3), activation='relu', name = 'conv4_1',padding='same'))
FCN32.add(Conv2D(512,(3, 3), activation='relu', name = 'conv4_2',padding='same'))
FCN32.add(Conv2D(512,(3, 3), activation='relu', name = 'conv4_3',padding='same'))
FCN32.add(MaxPooling2D(pool_size=(2,2), strides = (2,2), name = 'block4_pool'))

#block5
FCN32.add(Conv2D(512,(3, 3), activation='relu', name = 'conv5_1',padding='same'))
FCN32.add(Conv2D(512,(3, 3), activation='relu', name = 'conv5_2',padding='same'))
FCN32.add(Conv2D(512,(3, 3), activation='relu', name = 'conv5_3',padding='same'))
FCN32.add(MaxPooling2D(pool_size=(2,2), strides = (2,2), name = 'block5_pool'))

#block6
FCN32.add(Conv2D(4096,(7, 7), activation='relu', name = 'fc6',padding='same'))
FCN32.add(Dropout(0.5))
FCN32.add(Conv2D(4096,(1, 1), activation='relu', name = 'fc7',padding='same'))
FCN32.add(Dropout(0.5))


# Transformation
FCN32.add(Conv2D(n_classes,(1, 1), activation='linear', kernel_initializer='he_normal', padding='valid', strides=(1, 1), name= 'score_fr'))

#deconvolution
FCN32.add(Conv2DTranspose(n_classes,kernel_size = (64, 64),strides = (32,32), name = 'upsample'))
FCN32.add(Cropping2D(cropping = 16))
FCN32.add(Activation('softmax', name = 'ac1'))
FCN32.add(Conv2D(1,(3, 3), activation='relu', name = 'f',padding='same'))
FCN32.summary()
#compile model
FCN32.compile(loss="kullback_leibler_divergence", optimizer='adam', metrics=['accuracy'])


#transfer learning - VGGnet to FCN32

#user path to pascal-fcn32s-dag.mat 
#pascal-fcn32s-dag.mat can be found at http://www.vlfeat.org/matconvnet/pretrained/#semantic-segmentation


transfer_weights = loadmat('C://Users/jchin/Desktop/image_segmen/pascal-fcn32s-dag.mat', matlab_compatible=False, struct_as_record=False)
params = transfer_weights['params']

def transfer_learning(input_model):
    layer_names = [l.name for l in input_model.layers]
    for i in range(0, params.shape[1]-1, 2):
        t_name = '_'.join(params[0,i].name[0].split('_')[0:-1])
        if t_name in layer_names:
            kindex = layer_names.index(t_name)
            t_weights = params[0,i].value
            t_bias = params[0,i+1].value
            input_model.layers[kindex].set_weights([t_weights, t_bias[:,0]])
        else:
            print ('not found: ', str(t_name))
            
transfer_learning(FCN32)

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

FCN32.fit_generator(generator=generate_arrays_from_file(train_set_names, image_directory, segm_image_directory),
                     steps_per_epoch=steps_per_epoch,
                     epochs=n_epoch)
                 
                 
#validate model

n_steps = len(valid_set_names)
predicted_images = FCN32.predict_generator(generate_arrays_from_file(valid_set_names, image_directory, 
                                                                     segm_image_directory,validate= 1), steps =n_steps)

#Evaluate model accuracy

mean_accuracy = FCN32.evaluate_generator(generate_arrays_from_file(valid_set_names, image_directory, 
                                                                     segm_image_directory), steps =n_steps)
print('Accuracy of FCN32 model: ',mean_accuracy)

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

#Sample segmented image with FCN-32s model
 
# Read Image
inputImg = Image.open('C://Users/jchin/Desktop/image_segmen/VOC2012/JPEGImages/2007_002227.jpg')
inputImg = inputImg.resize((224, 224))
inputImgP = np.frombuffer(inputImg.tobytes(), dtype=np.uint8).reshape((224,224,3))
inputImgP = inputImgP[None,:]
inputImgP = inputImgP.astype('float32')

# Feed image to trained model and predict segmented image
preds = FCN32.predict(inputImgP)

#Plot input image
plt.subplot(1, 3, 1)
plt.imshow(Image.open('C://Users/jchin/Desktop/image_segmen/VOC2012/JPEGImages/2007_002227.jpg'))

#Plot segmented image
plt.subplot(1, 3, 2)
plt.imshow(Image.open('C://Users/jchin/Desktop/image_segmen/VOC2012/SegmentationClass/2007_002227.png'))

#Plot predicted segmented image
plt.subplot(1, 3, 3)
plt.imshow(preds)
