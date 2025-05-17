#import required libraries files
import os
import numpy as np
from keras.utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from keras.applications import VGG16
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import keras
from sklearn.metrics import accuracy_score
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from sklearn.metrics import confusion_matrix #class to calculate accuracy and other metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import nibabel as nib
import cv2
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
dataset_path = "Dataset"
def getID(name): #function to get ID of the MRI view as label
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index
#function to read labels from dataset
labels = []
for root, dirs, directory in os.walk(dataset_path):#now loop all files and get labels and then display all birds names
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name)
print("Brain Tumor Classes Found in Dataset")  
print(labels)
#now load dataset images
if os.path.exists('model/X.txt.npy'):#if dataset already process then load load it
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
else: #if not process the loop all images from dataset
    X = []
    Y = []
    for root, dirs, directory in os.walk(dataset_path):#loop all images from dataset
        for j in range(len(directory)):
            name = os.path.basename(root)
            if 'Thumbs.db' not in directory[j]:
                img = cv2.imread(root+"/"+directory[j])#read images from looping path
                img = cv2.resize(img, (32,32))#resize images
                X.append(img)#add image features to X
                label = getID(name)#get Image ID
                Y.append(label) #add image id as label                
    X = np.asarray(X)
    Y = np.asarray(Y)    
    np.save('model/X.txt',X)
    np.save('model/Y.txt',Y)
print("Dataset MRI Images Loading Completed")
print("Total images found in dataset : "+str(X.shape[0]))
#plot graph of different labels found in dataset
unique, count = np.unique(Y, return_counts = True)
height = count
bars = labels
y_pos = np.arange(len(bars))
plt.figure(figsize=(6,3))
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.xlabel("Alzheimer  Names")
plt.ylabel("Count")
plt.title("Dataset Class Label Graph")
plt.show()
#defining layers for 3D-UNET CNN model
def build_unet(inputs, ker_init, dropout):
    #defining conv3d cnn layer to build 3dunet
    conv1 = Conv3D(32, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = ker_init)(inputs)
    conv1 = Conv3D(32, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv1)
    
    pool = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv = Conv3D(64, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool)
    conv = Conv3D(64, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv)
    
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv)
    conv2 = Conv3D(128, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool1)
    conv2 = Conv3D(128, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv2)
    
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = Conv3D(256, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool2)
    conv3 = Conv3D(256, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv3)
    
    
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv5 = Conv3D(512, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool4)
    conv5 = Conv3D(512, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv5)
    drop5 = Dropout(dropout)(conv5)

    up7 = Conv3D(256, (2,2,2), activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling3D(size = (2,2,2))(drop5))
    merge7 = concatenate([conv3,up7])
    conv7 = Conv3D(256, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge7)
    conv7 = Conv3D(256, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv7)

    up8 = Conv3D(128, (2,2,2), activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling3D(size = (2,2,2))(conv7))
    merge8 = concatenate([conv2,up8])
    conv8 = Conv3D(128, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge8)
    conv8 = Conv3D(128, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv8)

    up9 = Conv3D(64, (2,2,2), activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling3D(size = (2,2,2))(conv8))
    merge9 = concatenate([conv,up9])
    conv9 = Conv3D(64, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge9)
    conv9 = Conv3D(64, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv9)
    
    up = Conv3D(32, (2,2,2), activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling3D(size = (2,2,2))(conv9))
    merge = concatenate([conv1,up])
    conv = Conv3D(32, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge)
    conv = Conv3D(32, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv)
    
    conv10 = Conv3D(4, (1,1,1), activation = 'softmax')(conv)
    
    return Model(inputs = inputs, outputs = conv10)

# dice loss as defined above for 4 classes
def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
   #     K.print_tensor(loss, message='loss value for class {} : '.format(SEGMENT_CLASSES[i]))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
#    K.print_tensor(total_loss, message=' total dice coef: ')
    return total_loss
 
# define per class evaluation of dice coef
def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,1] * y_pred[:,:,:,1]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,1])) + K.sum(K.square(y_pred[:,:,:,1])) + epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,2] * y_pred[:,:,:,2]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,2])) + K.sum(K.square(y_pred[:,:,:,2])) + epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,3] * y_pred[:,:,:,3]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,3])) + K.sum(K.square(y_pred[:,:,:,3])) + epsilon)

# Computing Precision 
def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
# Computing Sensitivity      
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# Computing Specificity
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())
#create & load unet 3d model
input_layer = Input((128, 128, 128, 3))
unet_model = build_unet(input_layer, 'he_normal', 0.2)
unet_model = keras.models.load_model('model/model_per_class.h5',custom_objects={ 'accuracy' : keras.metrics.MeanIoU(num_classes=4),
                                                   "dice_coef": dice_coef,
                                                   "precision": precision,
                                                   "sensitivity":sensitivity,
                                                   "specificity":specificity,
                                                   "dice_coef_necrotic": dice_coef_necrotic,
                                                   "dice_coef_edema": dice_coef_edema,
                                                   "dice_coef_enhancing": dice_coef_enhancing
                                                  }, compile=False)
unet_model.summary()
#dataset preprocessing such as shuffling and normalization
X = X.astype('float32')
X = X/255 #normalizing images
indices = np.arange(X.shape[0])
np.random.shuffle(indices)#shuffling images
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)
print("Dataset Normalization & Shuffling Process completed")
#now splitting dataset into train & test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
print()
print("Dataset train & test split as 80% dataset for training and 20% for testing")
print("Training Size (80%): "+str(X_train.shape[0])) #print training and test size
print("Testing Size (20%): "+str(X_test.shape[0]))
print()
#define global variables to calculate and store accuracy and other metrics
precision = []
recall = []
fscore = []
accuracy = []
#function to calculate various metrics such as accuracy, precision etc
def calculateMetrics(algorithm, predict, testY):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100  
    print(algorithm+' Accuracy  : '+str(a))
    print(algorithm+' Precision   : '+str(p))
    print(algorithm+' Recall      : '+str(r))
    print(algorithm+' FMeasure    : '+str(f))    
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(5, 4)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()
#train VGG16 on processed traion images
vgg16 = VGG16(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
for layer in vgg16.layers:
    layer.trainable = False
vgg16_model = Sequential()
vgg16_model.add(vgg16)
vgg16_model.add(Convolution2D(32, (1 , 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
vgg16_model.add(MaxPooling2D(pool_size = (1, 1)))
vgg16_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
vgg16_model.add(MaxPooling2D(pool_size = (1, 1)))
vgg16_model.add(Flatten())
vgg16_model.add(Dense(units = 256, activation = 'relu'))
vgg16_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
vgg16_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/vgg16_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/vgg16_weights.hdf5', verbose = 1, save_best_only = True)
    hist = vgg16_model.fit(X, Y, batch_size = 32, epochs = 10, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/vgg16_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    vgg16_model = load_model("model/vgg16_weights.hdf5")
#perform prediction on test images and then calculate accuracy and other metrics     
predict = vgg16_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
calculateMetrics("VGG16", predict, y_test1)#call function to calculate accuracy and other metrics
#training tensorflow, keras cnn proposed model
cnn_model = Sequential()
cnn_model.add(InputLayer(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
cnn_model.add(Conv2D(64, (5, 5), activation='relu', strides=(1, 1), padding='same'))
cnn_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
cnn_model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
cnn_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
cnn_model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
cnn_model.add(BatchNormalization())
cnn_model.add(Flatten())
cnn_model.add(Dense(units=100, activation='relu'))
cnn_model.add(Dense(units=100, activation='relu'))
cnn_model.add(Dropout(0.2))
cnn_model.add(Dense(units=y_train.shape[1], activation='softmax'))
cnn_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
if os.path.exists("model/cnn_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
    hist = cnn_model.fit(X_train, y_train, batch_size = 32, epochs = 10, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    cnn_model.load_weights("model/cnn_weights.hdf5")  
#perform prediction on test images and then calculate accuracy and other metrics    
predict = cnn_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
calculateMetrics("Proposed CNN Model", predict, y_test1)#call function to calculate accuracy and other metrics
#classification report for each class
from sklearn.metrics import classification_report
print(classification_report(y_test1, predict, target_names=labels))
#comparison graph between all algorithms
import pandas as pd
df = pd.DataFrame([['VGG16','Precision',precision[0]],['VGG16','Recall',recall[0]],['VGG16','F1 Score',fscore[0]],['VGG16','Accuracy',accuracy[0]],
                   ['Proposed CNN','Precision',precision[1]],['Proposed CNN','Recall',recall[1]],['Proposed CNN','F1 Score',fscore[1]],['Proposed CNN','Accuracy',accuracy[1]],
                  ],columns=['Algorithms','Metrics','Value'])
df.pivot_table(index="Algorithms", columns="Metrics", values="Value").plot(kind='bar', figsize=(5, 3))
plt.title("All Algorithms Performance Graph")
plt.show()
#displaying all algorithms performance in tabular format
columns = ["Algorithm Name","Precison","Recall","FScore","Accuracy"]
values = []
algorithm_names = ["VGG16", "Proposed CNN"]
for i in range(len(algorithm_names)):
    values.append([algorithm_names[i],precision[i],recall[i],fscore[i],accuracy[i]])    
temp = pd.DataFrame(values,columns=columns)
temp
f = open('model/cnn_history.pckl', 'rb')
data = pickle.load(f)
f.close()
cnn_accuracy = data['accuracy']
cnn_loss = data['loss']
f = open('model/vgg16_history.pckl', 'rb')
data = pickle.load(f)
f.close()
vgg_accuracy = data['accuracy']
vgg_loss = data['loss']
plt.figure(figsize=(8,4))
plt.grid(True)
plt.xlabel('EPOCH')
plt.ylabel('Accuracy/Loss Rate')
plt.plot(cnn_accuracy)
plt.plot(cnn_loss)
plt.plot(vgg_accuracy)
plt.plot(vgg_loss)
plt.legend(['Proposed CNN Accuracy', 'Proposed CNN Loss', 'VGG16 Accuracy', 'VGG16 Loss'], loc='upper left')
plt.title('Proposed CNN & VGG16 Training Accuracy & Loss Graph')
plt.show()
#function to convert image gto 3d format
def cv2_to_nibabel(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (128, 128))
    image = np.array(image)
    image = nib.Nifti1Image(image, affine=np.eye(4))
    return image
def getSegmentation(img_path):
    img = cv2.imread(img_path)
    img = cv2_to_nibabel(img)
    img.to_filename('image.nii')
    img = nib.load('image.nii')
    data = img.get_fdata()
    X = np.empty((1, 128, 128, 2))
    flair = data
    ce = data
    X[0,:,:,0] = flair
    X[0,:,:,1] = ce
    data = unet_model.predict(X/np.max(X), verbose=1)
    core = data[:,:,:,1]
    edema= data[:,:,:,2]
    enhancing = data[:,:,:,3]
    core = core[0]
    edema = edema[0]
    segment= enhancing[0]
    cv2.imwrite("segment.jpg", segment*255)
    return cv2.imread("segment.jpg")
    def classifyTumor(test_image, image):
    img = cv2.imread(test_image)
    img = cv2.resize(img, (32,32))#resize image
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255 #normalizing test image
    predict = cnn_model.predict(img)#now using  extension CNN + GRU to predict wild animals
    predict = np.argmax(predict)
    img = cv2.imread(test_image)
    img = cv2.resize(img, (600,400))
    image = cv2.resize(image, (600,400))
    cv2.putText(img, 'Prediction Output : '+labels[predict]+" Detected", (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.putText(image, '3D-UNET Segmented Image', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    return img, image, labels[predict]
img = getSegmentation("testImages/6.jpg")
classify_img, segment, label = classifyTumor("testImages/6.jpg", img) 
plt.figure()
f, axarr = plt.subplots(1,2, figsize=(8,4)) 
axarr[0].imshow(classify_img, cmap="gray")
axarr[0].title.set_text('Tumor Classification ('+label+")")
axarr[1].imshow(segment, cmap="gray")
axarr[1].title.set_text('Tumor Segmented Image')
plt.show()
img = getSegmentation("testImages/4.jpg")
classify_img, segment, label = classifyTumor("testImages/4.jpg", img) 
plt.figure()
f, axarr = plt.subplots(1,2, figsize=(8,4)) 
axarr[0].imshow(classify_img, cmap="gray")
axarr[0].title.set_text('Tumor Classification ('+label+")")
axarr[1].imshow(segment, cmap="gray")
axarr[1].title.set_text('Tumor Segmented Image')
plt.show()
img = getSegmentation("testImages/8.jpg")
classify_img, segment, label = classifyTumor("testImages/8.jpg", img) 
plt.figure()
f, axarr = plt.subplots(1,2, figsize=(8,4)) 
axarr[0].imshow(classify_img, cmap="gray")
axarr[0].title.set_text('Tumor Classification ('+label+")")
axarr[1].imshow(segment, cmap="gray")
axarr[1].title.set_text('Tumor Segmented Image')
plt.show()
img = getSegmentation("testImages/1.jpg")
classify_img, segment, label = classifyTumor("testImages/1.jpg", img) 
plt.figure()
f, axarr = plt.subplots(1,2, figsize=(8,4)) 
axarr[0].imshow(classify_img, cmap="gray")
axarr[0].title.set_text('Tumor Classification ('+label+")")
axarr[1].imshow(segment, cmap="gray")
axarr[1].title.set_text('Tumor Segmented Image')
plt.show()