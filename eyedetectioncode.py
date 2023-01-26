import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import random
import glob
import shutil
import splitfolders
import cv2
import numpy as np
import winsound
import pkg_resources

# splitting dataset into training,test & validation:
input_folder = 'C:/Users/KIIT/PycharmProjects/CNN Projects/Driver Drowsiness Detection/Eye_Dataset_Reduced/'
splitfolders.ratio(input_folder,
                   output="C:/Users/KIIT/PycharmProjects/CNN Projects/Driver Drowsiness Detection/Eye_Dataset_Reduced",
                   seed=42,
                   ratio=(.8, .1, .1), group_prefix=None)

DataDirectory = "C:/Users/KIIT/PycharmProjects/CNN Projects/Driver Drowsiness Detection/Eye_Dataset_Reduced/train"
outputClasses = ['closed_eye', 'open_eye']

new_img_size = 224
training_Data = []


# convert_to_rgb = []


# pushing the training dataset images into numpy arrays along-with their corresponding labels:
# Since we'll be incorporating transfer learning by using vgg16 model for training,we'll have to resize
# the images to (224*224*3) used in vgg16.
# We'll also be converting the default grayscale images into RGB.
def create_training_data():
    for category in outputClasses:
        path = os.path.join(DataDirectory, category)
        class_number = outputClasses.index(category)
        # print(class_number)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                convert_to_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                resized_img_array = cv2.resize(convert_to_rgb, (new_img_size, new_img_size))
                training_Data.append([resized_img_array, class_number])
            except Exception as e:
                pass


create_training_data()
print(len(training_Data))

# shuffling the dataset to avoid overfitting :
random.shuffle(training_Data)

X = []
Y = []

# Pushing labels & features in  training_Data in numpy arrays:

for features, label in training_Data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, new_img_size, new_img_size, 3)
Y = np.array(Y)

# Normalizing the features:
X = X / 255.0

# print(X.shape)

# Creating our model:

model = keras.applications.mobilenet.MobileNet()
model.summary()

# Incorporating transfer learning:
base_input = model.layers[0].input
# removing last three layers from default model
base_output = model.layers[-4].output
flat_layer = layers.Flatten()(base_output)
final_output = layers.Dense(1)(flat_layer)
final_output = layers.Activation('sigmoid')(final_output)

updated_model = keras.Model(inputs=base_input, outputs=final_output)
updated_model.summary()

# Now we'll be setting up the loss function and optimizer as well as splitting up the training_Data
# into training and validation:

updated_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
updated_model.fit(X, Y, epochs=1, validation_split=0.1)

# setting up frequency & duration for alarm
frequency = 2500
duration = 1000

# Detecting status of the eye:

# incorporating the CascadeClassifier for face and eye detection via opencv
# faceCascade_path = r'C:\Users\KIIT\PycharmProjects\CNN Projects\Driver Drowsiness Detection\Eye_Dataset_Reduced\haarcascade_frontalface_default.xml'
# faceCascade = cv2.CascadeClassifier(faceCascade_path)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# setting up a counter variable which will keep check of whether we need to activate the alarm or not
counter = 0
while True:
    ret, frame = cap.read()
    # cv2.imshow("Drowsiness Detection", frame)
    # # from the face we are extracting the region for eyes.
    # eyeCascade_path = r'C:\Users\KIIT\PycharmProjects\CNN Projects\Driver Drowsiness Detection\Eye_Dataset_Reduced\haarcascade_eye.xml'
    # haar_eye_xml = pkg_resources.resource_filename('cv2', 'data/haarcascade_eye.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#     eye_cascade = cv2.CascadeClassifier(eyeCascade_path)
    # converting image back to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    eyes_roi = []
    # final_image = []
    for x, y, w, h in eyes:
        # extracting the boundaries corresponding to every eye
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # enclosing the eye region in a bounding box by drawing a rectangle with eye boundaries as
        # vertices.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # now we'll extract the eye portion from the face image
        eyesss = eye_cascade.detectMultiScale(roi_gray)

        if len(eyesss) == 0:
            print("eyes are not detected")
        else:
            #  we are creating the region of interest that has the eye boundaries as end points.
            for (ex, ey, ew, eh) in eyesss:
                eyes_roi = roi_color[ey: ey + eh, ex:ex + ew]

    print(len(eyes_roi))
    # we have to give this extracted portion for training ,so we are resizing it to the value that our
    # default model will accept.
    if frame is not None:
        print(np.array(eyes_roi))
        final_image = cv2.resize(np.array(eyes_roi), (224, 224))
        # incorporating a fourth dimension
        final_image = np.expand_dims(final_image, axis=0)
        # normalizing the obtained image
        final_image = final_image / 255.0

        # predicting by passing the image through our trained model
        Predictions = updated_model.predict(final_image)

        if Predictions > 0:
            status = "Open Eyes"
            print("Open Eyes")
            # cv2.putText(frame,
            #             status,
            #             (158, 150),
            #             font, 3,
            #             (0, 255, 0),
            #             2,
            # cv2.LINE_4)
            x1, y1, w1, h1 = 0, 0, 175, 75

            # Draw black background rectangle
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
            # Add text
            cv2.putText(frame, 'Active', (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
        else:
            counter = counter + 1
            status = "Closed Eyes"
            if counter > 5:
                x1, y1, w1, h1 = 0, 0, 175, 75
                # Draw black background rectangle
                cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                # Add text
                cv2.putText(frame, 'Sleep Alert !!', (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255), 2)
                winsound.Beep(frequency, duration)
                counter = 0

    cv2.imshow("Drowsiness Detection", frame)

    cv2.waitKey(1)
#
# cap.release()
# cv2.destroyAllWindows()
