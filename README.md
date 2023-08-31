
# Face Recognition with Attendance System

This Project recognizes the face from the Webcam and marks the Attendance of that person in the Attendance csv file.

## Introduction

Face recognition is a method of identifying or verifying the identity of an individual using their face. There are various algorithms that can do face recognition but their accuracy might vary. Here I am going to describe how I do face recognition using **OpenCV**.

## OpenCV
### Overview
OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. OpenCV was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in the commercial products








## Pre-requisites
Hands-on knowledge of Python is essential before working on the concepts of OpenCV. Make sure that you have the following packages installed and running before starting this project.
 - Numpy
 - CMake , dlib and face_recognition
 - os
 - OpenCV


## Installing Packages in the Project


```bash
  pip install CMake # installing CMake 
  pip install dlib  # installing dlib 
  pip install face recognition  # installing face recognition
  pip install OpenCV  # installing OpenCV
```


## Import Libraries

Now that you have downloaded all the important libraries let’s import them to build the system

```bash 
import cv2
import numpy as np
import face_recognition as fr
import os
from datetime import datetime
```
 ### Now create a list to store person_name and image array.
 - traverse all image file present in path directory, read images, and append the image array to the Image list, and file-name to ClassNames.

```bash 
path = 'Attendance Images'
Images = []
ClassNames = []
Mylist = os.listdir(path)
for cl in Mylist:
    CurrentImage = cv2.imread(f'{path}/{cl}')
    Images.append(CurrentImage)
    ClassNames.append(os.path.splitext(cl)[0]) # As we want only the name of the image not type
# Print the names of the images in the List.
print(ClassNames)
```
### Create a function to encode all the train images and store them in a variable EncodeListKnown
```bash 
def findEcnoding(images):
    EncodeList = []
    for img in images:
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)  # convert Images BGR TO RGB
        encode = fr.face_encodings(img)[0]  # finding encoding of the store images
        EncodeList.append(encode)
    return EncodeList

EncodeListKnown = findEcnoding(Images)

# Print When Encoding is Complete (Hint)
print("Encoding Complete")
```

### Creating a function that will create a Attendance.csv file to store the attendance with time.

```bash
def MarkAttendance(name):
    with open('Attendance.csv' , 'r+') as f:  # As we want to read and write in the file at the same time
        MyDataList = f.readlines()
        NameList = []
        for line in MyDataList:
            entry = line.split(',')
            NameList.append(entry[0])
        if name not in NameList:
            now = datetime.now()
            Time = now.strftime("%H:%M:%S") # Give the time when the faces are matched
            f.writelines(f'\n{name},{Time}')
```
with open(“filename.csv”,’r+’) creates a file and ‘r+’ mode is used to open a file for reading and writing.

We first check if the name of the attendee is already available in attendance.csv we won’t write attendance again.

If the attendee’s name is not available in attendance.csv we will write the attendee name with a time of function call.

### Open Webcam for Real-Time Face Recognition
```bash 
# Open the WebCam

while True:
    # Read the image
    success , img = cap.read()

    # Resize the image to 1/4th of size so it takes less time by Machine to compare the image.
    imgs = cv2.resize(img , (0,0) , None , 0.25 , 0.25)

    imgs = cv2.cvtColor(imgs , cv2.COLOR_BGR2RGB)

    # Finding the location of the faces in the current Frame  , returns the coordinates of the faces.
    CurrentFaceLoc = fr.face_locations(imgs)

    # Find the encoding of each face in the frame with locations.
    CurrentFaceEncoding = fr.face_encodings(imgs , CurrentFaceLoc)

```
- Resize the image by 1/4 only for the recognition part. output frame will be of the original size.
- Resizing improves the Frame per Second.
- face_recognition.face_locations() is called on the resized image(imgS) .for face bounding box coordinates must be multiplied by 4 in order to overlay on the output frame.
- face_recognition.distance() returns an array of the distance of the test image with all images present in our train directory.
- The index of the minimum face distance will be the matching face.

### Create a function which compare the real time encode face with stored image.
``` bash
    # Creating a funtion to compare the Current Face and the store image of the Person
    for EncodeFace , FaceLoc in zip(CurrentFaceEncoding , CurrentFaceLoc):
        match = fr.compare_faces(EncodeListKnown,EncodeFace)
        FaceDist = fr.face_distance(EncodeListKnown , EncodeFace)

        # Return the value of each image after comparing with current face lower the value more chances of the face of same person.
        print(FaceDist)

```
###  Return the index having minimum value in the list

``` bash
       matchInd = np.argmin(FaceDist)
```
### Create a rectangle around the recognize face and return the name of the that person whose face is recognize and save it to the csv file.

```bash
      if match[matchInd]:
            name = ClassNames[matchInd].upper()
            print(name)
            y1,x2,y2,x1 = FaceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4 # Resize the image to original

            # Creating a rectangle around the face
            cv2.rectangle(img , (x1,y1) , (x2,y2) , (0,255,0) , 2)

            # Creating a rectangle to show name
            cv2.rectangle(img , (x1,y2-35) , (x2,y2) , (0,255,0) ,cv2.FILLED)

            # Show name on the image
            cv2.putText(img , name , (x1+6 , y2-6) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255,255,255) , 2)

            # Call MarkAttendance to save the name in Attendance.csv.
            MarkAttendance(name)

    cv2.imshow("WebCam", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
```
- After finding the matching name we call the markAttendance function.
- We put the matching name on the output frame using cv2.putText().






## Face Recognition Screenshots

![Bill_Gates](https://user-images.githubusercontent.com/83868776/177783968-c7b87449-6016-4552-892c-b2bbc3469bf7.png)

![elon_musk](https://user-images.githubusercontent.com/83868776/177784206-bf6b446b-f0f9-4787-acc6-ef75df43b37d.png)

## Attendance Screenshot

![atte](https://user-images.githubusercontent.com/83868776/177784711-f4a869d8-91f5-4041-b6f1-304be71b85d7.png)

### Thank You !!!
