# Imports the required Packages
import cv2
import numpy as np
import face_recognition as fr
import os
from datetime import datetime

# Create a list to store person_name and image array.

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

# Create a function to encode all the train images and store them in a variable EncodeListKnown

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

# Creating a function that will create a Attendance.csv file to store the attendance with time.
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

# Open the WebCam

cap = cv2.VideoCapture(0)

# When the WebCam is open take the picture and compare it with store images.

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

    # Creating a funtion to compare the Current Face and the store image of the Person
    for EncodeFace , FaceLoc in zip(CurrentFaceEncoding , CurrentFaceLoc):
        match = fr.compare_faces(EncodeListKnown,EncodeFace)
        FaceDist = fr.face_distance(EncodeListKnown , EncodeFace)

        # Return the value of each image after comparing with current face lower the value more chances of the face of same person.
        print(FaceDist)

        # Return the index having minimum value in the list
        matchInd = np.argmin(FaceDist)
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
cap.release()
cv2.destroyAllWindows()




