import cv2
import numpy as np

cam = cv2.VideoCapture(0)
fileName = input("Enter the name of the person: ")
dataset_path = "./data/"
offset = 20
model = cv2.CascadeClassifier("haarcascade_frontalface.xml")

# creating a list to save face data
faceData = []
skip = 0

while True:
    success, img = cam.read()
    if not success:
        print("Reading Camera Failed!!")
    # store all the gray images
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = model.detectMultiScale(img, 1.3, 5)
    # sorting the faces with the largest area(largest bounding area)

    faces = sorted(faces, key=lambda f: f[2] * f[3])
    # pick the largest face
    if len(faces) > 0:
        f = faces[-1]

        x, y, w, h = f
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # crop and save the largest face
        cropped_face = img[y - offset:y + h + offset, x - offset:x + w + offset]
        cropped_face = cv2.resize(cropped_face, (100, 100))
        skip += 1
        if (skip % 10 == 0):
            faceData.append(cropped_face)
            print("Saved so far " + str(len(faceData)))

    cv2.imshow("Image Window", img)
    #cv2.imshow("Cropped Face", cropped_face)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

#write the faceData on disk(basically save the data on the disk)

faceData = np.asarray(faceData)
m = faceData.shape[0]
faceData = faceData.reshape((m,-1))
print(faceData.shape)

#Save on the disk as np array

filepath = dataset_path + fileName + ".npy"
np.save(filepath, faceData)
print("Data saved successfully to " + filepath)

cam.release()
cv2.destroyAllWindows()
