import cv2
import numpy as np
import os
#
# # Data preparation
#
# dataset_path = "./data/"
# faceData = []
# labels = []
# nameMap = {}
# classId = 0
# offset = 20
#
# for f in os.listdir(dataset_path):
#     if f.endswith(".npy"):
#         nameMap[classId] = f[:-4]
#         # X- values
#         dataItem = np.load(dataset_path + f)
#         m = dataItem.shape[0]
#         faceData.append(dataItem)
#
#         # Y- values
#         target = classId * np.ones((m,))
#         classId+=1
#         labels.append(target)
#
# XT = np.concatenate(faceData, axis=0)
# yT = np.concatenate(faceData, axis=0).reshape((-1,1))
#
# # Algorithm
#
# def dist(p, q):
#     return np.sqrt(np.sum((p - q) ** 2))
#
#
# def knn(X, y, xt, k=5):
#     m = X.shape[0]
#     dlist = []
#
#     # Reshape xt to be compatible with X
#     xt_flattened = xt.flatten()
#
#     for i in range(m):
#         d = dist(X[i], xt_flattened)
#         dlist.append((d, y[i]))
#
#     dlist = sorted(dlist)
#     dlist = np.array(dlist[:k])
#     labels = dlist[:, 1]
#
#     labels, cnts = np.unique(labels, return_counts=True)
#     idx = cnts.argmax()
#     pred = labels[idx]
#
#     return int(pred)
#
#
# # Predictions
# cam = cv2.VideoCapture(0)
# model = cv2.CascadeClassifier("haarcascade_frontalface.xml")
# while True:
#     success, img = cam.read()
#     if not success:
#         print("Reading Camera Failed!!")
#     # store all the gray images
#
#     faces = model.detectMultiScale(img, 1.3, 5)
#
#     # pick the largest face
#
#     #render a box around each face and predicts its name
#     for f in faces:
#
#         x, y, w, h = f
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         # crop and save the largest face
#         cropped_face = img[y - offset:y + h + offset, x - offset:x + w + offset]
#         cropped_face = cv2.resize(cropped_face, (100, 100))
#
#         # Predictions using KNN Algorithm
#         classPredicted = knn(XT,yT,cropped_face,5)
#         namePredicted = nameMap[classPredicted]
#         cv2.putText(img, namePredicted, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2, cv2.LINE_AA)
#         cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
#     cv2.imshow("Prediction Window", img)
#
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
#
# cam.release()
# cv2.destroyAllWindows()
#

# Data preparation

dataset_path = "./data/"
faceData = []
labels = []
nameMap = {}
classId = 0
offset = 20

for f in os.listdir(dataset_path):
    if f.endswith(".npy"):
        nameMap[classId] = f[:-4]
        dataItem = np.load(dataset_path + f)
        m = dataItem.shape[0]
        faceData.append(dataItem)
        labels.extend([classId] * m)  # Assign the correct class label
        classId += 1

XT = np.concatenate(faceData, axis=0)
yT = np.array(labels)

def dist(p, q):
    return np.sqrt(np.sum((p - q) ** 2))


def knn(X, y, xt, k=5):
    m = X.shape[0]
    dlist = []

    # Reshape xt to be compatible with X
    xt_flattened = xt.flatten()

    for i in range(m):
        d = dist(X[i], xt_flattened)
        dlist.append((d, y[i]))

    dlist = sorted(dlist)
    dlist = np.array(dlist[:k])
    labels = dlist[:, 1]

    labels, cnts = np.unique(labels, return_counts=True)
    idx = cnts.argmax()
    pred = labels[idx]

    return int(pred)

# Predictions
cam = cv2.VideoCapture(0)
model = cv2.CascadeClassifier("haarcascade_frontalface.xml")
while True:
    success, img = cam.read()
    if not success:
        print("Reading Camera Failed!!")

    faces = model.detectMultiScale(img, 1.3, 5)

    for f in faces:
        x, y, w, h = f
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_face = img[y - offset:y + h + offset, x - offset:x + w + offset]
        cropped_face = cv2.resize(cropped_face, (100, 100))

        classPredicted = knn(XT, yT, cropped_face, 5)
        namePredicted = nameMap[classPredicted]
        cv2.putText(img, namePredicted, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Prediction Window", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()




