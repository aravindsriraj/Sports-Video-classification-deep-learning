import cv2 as cv
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from collections import deque

from tensorflow.python.ops.gen_array_ops import Where

model= load_model(r"C:\Users\MYPC\Desktop\computer vision\video classification\model.h5")
lb = pickle.loads(open(r"C:\Users\MYPC\Desktop\computer vision\video classification\labelpickle.pickle","rb").read())
mean = np.array([123.68,116.779,103.939][::1],dtype="float32")
Queue = deque(maxlen=128)
output_video = r"C:\Users\MYPC\Desktop\computer vision\demo_output.avi"

capture_video = cv.VideoCapture(r"C:\Users\MYPC\Desktop\computer vision\video classification\table tennis.mp4")
writer = None
(Width,Height) = None,None

while True:
    (taken,frame) = capture_video.read()
    if not taken:
        break
    if Width is None or Height is None:
        (Width,Height) = frame.shape[:2]

    output = frame.copy()
    frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    frame = cv.resize(frame,(224,224)).astype("float32")
    frame-=mean
    preds = model.predict(np.expand_dims(frame,axis=0))[0]
    Queue.append(preds)
    results = np.array(Queue).mean(axis=0)
    i = np.argmax(results)
    label = lb.classes_[i]
    text = "They are playing: {}".format(label)
    cv.putText(output,text,(45,60),cv.FONT_HERSHEY_SIMPLEX,1.25,(255,0,0),5)

    if writer is None:
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        writer = cv.VideoWriter(output_video,fourcc,30,(Width,Height),True)
    writer.write(output)
    cv.imshow("In progress",output)
    key = cv.waitKey(1) &  0xFF

    if key==ord("q"):
        break
    # print(Queue)

print("Finalizing....")
writer.release()
capture_video.release()