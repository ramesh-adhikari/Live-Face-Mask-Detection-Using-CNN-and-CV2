import cv2
import numpy as np
from keras.models import load_model
# model=load_model("./model-010.h5")
model=load_model("./model2-008.model")

results={0:'face without mask',1:'face with mask'}
GR_dict={0:(0,0,255),1:(0,255,0)}

rect_size = 4

# define a video capture object
cap = cv2.VideoCapture(0) 



# Haar Cascade Classifiers : We will implement our use case using the Haar Cascade classifier.
 # Haar Cascade classifier is an effective object detection approach which was proposed by Paul Viola and Michael Jones in their paper,
# “Rapid Object Detection using a Boosted Cascade of Simple Features” in 2001.
haarcascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

while True:
    # Capture the video frame
    # by frame
    (rval, im) = cap.read()
    im=cv2.flip(im,1,1) 

    
    rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    # FACE DETECTION USING HAAR CASCADE CLASSFIERS
    faces = haarcascade.detectMultiScale(rerect_size)
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f] 
        
        face_img = im[y:y+h, x:x+w]
        rerect_sized=cv2.resize(face_img,(150,150))
        normalized=rerect_sized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)

        
        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(im,(x,y),(x+w,y+h),GR_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),GR_dict[label],-1)
        cv2.putText(im, results[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    # Display the resulting frame
    cv2.imshow('LIVE',   im)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()