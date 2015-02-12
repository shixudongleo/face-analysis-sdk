Python warpper for face-analysis-sdk project:
--------------------------

###Resourcees:

http://github.com/ci2cv/face-analysis-sdk

https://github.com/yati-sagade/opencv-ndarray-conversion

###TODO:
1. landmark generator python warpper. 

###Dependencies(dev packages):
- Boost::Python
- Numpy
- OpenCV


###How to build:
```bash
$ mkdir build && cd build
$ cmake ..
$ make
```
After make, the desired python face landmark module is under lib folder(face_fit.so). 

###How to use face landmark detector:
```python
import cv2
from face_fit import FaceLandmarkDetector

img = cv2.imread('face_image.jpg')
ld = FaceLandmarkDetector()

landmarks = ld.get_face_landmarks_image_mode(img)
# if your are applying to video use 
# landmarks = ld.get_face_landmarks_video_mode(img)

print landmarks
```
