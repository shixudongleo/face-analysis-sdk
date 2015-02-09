Python warpper for face-analysis-sdk project:
--------------------------

###Resourcees:

http://github.com/ci2cv/face-analysis-sdk

https://github.com/yati-sagade/opencv-ndarray-conversion

###TODO:
1. landmark generator python warpper. 

###How to build:
```bash
$ mkdir build && cd build
$ cmake ..
$ make
```

###How to use face landmark detector:
```python
import cv2
from face_fit import FaceLandmarkDetector

img = cv2.imread('face_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ld = FaceLandmarkDetector()

landmarks = ld.get_face_landmarks(gray)
print landmarks
```
