# 🎭 SCAiLX Face Recognition Demo

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-SCAiLX-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Real-time **person and face detection** with **facial expression recognition** on the SCAiLX platform.

---


## 🚀 Quick Start
- Clone the repo first
```bash
# Clone with Git 
git clone https://github.com/VideologyInc/face-recognition-demo.git
```

- Check the depedencies
    <details>
    <summary>Dependencies</summary>
  
    - python  
    - tflite_runtime  
    - hailoplatform  
    - gstreamer  
    - opencv2  
    - numpy
    </details>


- Run demo
```bash
cd face-recognition-demo
python3 demo.py
```
---

## GStreamer Examples for Face Detection on Hailo Model
- Face Detection Model : yolov5s_personface.hef
```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! \
video/x-raw,format=YUY2,framerate=30/1,width=640,height=640 ! \
videoconvert  ! \
hailonet hef-path=models/yolov5s_personface.hef ! \
queue max-size-buffers=5 leaky=2 ! \
hailofilter so-path=/usr/lib/hailo-post-processes/libyolo_post.so \
function-name=yolov5_personface \
config-path=/opt/hailo/apps/multistream_detection/resources/configs/yolov5_personface.json qos=false ! \
queue max-size-buffers=5 leaky=2 ! \
hailooverlay qos=false ! videoconvert ! \
autovideosink
```
---
