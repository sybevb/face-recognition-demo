# SCAiLX Face Recognition  Demo

- person and face detection with 

## Dependencies
- python
- tflite_runtime
- hailoplatform
- gstreamer
- opencv2
- numpy

# Steps
1. git clone the repo on the SCAILX \
`curl -L https://github.com/git-lfs/git-lfs/releases/download/v3.6.1/git-lfs-linux-arm64-v3.6.1.tar.gz | tar -xz` \
`cp ./git-lfs*/git-lfs /usr/bin ` \
`git lfs install` \
`git lfs clone https://github.com/VideologyInc/face-recognition-demo.git ` \
2. check the dependencies available on the device
3. run the python script \
`python3 demo.py`

# Reference
1. https://videology-inc.atlassian.net/wiki/spaces/SUD/pages/63275010/Running+AI+models+in+Python+with+Tensorflow-lite