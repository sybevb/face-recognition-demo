# 🎭 SCAiLX Face Recognition Demo

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-SCAiLX-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Real-time **person and face detection** with **facial expression recognition** on the SCAiLX platform.

## 🚀 Quick Start
- Clone the repo first
```bash
# Clone with Git LFS
curl -L https://github.com/git-lfs/git-lfs/releases/download/v3.6.1/git-lfs-linux-arm64-v3.6.1.tar.gz | tar -xz
cp ./git-lfs*/git-lfs /usr/bin
git lfs install
git lfs clone https://github.com/VideologyInc/face-recognition-demo.git
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

# 📖 Reference
1. https://videology-inc.atlassian.net/wiki/spaces/SUD/pages/63275010/Running+AI+models+in+Python+with+Tensorflow-lite