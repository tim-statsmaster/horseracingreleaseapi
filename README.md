# Horse Racing Release Repository - v2.5


### What is this repository for? ###

* This repository has the following modules
	* 0 - Start/End/Camera Change Detection
	* 1 - Rail Mask Segmenation (Suspended)
	* 2 - Rail Pole Mask Segmenation (Suspended)
	* 3 - Semantic Semantation
	* 4 - Optical Flow
	* 5 - Cap, Jockey Body and Saddlecloth Detection
	* 6 - Tracking

## Requirement and Initial Setup

### **Hardware Requirement**
* At least 32 GB RAM 
* NVIDIA grahpics card with at least 11 GB memory

### **Initial Setup**

#### Step 1: Install Ananconda 3
* Install the latest version of Ananconda 3 from https://docs.anaconda.com/anaconda/install/index.html

#### Step 2: Create and activate an python 3.9 conda environment
```bash
conda create --name ["ENV_NAME"] python=3.9
conda activate ["ENV_NAME"]
```

#### Step 3: Install ``CUDA 11.3`` and ``pytorch 1.10.0``
```bash
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

#### Step 4: Install other dependencies
```bash
conda install ninja
python -m pip install -e detectron2-0.4
pip install tensorflow-gpu==2.10.1 tf_slim
pip install numpy==1.23 yacs==0.1.8 psutil pandas==1.5.2 opencv-python==4.7.0.68 pymongo==4.3.3 xlsxwriter

(Linux) pip install pycocotools==2.0.2 --no-cache-dir --no-binary :all:
(Windows) pip install pycocotools-windows

pip install memory-profiler
```

#### Step 5: Compile libraries for some modules
* compile the ``Optical Flow`` module
```bash
cd OpticalFlow 
rm -rf __pycache__ build *.egg-info dist *.so *.pyd 
python setup.py build_ext --inplace
cd ..
```

* compile the ``Detection`` modulde
```bash
cd Detection 
python setup.py build develop
cd ..
```

### Common Troubleshooting
#### Conda not found
* Check if anaconda3 paths are defined in environmental variables 
	* ``"C:\Users\[username]\anaconda3\Library\bin", "C:\Users\[username]\anaconda3\", "C:\Users\[username]\anaconda3\Scripts\"``

#### Visual Studio not found
* Check if visual studio paths are defined in environmental variables
	* ``"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.26.28801\bin\Hostx64\x64"``

#### CUDA/ NVIDIA driver 
* Try reboot/re-install the driver

#### Error: "Cannot open include file: 'corecrt.h': No such file or directory"
* Run Visual Studio Installer.
* Select Modify button.
* Go to "Individual Components" tab.
* Scroll down to "Compilers, build tools and runtimes".
* Tick "Windows Universal CRT SDK".
* Install.

## Run

* run the command with the necessary arguments for starting the release code
```command
python main.py --video_path [full path to video file] --model_dir [full path to models directory] --result [full path to result directory] --task [the task numbers separated by space] --distance [race distance] --track [race track] --jockeys [the list of jockeys that appear in the race] --xlsx
```
