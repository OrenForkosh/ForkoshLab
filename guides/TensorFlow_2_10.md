To add GPU support for TensorFlow 2.10, follow these detailed steps. TensorFlow 2.10 supports Ubuntu 20.04 and Ubuntu 22.04. Here's a step-by-step guide for a complete setup on Ubuntu.


---

**1. Recommended Ubuntu Version**  
- TensorFlow 2.10 is compatible with **Ubuntu 20.04**  and **Ubuntu 22.04** .
 
- **Ubuntu 20.04**  is the most commonly used and well-tested version for TensorFlow GPU setups.


---

**2. Check Hardware and Software Requirements**  
- **GPU** : TensorFlow 2.10 supports NVIDIA GPUs with CUDA Compute Capability 3.5 or higher. Check your GPU compatibility on NVIDIA's [CUDA GPUs list](https://developer.nvidia.com/cuda-gpus) .
 
- **Python** : Python 3.7–3.10.
 
- **CUDA Toolkit** : Version 11.2.
 
- **cuDNN Library** : Version 8.1.


---

**3. Install NVIDIA Drivers** 
Ensure your system has the latest NVIDIA drivers installed:

#### Check Existing Drivers 


```bash
nvidia-smi
```

If this command doesn't work, or the drivers are not installed, follow these steps:

#### Add NVIDIA's Driver Repository 


```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
```

#### Install the Latest Stable Driver 


```bash
sudo apt install -y nvidia-driver-525
```
Replace `525` with the recommended version for your GPU if different. Restart your machine after installation.

---

**4. Install CUDA Toolkit 11.2** 
#### Download CUDA 
 
1. Visit the [NVIDIA CUDA Toolkit 11.2 download page](https://developer.nvidia.com/cuda-11.2.0-download-archive) .
 
2. Select **Linux > x86_64 > Ubuntu > 20.04 (or 22.04)**  > `runfile (local)`.

#### Install CUDA Toolkit 


```bash
sudo sh cuda_11.2.2_460.32.03_linux.run
```

During the installation:
 
- Say **yes**  to the EULA.
 
- Say **no**  to installing the driver (as you've already installed it).

- Follow the prompts to complete the installation.

#### Set Environment Variables 
Add the following lines to your `~/.bashrc`:

```bash
export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Then reload the configuration:


```bash
source ~/.bashrc
```


---

**5. Install cuDNN 8.1** 
#### Download cuDNN 
 
1. Go to the [cuDNN download page](https://developer.nvidia.com/cudnn)  and log in with an NVIDIA Developer account.
 
2. Download cuDNN 8.1 for CUDA 11.2 as a `.tgz` file.

#### Install cuDNN 


```bash
tar -xzvf cudnn-linux-x86_64-8.x.x.x_cuda11-archive.tgz
sudo cp cuda/include/* /usr/local/cuda-11.2/include/
sudo cp cuda/lib64/* /usr/local/cuda-11.2/lib64/
```


---

**6. Verify CUDA and cuDNN** 
#### Check CUDA Version 


```bash
nvcc --version
```

#### Check cuDNN Installation 


```bash
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```


---

**7. Install TensorFlow with GPU Support** 
#### Set Up a Virtual Environment (Optional but Recommended) 


```bash
python3 -m venv tf_gpu_env
source tf_gpu_env/bin/activate
```

#### Install TensorFlow 


```bash
pip install tensorflow==2.10
```


---

**8. Test TensorFlow GPU Installation** 
Run the following Python script:


```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

If TensorFlow detects your GPU, the output should indicate at least one GPU is available.


---

**9. Troubleshooting**  
- If `nvidia-smi` shows errors, ensure the driver version is compatible with your GPU.
 
- If TensorFlow cannot detect the GPU:
  - Verify that the CUDA paths are correctly set.
 
  - Ensure TensorFlow and CUDA/cuDNN versions are compatible using the [TensorFlow compatibility guide]() .