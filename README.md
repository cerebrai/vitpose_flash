# VITPose_flash

## Overview
`vitpose_flash` demonstrates techniques to improve the inference time of ViTPose. We have taken the original implementation from the [OnePose](https://github.com/developer0hye/onepose) repository and improved the inference time using the techniques discussed below:

### Key Techniques:
1. **16-bit precision**: Leverages half-precision floating point calculations to speed up inference while reducing memory usage.
2. **Flash attention**: Uses more efficient attention mechanisms to enhance the model's performance.
3. **TensorRT backend**: Compiles the model using NVIDIA's TensorRT for optimized inference on supported hardware.

The details can be found in the medium article: [From VITPose to VITPose-Flash](https://medium.com/@safwan.comsats/from-vitpose-to-vitpose-flash-099b6dfb14f6)

For detailed documentation of the original `OnePose` project, refer to the [OnePose repository](https://github.com/developer0hye/onepose).

---

## Installation

### Prerequisites
This repository uses [Poetry](https://python-poetry.org/) for dependency management. Poetry is a tool that simplifies Python project setup, dependencies, and packaging.

To install Poetry, follow the official [installation guide](https://python-poetry.org/docs/#installation).

### Using Poetry
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vitpose_flash.git
   cd vitpose_flash
   ```
2. Create a Python 3.10 environment using Conda:
   ```bash
   conda create -n vitpose_flash python=3.10 -y
   conda activate vitpose_flash
   ```
3. Install the dependencies using Poetry:
   ```bash
   poetry install --no-root
   ```

### Using Pip
For users who prefer `pip`, a `requirements.txt` file is provided. It can be generated using Poetry for pip compatibility:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vitpose_flash.git
   cd vitpose_flash
   ```
2. Create a Python 3.10 environment using Conda:
   ```bash
   conda create -n vitpose_flash python=3.10 -y
   conda activate vitpose_flash
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Run the becnhmark
```bash
  python benchmark.py
   ```

### Notes regarding TensorRT compile
Compiling using TensorRT can take some time and resources. To store the compiled checkpoint for later usage, specify a path when creating the model as follows:
```python
model = onepose.create_model('ViTPose+_large_coco_wholebody',
                                compile_flag=True,
                                compiled_checkpoint_path="<Path-to-store-the-compiled-checkpoint>"
                                compile_batch_size=len(inputs)).cuda()
```
If the compiled checkpoint already exists, it will be loaded from the path; otherwise, a compiled checkpoint will be stored at the specified path. 

---

## Acknowledgments
This project is built on top of [OnePose](https://github.com/developer0hye/onepose). We extend our gratitude to the authors of OnePose for their work and encourage users to consult the original repository for additional resources and documentation.
