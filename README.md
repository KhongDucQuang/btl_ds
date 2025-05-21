## Setting Up the Development Environment

### Prerequisites

* Python 3.10+
* Git
* CUDA-capable GPU (for training)

### Installation Steps

1. **Clone the repository:**

```bash
git clone https://github.com/nontaphatfirm/Classroom-Behavior-Detector.git
cd Classroom-Behavior-Detector
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes all necessary packages including:

* `ultralytics` – YOLO framework
* `torch` / `torchvision` – PyTorch deep learning framework
* `gradio` – Web interface library
* `opencv-python` – Image processing
* `matplotlib` – Visualization tools
* `wandb` – Weights & Biases for experiment tracking
