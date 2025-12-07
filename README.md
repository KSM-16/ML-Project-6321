# Exploring the Role of Optimizers and Architectures in Active Learning

This repository implements an **active learning framework for image classification** using ResNet models on CIFAR datasets with different optimizers. The code includes functionality for uncertainty-based sampling, training, testing, and plotting performance metrics. This project has been completed as part of the course project requirement of **COMP 6321 - Machine Learning** for **Fall 2025** at Concordia University, Montreal, QC, Canada.

---
c
## Table of Contents

- [Folder Structure](#folder-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Collaborators](#collaborators)

---

## Folder Structure

```text
ML-Project-6321/
│
├── dataset/ # CIFAR-10 and CIFAR-100 dataset handling
│ ├── cifar.py
│ └── sampler.py # Subset samplers for active learning
│
├── model/ # Neural network models
│ ├── resnet.py # ResNet architectures
│ └── lossnet.py # Loss prediction module
│
├── plots/ # Generated performance plots
│
├── results/ # JSON files storing accuracies and labeled sizes
│
├── weights/ # Saved model weights
│
├── main.py # Main script to run active learning experiments
├── train.py # Training functions
├── test.py # Testing functions
├── utils.py # Utility functions for saving results, plotting, etc.
├── uncertainty_sampling.py # Functions to compute uncertainty for active learning
└── requirements.txt # Python dependencies
```


---

## Requirements

The project is compatible with **Python 3.8 or above**. Required Python packages are listed in `requirements.txt`.

**Key dependencies:**

- torch
- torchvision
- numpy
- pandas
- matplotlib
- tqdm
- scikit-learn

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/KSM-16/ML-Project-6321.git
cd ML-Project-6321

python3 -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage
```bash
python main.py --dataset cifar10 --save_path results/ --num_trial 5 --cycles 10 --num_epoch 200 --batch_size 128
```

---

## License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.

---

## Collaborators

This project was developed by students of Concordia University, Montreal, QC, Canada:

- [**Khadiza Sarwar Moury**](mailto:khadizasarwar.moury@mail.concordia.ca)
- [**Md Hasibul Hasan Shovo**](mailto:mdhasibulhasan.shovo@mail.concordia.ca)
- [**Sayed Abdullah Ali**](mailto:syedabdullah.ali@mail.concordia.ca)
- [**Youssef Midra**](mailto:youssef.midra@mail.concordia.ca)





