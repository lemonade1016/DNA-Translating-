
# DNA-Translating: Translating Visual Phenotypes into Functional DNA Probes

[](https://www.python.org/)

This repository contains the official implementation of the paper **"Translating Visual Phenotypes into Functional DNA Probes via an End-to-End Deep Learning Framework."** We introduce a novel deep learning framework that directly translates images of organisms into species-specific, functional DNA sequences without requiring prior genomic information.

## 📖 Overview

The escalating threat of invasive alien species (IAS) demands rapid and scalable identification methods. Current approaches, however, are often limited by their reliance on taxonomic expertise or laboratory infrastructure. This project bridges that gap by creating a deep learning pipeline that learns to generate unique DNA "fingerprints" directly from an organism's visual appearance.

Our end-to-end framework consists of a **ResNet-50 feature extractor** to interpret visual phenotypes and a custom **DNA Encoder** to translate these features into nucleotide sequences. A key innovation is our **Differentiable DNA Hybridization Predictor**, a pre-trained CNN that acts as a surrogate for biophysical simulations. This predictor provides a supervisory signal during training, guiding the encoder to produce DNA sequences with high target specificity and low off-target hybridization.


## 🏗️ Framework Architecture

The project is divided into two main components:

1.  **DNA Hybridization Predictor**: A pre-trained CNN (`train_predictor.py`) that predicts the hybridization yield between two DNA sequences. It is trained on a large synthetic dataset and serves as the loss function for the main model.
2.  **End-to-End Image-to-DNA Framework**: The primary model (`train_end_to_end.py`) that links a ResNet-50 backbone to our DNA Encoder. It takes an image as input and is trained using the Hybridization Predictor to output a highly specific DNA sequence.

## 📂 Repository Structure

```
DNA-Translating/
├── data/                                         
├── train_predictor.py          
├── train_end_to_end.py         
├── requirements.txt            
└── README.md
```

## 🚀 Getting Started

### Prerequisites

  * Python 3.8+
  * PyTorch
  * PyTorch Lightning
  * NumPy, Pandas

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/lemonade1016/DNA-Translating.git
    cd DNA-Translating
    ```

2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Training

1.  **Train the DNA Hybridization Predictor**:
    First, train the predictor model using the provided script. Make sure your training, validation, and test data paths are correctly configured within the script.

    ```bash
    python train_predictor.py
    ```

    This will save the best model weights (`final_cnn2_59nt.pth`), which are required for the next step.

2.  **Train the End-to-End Framework**:
    Once the predictor is trained, you can train the main image-to-DNA model. Ensure the path to the trained predictor model and the image dataset are correctly set in the script.

    ```bash
    python train_end_to_end.py
    ```

    This script will train the full pipeline and save the final model weights for generating DNA probes from new images.


## 📊 Results

Our framework demonstrates exceptional performance both in the generation of specific DNA probes and in the accuracy of its underlying predictive model.

#### End-to-End Framework Performance

The primary model successfully generates highly specific 59-nt DNA probes. Computational analysis shows a near-perfect distinction between same-class and different-class binding, indicating high molecular orthogonality.

| Evaluation Metric | Mean | Standard Deviation |
| :--- | :---: | :---: |
| **Same-Class Hybridization (↑)** | **0.973** | 0.0977 |
| **Different-Class Hybridization (↓)** | **0.0070** | 0.0411 |

These computational results were further validated by *in vitro* experiments, where a generated probe showed a **3- to 6-fold stronger signal** against its target species compared to non-targets, confirming the real-world functionality of our approach.

#### DNA Hybridization Predictor Performance

The success of the end-to-end framework is underpinned by our highly accurate DNA Hybridization Predictor. When benchmarked against other models, our custom CNN architecture demonstrated superior performance in predicting hybridization outcomes.

| Metric | **Ours** | QDA | RF | NN | RNN | CNN | RoBERTa |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AUROC (↑)** | **0.973** | 0.922 | 0.938 | 0.930 | 0.968 | 0.956 | 0.949 |
| **MSE (↓)** | **0.0438** | N/A | N/A | N/A | 0.7666 | 1.0955 | 1.3363 |

This state-of-the-art predictive accuracy ensures that the main framework receives a high-fidelity supervisory signal, enabling it to learn the complex sequence-to-function relationship effectively.

