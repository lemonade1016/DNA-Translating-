Of course, here is a professional README for your GitHub project.

-----

# DNA-Translating: Translating Visual Phenotypes into Functional DNA Probes

[](https://www.python.org/)

This repository contains the official implementation of the paper **"Translating Visual Phenotypes into Functional DNA Probes via an End-to-End Deep Learning Framework."** We introduce a novel deep learning framework that directly translates images of organisms into species-specific, functional DNA sequences without requiring prior genomic information.

## 📖 Overview

The escalating threat of invasive alien species (IAS) demands rapid and scalable identification methods. Current approaches, however, are often limited by their reliance on taxonomic expertise or laboratory infrastructure. This project bridges that gap by creating a deep learning pipeline that learns to generate unique DNA "fingerprints" directly from an organism's visual appearance.

Our end-to-end framework consists of a **ResNet-50 feature extractor** to interpret visual phenotypes and a custom **DNA Encoder** to translate these features into nucleotide sequences. A key innovation is our **Differentiable DNA Hybridization Predictor**, a pre-trained CNN that acts as a surrogate for biophysical simulations. This predictor provides a supervisory signal during training, guiding the encoder to produce DNA sequences with high target specificity and low off-target hybridization.

\!

## ✨ Key Features

  * **End-to-End Translation**: Directly converts organismal images into 59-nucleotide DNA probes.
  * **Genomic-Data-Free**: Operates without needing any pre-existing DNA or RNA sequences.
  * **High Specificity**: Achieves a mean intra-class hybridization yield of **0.973** versus a mean inter-class yield of **0.0070** computationally.
  * **Robust Performance**: Demonstrates strong classification F1-scores ranging from **0.84 to 0.98** across six Diptera families.
  * ***In Vitro* Validation**: The framework's predictions were successfully validated in lab experiments, confirming the functionality of the generated probes.

## 🏗️ Framework Architecture

The project is divided into two main components:

1.  **DNA Hybridization Predictor**: A pre-trained CNN (`train_predictor.py`) that predicts the hybridization yield between two DNA sequences. It is trained on a large synthetic dataset and serves as the loss function for the main model.
2.  **End-to-End Image-to-DNA Framework**: The primary model (`train_end_to_end.py`) that links a ResNet-50 backbone to our DNA Encoder. It takes an image as input and is trained using the Hybridization Predictor to output a highly specific DNA sequence.

## 📂 Repository Structure

```
DNA-Translating/
├── data/                       # Placeholder for training data
├── models/                     # Placeholder for saved model weights
├── train_predictor.py          # Script to train the DNA Hybridization Predictor
├── train_end_to_end.py         # Script to train the main Image-to-DNA model
├── requirements.txt            # Project dependencies
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
    git clone https://github.com/your-username/DNA-Translating.git
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

Our framework successfully generates highly specific DNA probes. Computationally, these probes exhibit strong binding to their intended class and minimal binding to other classes.

| Metric | Value |
| :--- | :--- |
| Mean Intra-Class Hybridization | 0.973 |
| Mean Inter-Class Hybridization | 0.0070 |
| Classification F1-Score (Diptera) | 0.84 - 0.98 |

These computational results were further validated by *in vitro* experiments, where a generated probe showed a **3- to 6-fold stronger signal** against its target species compared to non-targets.

## 📄 Citation

If you use this work in your research, please consider citing our paper:

```bibtex
@article{your_article_name,
  title={Translating Visual Phenotypes into Functional DNA Probes via an End-to-End Deep Learning Framework},
  author={Your Name, et al.},
  journal={Journal Name},
  year={Year},
  pages={Pages}
}
```

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
