# SSK2ST: A Semi-Supervised Kernel Two-Sample Test 
Python codes implementing the xssMMD test proposed in A Semi-Supervised Kernel Two-Sample Test by Gyumin Lee, Shubhanshu Shekhar, and Ilmun Kim: AISTATS 2026

## 1. Installation

Create a conda environment and install the required packages:

```sh
conda create --name ssk2st --yes python=3.8
conda activate ssk2st

# Install required Python packages
pip install -r requirements.txt
```
The requirements.txt should include: torch, numpy, scipy, scikit-learn, matplotlib, seaborn, tqdm, joblib, pandas, ucimlrepo, etc.

<br>

## 2. Numerical Analysis (Simulation)

### 2-1) Null Distribution Simulation

* File: ssk2st/null.py
* Purpose: Simulate the null distribution of xssMMD statistics under various scenarios.

Example command:

```sh
python null.py
```
Output: Null distribution histograms are saved in the figure/ directory.

<br>

### 2-2) Power Analysis

* File: power.py
* Purpose: Simulate the statistical power of various kernel-based two-sample tests as a function of sample size ratio, dimension, and perturbation.

Example command:

```sh
python ssk2st/power.py
```
Output: Power curves are saved in the figure/ directory.

<br>

## 3. Real Data Experiments
###3-1) HTRU2 Pulsar Dataset Experiment

* File: pulsar_test.py
* Purpose: Evaluate the performance of xssMMD tests on the HTRU2 pulsar/non-pulsar dataset.
* Description: This experiment analyzes the ability of the SS-xMMD tests to distinguish between pulsar and non-pulsar signals using the HTRU2 dataset. The dataset contains various features extracted from the signals, and the experiment assesses the statistical significance of the differences between the two classes.

Example command:

```sh
python ssk2st/pulsar_test.py
```

<br>

###3-2) CUB-200-2011 Dataset Experiment

* File: cub_test.py
* Purpose: Evaluate the performance of xssMMD tests on the CUB-200-2011 bird image/text dataset.
* Description: This experiment generates image and text embeddings to assess the ability of xssMMD tests to distinguish between different bird groups based on their visual and textual features. The goal is to evaluate the statistical significance of the differences between the embeddings of various bird groups.

Example command:

```sh
python ssk2st/cub_test.py
```

Output: Experimental statistics and results are saved as .npy files in the results/ directory.

<br>

###3-3) MNIST Dataset Experiment

* File: mnist_test.py
* Purpose: Evaluate the performance of xssMMD tests on the MNIST image dataset.
* Description: This experiment uses image data of the numbers to assess the ability of xssMMD tests to distinguish between different groups of digits when Gaussian noise is added. The goal is to evaluate the statistical significance of the differences between the embeddings of various number groups.

Example command:

```sh
python ssk2st/mnist_test.py
```

Output: Experimental statistics and results are saved as .npy files in the results/ directory.

<br>

## 4. Notes for Reproducibility
* You can modify key parameters (sample size, dimension, number of repetitions, etc.) at the top of each script.
* All results (figures, .npy, .pkl files, etc.) are automatically saved in subdirectories.
* The CUB-200-2011 datasets must be downloaded separately using the link provided below, and the corresponding paths in your local should be set in the scripts.

<br>

## 5. Acknowledgement
We appreciate the following repositories and datasets for their valuable code base and data:
* https://github.com/sshekhar17/PermFreeMMD
* https://github.com/ilmunk/ss-ustat
* CUB-200-2011 Dataset (https://www.vision.caltech.edu/datasets/cub_200_2011/)
* HTRU2 Dataset (https://archive.ics.uci.edu/dataset/372/htru2)

