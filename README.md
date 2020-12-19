# Stylometry Using Machine Learning

A machine learning project for classifying authors based on their writings.

## Dataset

The dataset contains the writings of three authors :
-   Edgar Allan Poe (EAP) : 7900 phrases.
-   HP Lovecraft (HPL) : 5635 phrases.
-   Mary Wollstonecraft Shelley (MWS) : 6044 phrases.

## Work Pipeline

-   Text Processing :
    -   Tokenization (converting into words or tokens).
    -   Stemming / Lemmatization (normalizing tokens).

-   Feature Extraction : 
    -   Classical : bag of words / n-grams / TF-IDF.
    -   Deep Learning : word embeddings.

-   Classification (better approach to train a binary classifier for each author to improve generalization) :
    -   Classical : Linear Regression / Naive Bayes _(BEST)_ / SVM / XGBoost.
    -   Deep Learning : RNNs.

## Installation

-   Install `requirements.txt` using PyPi:
    ```bash
    pip3 install -r requirements.txt
    ```

## Usage

-   For training linear classifier model :

    -   Edit `configs/lc_config.json`.

    -   Run `train.py` :
        ```bash
        python train.py train-lc
        ```

-   For training neural network model :

    -   Edit `configs/nn_config.json`.
    
    -   Run `train.py` :
        ```bash
        python train.py train-nn
        ```
