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

    -   Create `models` folder.

    -   Edit `configs/lc_config.json`.

    -   Run `train.py` :
        ```bash
        python train.py train-lc
        ```

-   For training neural network model :

    -   Create `models` and `w2v_models` folders.

    -   Download [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings and extract it into `w2v_models` folder.

    -   Edit `configs/nn_config.json`.
    
    -   Run `train.py` :
        ```bash
        python train.py train-nn
        ```

-   For inference on linear classifier model :
    ```bash
    python evaluate.py eval-lc --author1 /path/to/author1/text --author2 /path/to/author2/text --model /path/to/model/file
    ```

-   For inference on neural network model :
    ```bash
    python evaluate.py eval-nn --author1 /path/to/author1/text --author2 /path/to/author2/text --model /path/to/model/file --w2v_path /path/to/w2v/model
    ```
