# Objective

Finetune existing encoders to perform sentiment classification on the IMDB dataset

# Requirements

python3.4 or above

## Installation with PyPI

pip install torch, torchvision

pip install transformers

pip install scandir

pip install sentencepiece

# Experimental Results

| Model Architecture | Test Accuracy (%) |
| ----------------- | :-----------------: |
ELECTRA (base) encoder + classification head | 95.6 |
BERT (base-uncased) encoder + classification head | 93.8 |
RoBERTta (base) encoder + classification head | 95.4 |
XLNet (base) encoder + classification head | - |

### Training Details

- Initialise encoder with _<model>_
- Batch Size = 8
- Epochs = 2
- Learning Rate = 1e-5




