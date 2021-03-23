# Objective

Finetune existing encoders to perform sentiment classification on the IMDB dataset

# Requirements

python3.4 or above

## Installation with PyPI

pip install torch, torchvision

pip install transformers

pip install scandir

# Experimental Results

| Model Architecture | Test Accuracy (%) |
| ----------------- | :-----------------: |
Electra (base) encoder + classification head | 95.7 |
Bert (base-uncased) encoder + classification head | - |
Roberta (base) encoder + classification head | - |

### Training Details

- Initialise encoder with _<model>_
- Batch Size = 8
- Epochs = 2
- Learning Rate = 1e-5




