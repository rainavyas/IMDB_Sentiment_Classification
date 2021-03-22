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
BERT (base) encoder + classification head | - |

### Electra Training Details

- Initialise Electra encoder with _Electra Base_
- Batch Size = 8
- Epochs = 2
- Learning Rate = 1e-5




