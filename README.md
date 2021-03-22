# Objective

Use the Electra Language Model to perform sentiment classification on the IMDB dataset

# Requirements

python3.4 or above

## Installation with PyPI

pip install torch, torchvision

pip install transformers

pip install scandir

# Experimental Results

95.7% accuracy on Test dataset (25,000 data points)

## Training Details

- Initialise Electra encoder with _Electra Base_
- Batch Size = 8
- Epochs = 2
- Learning Rate = 1e-5




