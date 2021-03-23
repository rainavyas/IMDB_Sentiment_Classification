# Objective

Finetune existing encoders to perform sentiment classification on the IMDB dataset

# Requirements

python3.4 or above

## Necessary Packages (installation with PyPI)

pip install torch, torchvision

pip install transformers

pip install scandir

pip install sentencepiece

# Training

Fork the repository (and clone).

Run the _train.py_ scripts with desired arguments in your terminal. For example, to train an XLNet-based classifier:

_python ./train.py xlnet_trained.th xlnet --B=8 --lr=0.00001 --epochs=2_

# Experimental Results

| Model Architecture | Test Accuracy (%) |
| ----------------- | :-----------------: |
ELECTRA (base) encoder + classification head | 95.6 |
BERT (base-uncased) encoder + classification head | 93.8 |
RoBERTta (base) encoder + classification head | 95.4 |
XLNet (base) encoder + classification head | - | 95.9

Note that the XLNet tokenizer was adapted to force truncation to 512 tokens (to be comparable to other models)

### Training Details

- Initialise encoder with _model_
- Batch Size = 8
- Epochs = 2
- Learning Rate = 1e-5




