'''
Prepare the IMDB training data as tokenized pytorch
id tensor and attention mask
'''
import torch
import scandir
from transformers import ElectraTokenizer

def get_reviews(dir):
    review_files = [f.name for f in scandir.scandir(dir)]
    review_list = []
    for review_file in review_files:
        with open(dir+'/'+review_file, "r") as f:
            text = f.read()
            text = text.rstrip('\n')
        review_list.append(text)
    return review_list

def get_data(base_dir):
    neg = base_dir + '/neg'
    pos = base_dir + '/pos'

    neg_review_list = get_reviews(neg)
    pos_review_list = get_reviews(pos)
    review_list = neg_review_list + pos_review_list

    # Target labels
    labels = [0]*len(neg_review_list) + [1]*len(pos_review_list)
    labels = torch.LongTensor(labels)

    # Tokenize and prep input tensors
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
    encoded_inputs = tokenizer(review_list, padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs('input_ids')
    mask = encoded_inputs('attention_mask')

    return ids, mask, labels

def get_train():
    base_dir = '../data/train'
    return get_data(base_dir)

def get_test():
    base_dir = '../data/test'
    return get_data(base_dir)
