'''
Prepare the IMDB training data as tokenized pytorch
ids tensor and attention mask
'''
import torch
import torch.nn as nn
import scandir
from transformers import ElectraTokenizer, BertTokenizer, RobertaTokenizer, XLNetTokenizer

_DESCRIPTION = """\
Large Movie Review Dataset.
This is a dataset for binary sentiment classification containing substantially \
more data than previous benchmark datasets. We provide a set of 25,000 highly \
polar movie reviews for training, and 25,000 for testing. There is additional \
unlabeled data for use as well.\
"""

_CITATION = """\
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
"""

_DOWNLOAD_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

def get_reviews(dir):
    review_files = [f.name for f in scandir.scandir(dir)]
    review_list = []
    for review_file in review_files:
        with open(dir+'/'+review_file, "r", encoding="utf8") as f:
            text = f.read()
            text = text.rstrip('\n')
        review_list.append(text)
    return review_list

def get_data(base_dir, arch):

    allowed_arch = ['electra', 'bert', 'roberta', 'xlnet']
    if arch not in allowed_arch:
        raise Exception('Invalid architecture, only allowed: electra, bert, roberta')
    neg = base_dir + '/neg'
    pos = base_dir + '/pos'

    neg_review_list = get_reviews(neg)
    pos_review_list = get_reviews(pos)
    review_list = neg_review_list + pos_review_list

    # Target labels
    labels = [0]*len(neg_review_list) + [1]*len(pos_review_list)
    labels = torch.LongTensor(labels)

    # Tokenize and prep input tensors
    if arch == 'electra':
        tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
    elif arch == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif arch == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    elif arch == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

    encoded_inputs = tokenizer(review_list, padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']
    
    if arch == 'xlnet':
        # No truncation in xlnet (no max size) so do it manually for cuda memory sake
        ids = ids[:,0:512,:]
        mask = mask[:,0:512,:]

    return ids, mask, labels

def get_train(arch):
    base_dir = '../data/train'
    return get_data(base_dir, arch)

def get_test(arch):
    base_dir = '../data/test'
    return get_data(base_dir, arch)
