import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from datasets import load_dataset 
from tokenizers import Tokenizer
from tokenizers.models import WordLevel 
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


from pathlib import Path # ? Library which allows us to get the absolute path from the relative path.

# Padding Tokens, SOS, EOS, UNK
def get_all_sentences(ds, lang):
    ''' 
    Get all the sentences from the dataset for a particular language.
    ds -> Dataset 
    lang -> Language code for which we wish to extract the sentences. 
    '''
    for item in ds: # Each item is a pair 
        # ! Yield is used to define generators in Python. Instead of returning a value and terminating, it generates an object. 
        # ! When the function is called, the object is returned without executing the function. It is only executed when next() is called on the object. 
        # ! This is very good for memory efficiency and lazy evaluation as we saw on Big Data.
        # ! Compute values one at a time, and can be paused and resumed. -> Very useful for memory management.
        # ! You can start processing batches immediately, without having to wait for the entire dataset to be processed.
        yield item['translation'][lang] # ? Yield the translation for the language

# * 1. Build Tokenizer 
def get_or_build_tokenizer(config, ds, lang): 
    ''' 
    Get or build an existing tokenizer or build a new one if it doesn't exist. 
    '''
    # ? Construct the path to the tokenizer file using config and lang
    tokenizer_path = Path(config['tokenizer_path'].format(lang))
    # ? Check if the tokenizer file exists. If it doesn't, build a new tokenizer.
    if not Path.exists(tokenizer_path):
        # Build Tokenizer 
        tokenizer = Tokenizer(WordLevel(unk_token='<unk>')) # ? We create a new WordLevel tokenizer with unk as the unknown token
        tokenizer.pre_tokenizer = Whitespace() # ? We set the pre-tokenizer to split the sentence by whitespace
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],min_frequency=2) # ? Only consider words which appear twice in the training data. 
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer = trainer) # ? Tokeniser initialised, begin the training process
        tokenizer.save(str(tokenizer_path)) # ? Save the tokenizer to the path
    else: 
        tokenizer = Tokenizer.from_file(str(tokenizer_path)) # ? Load the tokenizer from the path
    return tokenizer

def get_ds(config): 
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split = 'train') # ? Load the dataset

    # * 1. Build the tokenizers 
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src']) # ? Get or build the tokenizer for the source & target lang. 
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # * 2. Split the dataset into 90:10 train:validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0 
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max Length of the source sentences: {max_len_src}')
    print(f'Max length of the target sentences: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])

