import torch 
from typing import Any
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lagn = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id("[SOS]")]).long() # Vocab size can be larger than 32 bit. 
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id("[EOS]")]).long()
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id("[PAD]")]).long()

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # ! Remember that we need to pad the sequences to the same length.
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # -2 for SOS and EOS tokens
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 2 # -2 for SOS and EOS tokens

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise Exception("Sequence Length Exceeded. Please increase the sequence length.")
        
        # ? Add SOS and EOS to the source text. 
        encoder_input = torch.cat(
            [
                self.sos_token, # Start of Sentence
                torch.tensor(enc_input_tokens, dtype = torch.int64), # Source Text 
                self.eos_token, # End of Sentence
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64)
            ]
        )

        # ? Add SOS to the target text. for decoder input
        decoder_input = torch.cat(
            [
                self.sos_token, # Start of Sentence
                torch.tensor(dec_input_tokens, dtype = torch.int64), # Target Text - No eos as it's the decoder. One token less than the encoder.
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )

        # ? Add EOS to the label (What we expect as output from the decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64), 
                self.eos_token, # End of Sentence 
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, 
            "decoder_input": decoder_input, 
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label, 
            "src_text": src_text, 
            "tgt_text": tgt_text,
            }
    

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0 # This will return a mask with 1s above the diagonal and 0s below the diagonal.
