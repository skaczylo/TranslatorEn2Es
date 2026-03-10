
import tiktoken
from  Transformer import CONTEXT_LENGTH
import torch
import numpy as np
from torch.utils.data import Dataset

class Tokenizer():
    """
    TOKENIZADOR
    Clase que permitirá dividir o tokenizar un string en tokens. 
    Se usará el algoritmo ya implementado de la liberia tiktoken y el vocabulario de GPT2
    """

    def __init__(self):
        self.gpt2 = tiktoken.get_encoding("gpt2")  #No confundir con el encoder del Transformer

        self.encoder = tiktoken.Encoding(
            name = "encoder",
            pat_str = self.gpt2._pat_str,
            mergeable_ranks = self.gpt2._mergeable_ranks,
            special_tokens = {
                **self.gpt2._special_tokens,
                "<START>": len(self.gpt2._mergeable_ranks) +1,
                "<END>": len(self.gpt2._mergeable_ranks) +2,
                "<PAD>": len(self.gpt2._mergeable_ranks) +3
            },
        )


    def __len__(self):
        return len(self.encoder._mergeable_ranks) + len(self.encoder._special_tokens)
    

    def pad_id(self):
        return self.encoder._special_tokens["<PAD>"]
    
    def end_of_text_id(self):
        return self.encoder._special_tokens["<END>"]
    
    def star_of_text_id(self):
        return self.encoder._special_tokens["<START>"]
    

    def add_pad_token(self,tokens):
        pad = [self.pad_id() for _ in range(CONTEXT_LENGTH-len(tokens))]
        return np.concatenate((tokens,pad))
    
    def encode(self,input: str):

        tokens  =self.encoder.encode("<START>"+input + "<END>",allowed_special="all") 

        if len(tokens) > CONTEXT_LENGTH:
            tokens =  tokens[:(CONTEXT_LENGTH-1)] + [self.tokenizer.end_of_text_id()]

        tokens = self.add_pad_token(tokens)
        
        return tokens

    def empty_predict(self):
        tokens = self.encoder.encode("<START>",allowed_special="all")
        tokens = self.add_pad_token(tokens)

        return tokens
    def decoder(self,input):


        return self.encoder.decode(input.cpu().numpy())
    


# Dataset
class Dataset_(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.X = self.data['en']
        self.y = self.data['es']
        self.tokenizer = tokenizer
        

    def __len__(self):
        return len(self.data)
    
    
    

    def __getitem__(self, idx):


        en_text = self.X.iloc[idx]
        es_text = self.y.iloc[idx]

    
        #COnvertimos a tensores
        es_tokens =  torch.tensor(self.tokenizer.encode(es_text),dtype=torch.long)
        en_tokens = torch.tensor(self.tokenizer.encode(en_text),dtype= torch.long)

        return en_tokens, es_tokens