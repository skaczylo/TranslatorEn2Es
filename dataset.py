import tiktoken
from  transformer import CONTEXT_LENGTH
import torch
import numpy as np
from torch.utils.data import Dataset
from tokenizers import Tokenizer as HFTokenizer

class Tokenizer():
    """
    TOKENIZADOR
    Clase que permitirá dividir o tokenizar un string en tokens.
    Se usará el algoritmo ya implementado de la liberia tiktoken y el vocabulario de GPT2
    """

    def __init__(self,path):
      # Cargamos el vocabulario que creamos en el paso anterior
        self.encoder = HFTokenizer.from_file(path)
        
        # Extraemos y guardamos los IDs de los tokens especiales para un acceso rápido
        self._pad_id = self.encoder.token_to_id("<PAD>")
        self._start_id = self.encoder.token_to_id("<START>")
        self._end_id = self.encoder.token_to_id("<END>")


    def __len__(self):
        return self.encoder.get_vocab_size()


    def pad_id(self):
        return self._pad_id

    def end_of_text_id(self):
        return self._end_id

    def star_of_text_id(self): # (Mantengo tu nombre de función 'star' por compatibilidad)
        return self._start_id


    def add_pad_token(self,tokens):
        pad = [self.pad_id() for _ in range(CONTEXT_LENGTH-len(tokens))]
        return np.concatenate((tokens,pad))

    def encode(self, input: str, pad=True):
       
        raw_tokens = self.encoder.encode(input).ids

        
        tokens = [self.star_of_text_id()] + raw_tokens + [self.end_of_text_id()]

       
        if len(tokens) > CONTEXT_LENGTH:
            tokens = tokens[:(CONTEXT_LENGTH - 1)] + [self.end_of_text_id()]
        
        # 4. Rellenamos con PAD si es necesario
        if pad:
            tokens = self.add_pad_token(tokens)

        return tokens

    def encode_batch(self, input, pad=True):
        batch_tokens = []
    
        for text in input:
            tokens = self.encode(text, pad=pad)
            batch_tokens.append(tokens)
        
        # Agrupamos todo de forma súper rápida en C con NumPy primero
        np_batch = np.array(batch_tokens)
        
        return torch.tensor(np_batch, dtype=torch.long)
        
    def empty_predict(self):
        # Genera un vector que solo tiene el START y el resto es PAD
        tokens = [self.star_of_text_id()]
        tokens = self.add_pad_token(tokens)
        return tokens

    def decode(self, input, skip_special_tokens=True):
        """
        Convierte una lista o tensor de IDs de vuelta a texto.
        skip_special_tokens=True oculta los <PAD>, <START> y <END> al imprimir.
        """
        # Si le pasas un tensor de PyTorch, lo convertimos a lista de Python
        if isinstance(input, torch.Tensor) or isinstance(input, np.ndarray):
            ids = input.tolist()
        else:
            ids = input
            
        # Decodificamos
        return self.encoder.decode(ids, skip_special_tokens=skip_special_tokens)



# Dataset
class Dataset(Dataset):
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


        return en_text, es_text
