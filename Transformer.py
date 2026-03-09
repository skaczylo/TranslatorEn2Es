"""
ARQUITECTURA TRANSFORMER.
En este archivo se pretende construir el modelo o la arquitectura del Transformer
usando Encoder-Decoder representado en el paper "Attention is all you need"

"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.nn import functional as F


CONTEXT_LENGTH = 300
D_EMBEDDING = 512
ATTENTION_HEADS = 8
DROPOUT = 0.0
NUMBER_ENCODERS = 6
NUMBER_DECODERS= 6



class AttentionHead(nn.Module):
    """Una sola capa de atencion"""

    def __init__(self, dimension):
        super().__init__()
        self.key = nn.Linear(in_features=D_EMBEDDING,out_features=dimension, bias=False)
        self.query = nn.Linear(in_features=D_EMBEDDING,out_features=dimension, bias=False)
        self.value = nn.Linear(in_features=D_EMBEDDING,out_features=dimension, bias = False)
        

        #self.dropout = nn.Dropout(DROPOUT)

    def forward(self,key_input,query_input,value_input, padding_mask = None, masked_attention = False):
        """
        padding_mask : Matriz máscara para evitar que los token <PAD> afecten al cálculo de la atencion
        masked_attention : False indica que no se triangula la matriz de atencion; True se triangula la matriz. 
        Se usa en el decoder
        """

        #B = numero de Batch; T = numero de "tokens"; C = dimension de cada token
        B, N, D_i = value_input.shape
        self.register_buffer('tril', torch.tril(torch.ones(N, N)))

        K = self.key(key_input) # (B, N , D_i)
        Q = self.key(query_input) #(B, N , D_i)
        V = self.key(value_input) #(B, N , D_i)

        #Calculamos la Atención QxK^T
        #Calculamos la "afinidad" Q x K^t
        #Como tenemos batches, usamos el operador @ que aplica la multiplicacion en las dos ultimas dimensiones B veces
        #K.transpose significa transponer la penultima dimensino T con la pultima dimension C. 
    
        scores = Q @ K.transpose(-2,-1) #(B, N , D_i) x (B,D_i,N) = (B,N,N)
        scores = scores /(D_i ** 0.5)

        #Mascara para que no pongan atencion en los tokens <PAD>
        if padding_mask is not None:
            scores = scores + padding_mask  
        
        #Mascara para que los tokens y_n del decoder no se fijen en los posterioes y_n+1,y_n+2,....
        if masked_attention is True: 
            scores = scores.masked_fill(self.tril[:N, :N] == 0, float('-inf')) # (B, T, T)

        scores = F.softmax(scores, dim=-1) #Aplicamos softmax sobre las filas, es decir, sobre la ultima dimension 

        attention = scores @ V

        return attention
    

class MultiHeadAttention(nn.Module):
    """Multiples capas de atencion"""

    def __init__(self, num_heads):
        super().__init__()

        """
        Block attention es la concatenacion de las capas de atencion
        nn.ModuleList permite tener una lista de modulos y que Pytorch sea "consciente" de que existen; si haces una lista [] normal, no los reconocería a la hora de entrenar
        """
        self.block_attention = nn.ModuleList([AttentionHead(D_EMBEDDING // num_heads) for _ in range(num_heads)])
        self.projection = nn.Linear(in_features=D_EMBEDDING,
                                    out_features=D_EMBEDDING,
                                    bias=False)  #Matriz Omega_O 

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self,key_input,query_input,value_input,padding_mask = None, masked_attention = False):
       

        outputs = [head(key_input = key_input ,
                        query_input = query_input,
                        value_input = value_input, 
                        padding_mask = padding_mask, 
                        masked_attention = masked_attention) 
                   for head in self.block_attention] #Lista de matrices Nx(D_Embedding / num_heads)

        outputs = torch.cat(outputs, dim =-1) # Matriz N x D_embedding (por cada batch) => (B, N, D_embedding)
        multiple_attention = self.projection(outputs)

        return multiple_attention
    

class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=D_EMBEDDING, out_features= 4*D_EMBEDDING),
            nn.ReLU(),
            nn.Linear(4*D_EMBEDDING, D_EMBEDDING),
            nn.Dropout(DROPOUT)
        )

    def forward(self,x):
        return self.mlp(x)


#==================================================
#DECODER 
#==================================================

class Decoder:

    def __init__(self):

        self.masked_attention_heads = MultiHeadAttention(num_heads= ATTENTION_HEADS)
        self.attention_heads = MultiHeadAttention(num_heads=ATTENTION_HEADS)
        self.mlp = MLP()
        self.LayerNorm1 = nn.LayerNorm(D_EMBEDDING)
        self.LayerNorm2 = nn.LayerNorm(D_EMBEDDING)
        self.LayerNorm3 = nn.LayerNorm(D_EMBEDDING)

        
    def forward(self,x,encoder_ouput,padding_mask =None):


        #Primeras capas de atencion 
        masked_attention = self.attention_heads(key_input = x,
                                                query_input = x,
                                                value_input = x,
                                                padding_mask = padding_mask)
        
        z = self.LayerNorm1(masked_attention + z)


        #Cross Attention con Encoder
        cross_attention = self.attention_heads(key_input = encoder_ouput,
                                               query_input = z,
                                               value_input = encoder_ouput,
                                               padding_mask = padding_mask,
                                               masked_attention=False)
        
        y = self.LayerNorm2(cross_attention + z)

        #MLP
        output = self.LayerNorm3(self.mlp(y) + y)

        return output



#=============================================
#ENCODER 
#=============================================


class Encoder(nn.Module):
    """
    Arquitectura Encoder
    """

    def __init__(self):
        super().__init__()
    
        self.attention_heads = MultiHeadAttention(num_heads= ATTENTION_HEADS)
        self.mlp = MLP()
        self.LayerNorm1 = nn.LayerNorm(D_EMBEDDING)
        self.LayerNorm2 = nn.LayerNorm(D_EMBEDDING)

    def forward(self,x,padding_mask = None):
        
        # x = ( B, N, D_i)

        attention = self.attention_heads(key_input = x,
                                         query_input = x,
                                         value_input = x,
                                         padding_mask = padding_mask)
        
        z = self.LayerNorm1(attention + x)
        output = self.LayerNorm2(self.mlp(z) + z) # ( B, N, D_i)

        return output



class Transformer(nn.Module):
    """
    Arquitectura del Transformer de tipo Encoder-Decoder
    """

    def __init__(self, vocab_size,pad_id):
        super().__init__()

   
        #Input embedding + Positional Encoding
        self.input_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=D_EMBEDDING, padding_idx= pad_id)
        #El objetivo e determina posición de palabra => numero de embeddings  CONTEXT_LENGT
        self.input_positional_encoding = nn.Embedding(num_embeddings= CONTEXT_LENGTH, embedding_dim=D_EMBEDDING)  

        #Output embedding + Positional Encoding (entrada del decoder)
        self.output_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=D_EMBEDDING, padding_idx= pad_id)
        self.output_positional_encoding = nn.Embedding(num_embeddings= CONTEXT_LENGTH, embedding_dim=D_EMBEDDING) 


        self.encoder_blocks = nn.Sequential(*[Encoder() for _ in range(NUMBER_ENCODERS)])
        self.decoder_blocks = nn.Sequential(*[Decoder() for _ in range(NUMBER_DECODERS)])
        self.linear = nn.Linear(in_features=D_EMBEDDING, out_features=vocab_size)
        


    def forward(self, x, y):

        B, num_tokens = x.shape
        
        #Input Embedding + Positional Embedding
        x_embedding = self.input_embedding_table(x) #(B, N, D_i)
        positional_encoding = self.input_positional_encoding(torch.arange(num_tokens))
        x_embedding = x_embedding + positional_encoding

        #Output Embedding + Positional Embedding 
        y_embedding = self.output_embedding_table(y)
        positional_encoding = self.input_positional_encoding(torch.arange(num_tokens))
        y_embedding = y_embedding + positional_encoding

        
        #Encoder-Decoder
        encoder_output = self.encoder_blocks(x_embedding)
        decoder_output = self.decoder_blocks(y_embedding,encoder_output)
        output = self.linear(decoder_output)
        

        return output






        





    





