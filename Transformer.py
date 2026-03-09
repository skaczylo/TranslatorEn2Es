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

    def forward(self,x,mask = None):

        #B = numero de Batch; T = numero de "tokens"; C = dimension de cada token
        B, N, D_i = x.shape

        K = self.key(x) # (B, N , D_i)
        Q = self.key(x) #(B, N , D_i)
        V = self.key(x) #(B, N , D_i)

        #Calculamos la Atención QxK^T
        """
        Calculamos la "afinidad" Q x K^t
        Como tenemos batches, usamos el operador @ que aplica la multiplicacion en las dos ultimas dimensiones B veces
        K.transpose significa transponer la penultima dimensino T con la pultima dimension C. 
        """
        scores = Q @ K.transpose(-2,-1) #(B, N , D_i) x (B,D_i,N) = (B,N,N)
        scores = scores /(D_i ** 0.5)

        if mask is not None:
            scores = scores + mask  #Sumamos la mascara para evitar que se fije en tokens <PAD>

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
        self.projection = nn.Linear(in_features=D_EMBEDDING,out_features=D_EMBEDDING,bias=False)  #Matriz Omega_O 

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self,x,mask = None):
        B, N, D_i = x.shape

        outputs = [head(x,mask) for head in self.block_attention] #Lista de matrices Nx(D_Embedding / num_heads)

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



class EncoderBlock(nn.Module):

    """
    Bloque que se apilará para formar el Encoder

    """

    def __init__(self):
        super().__init__()
    
        self.attention_heads = MultiHeadAttention(num_heads= ATTENTION_HEADS)
        self.mlp = MLP()
        self.LayerNorm = nn.LayerNorm(D_EMBEDDING)

    def forward(self,x,mask = None):
        
        # x = ( B, N, D_i)

        z = self.LayerNorm(self.attention_heads(x,mask) + x)
        output = self.LayerNorm(self.mlp(z) + z) # ( B, N, D_i)

        return output



class Encoder(nn.Module):

    def __init__(self, vocab_size,pad_id):
        super().__init__()

        """
        Embedding: Tabla E de dimension VOCAB_SIZE x D_EMBEDDING
        x_n tendrá dimension D_EMBEDDING
        Pytorch no multiplica matrices aquí, solo indica, dado un vector de indices de tokens, que filas de la matriz E extraer
        """
       
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=D_EMBEDDING, padding_idx= pad_id)

        """
        Positional Encoding: x_n = x_n + r_n -> r_ tiene dimensión D_EMBEDDING
        El objetivo es que permita determinar en que posición se encuentra cada palabra, por eso el numero de embeddings es CONTEXT_LENGT
        """
        self.position_embedding_table = nn.Embedding(num_embeddings= CONTEXT_LENGTH, embedding_dim=D_EMBEDDING)
        self.encoder_blocks = nn.Sequential(*[EncoderBlock() for _ in range(NUMBER_ENCODERS)])



        
    def forward(self, x, mask = None):

        B, num_tokens = x.shape
        
        #Input Embedding + Positional Embedding
        x_embedding = self.token_embedding_table(x) #(B, N, D_i)
        positional_encoding = self.position_embedding_table(torch.arange(num_tokens))
        x_embedding = x_embedding + positional_encoding

        #Bloques de Encoder encadenados
        output = self.encoder_blocks(x_embedding)
        return output






        





    





