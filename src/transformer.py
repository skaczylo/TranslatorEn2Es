"""
ARQUITECTURA TRANSFORMER.
En este archivo se pretende construir el modelo o la arquitectura del Transformer
usando Encoder-Decoder representado en el paper "Attention is all you need"

"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F


@dataclass
class TransformerConfig:
    # Arquitectura
    context_length : int   = 128
    d_embedding    : int   = 256
    attention_heads: int   = 8
    num_encoders   : int   = 4
    num_decoders   : int   = 4
    dropout        : float = 0.1

    # Vocabulario
    vocab_size     : int   = 16000
    pad_id         : int   = 0

    def __post_init__(self):
        assert self.d_embedding % self.attention_heads == 0, \
            f"d_embedding ({self.d_embedding}) debe ser divisible entre attention_heads ({self.attention_heads})"
        self.head_dim = self.d_embedding // self.attention_heads


class AttentionHead(nn.Module):
    """Una sola capa de atencion"""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.key   = nn.Linear(cfg.d_embedding, cfg.head_dim, bias=False)
        self.query = nn.Linear(cfg.d_embedding, cfg.head_dim, bias=False)
        self.value = nn.Linear(cfg.d_embedding, cfg.head_dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer('tril', torch.tril(torch.ones(cfg.context_length, cfg.context_length)))


    def forward(self,key_input,query_input,value_input, padding_mask = None, masked_attention = False):
        """
        padding_mask : Matriz máscara para evitar que los token <PAD> afecten al cálculo de la atencion
        masked_attention : False indica que no se triangula la matriz de atencion; True se triangula la matriz.
        Se usa en el decoder
        """

        #B = numero de Batch; T = numero de "tokens"; C = dimension de cada token
        B, N, D_i = query_input.shape
        
        K = self.key(key_input) # (B, N , D_i)
        Q = self.query(query_input) #(B, N , D_i)
        V = self.value(value_input) #(B, N , D_i)

        #Calculamos la Atención QxK^T
        #Calculamos la "afinidad" Q x K^t
        #Como tenemos batches, usamos el operador @ que aplica la multiplicacion en las dos ultimas dimensiones B veces
        #K.transpose significa transponer la penultima dimensino T con la pultima dimension C.

        scores = Q @ K.transpose(-2,-1) #(B, N , D_i) x (B,D_i,N) = (B,N,N)
        scores = scores /(D_i ** 0.5)

        #Mascara para que no pongan atencion en los tokens <PAD>
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, float('-inf'))
         
        #Mascara para que los tokens y_n del decoder no se fijen en los posterioes y_n+1,y_n+2,....
        if masked_attention is True:
            scores = scores.masked_fill(self.tril[:N, :N] == 0, float('-inf')) # (B, T, T)

        scores = F.softmax(scores, dim=-1) #Aplicamos softmax sobre las filas, es decir, sobre la ultima dimension

        scores = self.dropout(scores)

        attention = scores @ V

        return attention


class MultiHeadAttention(nn.Module):
    """Multiples capas de atencion"""

    def __init__(self, cfg: TransformerConfig):
        """
        Block attention es la concatenacion de las capas de atencion
        nn.ModuleList permite tener una lista de modulos y que Pytorch sea "consciente" de que existen; si haces una lista [] normal, no los reconocería a la hora de entrenar
        """
        super().__init__()
        # Renombrado: heads -> block_attention
        self.block_attention = nn.ModuleList([AttentionHead(cfg) for _ in range(cfg.attention_heads)])
        self.projection = nn.Linear(in_features=cfg.d_embedding, 
                                    out_features=cfg.d_embedding, 
                                    bias=False)
        self.dropout    = nn.Dropout(cfg.dropout)

    
    def forward(self,key_input,query_input,value_input,padding_mask = None, masked_attention = False):

        outputs = [head(key_input = key_input ,
                        query_input = query_input,
                        value_input = value_input,
                        padding_mask = padding_mask,
                        masked_attention = masked_attention)
                   for head in self.block_attention] #Lista de matrices Nx(D_Embedding / num_heads)

        outputs = torch.cat(outputs, dim =-1) # Matriz N x D_embedding (por cada batch) => (B, N, D_embedding)
        multiple_attention = self.projection(outputs)

        multiple_attention = self.dropout(multiple_attention)

        return multiple_attention


class MLP(nn.Module):

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        # Renombrado: net -> mlp
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_embedding, 4 * cfg.d_embedding),
            nn.ReLU(),
            nn.Linear(4 * cfg.d_embedding, cfg.d_embedding),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.mlp(x)


#==================================================
#DECODER
#==================================================

class Decoder(nn.Module):

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        
        self.masked_attention_heads = MultiHeadAttention(cfg)
        self.attention_heads  = MultiHeadAttention(cfg)
        self.mlp         = MLP(cfg)
        self.LayerNorm1  = nn.LayerNorm(cfg.d_embedding)
        self.LayerNorm2  = nn.LayerNorm(cfg.d_embedding)
        self.LayerNorm3  = nn.LayerNorm(cfg.d_embedding)

    def forward(self, y, y_padding_mask, encoder_output, encoder_padding_mask):
        y = self.LayerNorm1(y + self.masked_attention_heads(y, y, y,
                                             padding_mask=y_padding_mask,
                                             masked_attention=True))
        y = self.LayerNorm2(y + self.attention_heads(encoder_output, y, encoder_output,
                                            padding_mask=encoder_padding_mask))
        y = self.LayerNorm3(y + self.mlp(y))
        return y



#=============================================
#ENCODER
#=============================================

class Encoder(nn.Module):
    """
    Arquitectura Encoder
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
       
        self.attention_heads  = MultiHeadAttention(cfg)
        self.mlp   = MLP(cfg)
        self.LayerNorm1 = nn.LayerNorm(cfg.d_embedding)
        self.LayerNorm2 = nn.LayerNorm(cfg.d_embedding)

    def forward(self, x, padding_mask=None):
        x = x + self.attention_heads(x, x, x, padding_mask=padding_mask)
        x = self.LayerNorm1(x)
        x = x + self.mlp(x)
        x = self.LayerNorm2(x)
        return x
    

class Transformer(nn.Module):
    """
    Arquitectura del Transformer de tipo Encoder-Decoder
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

       
        # Embeddings (El ORDEN de declaración es VITAL para cargar el Optimizador)
        self.input_embedding_table = nn.Embedding(cfg.vocab_size, cfg.d_embedding, padding_idx=cfg.pad_id)
        self.input_positional_encoding = nn.Embedding(cfg.context_length, cfg.d_embedding)
        
        self.output_embedding_table = nn.Embedding(cfg.vocab_size, cfg.d_embedding, padding_idx=cfg.pad_id)
        self.output_positional_encoding = nn.Embedding(cfg.context_length, cfg.d_embedding)

        # Bloques (Esto se queda igual)
        self.encoder_blocks = nn.ModuleList([Encoder(cfg) for _ in range(cfg.num_encoders)])
        self.decoder_blocks = nn.ModuleList([Decoder(cfg) for _ in range(cfg.num_decoders)])
        
        # Linear Final
        self.linear = nn.Linear(cfg.d_embedding, cfg.vocab_size)

        # Inicialización de pesos
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _padding_mask(self, x):
        return (x == self.cfg.pad_id).unsqueeze(1)

    def encode(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device)
        out  = self.input_embedding_table(x) + self.input_positional_encoding(pos)
        mask = self._padding_mask(x)
        for block in self.encoder_blocks:
            out = block(out, padding_mask=mask)
        return out, mask

    def decode(self, y, encoder_output, encoder_mask):
        B, T = y.shape
        pos  = torch.arange(T, device=y.device)
        out  = self.output_embedding_table(y) + self.output_positional_encoding(pos)
        mask = self._padding_mask(y)
        for block in self.decoder_blocks:
            out = block(out, mask, encoder_output, encoder_mask)
        return out

    def forward(self, x, y):
        encoder_output, encoder_mask = self.encode(x)
        decoder_output = self.decode(y, encoder_output, encoder_mask)
        return self.linear(decoder_output)

    
    @torch.no_grad()
    def predict(self, x, start_id, end_id, max_new_tokens=None, top_k=10, device='cpu'):
        self.eval()
        max_new_tokens = max_new_tokens or self.cfg.context_length
        x = x[:, :self.cfg.context_length].to(device)
        y = torch.tensor([[start_id]], device=device)

        encoder_output, encoder_mask = self.encode(x)

        for _ in range(max_new_tokens):
            y_cond  = y[:, -self.cfg.context_length:]
            logits  = self.decode(y_cond, encoder_output, encoder_mask)
            logits  = self.linear(logits[:, -1, :])

           
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
            
            y = torch.cat([y, token], dim=1)
            if token.item() == end_id:
                break

        return y
    







        





    





