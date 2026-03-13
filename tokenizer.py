import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


RUTA_DATASET_JSON = "dataset.json" # Cambia esto por la ruta real de tu archivo
NOMBRE_TOKENIZER_SALIDA = "vocabulario.json"


def generador_de_textos(ruta_json):
    with open(ruta_json, 'r', encoding='utf-8') as f:
        datos = json.load(f)
        for par in datos:
            # Mandamos a entrenar tanto la frase en inglés como en español
            yield par['en']
            yield par['es']

#Tokenizador BPE
# Todo lo que no conozca lo marcará como <UNK> 

tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

tokenizer.pre_tokenizer = Whitespace() # separar por espacios en blanco antes de aplicar el BPE


# 15,000 tamaño vocab 
entrenador = BpeTrainer(
    vocab_size=15000, 
    special_tokens=["<PAD>", "<START>", "<END>", "<UNK>"]
)


print("Entrenando el tokenizador...")

tokenizer.train_from_iterator(generador_de_textos(RUTA_DATASET_JSON), trainer=entrenador)
tokenizer.save(NOMBRE_TOKENIZER_SALIDA)

print(f"Tamaño final del vocabulario: {tokenizer.get_vocab_size()} tokens.")








