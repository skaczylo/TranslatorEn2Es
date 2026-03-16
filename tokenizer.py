import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

RUTA_DATASET_JSON = "dataset_800k.json" 
NOMBRE_TOKENIZER_SALIDA = "vocabulario/vocabulario_32k.json"

def generador_de_textos(ruta_json):
    with open(ruta_json, 'r', encoding='utf-8') as f:
        datos = json.load(f)
        for par in datos:
            yield par['en']
            yield par['es']


tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

#ByteLevel para no perder los espacios
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()


entrenador = BpeTrainer(
    vocab_size=32000, 
    special_tokens=["<PAD>", "<START>", "<END>", "<UNK>"],
    initial_alphabet=ByteLevel.alphabet() #
)

tokenizer.train_from_iterator(generador_de_textos(RUTA_DATASET_JSON), trainer=entrenador)
tokenizer.save(NOMBRE_TOKENIZER_SALIDA)

print(f"Tamaño final del vocabulario: {tokenizer.get_vocab_size()} tokens.")








