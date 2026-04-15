# inference.py
import torch
from config import VOCAB_16K, BEST_MODEL_PATH
from transformer import TransformerConfig, Transformer
from data import TranslatorTokenizer
from dataclasses import dataclass
from train import load_model,load_tokenizer



@dataclass
class TranslatorConfig:
    
    checkpoint_path:  str = BEST_MODEL_PATH
    vocab_file:  str = VOCAB_16K
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_new_tokens: int = 128
    top_k: int = 10
    context_length: int = 128

    def __post_init__(self):
        if not self.vocab_file.exists():
            raise FileNotFoundError(f"Error: No se encontró el vocabulario en:\n{self.vocab_file}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Error: No se encontraron los pesos del modelo en:\n{self.checkpoint_path}")


class Translator:
    def __init__(self, cfg: TranslatorConfig = None, model_cfg: TransformerConfig = None):

        
        self.cfg = cfg if cfg is not None else TranslatorConfig()
        
        self.device = self.cfg.device

        
        #Tokenizer
        self.tokenizer = load_tokenizer(vocab_file=self.cfg.vocab_file)
        
        #Modelo
        model_cfg = TransformerConfig(vocab_size=len(self.tokenizer), pad_id=self.tokenizer.pad_id,context_length=self.cfg.context_length)
        
        self.model = load_model(cfg=model_cfg,checkpoint=self.cfg.checkpoint_path,device=self.device)
        
   

    def translate(self, text: str) -> str:
        """Recibe texto en inglés, devuelve la traducción en español."""
        if not text.strip():
            return ""

        input_ids = self.tokenizer.encode(text, pad=False)
        x = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        # Inferencia pura (llamando al modelo e inyectando max_tokens y top_k)
        y_pred = self.model.predict(
            x=x, 
            start_id=self.tokenizer.start_id, 
            end_id=self.tokenizer.end_id,
            max_new_tokens=self.cfg.max_new_tokens,
            top_k=self.cfg.top_k,
            device=self.device
        )

        # Postprocesamiento
        translation = self.tokenizer.decode(y_pred[0].tolist(), skip_special_tokens=True)
        return translation