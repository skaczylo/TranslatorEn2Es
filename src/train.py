import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm.auto import tqdm
from transformer import TransformerConfig, Transformer
from data import TranslatorTokenizer, TranslationDataset
from config import VOCAB_16K, BEST_MODEL_PATH


def get_scheduler(optimizer, d_model, warmup_steps=4000):
    def lr_lambda(step):
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def load_tokenizer(vocab_file = VOCAB_16K):

    if not vocab_file.exists():
        raise FileNotFoundError(f"No se encontró el vocabulario en: {vocab_file}")
    tokenizer = TranslatorTokenizer(path=str(vocab_file), context_length=128)
    return tokenizer
    

def load_model(cfg : TransformerConfig,checkpoint = BEST_MODEL_PATH,device = "cpu"):
    """
    Carga el modelo con los pesos de checkpoint y además el tokenizador con el vocabulario del modelo
    en vocab_file.
    """

    if not checkpoint.exists():
        raise FileNotFoundError(f"No se encontraron los pesos del modelo en: {checkpoint}")
    
    model = Transformer(cfg)

    checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model
    

class Trainer:
    """
    Clase que orquesta el entrenamiento, evaluación y guardado del modelo.
    """

    def __init__(self, model, train_loader, val_loader, device='cuda', results_path='./pesos', patience=3):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.results_path = results_path
        self.patience = patience

        self.criterion = nn.CrossEntropyLoss(ignore_index=model.cfg.pad_id)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        
        #Scheduler (Extrae el d_embedding del modelo)
        d_model = model.cfg.d_embedding
        def lr_lambda(step):
            step = max(step, 1)
            return (d_model ** -0.5) * min(step ** -0.5, step * 4000 ** -1.5)
            
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        
        
        # Variables de estado interno para el Early Stopping
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.start_epoch = 00

    @torch.no_grad()
    def estimate_loss(self, eval_batches=50):
        """Evalúa un puñado de batches para obtener una métrica rápida."""
        self.model.eval()
        out = {}
        loaders = {'train': self.train_loader, 'val': self.val_loader}
        
        for split, loader in loaders.items():
            total_loss = 0
            batches_evaluated = 0
            
            for x, y_full in loader:
                if batches_evaluated >= eval_batches:
                    break
                
                x = x.to(self.device)
                y_full = y_full.to(self.device)
                y_input = y_full[:, :-1]
                y_target = y_full[:, 1:]

                logits = self.model(x, y_input)
                B, T, C = logits.shape
                loss = self.criterion(logits.reshape(B*T, C), y_target.reshape(B*T))
                
                total_loss += loss.item()
                batches_evaluated += 1
                
            out[split] = total_loss / batches_evaluated
            
        self.model.train()
        return out

    def load_checkpoint(self, checkpoint_path):
        """Carga todos los estados para reanudar el entrenamiento exactamente donde se dejó."""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Recuperamos la memoria del Early Stopping y la época
        self.start_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', checkpoint['val_loss'])
        self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        
        print(f"Reanudando desde la época {self.start_epoch} | Mejor Val Loss histórico: {self.best_val_loss:.4f}")


    def train(self, total_epochs):
       

        for epoch in tqdm(range(self.start_epoch, total_epochs), desc="Training"):
            
            # --- 1. ENTRENAMIENTO ---
            self.model.train()
            total_train_loss = 0
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), 
                                desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)

            for batch_idx, (x, y_full) in progress_bar:
                x = x.to(self.device)
                y_full = y_full.to(self.device)

                y_input = y_full[:, :-1]
                y_target = y_full[:, 1:]

                logits = self.model(x, y_input)
                B, T, C = logits.shape
                loss = self.criterion(logits.reshape(B*T, C), y_target.reshape(B*T))
                
                total_train_loss += loss.item()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.6f}'})

            avg_train_loss = total_train_loss / len(self.train_loader)

            #Validacion
            losses = self.estimate_loss()
            val_loss = losses['val']

        
            # Checkpoint + early stopping
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': self.best_val_loss,
                'epochs_without_improvement': self.epochs_without_improvement
            }

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                

                save_name = os.path.join(self.results_path, f"best_model_epoch_{epoch+1}_val_{val_loss:.4f}.pth")
                torch.save(checkpoint, save_name)
            else:
                self.epochs_without_improvement += 1
                
                if self.epochs_without_improvement >= self.patience:
                    tqdm.write(f"Early stopping en la época {epoch+1}. Entrenamiento finalizado.")
                    break

            # --- 4. REPORTE ---
            tqdm.write(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f}\n")


