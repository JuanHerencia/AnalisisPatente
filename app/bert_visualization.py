import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
from typing import Dict, List, Tuple
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BertVisualizer:
    def __init__(self):
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Usando dispositivo: {self.device}")
            
            self.model_name = "anferico/bert-for-patents"
            
            # Configurar el modelo para usar atención cruzada
            config = AutoConfig.from_pretrained(
                self.model_name,
                output_attentions=True,
                output_hidden_states=True,
                attn_implementation="eager"
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                config=config
            ).to(self.device)
            
            self.model.eval()
            self.max_length = 512
            
        except Exception as e:
            logger.error(f"Error inicializando BertVisualizer: {str(e)}")
            raise

    def get_cross_attention_scores(self, text1: str, text2: str) -> Dict:
        try:
            # Tokenizar ambos textos
            tokens1 = self.tokenizer(
                text1, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=self.max_length,
                add_special_tokens=True
            )
            tokens2 = self.tokenizer(
                text2, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=self.max_length,
                add_special_tokens=True
            )
            
            # Mover tensores a GPU si está disponible
            tokens1 = {k: v.to(self.device) for k, v in tokens1.items()}
            tokens2 = {k: v.to(self.device) for k, v in tokens2.items()}
            
            # Procesar ambos textos
            with torch.no_grad():
                outputs1 = self.model(**tokens1, output_attentions=True)
                outputs2 = self.model(**tokens2, output_attentions=True)
                
                # Calcular atención cruzada usando la última capa de atención
                hidden1 = outputs1.last_hidden_state
                hidden2 = outputs2.last_hidden_state
                
                # Calcular scores de atención cruzada
                cross_attention = torch.matmul(hidden1, hidden2.transpose(-2, -1))
                cross_attention = F.softmax(cross_attention / np.sqrt(hidden1.size(-1)), dim=-1)
                
                # Promediar sobre los batch y head dimensions
                cross_attention = cross_attention.mean(dim=0)  # Promedio sobre batch
            
            # Obtener tokens y preparar resultado
            tokens1_text = self.tokenizer.convert_ids_to_tokens(tokens1['input_ids'][0])
            tokens2_text = self.tokenizer.convert_ids_to_tokens(tokens2['input_ids'][0])
            
            # Filtrar tokens PAD
            valid_indices1 = [i for i, t in enumerate(tokens1_text) if t != '[PAD]']
            valid_indices2 = [i for i, t in enumerate(tokens2_text) if t != '[PAD]']
            
            # Crear matriz de atención cruzada
            attention_matrix = cross_attention[valid_indices1][:, valid_indices2].cpu().numpy()
            
            return {
                'text1_tokens': [tokens1_text[i] for i in valid_indices1],
                'text2_tokens': [tokens2_text[i] for i in valid_indices2],
                'cross_attention': attention_matrix.tolist(),
                'is_special1': [tokens1_text[i] in ['[CLS]', '[SEP]'] for i in valid_indices1],
                'is_special2': [tokens2_text[i] in ['[CLS]', '[SEP]'] for i in valid_indices2]
            }
            
        except Exception as e:
            logger.error(f"Error en get_cross_attention_scores: {str(e)}")
            raise

    def process_texts(self, main_text: str, cited_text: str) -> Dict:
        try:
            if not main_text or not cited_text:
                raise ValueError("Textos vacíos o nulos")
                
            main_text = main_text.strip()
            cited_text = cited_text.strip()
            
            logger.info(f"Procesando textos - Principal: {len(main_text)} caracteres, Citado: {len(cited_text)} caracteres")
            
            cross_attention_results = self.get_cross_attention_scores(main_text, cited_text)
            
            return {
                'status': 'success',
                'analysis': cross_attention_results
            }
            
        except Exception as e:
            logger.error(f"Error en process_texts: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }