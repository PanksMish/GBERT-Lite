"""
GBERT-Lite: Lightweight Dual-Transformer Framework
Main model implementation
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, GPT2Model
from typing import Dict, Optional, Tuple


class DualEncoderFusion(nn.Module):
    """Dual-encoder fusion module combining DistilBERT and DistilGPT-2"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Load frozen encoders
        self.distilbert = DistilBertModel.from_pretrained(
            config['model']['distilbert_model']
        )
        self.distilgpt2 = GPT2Model.from_pretrained(
            config['model']['distilgpt2_model']
        )
        
        # Freeze encoders if specified
        if config['model']['freeze_encoders']:
            for param in self.distilbert.parameters():
                param.requires_grad = False
            for param in self.distilgpt2.parameters():
                param.requires_grad = False
        
        self.hidden_dim = config['model']['hidden_dim']
        self.fused_dim = self.hidden_dim * 2  # Concatenation of both encoders
    
    def forward(self, bert_input_ids: torch.Tensor, 
                bert_attention_mask: torch.Tensor,
                gpt_input_ids: torch.Tensor,
                gpt_attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dual encoders
        
        Args:
            bert_input_ids: Input IDs for DistilBERT
            bert_attention_mask: Attention mask for DistilBERT
            gpt_input_ids: Input IDs for DistilGPT-2
            gpt_attention_mask: Attention mask for DistilGPT-2
        
        Returns:
            Fused representation tensor
        """
        # DistilBERT encoding (bidirectional contextual)
        bert_outputs = self.distilbert(
            input_ids=bert_input_ids,
            attention_mask=bert_attention_mask
        )
        h_bert = bert_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # DistilGPT-2 encoding (autoregressive)
        gpt_outputs = self.distilgpt2(
            input_ids=gpt_input_ids,
            attention_mask=gpt_attention_mask
        )
        # Use mean pooling for GPT-2 (no [CLS] token)
        h_gpt = torch.mean(gpt_outputs.last_hidden_state, dim=1)
        
        # Fusion via concatenation
        z = torch.cat([h_bert, h_gpt], dim=-1)
        
        return z


class PropagationScorer(nn.Module):
    """LSTM-based temporal propagation scorer"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        input_dim = config['diffusion']['temporal_features']
        hidden_dim = config['diffusion']['lstm_hidden_dim']
        num_layers = config['diffusion']['lstm_num_layers']
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, propagation_features: torch.Tensor) -> torch.Tensor:
        """
        Estimate propagation potential
        
        Args:
            propagation_features: Temporal propagation features [batch, seq_len, features]
        
        Returns:
            Propagation score [batch, 1]
        """
        # If features are 2D, add sequence dimension
        if len(propagation_features.shape) == 2:
            propagation_features = propagation_features.unsqueeze(1)
        
        lstm_out, (h_n, c_n) = self.lstm(propagation_features)
        
        # Use final hidden state
        h_final = h_n[-1]  # [batch, hidden_dim]
        
        # Predict propagation score
        score = self.fc(h_final)
        score = self.sigmoid(score)
        
        return score


class ClassifierHead(nn.Module):
    """Lightweight classification head"""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classifier
        
        Args:
            x: Fused representation
        
        Returns:
            Falsity probability [batch, 1]
        """
        return self.classifier(x)


class GBERTLite(nn.Module):
    """
    GBERT-Lite: Lightweight Dual-Transformer Framework
    for Diffusion-Aware Misinformation Detection
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Dual encoder fusion
        self.dual_encoder = DualEncoderFusion(config)
        
        # Classification head
        fused_dim = self.dual_encoder.fused_dim
        classifier_hidden_dim = config['model']['classifier_hidden_dim']
        dropout = config['model']['dropout']
        
        self.classifier = ClassifierHead(
            fused_dim, 
            classifier_hidden_dim, 
            dropout
        )
        
        # Propagation scorer (optional)
        self.use_diffusion = config['diffusion']['enabled']
        if self.use_diffusion:
            self.propagation_scorer = PropagationScorer(config)
    
    def forward(self, batch: Dict[str, torch.Tensor], 
                compute_risk: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GBERT-Lite
        
        Args:
            batch: Dictionary containing input tensors
            compute_risk: Whether to compute diffusion-aware risk score
        
        Returns:
            Dictionary with predictions and optional risk scores
        """
        # Extract inputs
        bert_input_ids = batch['bert_input_ids']
        bert_attention_mask = batch['bert_attention_mask']
        gpt_input_ids = batch['gpt_input_ids']
        gpt_attention_mask = batch['gpt_attention_mask']
        
        # Dual-encoder fusion
        z = self.dual_encoder(
            bert_input_ids, bert_attention_mask,
            gpt_input_ids, gpt_attention_mask
        )
        
        # Falsity probability
        y_hat = self.classifier(z)
        
        outputs = {'falsity_prob': y_hat}
        
        # Compute risk score if requested and diffusion enabled
        if compute_risk and self.use_diffusion:
            if 'propagation_features' in batch:
                propagation_features = batch['propagation_features']
                s_hat = self.propagation_scorer(propagation_features)
                
                # Diffusion-aware risk score: R(x) = y_hat * s_hat
                risk_score = y_hat * s_hat
                outputs['risk_score'] = risk_score
                outputs['propagation_score'] = s_hat
            else:
                # If no propagation features, risk = falsity probability
                outputs['risk_score'] = y_hat
        
        return outputs
    
    def predict(self, batch: Dict[str, torch.Tensor], 
                threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make binary predictions
        
        Args:
            batch: Input batch
            threshold: Classification threshold
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch, compute_risk=False)
            probs = outputs['falsity_prob']
            preds = (probs >= threshold).long()
        
        return preds, probs
    
    def compute_risk_score(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute diffusion-aware risk score
        
        Args:
            batch: Input batch
        
        Returns:
            Risk scores
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch, compute_risk=True)
            risk_scores = outputs.get('risk_score', outputs['falsity_prob'])
        
        return risk_scores
    
    def get_trainable_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_parameters(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())