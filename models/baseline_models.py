"""
Baseline models for comparison
Implements: BERT, RoBERTa, DeBERTa, GBERT, BERT+GNN
"""

import torch
import torch.nn as nn
from transformers import (
    BertModel, RobertaModel, DebertaV2Model, 
    GPT2Model, AutoModel
)
from typing import Dict


class BERTBaseline(nn.Module):
    """Standard BERT baseline"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Fine-tune or freeze
        if config['model'].get('freeze_encoders', False):
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = self.bert(
            input_ids=batch['bert_input_ids'],
            attention_mask=batch['bert_attention_mask']
        )
        pooled = outputs.pooler_output
        falsity_prob = self.classifier(pooled)
        return {'falsity_prob': falsity_prob}


class RoBERTaBaseline(nn.Module):
    """RoBERTa baseline"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        
        if config['model'].get('freeze_encoders', False):
            for param in self.roberta.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = self.roberta(
            input_ids=batch['bert_input_ids'],
            attention_mask=batch['bert_attention_mask']
        )
        pooled = outputs.pooler_output
        falsity_prob = self.classifier(pooled)
        return {'falsity_prob': falsity_prob}


class DeBERTaBaseline(nn.Module):
    """DeBERTa-v3 baseline"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.deberta = DebertaV2Model.from_pretrained('microsoft/deberta-v3-base')
        
        if config['model'].get('freeze_encoders', False):
            for param in self.deberta.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = self.deberta(
            input_ids=batch['bert_input_ids'],
            attention_mask=batch['bert_attention_mask']
        )
        pooled = outputs.last_hidden_state[:, 0, :]
        falsity_prob = self.classifier(pooled)
        return {'falsity_prob': falsity_prob}


class GBERTBaseline(nn.Module):
    """GBERT baseline (full fine-tuning version)"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        
        # Don't freeze for full GBERT
        
        self.classifier = nn.Sequential(
            nn.Linear(768 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # BERT encoding
        bert_outputs = self.bert(
            input_ids=batch['bert_input_ids'],
            attention_mask=batch['bert_attention_mask']
        )
        h_bert = bert_outputs.pooler_output
        
        # GPT-2 encoding
        gpt_outputs = self.gpt2(
            input_ids=batch['gpt_input_ids'],
            attention_mask=batch['gpt_attention_mask']
        )
        h_gpt = torch.mean(gpt_outputs.last_hidden_state, dim=1)
        
        # Fusion
        fused = torch.cat([h_bert, h_gpt], dim=-1)
        falsity_prob = self.classifier(fused)
        
        return {'falsity_prob': falsity_prob}


class SimpleGNN(nn.Module):
    """Simple Graph Neural Network for propagation modeling"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Simple GNN forward pass
        x: node features [batch, num_nodes, input_dim]
        adj: adjacency matrix [batch, num_nodes, num_nodes]
        """
        # First layer
        x = self.conv1(x)
        x = torch.bmm(adj, x)  # Graph convolution
        x = self.relu(x)
        
        # Second layer
        x = self.conv2(x)
        x = torch.bmm(adj, x)
        x = self.relu(x)
        
        # Global pooling
        x = torch.mean(x, dim=1)
        
        return x


class BERTGNNBaseline(nn.Module):
    """BERT + GNN baseline"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        if config['model'].get('freeze_encoders', False):
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Simple GNN for propagation structure
        self.gnn = SimpleGNN(input_dim=768, hidden_dim=256)
        
        self.classifier = nn.Sequential(
            nn.Linear(768 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # BERT encoding
        bert_outputs = self.bert(
            input_ids=batch['bert_input_ids'],
            attention_mask=batch['bert_attention_mask']
        )
        h_bert = bert_outputs.pooler_output
        
        # GNN encoding (if propagation features available)
        if 'propagation_features' in batch:
            # Create simple graph structure
            batch_size = h_bert.size(0)
            num_nodes = 5  # Simplified
            
            # Node features (replicate BERT features)
            node_features = h_bert.unsqueeze(1).repeat(1, num_nodes, 1)
            
            # Simple adjacency (fully connected for simplicity)
            adj = torch.ones(batch_size, num_nodes, num_nodes).to(h_bert.device)
            adj = adj / num_nodes  # Normalize
            
            h_gnn = self.gnn(node_features, adj)
        else:
            # If no propagation data, use zeros
            h_gnn = torch.zeros(h_bert.size(0), 256).to(h_bert.device)
        
        # Fusion
        fused = torch.cat([h_bert, h_gnn], dim=-1)
        falsity_prob = self.classifier(fused)
        
        return {'falsity_prob': falsity_prob}


def get_baseline_model(model_name: str, config: Dict) -> nn.Module:
    """Factory function to get baseline models"""
    models = {
        'bert': BERTBaseline,
        'roberta': RoBERTaBaseline,
        'deberta': DeBERTaBaseline,
        'gbert': GBERTBaseline,
        'bert_gnn': BERTGNNBaseline
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    return models[model_name](config)