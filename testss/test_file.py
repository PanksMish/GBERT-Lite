"""
Unit tests for GBERT-Lite
"""

import pytest
import torch
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.gbert_lite import GBERTLite, DualEncoderFusion, ClassifierHead
from transformers import DistilBertTokenizer, GPT2Tokenizer


@pytest.fixture
def config():
    """Load test configuration"""
    config = {
        'model': {
            'distilbert_model': 'distilbert-base-uncased',
            'distilgpt2_model': 'distilgpt2',
            'freeze_encoders': True,
            'hidden_dim': 768,
            'classifier_hidden_dim': 256,
            'dropout': 0.1,
            'max_length': 64
        },
        'diffusion': {
            'enabled': False,
            'lstm_hidden_dim': 128,
            'lstm_num_layers': 2,
            'temporal_features': 10
        }
    }
    return config


@pytest.fixture
def sample_batch():
    """Create sample batch for testing"""
    batch_size = 2
    seq_length = 64
    
    batch = {
        'bert_input_ids': torch.randint(0, 30522, (batch_size, seq_length)),
        'bert_attention_mask': torch.ones(batch_size, seq_length),
        'gpt_input_ids': torch.randint(0, 50257, (batch_size, seq_length)),
        'gpt_attention_mask': torch.ones(batch_size, seq_length),
        'labels': torch.tensor([0, 1])
    }
    return batch


def test_dual_encoder_fusion(config, sample_batch):
    """Test dual encoder fusion module"""
    fusion = DualEncoderFusion(config)
    
    # Forward pass
    fused = fusion(
        sample_batch['bert_input_ids'],
        sample_batch['bert_attention_mask'],
        sample_batch['gpt_input_ids'],
        sample_batch['gpt_attention_mask']
    )
    
    # Check output shape
    assert fused.shape == (2, 1536)  # batch_size=2, fused_dim=768*2
    
    # Check if encoders are frozen
    if config['model']['freeze_encoders']:
        for param in fusion.distilbert.parameters():
            assert not param.requires_grad
        for param in fusion.distilgpt2.parameters():
            assert not param.requires_grad


def test_classifier_head():
    """Test classifier head"""
    classifier = ClassifierHead(input_dim=1536, hidden_dim=256, dropout=0.1)
    
    # Test forward pass
    x = torch.randn(2, 1536)
    output = classifier(x)
    
    # Check output shape and range
    assert output.shape == (2, 1)
    assert torch.all(output >= 0) and torch.all(output <= 1)


def test_gbert_lite_model(config, sample_batch):
    """Test full GBERT-Lite model"""
    model = GBERTLite(config)
    
    # Forward pass
    outputs = model(sample_batch, compute_risk=False)
    
    # Check outputs
    assert 'falsity_prob' in outputs
    assert outputs['falsity_prob'].shape == (2, 1)
    assert torch.all(outputs['falsity_prob'] >= 0)
    assert torch.all(outputs['falsity_prob'] <= 1)


def test_gbert_lite_predictions(config, sample_batch):
    """Test model predictions"""
    model = GBERTLite(config)
    model.eval()
    
    # Make predictions
    preds, probs = model.predict(sample_batch, threshold=0.5)
    
    # Check outputs
    assert preds.shape == (2,)
    assert probs.shape == (2, 1)
    assert torch.all((preds == 0) | (preds == 1))


def test_parameter_counts(config):
    """Test parameter counting"""
    model = GBERTLite(config)
    
    total_params = model.get_total_parameters()
    trainable_params = model.get_trainable_parameters()
    
    # Check counts
    assert total_params > 0
    assert trainable_params > 0
    
    # With frozen encoders, trainable should be much less than total
    if config['model']['freeze_encoders']:
        assert trainable_params < total_params * 0.1


def test_model_save_load(config, tmp_path):
    """Test model saving and loading"""
    model = GBERTLite(config)
    
    # Save model
    save_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), save_path)
    
    # Load model
    model2 = GBERTLite(config)
    model2.load_state_dict(torch.load(save_path))
    
    # Check parameters match
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_tokenizers():
    """Test tokenizer loading"""
    bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    
    # Test tokenization
    text = "This is a test sentence."
    
    bert_tokens = bert_tokenizer(text, return_tensors='pt')
    gpt_tokens = gpt_tokenizer(text, return_tensors='pt')
    
    assert 'input_ids' in bert_tokens
    assert 'attention_mask' in bert_tokens
    assert 'input_ids' in gpt_tokens
    assert 'attention_mask' in gpt_tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])