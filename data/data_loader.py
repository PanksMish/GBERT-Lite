"""
Data loader module for GBERT-Lite
Handles loading and preprocessing of Fake and Real News Dataset and FakeNewsNet
"""

import os
import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import networkx as nx
from tqdm import tqdm


class TextPreprocessor:
    """Preprocessor for cleaning and normalizing news articles"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        if self.config.get('lowercase', True):
            text = text.lower()
        
        # Remove URLs
        if self.config.get('remove_urls', True):
            text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove HTML tags
        if self.config.get('remove_html', True):
            text = re.sub(r'<.*?>', '', text)
        
        # Remove emojis
        if self.config.get('remove_emojis', True):
            emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags
                "]+", flags=re.UNICODE)
            text = emoji_pattern.sub(r'', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts"""
        return [self.clean_text(text) for text in texts]


class FakeRealNewsDataset(Dataset):
    """Dataset class for Fake and Real News Dataset"""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 bert_tokenizer, gpt_tokenizer, max_length: int = 64):
        self.texts = texts
        self.labels = labels
        self.bert_tokenizer = bert_tokenizer
        self.gpt_tokenizer = gpt_tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize for BERT
        bert_encoding = self.bert_tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize for GPT
        gpt_encoding = self.gpt_tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'bert_input_ids': bert_encoding['input_ids'].squeeze(0),
            'bert_attention_mask': bert_encoding['attention_mask'].squeeze(0),
            'gpt_input_ids': gpt_encoding['input_ids'].squeeze(0),
            'gpt_attention_mask': gpt_encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class FakeNewsNetDataset(Dataset):
    """Dataset class for FakeNewsNet with propagation data"""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 propagation_features: np.ndarray,
                 bert_tokenizer, gpt_tokenizer, max_length: int = 64):
        self.texts = texts
        self.labels = labels
        self.propagation_features = propagation_features
        self.bert_tokenizer = bert_tokenizer
        self.gpt_tokenizer = gpt_tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        prop_feat = self.propagation_features[idx]
        
        # Tokenize for BERT
        bert_encoding = self.bert_tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize for GPT
        gpt_encoding = self.gpt_tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'bert_input_ids': bert_encoding['input_ids'].squeeze(0),
            'bert_attention_mask': bert_encoding['attention_mask'].squeeze(0),
            'gpt_input_ids': gpt_encoding['input_ids'].squeeze(0),
            'gpt_attention_mask': gpt_encoding['attention_mask'].squeeze(0),
            'propagation_features': torch.tensor(prop_feat, dtype=torch.float32),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DataLoaderFactory:
    """Factory class for creating data loaders"""
    
    def __init__(self, config: Dict, bert_tokenizer, gpt_tokenizer):
        self.config = config
        self.bert_tokenizer = bert_tokenizer
        self.gpt_tokenizer = gpt_tokenizer
        self.preprocessor = TextPreprocessor(config['preprocessing'])
        self.data_dir = Path(config['dataset']['data_dir'])
        self.cache_dir = Path(config['dataset']['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_fake_real_news(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load Fake and Real News Dataset from Kaggle"""
        fake_path = self.data_dir / "Fake.csv"
        real_path = self.data_dir / "True.csv"
        
        if not fake_path.exists() or not real_path.exists():
            print("Dataset not found. Please download from Kaggle:")
            print("https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
            print(f"Place Fake.csv and True.csv in {self.data_dir}")
            raise FileNotFoundError("Dataset files not found")
        
        # Load datasets
        fake_df = pd.read_csv(fake_path)
        real_df = pd.read_csv(real_path)
        
        # Add labels
        fake_df['label'] = 1  # Fake news
        real_df['label'] = 0  # Real news
        
        # Combine
        df = pd.concat([fake_df, real_df], ignore_index=True)
        
        # Combine title and text
        df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        
        # Clean text
        print("Preprocessing texts...")
        df['text'] = self.preprocessor.preprocess_batch(df['text'].tolist())
        
        # Remove duplicates
        if self.config['preprocessing'].get('remove_duplicates', True):
            df = df.drop_duplicates(subset=['text'])
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def load_fakenewsnet(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load FakeNewsNet dataset with propagation features"""
        data_path = self.data_dir / "fakenewsnet"
        
        if not data_path.exists():
            print("FakeNewsNet not found. Please download from:")
            print("https://github.com/KaiDMML/FakeNewsNet")
            print(f"Place in {data_path}")
            raise FileNotFoundError("FakeNewsNet not found")
        
        # Load data (simplified version)
        # In practice, you'd parse the full FakeNewsNet structure
        articles = []
        labels = []
        propagation_features = []
        
        for label_dir in ['fake', 'real']:
            label = 1 if label_dir == 'fake' else 0
            label_path = data_path / label_dir
            
            if label_path.exists():
                for article_file in label_path.glob('*.json'):
                    with open(article_file, 'r') as f:
                        data = json.load(f)
                        articles.append(data.get('text', ''))
                        labels.append(label)
                        # Extract propagation features
                        prop_feat = self._extract_propagation_features(data)
                        propagation_features.append(prop_feat)
        
        df = pd.DataFrame({
            'text': articles,
            'label': labels
        })
        
        # Preprocess
        df['text'] = self.preprocessor.preprocess_batch(df['text'].tolist())
        
        propagation_features = np.array(propagation_features)
        
        return df, propagation_features
    
    def _extract_propagation_features(self, data: Dict) -> np.ndarray:
        """Extract propagation features from article data"""
        # Simplified feature extraction
        # In practice, extract: repost counts, temporal gaps, user influence, etc.
        features = np.zeros(self.config['diffusion']['temporal_features'])
        
        if 'retweets' in data:
            features[0] = len(data['retweets'])
        if 'likes' in data:
            features[1] = data.get('likes', 0)
        # Add more features as needed
        
        return features
    
    def create_dataloaders(self, seed: int = 42) -> Dict[str, DataLoader]:
        """Create train, validation, and test dataloaders"""
        dataset_name = self.config['dataset']['name']
        
        if dataset_name == 'fake_real_news':
            df = self.load_fake_real_news()
            propagation_features = None
        elif dataset_name == 'fakenewsnet':
            df, propagation_features = self.load_fakenewsnet()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Split data
        train_split = self.config['dataset']['train_split']
        val_split = self.config['dataset']['val_split']
        
        # First split: train and temp
        train_df, temp_df = train_test_split(
            df, 
            train_size=train_split, 
            random_state=seed,
            stratify=df['label']
        )
        
        # Second split: val and test
        val_size = val_split / (1 - train_split)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size,
            random_state=seed,
            stratify=temp_df['label']
        )
        
        print(f"Dataset split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Create datasets
        max_length = self.config['model']['max_length']
        batch_size = self.config['training']['batch_size']
        
        if propagation_features is None:
            train_dataset = FakeRealNewsDataset(
                train_df['text'].tolist(),
                train_df['label'].tolist(),
                self.bert_tokenizer,
                self.gpt_tokenizer,
                max_length
            )
            val_dataset = FakeRealNewsDataset(
                val_df['text'].tolist(),
                val_df['label'].tolist(),
                self.bert_tokenizer,
                self.gpt_tokenizer,
                max_length
            )
            test_dataset = FakeRealNewsDataset(
                test_df['text'].tolist(),
                test_df['label'].tolist(),
                self.bert_tokenizer,
                self.gpt_tokenizer,
                max_length
            )
        else:
            # Split propagation features accordingly
            train_indices = train_df.index
            val_indices = val_df.index
            test_indices = test_df.index
            
            train_dataset = FakeNewsNetDataset(
                train_df['text'].tolist(),
                train_df['label'].tolist(),
                propagation_features[train_indices],
                self.bert_tokenizer,
                self.gpt_tokenizer,
                max_length
            )
            val_dataset = FakeNewsNetDataset(
                val_df['text'].tolist(),
                val_df['label'].tolist(),
                propagation_features[val_indices],
                self.bert_tokenizer,
                self.gpt_tokenizer,
                max_length
            )
            test_dataset = FakeNewsNetDataset(
                test_df['text'].tolist(),
                test_df['label'].tolist(),
                propagation_features[test_indices],
                self.bert_tokenizer,
                self.gpt_tokenizer,
                max_length
            )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # For latency measurement
            shuffle=False,
            num_workers=0
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }