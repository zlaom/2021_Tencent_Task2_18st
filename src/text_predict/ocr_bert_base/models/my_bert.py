# define our model
import torch.nn as nn
from transformers import AutoModel
class MyBertClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config['model_name'])
        self.classifier = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'], config['output_dim'])
        )
    def forward(self, input_id, attention_mask):
        x = self.bert(input_id, attention_mask).pooler_output
        self.featuremap = x # 核心代码
        x = self.classifier(x)
        return x