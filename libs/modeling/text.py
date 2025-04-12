import torch
import torch.nn as nn
from transformers import CLIPTokenizer  # pip install transformers==4.19.2
from .clip import CLIPModel

def data_split_dir(file, data_split, mode, split_num):
        if mode == 'train':
            file_dir = file.format(mode='train',r1=data_split, r2=100-data_split, r3=split_num)
        else:
            file_dir = file.format(mode='test',r1=data_split, r2=100-data_split, r3=split_num)
        return file_dir

class TextFeatures(nn.Module):
    def __init__(
        self,
        model_path,
        subset_file,
        data_split,
        emb_dim,
        split_num,
        freeze_txt_model=True,
    ):
        super().__init__()
        self.train_classes = self._load_classes(subset_file, data_split, 'train', split_num)
        self.test_classes = self._load_classes(subset_file, data_split, 'test', split_num)

        self.tokenizer = CLIPTokenizer.from_pretrained(model_path)
        self.txt_model = CLIPModel.from_pretrained(model_path, ignore_mismatched_sizes=True).float()

        if isinstance(self.txt_model.text_projection, nn.Linear):
            self.txt_model.text_projection = nn.Linear(self.txt_model.text_embed_dim, emb_dim, bias=False)
            nn.init.normal_(self.txt_model.text_projection.weight, std=self.txt_model.text_embed_dim ** -0.5)
        else:
            self.txt_model.text_projection = nn.Parameter(torch.empty(self.txt_model.text_embed_dim, emb_dim))
            nn.init.normal_(self.txt_model.text_projection, std=self.txt_model.text_embed_dim ** -0.5)


        if freeze_txt_model:
            self.txt_model.requires_grad_(False)
            self.txt_model.text_projection.requires_grad_(True)

    
    def extract_text_emb(self, cls_name, is_prompt=False):
        if is_prompt:
            train_prompt = self.get_prompt(cls_name)
        else:
            train_prompt = cls_name 
        device = next(iter(self.txt_model.parameters())).device
        texts = self.tokenizer(train_prompt, padding=True, return_tensors="pt").to(device)

        if hasattr(self.txt_model, 'get_text_features'):
            text_emb = self.txt_model.get_text_features(**texts)
        else:
            text_emb = self.txt_model.encode_text(train_prompt)
            
        return text_emb
        
    def txt_read(self, file_path, sort=False):
        with open(file_path, 'r') as f:
            cls_name = [cls_name.strip('\n') for cls_name in f.readlines()]

        if sort:
            cls_name = sorted(cls_name)
        
        split_dict = {cls_name: i for i, cls_name in enumerate(cls_name)}

        return split_dict

    def get_prompt(self, cls_name):
        prompt_cls_name = [f'a video of action {c}' for c in cls_name]
        
        return prompt_cls_name
    
    def _load_classes(self, subset_file, data_split, mode, split_num):
        file_path = data_split_dir(subset_file, data_split, mode, split_num)
        try:
            with open(file_path, 'r') as f:
                cls_name = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            raise ValueError(f"No class file: {file_path}")

        return cls_name

    def forward(self, batch_size, mode):
        
        cls_name = self.train_classes if mode == 'train' else self.test_classes
        text_emb = self.extract_text_emb(cls_name, None) 

        if len(text_emb.size()) == 2:
            text_emb = text_emb.expand(batch_size, -1, -1)
        
        split_num_cls = text_emb.size(1)

        return text_emb, split_num_cls