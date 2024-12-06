import os
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import InternLMForCausalLM, InternLMTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

# 加载模型和数据
model_path = './InternVL2-2B'
tokenizer = InternLMTokenizer.from_pretrained(model_path)
model = InternLMForCausalLM.from_pretrained(model_path)

# 数据集
train_dataset = load_dataset("json", data_files={"train": "./processed_vizwiz/train/*.json"})
train_dataloader = DataLoader(train_dataset['train'], batch_size=16, shuffle=True)

# 优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*3)

# 训练循环
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(3):
    model.train()
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
        inputs = tokenizer(batch['image'], padding=True, truncation=True, return_tensors="pt").to(device)
        labels = tokenizer(batch['conversations'][1]['value'], padding=True, truncation=True, return_tensors="pt").to(device)

        outputs = model(**inputs, labels=labels["input_ids"])
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f"Loss: {loss.item()}")

# 保存模型
model.save_pretrained('./output')
tokenizer.save_pretrained('./output')
