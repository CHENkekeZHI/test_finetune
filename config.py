import os

# 模型配置
model_config = {
    "model": {
        "type": "InternVL_V1_5",
        "model_path": "./InternVL2-2B",  # 下载模型的路径
        "freeze_llm": True,  # 冻结 LLM
        "freeze_visual_encoder": True,  # 冻结视觉编码器
        "quantization_llm": True,  # 对 LLM 进行量化
        "quantization_vit": False,  # 不量化视觉编码器
        "llm_lora": {
            "type": "LoraConfig",
            "r": 128,
            "lora_alpha": 256,
            "lora_dropout": 0.05,
            "task_type": "CAUSAL_LM"
        }
    },
    "training": {
        "batch_size": 16,
        "epochs": 3,
        "learning_rate": 5e-5,
        "optimizer": "AdamW",
        "max_seq_length": 128
    }
}

# 输出路径
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

