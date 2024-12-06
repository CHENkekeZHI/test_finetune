from transformers import InternLMForCausalLM, InternLMTokenizer
import torch

# 加载训练好的模型
model_path = './output'
tokenizer = InternLMTokenizer.from_pretrained(model_path)
model = InternLMForCausalLM.from_pretrained(model_path)

# 测试文本
test_input = "<image>\nWhat color is this?"
inputs = tokenizer(test_input, return_tensors="pt")
inputs = {key: value.to(model.device) for key, value in inputs.items()}

# 生成预测
with torch.no_grad():
    outputs = model.generate(inputs['input_ids'], max_length=50)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result)
