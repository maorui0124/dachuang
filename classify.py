import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import tempfile
import os

# 1. 加载数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. 准备数据
train_texts = train_df['TEXT'].tolist()
train_labels = train_df['label'].tolist()
test_texts = test_df['TEXT'].tolist()
test_labels = test_df['label'].tolist()

# 3. 使用BERT的tokenizer将文本转为BERT模型需要的格式
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', mirror='tuna')

def tokenize_function(examples):
    return tokenizer(examples['TEXT'], padding='max_length', truncation=True, max_length=128)

# 转换为huggingface的数据集格式
train_dataset = Dataset.from_dict({"TEXT": train_texts, "label": train_labels})
test_dataset = Dataset.from_dict({"TEXT": test_texts, "label": test_labels})

# Tokenize数据
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 4. 加载BERT模型并准备训练
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# 5. 设置训练参数
# 使用临时目录来避免占用过多的磁盘空间，并禁用检查点保存
training_args = TrainingArguments(
    output_dir=tempfile.mkdtemp(),  # 使用临时目录
    num_train_epochs=3,              # 训练的轮次
    per_device_train_batch_size=16,  # 每个设备上的训练批量大小
    per_device_eval_batch_size=64,   # 每个设备上的评估批量大小
    warmup_steps=500,                # 热身步骤
    weight_decay=0.01,               # 权重衰减
    logging_dir=tempfile.mkdtemp(),  # 临时日志目录
    logging_steps=10,
    evaluation_strategy="epoch",     # 每个epoch结束后进行评估
    save_strategy="epoch",           # 每个epoch保存一次模型
    save_total_limit=2,              # 限制保存的模型个数
#    load_best_model_at_end=True,     # 如果需要，可以加载最好的模型
)

# 6. 创建Trainer对象
trainer = Trainer(
    model=model,                         # 要训练的模型
    args=training_args,                  # 训练参数
    train_dataset=train_dataset,         # 训练数据集
    eval_dataset=test_dataset,           # 测试数据集
    compute_metrics=lambda p: {
        'accuracy': accuracy_score(p.predictions.argmax(axis=-1), p.label_ids)
    }
)

# 7. 开始训练
trainer.train()

# 8. 保存模型
output_model_dir = "saved_model"
model.save_pretrained(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
print(f"模型已保存至 {output_model_dir}")

# 9. 在测试集上进行评估
results = trainer.evaluate()

print("测试集评估结果:", results)

# 10. 预测测试集
predictions = trainer.predict(test_dataset)
pred_labels = predictions.predictions.argmax(axis=-1)

# 11. 输出测试集的分类报告
print("分类报告:")
print(classification_report(test_labels, pred_labels, target_names=['安全', '冒犯']))




