import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ============================================================
# 1. CẤU HÌNH
# ============================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Kiểm tra GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# 2. LOAD & CHUẨN BỊ DỮ LIỆU
# ============================================================
dataset_id = "hihihohohehe/vifactcheck-normalized"
print(f"\nLoading dataset: {dataset_id}")
dataset = load_dataset(dataset_id)

# Chọn số lượng classes: 16 hoặc 8
# Thử với 16 classes trước (New Topic 1)
USE_16_CLASSES = True  # Đổi thành False để dùng 8 classes

if USE_16_CLASSES:
    topic_column = "New Topic 1"
    print("\n🎯 Using 16 classes (New Topic 1)")
else:
    topic_column = "New Topic 2"
    print("\n🎯 Using 8 classes (New Topic 2)")

# Tạo label mapping
unique_labels = sorted(list(set(dataset['train'][topic_column])))
num_labels = len(unique_labels)
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

print(f"Number of labels: {num_labels}")
print(f"Labels: {unique_labels}\n")

# ============================================================
# 3. THÊM CỘT LABEL VÀO DATASET (QUAN TRỌNG!)
# ============================================================
def add_label_column(example):
    """Map topic string sang label integer"""
    example['label'] = label2id[example[topic_column]]
    return example

print("Adding label column to dataset...")
dataset = dataset.map(add_label_column)

# Kiểm tra distribution
from collections import Counter
train_label_dist = Counter(dataset['train']['label'])
print(f"Train label distribution: {dict(sorted(train_label_dist.items()))}")

# ============================================================
# 4. TOKENIZER & PREPROCESSING
# ============================================================
# Ưu tiên load local model nếu có
LOCAL_MODEL_PATH = "./phobert-base-v2"
if os.path.exists(LOCAL_MODEL_PATH):
    model_ckpt = LOCAL_MODEL_PATH
    print(f"\n✅ Loading LOCAL model: {model_ckpt}")
else:
    model_ckpt = "vinai/phobert-base-v2"
    print(f"\n⬇️ Loading HuggingFace model: {model_ckpt}")

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def preprocess_function(examples):
    """
    Tokenize với dynamic padding (không pad đến max_length ngay)
    Sẽ pad trong batch khi training (hiệu quả hơn)
    """
    return tokenizer(
        examples["Statement"], 
        truncation=True,
        max_length=256,
        # KHÔNG dùng padding="max_length" ở đây!
    )

print("Tokenizing dataset...")
# QUAN TRỌNG: Chỉ xóa các cột không cần, GIỮ LẠI 'label'
columns_to_remove = [col for col in dataset['train'].column_names if col != 'label']
tokenized_datasets = dataset.map(
    preprocess_function, 
    batched=True,
    remove_columns=columns_to_remove
)

print(f"Tokenized dataset: {tokenized_datasets}")
print(f"Columns in tokenized dataset: {tokenized_datasets['train'].column_names}")

# ============================================================
# 5. ĐỊNH NGHĨA METRICS
# ============================================================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        "accuracy": acc,
        "f1": f1
    }

# ============================================================
# 6. KHỞI TẠO MODEL
# ============================================================
print(f"\nLoading model: {model_ckpt}")
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt, 
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# ============================================================
# 7. TRAINING ARGUMENTS
# ============================================================
output_dir = f"./draft/finetuned-phobert-{num_labels}classes"

training_args = TrainingArguments(
    output_dir=output_dir,
    
    # Learning rate thấp hơn cho Vietnamese
    learning_rate=5e-6,  # Giảm từ 2e-5 xuống 5e-6
    
    # Batch size
    per_device_train_batch_size=16,  # Giảm nếu bị OOM
    per_device_eval_batch_size=32,
    
    # Epochs
    num_train_epochs=10,  # Tăng lên 10 epochs
    
    # Regularization
    weight_decay=0.01,
    warmup_ratio=0.1,  # Warmup 10% steps đầu
    
    # Evaluation & Saving
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,  # Chỉ giữ 2 checkpoints tốt nhất
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    
    # Mixed Precision (chỉ bật nếu có GPU)
    fp16=torch.cuda.is_available(),
    
    # Data loading
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    
    # Logging
    logging_dir='./logs',
    logging_steps=50,
    logging_strategy="steps",
    
    # Other
    seed=seed,
    report_to="none",
    disable_tqdm=False,
)

print(f"\nTraining arguments:")
print(f"  - Learning rate: {training_args.learning_rate}")
print(f"  - Batch size: {training_args.per_device_train_batch_size}")
print(f"  - Epochs: {training_args.num_train_epochs}")
print(f"  - FP16: {training_args.fp16}")
print(f"  - Output dir: {output_dir}\n")

# ============================================================
# 8. DATA COLLATOR (DYNAMIC PADDING)
# ============================================================
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True  # Dynamic padding trong batch (không cần max_length)
)

# ============================================================
# 9. TRAINER
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ============================================================
# 10. TRAINING
# ============================================================
print("=" * 60)
print("STARTING TRAINING...")
print("=" * 60)

trainer.train()

print("\n" + "=" * 60)
print("TRAINING COMPLETED!")
print("=" * 60)

# ============================================================
# 11. EVALUATION ON TEST SET
# ============================================================
print("\n" + "=" * 60)
print("EVALUATING ON TEST SET...")
print("=" * 60)

test_results = trainer.evaluate(tokenized_datasets["test"])
print(f"\n===== TEST METRICS =====")
print(f"Accuracy : {test_results['eval_accuracy']:.4f}")
print(f"F1-score : {test_results['eval_f1']:.4f}")

# Detailed classification report
print("\n===== CLASSIFICATION REPORT =====")
predictions = trainer.predict(tokenized_datasets["test"])
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

print(classification_report(
    y_true, 
    y_pred, 
    target_names=[id2label[i] for i in range(num_labels)],
    digits=4
))

# ============================================================
# 12. SAVE FINAL MODEL
# ============================================================
final_model_path = f"./draft/final_phobert_{num_labels}classes"
trainer.save_model(final_model_path)
print(f"\n✅ Model saved to: {final_model_path}")

# ============================================================
# 13. TEST INFERENCE
# ============================================================
print("\n" + "=" * 60)
print("TESTING INFERENCE...")
print("=" * 60)

# Load saved model
model = AutoModelForSequenceClassification.from_pretrained(final_model_path)
model.to(device)
model.eval()

def predict_topic(text):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=256
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        pred_idx = torch.argmax(outputs.logits, dim=-1).item()
    
    return id2label[pred_idx]

# Test samples
test_samples = [
    "Bộ Giáo dục công bố điểm chuẩn đại học năm 2024",
    "Đội tuyển Việt Nam giành chiến thắng 3-0 trước Thái Lan",
    "Giá vàng hôm nay tăng mạnh lên mức cao nhất trong năm",
    "Khoa học gia phát hiện loại virus mới gây nguy hiểm",
]

print("\nSample predictions:")
for i, text in enumerate(test_samples, 1):
    pred = predict_topic(text)
    print(f"\n{i}. Text: {text}")
    print(f"   Predicted: {pred}")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
