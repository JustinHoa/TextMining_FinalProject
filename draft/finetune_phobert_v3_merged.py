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
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn

# ============================================================
# 1. CẤU HÌNH
# ============================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# 2. LOAD DỮ LIỆU
# ============================================================
dataset_id = "hihihohohehe/vifactcheck-normalized"
print(f"\nLoading dataset: {dataset_id}")
dataset = load_dataset(dataset_id)

# ============================================================
# 3. CHIẾN LƯỢC GỘP LABELS
# ============================================================
print("\n" + "="*60)
print("APPLYING LABEL MERGING STRATEGY")
print("="*60)

# Mapping gộp labels theo chiến lược đề xuất
LABEL_MERGE_MAPPING = {
    # Nhóm 1: Kinh tế & Thị trường
    "BẤT ĐỘNG SẢN": "KINH TẾ",  # Gộp vào KINH TẾ
    
    # Nhóm 2: Văn hóa - Giải trí - Đời sống
    "DU LỊCH": "VĂN HÓA",  # Du lịch thường gắn với văn hóa
    "GIỚI TRẺ": "XÃ HỘI ĐỜI SỐNG",  # Lối sống giới trẻ
    
    # Nhóm 3: Chính trị - Xã hội - Thời sự
    "QUÂN SỰ": "THẾ GIỚI",  # Quân sự thường là tin quốc tế
    "THỜI SỰ": "XÃ HỘI ĐỜI SỐNG",  # Thời sự tổng hợp -> Xã hội
    
    # Các labels giữ nguyên
    "CHÍNH TRỊ": "CHÍNH TRỊ",
    "GIÁO DỤC": "GIÁO DỤC",
    "GIẢI TRÍ": "GIẢI TRÍ",
    "KHOA HỌC CÔNG NGHỆ": "KHOA HỌC CÔNG NGHỆ",
    "KINH TẾ": "KINH TẾ",
    "PHÁP LUẬT": "PHÁP LUẬT",
    "SỨC KHỎE": "SỨC KHỎE",
    "THẾ GIỚI": "THẾ GIỚI",
    "THỂ THAO": "THỂ THAO",
    "VĂN HÓA": "VĂN HÓA",
    "XÃ HỘI ĐỜI SỐNG": "XÃ HỘI ĐỜI SỐNG",
}

print("\nLabel merging strategy:")
print("  BẤT ĐỘNG SẢN (17) → KINH TẾ")
print("  DU LỊCH (20) → VĂN HÓA")
print("  GIỚI TRẺ (27) → XÃ HỘI ĐỜI SỐNG")
print("  QUÂN SỰ (15) → THẾ GIỚI")
print("  THỜI SỰ (115) → XÃ HỘI ĐỜI SỐNG")

def merge_labels(example):
    """Gộp labels theo mapping"""
    old_label = example["New Topic 1"]
    new_label = LABEL_MERGE_MAPPING.get(old_label, old_label)
    example["Merged Topic"] = new_label
    return example

print("\nApplying label merging...")
dataset = dataset.map(merge_labels)

# Tạo label mapping mới
unique_labels = sorted(list(set(dataset['train']['Merged Topic'])))
num_labels = len(unique_labels)
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

print(f"\n✅ After merging: {num_labels} classes")
print(f"Labels: {unique_labels}\n")

# ============================================================
# 4. THÊM CỘT LABEL
# ============================================================
def add_label_column(example):
    example['label'] = label2id[example['Merged Topic']]
    return example

dataset = dataset.map(add_label_column)

# Kiểm tra distribution
from collections import Counter
train_label_dist = Counter(dataset['train']['label'])
print("Train label distribution:")
for label_id in sorted(train_label_dist.keys()):
    label_name = id2label[label_id]
    count = train_label_dist[label_id]
    print(f"  {label_id:2d}. {label_name:25s}: {count:4d} samples")

# ============================================================
# 5. CLASS WEIGHTS CHO IMBALANCED DATA
# ============================================================
print("\n🔧 Computing class weights...")
train_labels = np.array(dataset['train']['label'])
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)
print(f"Class weights computed: {class_weights.round(2)}")

# ============================================================
# 6. CUSTOM TRAINER VỚI WEIGHTED LOSS + LABEL SMOOTHING
# ============================================================
class WeightedLabelSmoothingTrainer(Trainer):
    def __init__(self, label_smoothing=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Weighted Cross Entropy với Label Smoothing
        loss_fct = nn.CrossEntropyLoss(
            weight=class_weights_tensor,
            label_smoothing=self.label_smoothing
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# ============================================================
# 7. TOKENIZER & PREPROCESSING
# ============================================================
LOCAL_MODEL_PATH = "./phobert-base-v2"
if os.path.exists(LOCAL_MODEL_PATH):
    model_ckpt = LOCAL_MODEL_PATH
    print(f"\n✅ Loading LOCAL model: {model_ckpt}")
else:
    model_ckpt = "vinai/phobert-base-v2"
    print(f"\n⬇️ Loading HuggingFace model: {model_ckpt}")

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def preprocess_function(examples):
    return tokenizer(
        examples["Statement"], 
        truncation=True,
        max_length=256,
    )

print("Tokenizing dataset...")
columns_to_remove = [col for col in dataset['train'].column_names if col != 'label']
tokenized_datasets = dataset.map(
    preprocess_function, 
    batched=True,
    remove_columns=columns_to_remove
)

print(f"Tokenized dataset: {tokenized_datasets}")
print(f"Columns: {tokenized_datasets['train'].column_names}")

# ============================================================
# 8. METRICS
# ============================================================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    acc = accuracy_score(labels, predictions)
    f1_weighted = f1_score(labels, predictions, average='weighted')
    f1_macro = f1_score(labels, predictions, average='macro')
    
    return {
        "accuracy": acc,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
    }

# ============================================================
# 9. KHỞI TẠO MODEL
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
# 10. TRAINING ARGUMENTS - TỐI ƯU HÓA
# ============================================================
output_dir = f"./draft/finetuned-phobert-merged-{num_labels}classes"

training_args = TrainingArguments(
    output_dir=output_dir,
    
    # Learning rate schedule
    learning_rate=2e-5,  # Tăng lên vì ít classes hơn
    warmup_ratio=0.15,  # Warmup 15% để ổn định
    lr_scheduler_type="cosine",  # Cosine decay
    
    # Batch size
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,  # Effective batch = 32
    
    # Epochs
    num_train_epochs=20,  # Tăng lên 20 epochs
    
    # Regularization
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    # Evaluation & Saving
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    
    # Mixed Precision
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

print(f"\n📋 Training Configuration:")
print(f"  - Number of classes: {num_labels}")
print(f"  - Learning rate: {training_args.learning_rate}")
print(f"  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  - Epochs: {training_args.num_train_epochs}")
print(f"  - Warmup ratio: {training_args.warmup_ratio}")
print(f"  - LR scheduler: {training_args.lr_scheduler_type}")
print(f"  - Label smoothing: 0.1")
print(f"  - Class weighting: Enabled")
print(f"  - FP16: {training_args.fp16}")
print(f"  - Output dir: {output_dir}\n")

# ============================================================
# 11. DATA COLLATOR
# ============================================================
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True
)

# ============================================================
# 12. TRAINER
# ============================================================
trainer = WeightedLabelSmoothingTrainer(
    label_smoothing=0.1,  # Label smoothing = 0.1
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ============================================================
# 13. TRAINING
# ============================================================
print("=" * 60)
print("🚀 STARTING TRAINING WITH OPTIMIZATIONS")
print("=" * 60)
print("Optimizations applied:")
print("  ✅ Label merging (16 → 11 classes)")
print("  ✅ Class weighting")
print("  ✅ Label smoothing (0.1)")
print("  ✅ Cosine LR schedule")
print("  ✅ Gradient accumulation")
print("=" * 60 + "\n")

trainer.train()

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETED!")
print("=" * 60)

# ============================================================
# 14. EVALUATION ON TEST SET
# ============================================================
print("\n" + "=" * 60)
print("📊 EVALUATING ON TEST SET...")
print("=" * 60)

test_results = trainer.evaluate(tokenized_datasets["test"])
print(f"\n===== TEST METRICS =====")
print(f"Accuracy    : {test_results['eval_accuracy']:.4f}")
print(f"F1 Weighted : {test_results['eval_f1_weighted']:.4f}")
print(f"F1 Macro    : {test_results['eval_f1_macro']:.4f}")

# Detailed classification report
print("\n===== CLASSIFICATION REPORT =====")
predictions = trainer.predict(tokenized_datasets["test"])
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

print(classification_report(
    y_true, 
    y_pred, 
    target_names=[id2label[i] for i in range(num_labels)],
    digits=4,
    zero_division=0
))

# Per-class analysis
print("\n===== PER-CLASS ANALYSIS =====")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
for i in range(num_labels):
    total = cm[i].sum()
    correct = cm[i, i]
    acc = correct / total if total > 0 else 0
    print(f"{id2label[i]:25s}: {correct:3d}/{total:3d} = {acc:.2%}")

# ============================================================
# 15. SAVE FINAL MODEL
# ============================================================
final_model_path = f"./draft/final_phobert_merged_{num_labels}classes"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)

# Lưu label mapping
import json
mapping_path = f"{final_model_path}/label_mapping.json"
with open(mapping_path, 'w', encoding='utf-8') as f:
    json.dump({
        "label2id": label2id,
        "id2label": id2label,
        "merge_mapping": LABEL_MERGE_MAPPING
    }, f, ensure_ascii=False, indent=2)

print(f"\n✅ Model saved to: {final_model_path}")
print(f"✅ Label mapping saved to: {mapping_path}")

# ============================================================
# 16. TEST INFERENCE
# ============================================================
print("\n" + "=" * 60)
print("🧪 TESTING INFERENCE...")
print("=" * 60)

model = AutoModelForSequenceClassification.from_pretrained(final_model_path)
model.to(device)
model.eval()

def predict_topic(text, return_probs=False):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=256
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_idx].item()
    
    if return_probs:
        return id2label[pred_idx], confidence, probs[0].cpu().numpy()
    return id2label[pred_idx], confidence

# Test samples covering all merged classes
test_samples = [
    "Bộ Giáo dục công bố điểm chuẩn đại học năm 2024",
    "Đội tuyển Việt Nam giành chiến thắng 3-0 trước Thái Lan",
    "Giá vàng hôm nay tăng mạnh lên mức cao nhất trong năm",
    "Khoa học gia phát hiện loại virus mới gây nguy hiểm",
    "Chung cư mới mở bán tại quận 2 với giá ưu đãi",  # BĐS → KINH TẾ
    "Lễ hội hoa xuân 2024 sẽ được tổ chức tại Hà Nội",  # Du lịch → VĂN HÓA
    "Giới trẻ Sài Gòn đổ xô đi check-in quán cà phê mới",  # Giới trẻ → XHĐ
    "Tướng Mỹ cảnh báo về căng thẳng ở Biển Đông",  # Quân sự → THẾ GIỚI
    "Tai nạn giao thông nghiêm trọng trên cao tốc",  # Thời sự → XHĐ
    "Quốc hội thông qua luật mới về đất đai",  # Chính trị
]

print("\nSample predictions:")
for i, text in enumerate(test_samples, 1):
    pred, conf = predict_topic(text)
    print(f"\n{i}. Text: {text}")
    print(f"   Predicted: {pred} (confidence: {conf:.2%})")

print("\n" + "=" * 60)
print("✅ DONE!")
print("=" * 60)
print(f"\nFinal model saved at: {final_model_path}")
print(f"Number of classes: {num_labels}")
print(f"Test accuracy: {test_results['eval_accuracy']:.2%}")
print(f"Test F1 (weighted): {test_results['eval_f1_weighted']:.2%}")
