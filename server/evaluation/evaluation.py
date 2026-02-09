import json
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. CẤU HÌNH & MAPPING ---
# Quy ước nhãn từ Dataset: 0: True, 1: False, 2: Uncertain
LABEL_MAPPING = {
    "ĐÚNG": 0,
    "SAI": 1,
    "KHÔNG ĐỦ THÔNG TIN": 2
}

def map_prediction_to_label(status_text):
    """Chuyển đổi text output của model sang số integer"""
    if not status_text:
        return 2 # Mặc định nếu lỗi là Uncertain
    status_upper = status_text.upper().strip()
    return LABEL_MAPPING.get(status_upper, 2)

# --- 2. LOAD DỮ LIỆU ---

print("-> Đang load Dataset từ HuggingFace...")
# Load dataset (chỉ load split train vì bạn đang test trên đó)
dataset = load_from_disk("../data/vifactcheck-normalized")

print("-> Đang đọc file kết quả JSON...")
# Giả sử bạn lưu nội dung JSON bạn cung cấp vào file 'results.json'
try:
    with open('results.json', 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy file 'results.json'. Hãy tạo file này từ output của bạn.")
    exit()

y_true = [] # Nhãn thực tế
y_pred = [] # Nhãn dự đoán

print(f"-> Đang đối chiếu {len(predictions_data)} mẫu dữ liệu...")

for item in predictions_data:
    try:
        # 1. Lấy ID để tìm Ground Truth
        # JSON ID là 1, 2... nhưng Dataset Index là 0, 1... => index = id - 1
        idx = item['id'] - 1
        
        # Lấy nhãn thực tế từ dataset
        ground_truth = dataset[idx]['labels']
        
        # 2. Lấy dự đoán từ JSON
        # Đường dẫn: output -> response -> status
        if 'output' in item and 'response' in item['output']:
            pred_status = item['output']['response']['status']
            pred_label = map_prediction_to_label(pred_status)
            
            y_true.append(ground_truth)
            y_pred.append(pred_label)
        else:
            print(f"⚠️ Warning: Mẫu ID {item['id']} thiếu trường output/response.")
            
    except IndexError:
        print(f"❌ Lỗi: ID {item['id']} vượt quá số lượng dòng của dataset.")
    except Exception as e:
        print(f"❌ Lỗi xử lý ID {item.get('id', 'Unknown')}: {e}")

# --- 3. TÍNH TOÁN METRICS ---

if not y_true:
    print("Không có dữ liệu để tính toán.")
    exit()

# Tính Accuracy
accuracy = accuracy_score(y_true, y_pred)

# Tính Precision, Recall, F1 (dùng average='weighted' hoặc 'macro' cho bài toán đa lớp)
# 'weighted': Tính trung bình có trọng số theo số lượng mẫu của mỗi lớp (tốt nếu dữ liệu mất cân bằng)
# 'macro': Tính trung bình cộng đơn thuần (coi trọng các lớp như nhau)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

print("\n" + "="*40)
print("   KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH FACT-CHECK")
print("="*40)
print(f"✅ Accuracy  (Độ chính xác tổng): {accuracy:.4f}")
print(f"🎯 Precision (Độ chính xác):      {precision:.4f}")
print(f"🔍 Recall    (Độ bao phủ):        {recall:.4f}")
print(f"⚖️ F1-Score  (Trung hòa P&R):     {f1:.4f}")
print("-" * 40)

# Báo cáo chi tiết từng lớp
target_names = ['SUPPORTED (0)', 'REFUTED (1)', 'UNCERTAIN (2)']
# Lưu ý: Nếu tập test của bạn chỉ có nhãn 0 và 1, report sẽ tự điều chỉnh
unique_labels = sorted(list(set(y_true) | set(y_pred)))
current_names = [target_names[i] for i in unique_labels]

print("\n📊 Báo cáo chi tiết từng lớp:")
print(classification_report(y_true, y_pred, target_names=current_names, zero_division=0))

# --- 4. VẼ CONFUSION MATRIX (Tùy chọn) ---
try:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred: SUP', 'Pred: REF', 'Pred: NEI'],
                yticklabels=['True: SUP', 'True: REF', 'True: NEI'])
    plt.title('Confusion Matrix')
    plt.ylabel('Thực tế (Ground Truth)')
    plt.xlabel('Dự đoán (Prediction)')
    
    # Lưu ảnh thay vì show nếu chạy trên server
    plt.savefig('confusion_matrix.png')
    print("🖼️ Đã lưu biểu đồ nhầm lẫn vào 'confusion_matrix.png'")
    # plt.show() 
except Exception as e:
    print(f"Không thể vẽ biểu đồ: {e}")