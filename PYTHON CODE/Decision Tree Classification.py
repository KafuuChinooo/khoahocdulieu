import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, 
    accuracy_score, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt


# Đọc dữ liệu huấn luyện và tách thành x và y
df = pd.read_csv('data sau encode.csv')
x = df.drop('ngung_su_dung', axis=1)
y = df['ngung_su_dung']

xtrain, xtest1, ytrain, ytest1 = train_test_split(x, y, test_size=0.3, random_state=42)

# Xây dựng và đánh giá mô hình
model = DecisionTreeClassifier()
model = model.fit(xtrain, ytrain)

# Dự đoán trên tập kiểm tra (xtest)
Treepred = model.predict(xtest1)  
Treeprob = model.predict_proba(xtest1)[:, 1]

# In classification report
print("Classification Report:")
print(classification_report(ytest1, Treepred))

# Tính confusion matrix dạng số lượng và chuyển sang %
cm = confusion_matrix(ytest1, Treepred)

# In ma trận nhầm lẫn theo định dạng bảng tùy chỉnh (dạng số lượng)
print("\nMa trận nhầm lẫn:")
print(f"{'':<10} {'Không':<10} {'Có':<10}")
print(f"{'Không':<10} {cm[0, 0]:<10} {cm[0, 1]:<10}")
print(f"{'Có':<10} {cm[1, 0]:<10} {cm[1, 1]:<10}")

# Tính và in AUC
auc = roc_auc_score(ytest1, Treeprob)
print(f"\nAUC: {auc:.2f}")

# Tính và in CA (Classification Accuracy)
accuracy = accuracy_score(ytest1, Treepred)
print(f"Classification Accuracy (CA): {accuracy * 100:.2f}%")

# Hiển thị ma trận nhầm lẫn trên tập kiểm tra
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Không", "Có"])
disp.plot(cmap='viridis')  # Sử dụng colormap 'viridis' để đồng nhất
plt.show()