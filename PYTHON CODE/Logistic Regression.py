import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,ConfusionMatrixDisplay,
    roc_auc_score,
    accuracy_score
)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# Đọc dữ liệu huấn luyện và tách thành x và y
df = pd.read_csv('demo/train_data.csv')
x = df.drop('ngung_su_dung', axis=1)
y = df['ngung_su_dung']

# Huấn luyện Logistic Regression
logr = linear_model.LogisticRegression()
logr.fit(x, y)

# Đọc dữ liệu kiểm tra
test = pd.read_csv("demo/test_data.csv")

# Tách dữ liệu kiểm tra thành đầu vào và nhãn thực tế
xtest = test.drop('ngung_su_dung', axis=1)
ytest = test['ngung_su_dung']

# Thực hiện dự đoán
predicted = logr.predict(xtest)
probabilities = logr.predict_proba(xtest)[:, 1]  # Xác suất dự đoán cho lớp 1


# In classification report
print("Classification Report:")
print(classification_report(ytest, predicted))

# Tính confusion matrix dạng số lượng và chuyển sang %
cm = confusion_matrix(ytest, predicted)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# In ma trận nhầm lẫn theo định dạng bảng tùy chỉnh (dạng số lượng)
print("\nConfusion Matrix:")
print(f"{'':<10} {'Không':<10} {'Có':<10}")
print(f"{'Không':<10} {cm[0, 0]:<10} {cm[0, 1]:<10}")
print(f"{'Có':<10} {cm[1, 0]:<10} {cm[1, 1]:<10}")
# Tính và in AUC
auc = roc_auc_score(ytest, probabilities)
print(f"\nAUC: {auc:.2f}")

# Tính và in CA (Classification Accuracy)
accuracy = accuracy_score(ytest, predicted)
print(f"Classification Accuracy (CA): {accuracy * 100:.2f}%")
# Hiển thị ma trận nhầm lẫn trên tập kiểm tra
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Không", "Có"])
disp.plot(cmap='viridis')  # Sử dụng colormap 'viridis' để đồng nhất
plt.show()