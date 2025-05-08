
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("demo/train_data.csv")
x = df.drop('ngung_su_dung', axis=1) 
y = df['ngung_su_dung']  

svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(x, y)

# Đọc dữ liệu kiểm tra
test = pd.read_csv("demo/test_data.csv")

# Tách dữ liệu kiểm tra thành đầu vào và nhãn thực tế
xtest = test.drop('ngung_su_dung', axis=1)
ytest = test['ngung_su_dung']

# Đọc dữ liệu huấn luyện và tách thành x và y
df = pd.read_csv('data sau encode.csv')
x = df.drop('ngung_su_dung', axis=1)
y = df['ngung_su_dung']

xtrain, xtest1, ytrain, ytest1 = train_test_split(x, y, test_size=0.3, random_state=42)

predicted = svm_model.predict(xtest)
probabilities = svm_model.predict_proba(xtest)[:, 1] 

accuracy = accuracy_score(ytest, predicted)

f1 = f1_score(ytest, predicted)

cm = confusion_matrix(ytest, predicted)

auc_score = roc_auc_score(ytest, probabilities)
print("\nClassification Report:\n", classification_report(ytest, predicted))

print("\nMa trận nhầm lẫn:")
print(f"{'':<10} {'Không':<10} {'Có':<10}")
print(f"{'Không':<10} {cm[0, 0]:<10} {cm[0, 1]:<10}")
print(f"{'Có':<10} {cm[1, 0]:<10} {cm[1, 1]:<10}")

print("F1 Score:", f1)
print("AUC:", auc_score)

print("Classification Accuracy (CA):", accuracy)


from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Hiển thị ma trận nhầm lẫn trên tập kiểm tra
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Không", "Có"])
disp.plot(cmap='viridis')  # Sử dụng colormap 'viridis' để đồng nhất
plt.show()
