import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv("demo/data encode.csv", index_col=0)

# Lọc dữ liệu theo điều kiện
x = df.drop(['ngung_su_dung'], axis=1)

# Bước 1: Tính ma trận tương quan
corr_matrix = x.corr()

# Bước 2: Tạo mask để chỉ hiển thị một nửa heatmap (tam giác trên hoặc dưới)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Tam giác trên
# mask = np.tril(np.ones_like(corr_matrix, dtype=bool))  # Tam giác dưới (nếu cần)

# Bước 3: Vẽ heatmap với mask
plt.figure(figsize=(10, 20))
sns.heatmap(
    corr_matrix, 
    mask=mask,   # Áp dụng mask
    annot=True,  # Hiển thị giá trị
    cmap='coolwarm', 
    fmt='.2f', 
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}  # Điều chỉnh thanh màu
)
plt.title("Heatmap tương quan giữa các biến")
plt.show()
