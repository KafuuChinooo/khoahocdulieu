#Khai báo thư viện
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, classification_report, confusion_matrix,ConfusionMatrixDisplay, roc_auc_score, accuracy_score

# Đọc dữ liệu
df = pd.read_csv("demo/data encode.csv", index_col=0)

x = df.drop(['ngung_su_dung'], axis='columns')
y = df['ngung_su_dung']
print(f"Kích thước dữ liệu X: {x.shape}")

# Chạy KMeans cho số cụm từ 2 đến 10 và tính silhouette score trung bình
silhouette_scores = []

for num_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(x)
    cluster_labels = kmeans.labels_
    
    # Tính silhouette score trung bình
    silhouette_avg = silhouette_score(x, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    
    print(f"Silhouette Score trung bình cho {num_clusters} cụm: {silhouette_avg:.2f}")
    
    # Vẽ biểu đồ Silhouette cho từng số cụm
    silhouette_values = silhouette_samples(x, cluster_labels)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10
    
    for i in range(num_clusters):
        ith_cluster_silhouette_values = silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = cm.viridis(float(i) / num_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title(f"Silhouette plot for K-means clustering with n_clusters = {num_clusters}\n"
                 f"Average silhouette score: {silhouette_avg:.2f}")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.show()

# Vẽ biểu đồ Silhouette trung bình cho từng số cụm từ 2 đến 10
plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), silhouette_scores, 'bo-')
plt.xlabel('Số lượng cụm')
plt.ylabel('Silhouette Score trung bình')
plt.title('Silhouette Score trung bình cho các số lượng cụm từ 2 đến 10')
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(x)
x['Cluster'] = kmeans.labels_
# Tạo DataFrame mới bao gồm cột target 'ngung_su_dung' và nhãn cụm

df['du_doan'] = x['Cluster']

accuracy = accuracy_score(y, x['Cluster'])*100
print("Độ chính xác: ", accuracy)
accuracy_row = pd.DataFrame([[f"Độ chính xác: {accuracy:.2f}%"] + [""] * (df.shape[1] - 1)], 
                            columns=df.columns)
accuracy1 = pd.concat([accuracy_row, df], ignore_index=True)
accuracy1.to_csv('data_co_du_doan.csv', index=False, encoding='utf-8-sig')
print(num_clusters)
