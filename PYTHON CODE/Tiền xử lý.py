import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
df = pd.read_csv('New folder\WA_Fn-UseC_-Telco-Customer-Churn.csv')
print("Kích thước dữ liệu:", df.shape)
print(df.isnull().sum())
# Xóa các hàng có giá trị NaN trong 'tong_chi_phi'
df = df.dropna(subset=['tong_chi_phi'])
print(df.shape)

# Xóa cột 'IDKhachHang'
df = df.drop('IDKhachHang', axis=1)

# Tìm và mã hóa tất cả các cột không phải số
label_encoders = {}  # Lưu các bộ mã hóa nếu cần sử dụng lại
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Lưu bộ mã hóa cho cột này (nếu cần giải mã sau)

df.to_csv("demo/data encode.csv")
# Chuẩn hóa các cột số
cot_num = ['so_thang_su_dung', 'phi_hang_thang', 'tong_chi_phi']
scaler = StandardScaler()
df[cot_num] = scaler.fit_transform(df[cot_num])



# Kiểm tra kết quả
print(df.head())
df.to_csv('demo/data sau encode chuan hoa.csv')
# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_data, test_data = train_test_split(df, test_size=0.3, random_state=42)

# Lưu dữ liệu thành các tệp CSV
train_data.to_csv('demo/train_data.csv', index=False)
test_data.to_csv('demo/test_data.csv', index=False)

print("hoàn tất")

