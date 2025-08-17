import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Tạo dataset giả lập
data = {
    'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
    'Production Quantity': [10, 15, 20, 25, 30],
    'Fabric Used (meters)': [50, 75, 100, 125, 150],
    'Wood Used (cubic meters)': [2, 3, 4, 5, 6],
    'Foam Used (kg)': [20, 30, 40, 50, 60],
    'Labor Hours': [100, 150, 200, 250, 300]
}

# Chuyển đổi dữ liệu thành DataFrame
df = pd.DataFrame(data)

# Chọn biến độc lập (features) và biến phụ thuộc (target)
X = df[['Production Quantity']]
y_fabric = df['Fabric Used (meters)']
y_wood = df['Wood Used (cubic meters)']
y_foam = df['Foam Used (kg)']
y_labor = df['Labor Hours']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_fabric_train, y_fabric_test = train_test_split(X, y_fabric, test_size=0.2, random_state=42)
X_train, X_test, y_wood_train, y_wood_test = train_test_split(X, y_wood, test_size=0.2, random_state=42)
X_train, X_test, y_foam_train, y_foam_test = train_test_split(X, y_foam, test_size=0.2, random_state=42)
X_train, X_test, y_labor_train, y_labor_test = train_test_split(X, y_labor, test_size=0.2, random_state=42)

# Tạo mô hình hồi quy tuyến tính
model_fabric = LinearRegression()
model_wood = LinearRegression()
model_foam = LinearRegression()
model_labor = LinearRegression()

# Huấn luyện mô hình
model_fabric.fit(X_train, y_fabric_train)
model_wood.fit(X_train, y_wood_train)
model_foam.fit(X_train, y_foam_train)
model_labor.fit(X_train, y_labor_train)

# Streamlit UI
st.title("Giới thiệu về tôi")
st.image("m1.PNG")
st.image("m2.PNG")
st.image("m3.PNG")
st.image("m4.PNG")
st.image("m5.PNG")
st.image("m6.PNG")
st.image("m7.PNG")
st.image("m8.PNG")
st.subheader("Dữ liệu đầu vào")
st.write("Chúng ta sẽ lấy dữ liệu thực tế sản phẩm làm mẫu hoặc đã làm một vài lần để sự đoán vật tư cho các lô hàng tiếp theo.Dưới đây model nhỏ để dự đoán số vật tư cho ghế sofa.Gồm các cột dữ liệu:Ngày,Sản phẩm,Vải,Gỗ,Foamvà Giờ công")
st.write("Ứng dụng này hướng đến cho người dùng một cách dễ dàng nhất và tiện lợi")
st.title("Dự đoán lượng vật tư cần thiết")
st.write("Nhập số lượng ghế sofa cần sản xuất để dự đoán lượng vật tư cần thiết.")

# Nhập số lượng ghế sofa
quantity = st.number_input("Số lượng ghế sofa:", min_value=1, value=35)

# Dự đoán lượng vật tư cần thiết
if st.button("Dự đoán"):
    new_production = pd.DataFrame({'Production Quantity': [quantity]})
    fabric_needed = model_fabric.predict(new_production)
    wood_needed = model_wood.predict(new_production)
    foam_needed = model_foam.predict(new_production)
    labor_needed = model_labor.predict(new_production)

    st.write(f"### Kết quả dự đoán cho {quantity} ghế sofa:")
    st.write(f"- **Vải cần thiết (mét):** {fabric_needed[0]:.2f}")
    st.write(f"- **Gỗ cần thiết (m³):** {wood_needed[0]:.2f}")
    st.write(f"- **Mút cần thiết (kg):** {foam_needed[0]:.2f}")
    st.write(f"- **Giờ lao động cần thiết:** {labor_needed[0]:.2f}")

# Thêm chữ cảm ơn ở cuối trang
st.markdown("<h2 style='text-align: center;'>𝒯𝒽𝒶𝓃𝓀 𝓎𝑜𝓊!</h2>", unsafe_allow_html=True)

