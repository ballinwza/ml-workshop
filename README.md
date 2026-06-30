# 🚀 Machine Learning Workshop


![FastAPI](https://img.shields.io/badge/fastapi-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![ScikitLearn](https://img.shields.io/badge/sklearn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)




Repository นี้สร้างขึ้นมาเพื่อเรียนรู้การสร้าง, วิเคราะห์ข้อมูล และจำลองการทำงานของ Machine Learning models ผ่าน FastAPI

## 🛠️ Tech Stack

- **Backend:** FastAPI
- **Machine Learning:** Tensorflow, Scikit-learn, Pandas, Numpy, Matplotlib
- **Tools & DevOps:** Docker, GitHub Actions, Pip-tools, Swagger, Render



## 🚀 Getting Started

สามารถเข้าใช้งานผ่าน web browser ได้โดยตรงที่ [https://tr-ml-workshop-2.onrender.com/docs](https://tr-ml-workshop-2.onrender.com/docs) (ต้องรอให้ server run เมื่อเข้าใช้งานครั้งแรก)

## ✨ API Method

อธิบายรายละเอียด API ทั้งหมดที่เปิดให้ใช้งาน:

| Router | Endpoint | Description |
| ----------- | ----------- | ----------- |
| Prediction  | /iris  | ทำนายสายพันธุ์ดอก Iris จาก ความกว้างxยาว กลีบดอก, ความกว้างxยาว กลีบเลี้ยง หน่วย (cm)|
|   | /california-house  | ทำนายราคาบ้านใน California จากรายได้คนบริเวณนั้น, จำนวนห้องนอน, จำนวนห้อง และพิกัดที่ตั้ง  |
|   | /fasion  | ทำนายประเภทเครื่องแต่งกายจากรูปภาพขนาด 28x28 px แบบ grey scale |
| Download  | /fashion-example  | โหลดตัวอย่างรูปภาพที่ใช้ทำนายใน fashion มี 10รูป  |

