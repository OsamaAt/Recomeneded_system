# 🎬 Movie Recommender System

This project is a smart movie recommender system using machine learning and collaborative filtering techniques.  
It suggests movies to users based on their past ratings.

---

## ⚙️ Features

- Personalized movie recommendations using SVD (Collaborative Filtering)
- Interactive user interface with Streamlit
- Trained on MovieLens 100K dataset

---

## 🔧 Requirements

- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, scikit-surprise, streamlit

---

## 📥 How to Run the Project

### 1. Install dependencies:
```bash
pip install -r requirements.txt
````

### 2. Download the dataset:

Download [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) and place the files in a folder named `data/` inside the project folder.

### 3. Project directory structure:

```
movie-recommender/
├── app.py
├── model_code.ipynb
├── data/
│   ├── u.data
│   ├── u.item
│   └── ...
├── requirements.txt
└── README.md
```

### 4. Run the app:

```bash
streamlit run app.py
```

---

## 📌 Notes

* Make sure your code uses relative paths like:

```python
pd.read_csv("data/u.data")
```

* If the `data/` folder is missing, the app will show an error message.

---

## 🧾 License

This project is licensed under the MIT License.

# Author : OsamaAt

`````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
## ✅ . **النسخة العربية –
````markdown
# 🎬 نظام توصية الأفلام

هذا المشروع هو نظام توصية أفلام ذكي يعتمد على تقنيات الذكاء الاصطناعي وتعلم الآلة، يقدّم اقتراحات للمستخدم بناءً على تقييماته السابقة. يستخدم طريقتين شهيرتين:

- **التوصية بناءً على تشابه المستخدمين (Collaborative Filtering - SVD)**
- **واجهة استخدام تفاعلية باستخدام Streamlit**

---

## ⚙️ المميزات

- توصية أفلام مخصصة لكل مستخدم
- واجهة سهلة وبسيطة
- نموذج SVD مدرّب على بيانات MovieLens 100K

---

## 🔧 المتطلبات

- Python 3.8 أو أحدث
- المكتبات: pandas, numpy, scikit-learn, scikit-surprise, streamlit

---

## 📥 طريقة تشغيل المشروع

### 1. تثبيت المكتبات المطلوبة:
```bash
pip install -r requirements.txt
````

### 2. تحميل البيانات:

قم بتحميل [مجموعة بيانات MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) وضع الملفات داخل مجلد باسم `data/` داخل مجلد المشروع.

### 3. هيكل المجلد الصحيح:

```
movie-recommender/
├── app.py
├── model_code.ipynb
├── data/
│   ├── u.data
│   ├── u.item
│   └── ...
├── requirements.txt
└── README.md
```

### 4. تشغيل التطبيق:

```bash
streamlit run app.py
```

---

## 📌 ملاحظات هامة

* تأكد من أن المسارات في الكود تستخدم الشكل التالي:

```python
pd.read_csv("data/u.data")
```

* إذا لم يكن مجلد `data/` موجودًا، ستظهر لك رسالة خطأ في التطبيق.

---

## 🧾 الترخيص

هذا المشروع متاح تحت رخصة MIT.

# المؤلف : OsamaAt
````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
