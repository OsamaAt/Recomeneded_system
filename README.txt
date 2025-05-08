#🎬 Movie Recommender System
A simple yet smart movie recommendation system using machine learning.
It provides personalized movie suggestions based on previous user ratings.
It includes:

Collaborative Filtering using SVD

An interactive web app built with Streamlit

## ✅ How to Run:

 1 Install required libraries:
pip install -r requirements.txt

 2 Download the MovieLens 100K dataset from:
🔗 https://grouplens.org/datasets/movielens/100k/

 3 Create a data/ folder inside your project and place the following files:
 movie-recommender/
├── app.py
├── data/
│   ├── u.data
│   ├── u.item
│   └── ...

 4 Run the app:
streamlit run app.py

# ❗ Make sure your code uses relative paths like:
pd.read_csv("data/u.data")

# Author : OsamaAt

--------------------------------------------------------------------------------------------------------------------------------------------------------------
# 🎬 نظام توصية الأفلام (Movie Recommender System)

نظام توصية أفلام بسيط وذكي يعتمد على تقنيات تعلم الآلة (Machine Learning)، ويوفر للمستخدم اقتراحات أفلام مناسبة بناءً على تقييماته السابقة.  
يدعم طريقتين:
- التوصية بناءً على تشابه المستخدمين (Collaborative Filtering باستخدام SVD)
- واجهة تفاعلية باستخدام Streamlit

---

## ✅ كيفية التشغيل (عربي)

1. **ثبّت المكتبات المطلوبة**:
```bash
pip install -r requirements.txt

2;حمّل بيانات MovieLens 100K من الرابط التالي:
🔗 https://grouplens.org/datasets/movielens/100k/

3 قم بإنشاء مجلد data/ 
داخل مشروعك وضع الملفات التالية
movie-recommender/
├── app.py
├── data/
│   ├── u.data
│   ├── u.item
│   └── ...

4 شغّل التطبيق:
streamlit run app.py

❗ تأكد أن المسارات داخل الكود تستخدم:
pd.read_csv("data/u.data")

# المؤلف : OsamaAt
