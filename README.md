# 🌍 Climate Change Sentiment Dashboard

This project analyzes public comments on NASA's climate change Facebook posts using NLP and Machine Learning techniques. It includes a fully interactive **Streamlit dashboard** and a modeling notebook that predicts public sentiment and engagement.

[🔗 Live App]((https://climatecommentsanalysis-c97nhzermwtwsrdbgo54rs.streamlit.app/)  
[📓 View Notebook](./ClimateCommentsAnalysis.ipynb)

---

## 📌 Project Objective

To model and visualize public sentiment on climate change using comments from NASA’s social media, and predict engagement levels (likes) based on comment content.

---

## 🧾 Dataset Summary

- **Source**: NASA Climate Facebook Page  
- **Size**: 504 comments  
- **Fields**: `Date`, `Text`, `LikesCount`, `CommentsCount`, `CleanText`, `Sentiment`, `EngagementLevel`

---

## ⚙️ Tools and Libraries

- Python, Pandas, NumPy
- Scikit-learn, NLTK (VADER), WordCloud
- Seaborn, Matplotlib
- Streamlit

---

## 📊 Dashboard Features

- 🥧 **Sentiment Distribution Pie Chart**
- 📈 **Sentiment Trend Over Time**
- 🗓️ **Monthly Comment Volume**
- 💬 **Likes vs Sentiment Boxplot**
- ☁️ **Word Clouds (positive/negative)**
- 🔡 **Top Keywords Bar Graph**
- 📊 **Correlation Heatmap**
- 🧠 **Live Comment Prediction**
- 📋 **Classification Report**
- 🔄 **Confusion Matrix & ROC Curve**

---

## 🔮 Model Summary

- **Text Vectorizer**: TF-IDF
- **Sentiment Analysis**: VADER
- **Classifier**: RandomForest (multi-class for engagement prediction)
- **Evaluation**: ROC Curve, Confusion Matrix, Classification Report

---

## 🚀 Run Locally

```bash
git clone https://github.com/your-username/climate-change-sentiment-app.git
cd climate-change-sentiment-app
pip install -r requirements.txt
streamlit run app.py
