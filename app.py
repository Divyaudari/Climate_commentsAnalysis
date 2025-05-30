# Streamlit Climate Change Sentiment Dashboard App

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import hstack
from wordcloud import WordCloud
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# --- Styling ---
st.set_page_config(page_title="Climate Sentiment Dashboard", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f4f4f4; }
    .reportview-container .markdown-text-container {
        font-family: 'Segoe UI', sans-serif;
    }
    .sidebar .sidebar-content { background-color: #e8f0fe; }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading and Preprocessing ---
df = pd.read_csv("climate_nasa.csv")
df['Date'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['date', 'text'], inplace=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['CleanText'] = df['text'].apply(clean_text)

sid = SentimentIntensityAnalyzer()
df['Sentiment'] = df['CleanText'].apply(lambda x: sid.polarity_scores(x)['compound'])
df['SentimentLabel'] = df['Sentiment'].apply(lambda x: 'positive' if x > 0.05 else 'negative' if x < -0.05 else 'neutral')

def bucket_likes(x):
    if x == 0:
        return 'none'
    elif x <= 2:
        return 'low'
    elif x <= 10:
        return 'medium'
    else:
        return 'high'

df['EngagementLevel'] = df['likesCount'].apply(bucket_likes)


# --- Feature Preparation ---
tfidf = TfidfVectorizer(max_features=300)
X_text = tfidf.fit_transform(df['CleanText'])
X_sentiment = df[['Sentiment']].values
X = hstack([X_text, X_sentiment])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['EngagementLevel'])
y_bin = label_binarize(y, classes=[0, 1, 2, 3])

X_train, X_test, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=0.2, random_state=42)
model = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, y_train_bin)
y_score = model.predict_proba(X_test)
y_pred = model.predict(X_test)
y_pred_labels = label_encoder.inverse_transform(np.argmax(y_pred, axis=1))
y_test_labels = label_encoder.inverse_transform(np.argmax(y_test_bin, axis=1))

# --- Streamlit App Layout ---
st.title("🌍 Climate Change Public Sentiment Dashboard")

menu = st.sidebar.radio("📂 Navigation", ["🏠 Home", "📊 Dashboard", "🔮 Predict Comment"])

if menu == "🏠 Home":
    st.markdown("""
    ## 👋 Welcome to the NASA Climate Sentiment Dashboard
    This interactive tool helps analyze public opinions on climate change.
    - 🌡️ View sentiment trends
    - 💬 Explore comment behavior
    - 🤖 Predict likes and engagement from new comments
    """)

elif menu == "📊 Dashboard":
    st.header("📈 Visual Analytics")
    with st.expander("🎯 Sentiment Distribution"):
        sentiment_counts = df['SentimentLabel'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['green', 'gray', 'red'])
        ax1.axis('equal')
        st.pyplot(fig1)

    with st.expander("📅 Sentiment Over Time"):
        df['Year'] = df['Date'].dt.year
        yearly_sentiment = df.groupby(['Year', 'SentimentLabel']).size().unstack().fillna(0)
        st.bar_chart(yearly_sentiment)

    with st.expander("🗓️ Monthly Comment Volume"):
        df['Month'] = df['Date'].dt.to_period('M').astype(str)
        monthly_comments = df.groupby('Month').size()
        st.line_chart(monthly_comments)

    with st.expander("📉 Likes by Sentiment"):
        fig2, ax2 = plt.subplots()
        sns.boxplot(data=df, x='SentimentLabel', y='likesCount', ax=ax2)
        ax2.set_yscale('log')
        st.pyplot(fig2)

    with st.expander("☁️ Word Clouds"):
        col1, col2 = st.columns(2)
        with col1:
            positive_text = ' '.join(df[df['SentimentLabel'] == 'positive']['CleanText'])
            wordcloud_pos = WordCloud(width=600, height=400, background_color='white').generate(positive_text)
            st.image(wordcloud_pos.to_array(), caption='Positive Comments')
        with col2:
            negative_text = ' '.join(df[df['SentimentLabel'] == 'negative']['CleanText'])
            wordcloud_neg = WordCloud(width=600, height=400, background_color='black', colormap='Reds').generate(negative_text)
            st.image(wordcloud_neg.to_array(), caption='Negative Comments')

    with st.expander("🔡 Top Keywords"):
        count_vect = CountVectorizer(stop_words='english')
        word_matrix = count_vect.fit_transform(df['CleanText'])
        word_freq = word_matrix.toarray().sum(axis=0)
        words = count_vect.get_feature_names_out()
        freq_df = pd.DataFrame({'word': words, 'frequency': word_freq}).sort_values(by='frequency', ascending=False).head(20)
        fig3, ax3 = plt.subplots()
        sns.barplot(data=freq_df, x='frequency', y='word', ax=ax3, palette='viridis')
        st.pyplot(fig3)

    with st.expander("📊 Feature Correlation"):
        fig4, ax4 = plt.subplots()
        sns.heatmap(df[['likesCount', 'commentsCount', 'Sentiment']].corr(), annot=True, cmap='coolwarm', ax=ax4)
        st.pyplot(fig4)

elif menu == "🔮 Predict Comment":
    st.header("🧠 Predict Sentiment & Engagement")
    comment = st.text_area("✏️ Enter a comment about climate change")

    if st.button("Analyze"):
        if not comment.strip():
            st.warning("Please enter a comment.")
        else:
            cleaned = clean_text(comment)
            sentiment_score = sid.polarity_scores(cleaned)['compound']
            sentiment_label = 'positive' if sentiment_score > 0.05 else 'negative' if sentiment_score < -0.05 else 'neutral'
            input_vec = tfidf.transform([cleaned])
            input_final = hstack([input_vec, np.array([[sentiment_score]])])
            prediction = model.predict(input_final)
            engagement = label_encoder.inverse_transform(np.argmax(prediction, axis=1))[0]

            st.success(f"✅ Sentiment: **{sentiment_label}**")
            st.info(f"📈 Predicted Engagement: **{engagement}**")

            st.subheader("📋 Classification Report")
            report = classification_report(y_test_labels, y_pred_labels, target_names=label_encoder.classes_, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

            st.subheader("📊 Confusion Matrix")
            fig_cm, ax_cm = plt.subplots()
            conf_matrix = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax_cm)
            st.pyplot(fig_cm)

            st.subheader("📈 ROC Curve")
            fpr, tpr, roc_auc = {}, {}, {}
            for i in range(len(label_encoder.classes_)):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            fig_roc, ax_roc = plt.subplots()
            for i, color in zip(range(len(label_encoder.classes_)), ['blue', 'green', 'orange', 'red']):
                ax_roc.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')
            ax_roc.plot([0, 1], [0, 1], 'k--')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('ROC Curve')
            ax_roc.legend(loc='lower right')
            st.pyplot(fig_roc)
