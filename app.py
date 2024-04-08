# Load EDA Pkgs
import pandas as pd
import numpy as np

# Load Data Viz Pkgs
import seaborn as sns

# Load Text Cleaning Pkgs
import neattext.functions as nfx

# Load ML Pkgs
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

# Load Dataset
df = pd.read_csv("data/emotion_dataset_raw.csv")

# Data Cleaning
df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)

# Features & Labels
Xfeatures = df['Clean_Text']
ylabels = df['Emotion']

# Split Data
x_train, x_test, y_train, y_test = train_test_split(Xfeatures, ylabels, test_size=0.3, random_state=42)

# LogisticRegression Pipeline
pipe_lr = Pipeline(steps=[('cv', CountVectorizer()), ('lr', LogisticRegression())])

# Train and Fit Data
pipe_lr.fit(x_train, y_train)

# Check Accuracy
accuracy = pipe_lr.score(x_test, y_test)
print("Model Accuracy:", accuracy)

# Save Model & Pipeline
pipeline_file = open("emotion_classifier_pipe_lr.pkl", "wb")
joblib.dump(pipe_lr, pipeline_file)
pipeline_file.close()

# Ask for user input and make a prediction
user_input = input("Give sentence: ")
prediction = pipe_lr.predict([user_input])
print("Predicted Emotion:", prediction[0])
