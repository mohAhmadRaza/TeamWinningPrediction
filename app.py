# 🧠 Libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import gradio as gr
import kagglehub

# 🎯 Step 1: Load Dataset
path = kagglehub.dataset_download("wyattowalsh/basketball")
csv_folder = os.path.join(path, "csv")
df = pd.read_csv(os.path.join(csv_folder, "game.csv"))

# 🧹 Step 2: Prepare Data
home_df = df[[
    'game_id', 'team_abbreviation_home', 'wl_home', 'fg3_pct_home', 'tov_home',
    'reb_home', 'ast_home', 'pts_home', 'stl_home', 'blk_home'
]].copy()
home_df.columns = ['game_id', 'team', 'win_label', 'fg3_pct', 'tov', 'reb', 'ast', 'pts', 'stl', 'blk']
home_df['is_home'] = 1

away_df = df[[
    'game_id', 'team_abbreviation_away', 'wl_away', 'fg3_pct_away', 'tov_away',
    'reb_away', 'ast_away', 'pts_away', 'stl_away', 'blk_away'
]].copy()
away_df.columns = ['game_id', 'team', 'win_label', 'fg3_pct', 'tov', 'reb', 'ast', 'pts', 'stl', 'blk']
away_df['is_home'] = 0

data = pd.concat([home_df, away_df], ignore_index=True)
data = data.dropna()
data['win'] = data['win_label'].apply(lambda x: 1 if x == 'W' else 0)

features = ['fg3_pct', 'tov', 'reb', 'ast', 'pts', 'stl', 'blk', 'is_home']
X = data[features]
y = data['win']

# 🔍 Step 3: Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 📊 Step 4: Evaluation Metrics

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# Feature Importance
importances = model.feature_importances_
feature_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=feature_df, x='Importance', y='Feature')
plt.title("🏀 What Factors Lead to NBA Wins?")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# 🧠 Step 5: Define Prediction Function
def predict_win(fg3_pct, tov, reb, ast, pts, stl, blk, is_home):
    input_data = {
        'fg3_pct': fg3_pct,
        'tov': tov,
        'reb': reb,
        'ast': ast,
        'pts': pts,
        'stl': stl,
        'blk': blk,
        'is_home': 1 if is_home == "Yes" else 0
    }
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        result = "🏆 WIN — Confidence: {:.2%}".format(prob)
        gif_html = "<img src='https://media.giphy.com/media/l0MYC0LajbaPoEADu/giphy.gif' width='300'>"
    else:
        result = "❌ LOSS — Confidence: {:.2%}".format(prob)
        gif_html = "<img src='https://media.giphy.com/media/l3vR85PnGsBwu1PFK/giphy.gif' width='300'>"
    
    return result, gif_html, "confusion_matrix.png", "feature_importance.png"

# 🎨 Step 6: Gradio Interface

inputs = [
    gr.Slider(0.0, 1.0, value=0.35, label="3-Point % (fg3_pct)"),
    gr.Slider(0, 25, value=12, label="Turnovers (tov)"),
    gr.Slider(20, 60, value=44, label="Rebounds (reb)"),
    gr.Slider(5, 40, value=22, label="Assists (ast)"),
    gr.Slider(60, 150, value=110, label="Points (pts)"),
    gr.Slider(0, 20, value=7, label="Steals (stl)"),
    gr.Slider(0, 15, value=5, label="Blocks (blk)"),
    gr.Radio(["Yes", "No"], label="Is it a home game?")
]

intro = f"""
<style>
body {{
    background-color: #f9fbfd;
    font-family: 'Segoe UI', sans-serif;
}}
</style>
## 🏀 NBA Win Predictor
Welcome! This app uses real NBA game data and machine learning to predict whether a team will **WIN** or **LOSE** based on your input stats.
---
### 👨‍💻 Built by: Ahmad Raza  
- Software Engineer 3.82 CGPA
- AI/ML and Full Stack Developer
- Stanford Section Leader 2025
- NASA's Hackathon Nominee 2024
- 16 Hackathons Participant
- 350+ LeetCode
- 📧 [Linkedin](https://www.linkedin.com/in/ahmad-raza-6667552b1/)
- 📧 [Github](https://www.github.com/in/mohahmadraza/)
---
### 📊 Model Accuracy: **{accuracy:.2%}**
Trained on thousands of NBA games using Random Forests.
"""

# 🖥️ Interface
gr.Interface(
    fn=predict_win,
    inputs=inputs,
    outputs=[
        gr.Text(label="Prediction"),
        gr.HTML(label="Reaction GIF"),  # 👈 Fixed this
        gr.Image(type="filepath", label="Confusion Matrix"),
        gr.Image(type="filepath", label="Feature Importance")
    ],
    title="🏀 NBA Game Outcome Predictor",
    description=intro,
    theme="default"
).launch(share=True)
