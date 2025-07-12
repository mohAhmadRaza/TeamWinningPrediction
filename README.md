# 🏀 NBA Win Predictor — Project Repository Guide
<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/d50e2b82-74bd-402c-be53-82338f8ffcd5" />
---

## ✅ 1. 📁 Folder Structure for GitHub

```
nba-win-predictor/
│
├── app.py                     ← Complete code (your Gradio app)
├── confusion_matrix.png       ← Generated image
├── feature_importance.png     ← Generated image
├── README.md                  ← Main project documentation
├── requirements.txt           ← Dependencies
├── assets/
│   ├── win_sample.png         ← Screenshot of WIN prediction (you’ll upload)
│   └── loss_sample.png        ← Screenshot of LOSS prediction
└── notebook/
    └── nba_win_predictor_colab.ipynb ← Optional: Upload Colab notebook
```

````markdown
# 🏀 NBA Win Predictor
<img width="1908" height="922" alt="Image" src="https://github.com/user-attachments/assets/fec219d2-2613-4ef2-9f47-e6ebd0507961" />
This project uses machine learning and real NBA statistics to **predict whether a basketball team will win** based on in-game features such as shooting %, assists, turnovers, and more. It also visualizes important factors contributing to victories in the NBA.

## 🌟 Live Demo

▶️ **Gradio App**: [Click to Launch App](https://huggingface.co/spaces/mohAhmad/GameWinningPrediction)

-- WIN Example
-- LOSS Example

---

## 📊 Dataset Used

- **Source**: [Kaggle - NBA Game Stats](https://www.kaggle.com/datasets/wyattowalsh/basketball)
- **Download**: Handled via `kagglehub`
- **Size**: ~700MB
- **Files Used**: `game.csv` (contains home/away team stats, win/loss, etc.)

---

## 🚀 Model Summary

| Model         | Type                 |
|---------------|----------------------|
| Algorithm     | RandomForestClassifier |
| Accuracy      | `73.57%`             |
| Labels        | Win (1), Loss (0)    |
| Important Features | 3pt %, rebounds, assists, points, steals, home/away |

---

## 💡 How It Works

📥 Input features:
- 3-Point %
- Turnovers
- Rebounds
- Assists
- Total Points
- Steals
- Blocks
- Is Home Game?

📤 Output:
- Prediction: **WIN** or **LOSS**
- Confidence score
- 🎉 Animated Reaction (GIF)
- 📊 Confusion Matrix
- 📈 Feature Importance

---

## 🧪 Resources Used

| Resource | Purpose |
|---------|--------|
| `kagglehub` | Download dataset |
| `pandas` | Data handling |
| `sklearn` | ML model |
| `gradio` | Frontend |
| `matplotlib/seaborn` | Charts |
| `colab` | Development platform |

---

## 📦 How to Run Locally

```bash
git clone https://github.com/your-username/nba-win-predictor.git
cd nba-win-predictor
pip install -r requirements.txt
python app.py
````

---

## 🤖 Deployment

You can deploy this on:

* [Hugging Face Spaces](https://huggingface.co/spaces/mohAhmad/GameWinningPrediction)

---

## 📝 Colab Notebook

Open the complete training pipeline here:
👉 [Colab Link](https://colab.research.google.com/drive/1AFOSZ3WskK4YqJnaQyr_iFNOTgz0n-3Z#scrollTo=f2f387yDQr3L)

---

### ✅ WIN Prediction

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/d74b2d49-2471-4c89-afef-796ea56fa895" />

### ❌ LOSS Prediction

---

## ⭐️ Show Your Support

If you found this useful, please ⭐️ the repository and connect on [LinkedIn](https://linkedin.com/in/ahmadkhushi)!
