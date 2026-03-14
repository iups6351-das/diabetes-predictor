import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    for c in cols:
        df[c] = df[c].replace(0, df[c].replace(0, np.nan).median())
    return df

@st.cache_resource
def train_model(df):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_split=10, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

FOODS = {"White Rice (1 cup)": 73, "Brown Rice (1 cup)": 55, "Roti (2 pieces)": 62, "Dal (1 bowl)": 29, "Samosa (2 pieces)": 55, "Idli (3 pieces)": 46, "Dosa (1 piece)": 57, "Biryani (1 plate)": 65, "Rasgulla (2 pieces)": 65, "Poha (1 bowl)": 55, "Upma (1 bowl)": 52, "Paratha (2 pieces)": 60}

st.set_page_config(page_title="Indian Diet Diabetes Risk", page_icon="🍛")
st.title("🍛 Indian Diet & Diabetes Risk Predictor")
st.markdown("*Built with real ML model — 76.6% accuracy*")
df = load_data()
model, acc = train_model(df)
st.sidebar.header("Your Health Details")
age = st.sidebar.slider("Age", 18, 80, 28)
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 28.0)
glucose = st.sidebar.slider("Glucose Level", 50, 200, 100)
bp = st.sidebar.slider("Blood Pressure", 40, 130, 72)
pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 1)
st.subheader("What did you eat today?")
selected = st.multiselect("Select Indian foods you eat regularly:", list(FOODS.keys()))
avg_gi = np.mean([FOODS[f] for f in selected]) if selected else 55
simulated_glucose = glucose + (avg_gi - 55) * 0.5
person = pd.DataFrame([{"Pregnancies": pregnancies, "Glucose": simulated_glucose, "BloodPressure": bp, "SkinThickness": 29, "Insulin": 125, "BMI": bmi, "DiabetesPedigreeFunction": 0.37, "Age": age}])
prob = model.predict_proba(person)[0][1] * 100
result = model.predict(person)[0]
st.subheader("Your Diabetes Risk")
if result == 1:
    st.error(f"Higher Risk Detected — {prob:.1f}% probability")
else:
    st.success(f"Lower Risk — {prob:.1f}% probability")
st.progress(int(prob))
if selected:
    food_data = pd.DataFrame({"Food": selected, "GI Score": [FOODS[f] for f in selected], "Risk": ["High" if FOODS[f]>60 else "Medium" if FOODS[f]>50 else "Low" for f in selected]})
    st.dataframe(food_data, use_container_width=True)
st.caption(f"Model accuracy: {acc*100:.1f}%")
