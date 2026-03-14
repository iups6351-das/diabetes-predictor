import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO

# ── Data & Model ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    for c in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
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

# ── Food Database ─────────────────────────────────────────────────────────────
# Format: "Food Name": (GI score, unit_label, calories_per_unit)
FOODS = {
    "White Rice (per cup)":        (73, "cups", 206),
    "Brown Rice (per cup)":        (55, "cups", 216),
    "Basmati Rice (per cup)":      (58, "cups", 190),
    "Roti (per piece)":            (62, "pieces", 80),
    "Paratha (per piece)":         (60, "pieces", 150),
    "Puri (per piece)":            (65, "pieces", 90),
    "Naan (per piece)":            (68, "pieces", 260),
    "Bhatura (per piece)":         (70, "pieces", 300),
    "Idli (per piece)":            (46, "pieces", 39),
    "Dosa (per piece)":            (57, "pieces", 120),
    "Uttapam (per piece)":         (50, "pieces", 110),
    "Poha (per bowl)":             (55, "bowls", 180),
    "Upma (per bowl)":             (52, "bowls", 170),
    "Aloo Paratha (per piece)":    (63, "pieces", 200),
    "Pongal (per bowl)":           (48, "bowls", 220),
    "Sabudana Khichdi (per bowl)": (67, "bowls", 250),
    "Dal (per bowl)":              (29, "bowls", 130),
    "Rajma (per bowl)":            (29, "bowls", 140),
    "Chana Masala (per bowl)":     (28, "bowls", 150),
    "Sambar (per bowl)":           (35, "bowls", 90),
    "Moong Dal (per bowl)":        (31, "bowls", 110),
    "Black Chana (per bowl)":      (30, "bowls", 120),
    "Aloo Sabzi (per bowl)":       (65, "bowls", 150),
    "Mixed Veg Curry (per bowl)":  (40, "bowls", 120),
    "Palak Paneer (per bowl)":     (35, "bowls", 200),
    "Bhindi (per bowl)":           (20, "bowls", 80),
    "Baingan Bharta (per bowl)":   (25, "bowls", 90),
    "Samosa (per piece)":          (55, "pieces", 130),
    "Pakora (per piece)":          (52, "pieces", 50),
    "Vada Pav (per piece)":        (68, "pieces", 290),
    "Pav Bhaji (per plate)":       (67, "plates", 400),
    "Bhel Puri (per bowl)":        (58, "bowls", 180),
    "Sev Puri (per piece)":        (60, "pieces", 50),
    "Kachori (per piece)":         (65, "pieces", 160),
    "Dhokla (per piece)":          (49, "pieces", 60),
    "Biryani (per plate)":         (65, "plates", 500),
    "Pulao (per plate)":           (62, "plates", 350),
    "Curd Rice (per bowl)":        (50, "bowls", 200),
    "Lemon Rice (per bowl)":       (58, "bowls", 220),
    "Tamarind Rice (per bowl)":    (60, "bowls", 230),
    "Rasgulla (per piece)":        (65, "pieces", 100),
    "Gulab Jamun (per piece)":     (76, "pieces", 140),
    "Jalebi (per piece)":          (85, "pieces", 60),
    "Kheer (per bowl)":            (70, "bowls", 250),
    "Halwa (per bowl)":            (72, "bowls", 300),
    "Ladoo (per piece)":           (74, "pieces", 180),
    "Barfi (per piece)":           (71, "pieces", 100),
    "Peda (per piece)":            (73, "pieces", 90),
    "Chai with sugar (per cup)":   (55, "cups", 50),
    "Lassi sweet (per glass)":     (62, "glasses", 180),
    "Nimbu Pani with sugar (per glass)": (58, "glasses", 60),
    "Coconut Water (per glass)":   (54, "glasses", 45),
    "Sprouts Salad (per bowl)":    (22, "bowls", 80),
    "Raita (per bowl)":            (28, "bowls", 70),
    "Buttermilk plain (per glass)":(25, "glasses", 40),
    "Plain Curd (per bowl)":       (30, "bowls", 100),
}

FOOD_CATEGORIES = {
    "🍚 Grains & Rice": ["White Rice (per cup)", "Brown Rice (per cup)", "Basmati Rice (per cup)", "Roti (per piece)", "Paratha (per piece)", "Puri (per piece)", "Naan (per piece)", "Bhatura (per piece)"],
    "🌅 Breakfast": ["Idli (per piece)", "Dosa (per piece)", "Uttapam (per piece)", "Poha (per bowl)", "Upma (per bowl)", "Aloo Paratha (per piece)", "Pongal (per bowl)", "Sabudana Khichdi (per bowl)"],
    "🫘 Dal & Pulses": ["Dal (per bowl)", "Rajma (per bowl)", "Chana Masala (per bowl)", "Sambar (per bowl)", "Moong Dal (per bowl)", "Black Chana (per bowl)"],
    "🥦 Vegetables": ["Aloo Sabzi (per bowl)", "Mixed Veg Curry (per bowl)", "Palak Paneer (per bowl)", "Bhindi (per bowl)", "Baingan Bharta (per bowl)"],
    "🥪 Snacks & Street Food": ["Samosa (per piece)", "Pakora (per piece)", "Vada Pav (per piece)", "Pav Bhaji (per plate)", "Bhel Puri (per bowl)", "Sev Puri (per piece)", "Kachori (per piece)", "Dhokla (per piece)"],
    "🍛 Rice Dishes": ["Biryani (per plate)", "Pulao (per plate)", "Curd Rice (per bowl)", "Lemon Rice (per bowl)", "Tamarind Rice (per bowl)"],
    "🍮 Sweets": ["Rasgulla (per piece)", "Gulab Jamun (per piece)", "Jalebi (per piece)", "Kheer (per bowl)", "Halwa (per bowl)", "Ladoo (per piece)", "Barfi (per piece)", "Peda (per piece)"],
    "🥤 Drinks": ["Chai with sugar (per cup)", "Lassi sweet (per glass)", "Nimbu Pani with sugar (per glass)", "Coconut Water (per glass)"],
    "🥗 Healthy Options": ["Sprouts Salad (per bowl)", "Raita (per bowl)", "Buttermilk plain (per glass)", "Plain Curd (per bowl)"],
}

# ── Exercise Database ─────────────────────────────────────────────────────────
EXERCISES = {
    "🏃 Jogging":             {"cal_per_hr": 500, "glucose_reduction": 15, "intensity": "Medium"},
    "🚶 Walking":             {"cal_per_hr": 280, "glucose_reduction": 8,  "intensity": "Low"},
    "🏊 Swimming":            {"cal_per_hr": 600, "glucose_reduction": 18, "intensity": "High"},
    "🏋️ Gym / Weightlifting": {"cal_per_hr": 400, "glucose_reduction": 10, "intensity": "Medium"},
    "🚴 Cycling":             {"cal_per_hr": 550, "glucose_reduction": 16, "intensity": "Medium"},
    "⚽ Football / Cricket":  {"cal_per_hr": 480, "glucose_reduction": 14, "intensity": "Medium"},
    "🧘 Yoga":                {"cal_per_hr": 200, "glucose_reduction": 5,  "intensity": "Low"},
    "💃 Dancing / Zumba":     {"cal_per_hr": 400, "glucose_reduction": 12, "intensity": "Medium"},
    "🏸 Badminton":           {"cal_per_hr": 450, "glucose_reduction": 13, "intensity": "Medium"},
    "🤸 HIIT / Aerobics":     {"cal_per_hr": 700, "glucose_reduction": 20, "intensity": "High"},
    "🧗 Stair Climbing":      {"cal_per_hr": 500, "glucose_reduction": 15, "intensity": "Medium"},
    "🏇 Running (fast)":      {"cal_per_hr": 750, "glucose_reduction": 22, "intensity": "High"},
}

DIET_ADVICE = {
    "high": {
        "avoid": ["White Rice", "Maida items", "Jalebi", "Gulab Jamun", "Biryani", "Puri", "Bhatura", "Naan"],
        "reduce": ["Paratha", "Samosa", "Vada Pav", "Kheer", "Chai with sugar"],
        "eat_more": ["Brown Rice", "Roti", "Dal", "Rajma", "Chana", "Bhindi", "Sprouts", "Buttermilk", "Moong Dal"],
        "tips": [
            "Eat small portions every 3-4 hours instead of big meals",
            "Walk for 30 minutes after meals, especially after dinner",
            "Replace white rice with brown rice or millets",
            "Drink water before meals to reduce portion size",
            "Avoid sweets and sugary drinks completely",
            "Include methi (fenugreek) seeds — soak overnight and eat in morning",
        ]
    },
    "low": {
        "avoid": ["Jalebi", "Gulab Jamun", "Ladoo", "Barfi"],
        "reduce": ["White Rice", "Puri", "Bhatura"],
        "eat_more": ["Dal", "Rajma", "Brown Rice", "Vegetables", "Sprouts", "Curd"],
        "tips": [
            "Maintain your current healthy diet — you're doing well!",
            "Stay active with at least 30 minutes of exercise daily",
            "Monitor glucose levels once a year as a precaution",
            "Reduce refined sugar intake gradually",
            "Include more fiber — vegetables, whole grains, pulses",
        ]
    }
}

# ── PDF Generator ────────────────────────────────────────────────────────────
def generate_pdf_report(name, age, bmi, glucose, bp, prob, risk_label, food_items, advice_key, exercise_summary):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.units import cm

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    story = []

    title_s = ParagraphStyle('t', fontSize=20, fontName='Helvetica-Bold', spaceAfter=6)
    sub_s   = ParagraphStyle('s', fontSize=11, spaceAfter=12, textColor=colors.HexColor('#555555'))
    sec_s   = ParagraphStyle('h', fontSize=13, fontName='Helvetica-Bold', spaceBefore=14, spaceAfter=6)
    body_s  = ParagraphStyle('b', fontSize=11, spaceAfter=4, leading=16)

    story.append(Paragraph("Indian Diet & Diabetes Risk Report", title_s))
    story.append(Paragraph(f"Prepared for: {name if name else 'Patient'}", sub_s))
    story.append(Spacer(1, 0.3*cm))

    health_data = [
        ['Parameter', 'Your Value', 'Normal Range'],
        ['Age', str(age), '—'],
        ['BMI', f'{bmi:.1f}', '18.5 – 24.9'],
        ['Glucose Level', str(glucose), '70 – 100 mg/dL'],
        ['Blood Pressure', str(bp), '60 – 80 mmHg'],
        ['Diabetes Risk', f'{prob:.1f}%', '< 30% (Low Risk)'],
    ]
    t = Table(health_data, colWidths=[5*cm, 5*cm, 6*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 11),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))
    risk_color = colors.HexColor('#e74c3c') if advice_key == 'high' else colors.HexColor('#27ae60')
    story.append(Paragraph(f"Risk: {risk_label}", ParagraphStyle('r', fontSize=13, fontName='Helvetica-Bold', textColor=risk_color, spaceAfter=8)))

    if food_items:
        story.append(Paragraph("Foods Consumed", sec_s))
        rows = [['Food Item', 'Qty', 'GI Score', 'Calories', 'Risk Level']]
        for fname, qty in food_items:
            gi, unit, cal = FOODS[fname]
            rows.append([fname, f"{qty} {unit}", str(gi), f"{int(cal*qty)} kcal", 'High' if gi>60 else 'Medium' if gi>50 else 'Low'])
        ft = Table(rows, colWidths=[5.5*cm, 2.5*cm, 2*cm, 2.5*cm, 2.5*cm])
        ft.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
            ('PADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(ft)

    if exercise_summary:
        story.append(Paragraph("Exercise Activity", sec_s))
        ex_rows = [['Exercise', 'Duration', 'Calories Burned', 'Glucose Reduction']]
        for ex_name, hrs in exercise_summary:
            ex = EXERCISES[ex_name]
            ex_rows.append([ex_name, f"{hrs} hrs", f"{int(ex['cal_per_hr']*hrs)} kcal", f"-{round(ex['glucose_reduction']*hrs,1)} mg/dL"])
        et = Table(ex_rows, colWidths=[5*cm, 3*cm, 4*cm, 4*cm])
        et.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#145a32')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#eafaf1'), colors.white]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(et)

    advice = DIET_ADVICE[advice_key]
    story.append(Paragraph("Diet Recommendations", sec_s))
    story.append(Paragraph("Eat More: " + ", ".join(advice['eat_more']), body_s))
    story.append(Paragraph("Reduce: " + ", ".join(advice['reduce']), body_s))
    story.append(Paragraph("Avoid: " + ", ".join(advice['avoid']), body_s))
    story.append(Paragraph("Health Tips", sec_s))
    for tip in advice['tips']:
        story.append(Paragraph(f"• {tip}", body_s))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Disclaimer: For educational purposes only. Consult a doctor for medical advice.", ParagraphStyle('d', fontSize=9, textColor=colors.HexColor('#888888'))))
    doc.build(story)
    buf.seek(0)
    return buf

# ════════════════════════════════════════════════════════════════════════════
# APP LAYOUT
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Indian Diet Diabetes Risk", page_icon="🍛", layout="wide")
st.title("🍛 Indian Diet & Diabetes Risk Predictor")
st.markdown("*Built with real ML model — 76.6% accuracy*")

df = load_data()
model, acc = train_model(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Your Health Details")
name = st.sidebar.text_input("Your Name (optional)", placeholder="e.g. Rahul Sharma")
st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ BMI Calculator")
with st.sidebar.expander("Calculate my BMI"):
    height_cm = st.number_input("Height (cm)", min_value=100, max_value=220, value=165)
    weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    calc_bmi = weight_kg / ((height_cm / 100) ** 2)
    bmi_cat = "Underweight" if calc_bmi < 18.5 else "Normal" if calc_bmi < 25 else "Overweight" if calc_bmi < 30 else "Obese"
    st.metric("Your BMI", f"{calc_bmi:.1f}", bmi_cat)

st.sidebar.markdown("---")
age = st.sidebar.slider("Age", 18, 80, 28)
bmi = st.sidebar.slider("BMI", 15.0, 50.0, float(round(calc_bmi, 1)))
glucose = st.sidebar.slider("Glucose Level (mg/dL)", 50, 200, 100)
bp = st.sidebar.slider("Blood Pressure (mmHg)", 40, 130, 72)
pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 1)

# ── Food Selection with Portions ──────────────────────────────────────────────
st.subheader("🥘 What did you eat today?")
st.markdown("Check a food item, then adjust the quantity:")

selected_with_qty = []

for category, items in FOOD_CATEGORIES.items():
    with st.expander(category):
        for item in items:
            gi, unit, cal = FOODS[item]
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                checked = st.checkbox(item, key=f"chk_{item}")
            with c2:
                qty = st.number_input(
                    f"qty", min_value=0.5, max_value=10.0, value=1.0, step=0.5,
                    key=f"qty_{item}", label_visibility="collapsed"
                ) if checked else 1.0
            with c3:
                if checked:
                    gi_icon = "🔴" if gi > 60 else "🟡" if gi > 50 else "🟢"
                    st.caption(f"{gi_icon} GI:{gi} | {int(cal*qty)} kcal")
            if checked:
                selected_with_qty.append((item, qty))

# ── Exercise Tracker ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🏃 Exercise Tracker")
st.markdown("Select activities you did today and set the duration:")

exercise_summary = []
total_cal_burned = 0
total_glucose_reduction = 0.0

ex_cols = st.columns(3)
for i, (ex_name, ex_data) in enumerate(EXERCISES.items()):
    with ex_cols[i % 3]:
        ex_checked = st.checkbox(ex_name, key=f"ex_{ex_name}")
        if ex_checked:
            hrs = st.slider(
                "Duration (hours)", min_value=0.5, max_value=5.0,
                value=1.0, step=0.5, key=f"hrs_{ex_name}"
            )
            cal_b = int(ex_data['cal_per_hr'] * hrs)
            gluc_r = round(ex_data['glucose_reduction'] * hrs, 1)
            intensity_color = "🔴" if ex_data['intensity'] == "High" else "🟡" if ex_data['intensity'] == "Medium" else "🟢"
            st.caption(f"{intensity_color} {ex_data['intensity']} | 🔥 {cal_b} kcal | 📉 -{gluc_r} mg/dL")
            exercise_summary.append((ex_name, hrs))
            total_cal_burned += cal_b
            total_glucose_reduction += gluc_r

if exercise_summary:
    st.success(f"💪 Total exercise today: **{total_cal_burned} kcal burned** | Glucose reduced by **{total_glucose_reduction:.1f} mg/dL**")

# ── Prediction ────────────────────────────────────────────────────────────────
if selected_with_qty:
    total_qty = sum(q for _, q in selected_with_qty)
    avg_gi = sum(FOODS[f][0] * q for f, q in selected_with_qty) / total_qty
else:
    avg_gi = 55.0

simulated_glucose = max(50, glucose + (avg_gi - 55) * 0.5 - total_glucose_reduction)

person = pd.DataFrame([{
    "Pregnancies": pregnancies, "Glucose": simulated_glucose,
    "BloodPressure": bp, "SkinThickness": 29, "Insulin": 125,
    "BMI": bmi, "DiabetesPedigreeFunction": 0.37, "Age": age
}])
prob = model.predict_proba(person)[0][1] * 100
result = model.predict(person)[0]
advice_key = "high" if result == 1 else "low"
risk_label = f"Higher Risk — {prob:.1f}% probability" if result == 1 else f"Lower Risk — {prob:.1f}% probability"

# ── Results ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Your Diabetes Risk")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Risk Score", f"{prob:.1f}%", "High Risk" if result == 1 else "Low Risk")
c2.metric("Diet Avg GI", f"{avg_gi:.0f}", "High" if avg_gi > 60 else "Normal")
c3.metric("Calories Burned", f"{total_cal_burned} kcal")
c4.metric("Model Accuracy", f"{acc*100:.1f}%")

if result == 1:
    st.error(f"⚠️ {risk_label}")
else:
    st.success(f"✅ {risk_label}")
st.progress(int(prob))

if total_glucose_reduction > 0:
    st.info(f"🏃 Your exercise today lowered your simulated glucose by {total_glucose_reduction:.1f} mg/dL — great work!")

# ── Charts ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Charts")

tab1, tab2, tab3, tab4 = st.tabs(["Risk Gauge", "Health vs Normal", "Food GI Chart", "Exercise Impact"])

with tab1:
    fig, ax = plt.subplots(figsize=(7, 3))
    bar_color = '#e74c3c' if prob > 50 else '#f39c12' if prob > 30 else '#27ae60'
    ax.barh(['Risk'], [prob], color=bar_color, height=0.4)
    ax.barh(['Risk'], [100 - prob], left=[prob], color='#ecf0f1', height=0.4)
    ax.axvline(x=30, color='#f39c12', linestyle='--', alpha=0.7, label='Caution (30%)')
    ax.axvline(x=50, color='#e74c3c', linestyle='--', alpha=0.7, label='High Risk (50%)')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Risk Probability (%)')
    ax.set_title(f'Your Diabetes Risk: {prob:.1f}%', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab2:
    cats = ['Age\n(norm)', 'BMI', 'Glucose', 'Blood\nPressure']
    your_vals = [age/80*100, bmi/50*100, glucose/200*100, bp/130*100]
    norm_vals = [35/80*100, 22/50*100, 85/200*100, 70/130*100]
    x = np.arange(len(cats))
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.bar(x - 0.2, your_vals, 0.35, label='You', color='#3498db')
    ax2.bar(x + 0.2, norm_vals, 0.35, label='Normal', color='#2ecc71')
    ax2.set_xticks(x); ax2.set_xticklabels(cats)
    ax2.set_ylabel('% of max range')
    ax2.set_title('Your Values vs Normal', fontsize=13, fontweight='bold')
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

with tab3:
    if selected_with_qty:
        labels = [f[:25] for f, _ in selected_with_qty]
        gi_vals = [FOODS[f][0] for f, _ in selected_with_qty]
        colors_list = ['#e74c3c' if g > 60 else '#f39c12' if g > 50 else '#27ae60' for g in gi_vals]
        fig3, ax3 = plt.subplots(figsize=(8, max(3, len(labels)*0.55)))
        ax3.barh(labels, gi_vals, color=colors_list)
        ax3.axvline(x=55, color='gray', linestyle='--', alpha=0.7)
        ax3.axvline(x=70, color='red', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Glycemic Index')
        ax3.set_title('GI Score of Your Foods', fontsize=13, fontweight='bold')
        ax3.legend(handles=[
            mpatches.Patch(color='#e74c3c', label='High GI (>60)'),
            mpatches.Patch(color='#f39c12', label='Medium GI (50–60)'),
            mpatches.Patch(color='#27ae60', label='Low GI (<50)'),
        ])
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
    else:
        st.info("Select some foods above to see the GI chart.")

with tab4:
    if exercise_summary:
        ex_names = [e for e, _ in exercise_summary]
        ex_cals  = [int(EXERCISES[e]['cal_per_hr'] * h) for e, h in exercise_summary]
        fig4, ax4 = plt.subplots(figsize=(7, max(3, len(ex_names)*0.7)))
        ax4.barh(ex_names, ex_cals, color='#27ae60')
        ax4.set_xlabel('Calories Burned')
        ax4.set_title('Calories Burned by Exercise', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()
    else:
        st.info("Select exercises above to see your impact chart.")

# ── Diet Recommendations ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🥗 Diet Recommendations")
advice = DIET_ADVICE[advice_key]
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### ✅ Eat More")
    for item in advice['eat_more']:
        st.markdown(f"- {item}")
with c2:
    st.markdown("### ⚠️ Reduce")
    for item in advice['reduce']:
        st.markdown(f"- {item}")
with c3:
    st.markdown("### ❌ Avoid")
    for item in advice['avoid']:
        st.markdown(f"- {item}")
st.markdown("### 💡 Health Tips")
for tip in advice['tips']:
    st.info(tip)

# ── Food Table ────────────────────────────────────────────────────────────────
if selected_with_qty:
    st.markdown("---")
    st.subheader("🍽️ Your Food Analysis")
    food_df = pd.DataFrame({
        "Food":     [f for f, _ in selected_with_qty],
        "Quantity": [f"{qty} {FOODS[f][1]}" for f, qty in selected_with_qty],
        "GI Score": [FOODS[f][0] for f, _ in selected_with_qty],
        "Calories": [f"{int(FOODS[f][2]*qty)} kcal" for f, qty in selected_with_qty],
        "Risk":     ["🔴 High" if FOODS[f][0]>60 else "🟡 Medium" if FOODS[f][0]>50 else "🟢 Low" for f,_ in selected_with_qty],
    })
    st.dataframe(food_df, use_container_width=True)
    total_cals = sum(int(FOODS[f][2]*qty) for f, qty in selected_with_qty)
    net_cals = total_cals - total_cal_burned
    st.metric("Total Calories", f"{total_cals} kcal consumed",
              delta=f"-{total_cal_burned} kcal burned = {net_cals} kcal net" if total_cal_burned > 0 else None)

# ── PDF Download ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📄 Download Your Report")
try:
    import reportlab
    pdf_buf = generate_pdf_report(name, age, bmi, glucose, bp, prob, risk_label, selected_with_qty, advice_key, exercise_summary)
    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_buf,
        file_name=f"diabetes_risk_{name.replace(' ','_') if name else 'report'}.pdf",
        mime="application/pdf"
    )
except ImportError:
    st.warning("Add 'reportlab' to requirements.txt to enable PDF download.")

st.caption(f"Model accuracy: {acc*100:.1f}% | For educational purposes only. Consult a doctor for medical advice.")
