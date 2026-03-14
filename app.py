import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO
import textwrap

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

# ── Food Database (expanded) ─────────────────────────────────────────────────
FOODS = {
    # Grains & Rice
    "White Rice (1 cup)": 73, "Brown Rice (1 cup)": 55, "Basmati Rice (1 cup)": 58,
    "Roti (2 pieces)": 62, "Paratha (2 pieces)": 60, "Puri (3 pieces)": 65,
    "Naan (1 piece)": 68, "Bhatura (1 piece)": 70,
    # Breakfast
    "Idli (3 pieces)": 46, "Dosa (1 piece)": 57, "Uttapam (1 piece)": 50,
    "Poha (1 bowl)": 55, "Upma (1 bowl)": 52, "Aloo Paratha (1 piece)": 63,
    "Pongal (1 bowl)": 48, "Sabudana Khichdi (1 bowl)": 67,
    # Lentils & Pulses
    "Dal (1 bowl)": 29, "Rajma (1 bowl)": 29, "Chana Masala (1 bowl)": 28,
    "Sambar (1 bowl)": 35, "Moong Dal (1 bowl)": 31, "Black Chana (1 bowl)": 30,
    # Vegetables
    "Aloo Sabzi (1 bowl)": 65, "Mixed Veg Curry (1 bowl)": 40,
    "Palak Paneer (1 bowl)": 35, "Bhindi (1 bowl)": 20, "Baingan Bharta (1 bowl)": 25,
    # Snacks & Street Food
    "Samosa (2 pieces)": 55, "Pakora (5 pieces)": 52, "Vada Pav (1 piece)": 68,
    "Pav Bhaji (1 plate)": 67, "Bhel Puri (1 bowl)": 58, "Sev Puri (4 pieces)": 60,
    "Kachori (2 pieces)": 65, "Dhokla (4 pieces)": 49,
    # Rice Dishes
    "Biryani (1 plate)": 65, "Pulao (1 plate)": 62, "Curd Rice (1 bowl)": 50,
    "Lemon Rice (1 bowl)": 58, "Tamarind Rice (1 bowl)": 60,
    # Sweets & Desserts
    "Rasgulla (2 pieces)": 65, "Gulab Jamun (2 pieces)": 76,
    "Jalebi (4 pieces)": 85, "Kheer (1 bowl)": 70, "Halwa (1 bowl)": 72,
    "Ladoo (2 pieces)": 74, "Barfi (2 pieces)": 71, "Peda (3 pieces)": 73,
    # Drinks
    "Chai with sugar (1 cup)": 55, "Lassi sweet (1 glass)": 62,
    "Nimbu Pani with sugar (1 glass)": 58, "Coconut Water (1 glass)": 54,
    # Healthy Options
    "Sprouts Salad (1 bowl)": 22, "Raita (1 bowl)": 28,
    "Buttermilk plain (1 glass)": 25, "Plain Curd (1 bowl)": 30,
}

FOOD_CATEGORIES = {
    "🍚 Grains & Rice": ["White Rice (1 cup)", "Brown Rice (1 cup)", "Basmati Rice (1 cup)", "Roti (2 pieces)", "Paratha (2 pieces)", "Puri (3 pieces)", "Naan (1 piece)", "Bhatura (1 piece)"],
    "🌅 Breakfast": ["Idli (3 pieces)", "Dosa (1 piece)", "Uttapam (1 piece)", "Poha (1 bowl)", "Upma (1 bowl)", "Aloo Paratha (1 piece)", "Pongal (1 bowl)", "Sabudana Khichdi (1 bowl)"],
    "🫘 Dal & Pulses": ["Dal (1 bowl)", "Rajma (1 bowl)", "Chana Masala (1 bowl)", "Sambar (1 bowl)", "Moong Dal (1 bowl)", "Black Chana (1 bowl)"],
    "🥦 Vegetables": ["Aloo Sabzi (1 bowl)", "Mixed Veg Curry (1 bowl)", "Palak Paneer (1 bowl)", "Bhindi (1 bowl)", "Baingan Bharta (1 bowl)"],
    "🥪 Snacks & Street Food": ["Samosa (2 pieces)", "Pakora (5 pieces)", "Vada Pav (1 piece)", "Pav Bhaji (1 plate)", "Bhel Puri (1 bowl)", "Sev Puri (4 pieces)", "Kachori (2 pieces)", "Dhokla (4 pieces)"],
    "🍛 Rice Dishes": ["Biryani (1 plate)", "Pulao (1 plate)", "Curd Rice (1 bowl)", "Lemon Rice (1 bowl)", "Tamarind Rice (1 bowl)"],
    "🍮 Sweets": ["Rasgulla (2 pieces)", "Gulab Jamun (2 pieces)", "Jalebi (4 pieces)", "Kheer (1 bowl)", "Halwa (1 bowl)", "Ladoo (2 pieces)", "Barfi (2 pieces)", "Peda (3 pieces)"],
    "🥤 Drinks": ["Chai with sugar (1 cup)", "Lassi sweet (1 glass)", "Nimbu Pani with sugar (1 glass)", "Coconut Water (1 glass)"],
    "🥗 Healthy Options": ["Sprouts Salad (1 bowl)", "Raita (1 bowl)", "Buttermilk plain (1 glass)", "Plain Curd (1 bowl)"],
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
def generate_pdf_report(name, age, bmi, glucose, bp, prob, risk_label, selected_foods, advice_key):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.units import cm

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('title', fontSize=20, fontName='Helvetica-Bold', spaceAfter=6, textColor=colors.HexColor('#1a1a2e'))
    sub_style = ParagraphStyle('sub', fontSize=11, fontName='Helvetica', spaceAfter=12, textColor=colors.HexColor('#555555'))
    section_style = ParagraphStyle('section', fontSize=13, fontName='Helvetica-Bold', spaceBefore=14, spaceAfter=6, textColor=colors.HexColor('#1a1a2e'))
    body_style = ParagraphStyle('body', fontSize=11, fontName='Helvetica', spaceAfter=4, leading=16)

    story.append(Paragraph("🍛 Indian Diet & Diabetes Risk Report", title_style))
    story.append(Paragraph(f"Prepared for: {name if name else 'Patient'}", sub_style))
    story.append(Spacer(1, 0.3*cm))

    risk_color = colors.HexColor('#e74c3c') if advice_key == 'high' else colors.HexColor('#27ae60')
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
        ('BACKGROUND', (0,5), (-1,5), colors.HexColor('#fff3cd') if advice_key == 'high' else colors.HexColor('#d4edda')),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph(f"Risk Assessment: {risk_label}", ParagraphStyle('risk', fontSize=14, fontName='Helvetica-Bold', textColor=risk_color, spaceAfter=8)))

    if selected_foods:
        story.append(Paragraph("Foods You Selected", section_style))
        food_rows = [['Food Item', 'GI Score', 'Risk Level']]
        for f in selected_foods:
            gi = FOODS[f]
            r = 'High' if gi > 60 else 'Medium' if gi > 50 else 'Low'
            food_rows.append([f, str(gi), r])
        ft = Table(food_rows, colWidths=[8*cm, 3*cm, 4*cm])
        ft.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(ft)
        story.append(Spacer(1, 0.3*cm))

    advice = DIET_ADVICE[advice_key]
    story.append(Paragraph("Diet Recommendations", section_style))
    story.append(Paragraph("✅ Eat more:", ParagraphStyle('green', fontSize=11, fontName='Helvetica-Bold', textColor=colors.HexColor('#27ae60'), spaceAfter=3)))
    story.append(Paragraph(", ".join(advice['eat_more']), body_style))
    story.append(Paragraph("⚠️ Reduce:", ParagraphStyle('orange', fontSize=11, fontName='Helvetica-Bold', textColor=colors.HexColor('#e67e22'), spaceAfter=3)))
    story.append(Paragraph(", ".join(advice['reduce']), body_style))
    story.append(Paragraph("❌ Avoid:", ParagraphStyle('red', fontSize=11, fontName='Helvetica-Bold', textColor=colors.HexColor('#e74c3c'), spaceAfter=3)))
    story.append(Paragraph(", ".join(advice['avoid']), body_style))

    story.append(Paragraph("Health Tips", section_style))
    for tip in advice['tips']:
        story.append(Paragraph(f"• {tip}", body_style))

    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("⚠️ Disclaimer: This report is for educational purposes only. Please consult a qualified doctor for medical advice.", ParagraphStyle('disc', fontSize=9, textColor=colors.HexColor('#888888'))))

    doc.build(story)
    buf.seek(0)
    return buf

# ── App Layout ────────────────────────────────────────────────────────────────
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
    bmi_category = "Underweight" if calc_bmi < 18.5 else "Normal" if calc_bmi < 25 else "Overweight" if calc_bmi < 30 else "Obese"
    st.metric("Your BMI", f"{calc_bmi:.1f}", bmi_category)
    st.info(f"Category: **{bmi_category}**")

st.sidebar.markdown("---")
age = st.sidebar.slider("Age", 18, 80, 28)
bmi = st.sidebar.slider("BMI", 15.0, 50.0, float(round(calc_bmi, 1)))
glucose = st.sidebar.slider("Glucose Level (mg/dL)", 50, 200, 100)
bp = st.sidebar.slider("Blood Pressure (mmHg)", 40, 130, 72)
pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 1)

# ── Food Selection ────────────────────────────────────────────────────────────
st.subheader("🥘 What did you eat today?")
st.markdown("Select foods by category:")

selected = []
cols = st.columns(3)
for i, (category, items) in enumerate(FOOD_CATEGORIES.items()):
    with cols[i % 3]:
        with st.expander(category):
            for item in items:
                if st.checkbox(item, key=item):
                    selected.append(item)

# ── Prediction ────────────────────────────────────────────────────────────────
avg_gi = np.mean([FOODS[f] for f in selected]) if selected else 55
simulated_glucose = glucose + (avg_gi - 55) * 0.5
person = pd.DataFrame([{
    "Pregnancies": pregnancies, "Glucose": simulated_glucose,
    "BloodPressure": bp, "SkinThickness": 29, "Insulin": 125,
    "BMI": bmi, "DiabetesPedigreeFunction": 0.37, "Age": age
}])
prob = model.predict_proba(person)[0][1] * 100
result = model.predict(person)[0]
advice_key = "high" if result == 1 else "low"
risk_label = f"Higher Risk — {prob:.1f}% probability" if result == 1 else f"Lower Risk — {prob:.1f}% probability"

st.markdown("---")
st.subheader("📊 Your Diabetes Risk")

col1, col2, col3 = st.columns(3)
col1.metric("Risk Score", f"{prob:.1f}%", "High Risk" if result == 1 else "Low Risk")
col2.metric("Avg GI of your diet", f"{avg_gi:.0f}", "High" if avg_gi > 60 else "Normal")
col3.metric("Model Accuracy", f"{acc*100:.1f}%")

if result == 1:
    st.error(f"⚠️ {risk_label}")
else:
    st.success(f"✅ {risk_label}")
st.progress(int(prob))

# ── Charts ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Your Risk Factor Chart")

tab1, tab2, tab3 = st.tabs(["Risk Gauge", "Your Health vs Normal", "Food GI Chart"])

with tab1:
    fig, ax = plt.subplots(figsize=(7, 3.5))
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
    categories = ['Age\n(normalized)', 'BMI', 'Glucose', 'Blood\nPressure']
    your_vals = [age/80*100, bmi/50*100, glucose/200*100, bp/130*100]
    normal_vals = [35/80*100, 22/50*100, 85/200*100, 70/130*100]
    x = np.arange(len(categories))
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    bars1 = ax2.bar(x - 0.2, your_vals, 0.35, label='You', color='#3498db')
    bars2 = ax2.bar(x + 0.2, normal_vals, 0.35, label='Normal', color='#2ecc71')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel('% of max range')
    ax2.set_title('Your Values vs Normal Range', fontsize=13, fontweight='bold')
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

with tab3:
    if selected:
        fig3, ax3 = plt.subplots(figsize=(8, max(3, len(selected) * 0.5)))
        gi_vals = [FOODS[f] for f in selected]
        colors_list = ['#e74c3c' if g > 60 else '#f39c12' if g > 50 else '#27ae60' for g in gi_vals]
        labels = [f[:30] for f in selected]
        ax3.barh(labels, gi_vals, color=colors_list)
        ax3.axvline(x=55, color='gray', linestyle='--', alpha=0.7, label='Low GI threshold (55)')
        ax3.axvline(x=70, color='red', linestyle='--', alpha=0.7, label='High GI threshold (70)')
        ax3.set_xlabel('Glycemic Index')
        ax3.set_title('GI Score of Your Foods', fontsize=13, fontweight='bold')
        ax3.legend()
        red_p = mpatches.Patch(color='#e74c3c', label='High GI (>60)')
        orange_p = mpatches.Patch(color='#f39c12', label='Medium GI (50-60)')
        green_p = mpatches.Patch(color='#27ae60', label='Low GI (<50)')
        ax3.legend(handles=[red_p, orange_p, green_p])
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
    else:
        st.info("Select some foods above to see their GI chart.")

# ── Diet Recommendations ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🥗 Diet Recommendations for You")
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
if selected:
    st.markdown("---")
    st.subheader("🍽️ Your Food Analysis")
    food_data = pd.DataFrame({
        "Food": selected,
        "GI Score": [FOODS[f] for f in selected],
        "Risk Level": ["🔴 High" if FOODS[f] > 60 else "🟡 Medium" if FOODS[f] > 50 else "🟢 Low" for f in selected]
    })
    st.dataframe(food_data, use_container_width=True)

# ── PDF Download ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📄 Download Your Report")
st.markdown("Get a full PDF report with your risk score, food analysis, and diet plan.")

try:
    import reportlab
    pdf_buf = generate_pdf_report(name, age, bmi, glucose, bp, prob, risk_label, selected, advice_key)
    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_buf,
        file_name=f"diabetes_risk_report_{name.replace(' ','_') if name else 'report'}.pdf",
        mime="application/pdf"
    )
except ImportError:
    st.warning("PDF download requires 'reportlab'. Add it to requirements.txt to enable this feature.")

st.caption(f"Model accuracy: {acc*100:.1f}% | This app is for educational purposes only. Consult a doctor for medical advice.")
