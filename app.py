import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO

# ── Data & Model ─────────────────────────────────────────────────────────────
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
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_split=10, random_state=42)
    m.fit(X_tr, y_tr)
    return m, accuracy_score(y_te, m.predict(X_te))

# ── Food Database ─────────────────────────────────────────────────────────────
# Format: name: (GI, unit, cal, protein_g, carb_g, fat_g, fiber_g, vit_c_mg, calcium_mg, iron_mg)
FOODS = {
    "White Rice (per cup)":        (73,"cups",206, 4.3, 44.5, 0.4, 0.6,  0,  16, 0.2),
    "Brown Rice (per cup)":        (55,"cups",216, 5.0, 44.8, 1.8, 3.5,  0,  20, 1.0),
    "Basmati Rice (per cup)":      (58,"cups",190, 4.4, 41.0, 0.5, 0.7,  0,  15, 0.5),
    "Roti (per piece)":            (62,"pieces",80, 2.7, 15.0, 1.0, 1.2,  0,  10, 0.8),
    "Paratha (per piece)":         (60,"pieces",150,3.5, 20.0, 6.0, 1.5,  0,  15, 1.0),
    "Puri (per piece)":            (65,"pieces",90, 2.0, 12.0, 4.0, 0.5,  0,   8, 0.5),
    "Naan (per piece)":            (68,"pieces",260,8.7, 45.0, 5.0, 1.8,  0,  30, 2.5),
    "Bhatura (per piece)":         (70,"pieces",300,7.0, 40.0,12.0, 1.2,  0,  20, 1.5),
    "Idli (per piece)":            (46,"pieces",39, 1.5,  8.0, 0.2, 0.5,  0,   8, 0.3),
    "Dosa (per piece)":            (57,"pieces",120,3.0, 20.0, 3.0, 0.8,  0,  10, 0.5),
    "Uttapam (per piece)":         (50,"pieces",110,4.0, 18.0, 2.5, 1.0,  2,  25, 0.8),
    "Poha (per bowl)":             (55,"bowls",180, 3.5, 36.0, 2.5, 1.5,  4,  15, 2.0),
    "Upma (per bowl)":             (52,"bowls",170, 4.0, 30.0, 5.0, 2.0,  2,  20, 1.5),
    "Aloo Paratha (per piece)":    (63,"pieces",200,4.5, 30.0, 7.0, 2.5,  8,  20, 1.2),
    "Pongal (per bowl)":           (48,"bowls",220, 6.0, 38.0, 5.0, 2.0,  0,  30, 1.8),
    "Sabudana Khichdi (per bowl)": (67,"bowls",250, 2.5, 50.0, 6.0, 0.5,  0,  10, 0.5),
    "Dal (per bowl)":              (29,"bowls",130, 9.0, 22.0, 0.5, 4.0,  2,  30, 2.5),
    "Rajma (per bowl)":            (29,"bowls",140,10.0, 24.0, 0.5, 6.0,  2,  50, 3.5),
    "Chana Masala (per bowl)":     (28,"bowls",150,10.0, 25.0, 2.0, 6.5,  5,  60, 4.0),
    "Sambar (per bowl)":           (35,"bowls",90,  5.0, 15.0, 1.5, 3.5, 10,  40, 2.0),
    "Moong Dal (per bowl)":        (31,"bowls",110, 8.0, 18.0, 0.5, 3.0,  2,  25, 2.0),
    "Black Chana (per bowl)":      (30,"bowls",120, 9.0, 20.0, 2.0, 5.5,  2,  45, 3.0),
    "Aloo Sabzi (per bowl)":       (65,"bowls",150, 3.0, 28.0, 3.5, 2.5, 12,  15, 1.0),
    "Mixed Veg Curry (per bowl)":  (40,"bowls",120, 4.0, 18.0, 4.0, 3.5, 20,  40, 1.5),
    "Palak Paneer (per bowl)":     (35,"bowls",200,10.0, 10.0,14.0, 2.5, 15, 200, 2.5),
    "Bhindi (per bowl)":           (20,"bowls",80,  2.5, 14.0, 1.0, 4.0, 18,  80, 0.8),
    "Baingan Bharta (per bowl)":   (25,"bowls",90,  2.5, 12.0, 4.5, 3.5,  5,  20, 0.8),
    "Samosa (per piece)":          (55,"pieces",130,3.0, 18.0, 5.5, 1.5,  5,  10, 0.8),
    "Pakora (per piece)":          (52,"pieces",50, 1.5,  6.0, 2.5, 0.5,  2,   8, 0.4),
    "Vada Pav (per piece)":        (68,"pieces",290,7.0, 42.0,10.0, 2.5,  5,  30, 2.0),
    "Pav Bhaji (per plate)":       (67,"plates",400,9.0, 55.0,14.0, 5.0, 20,  60, 3.0),
    "Bhel Puri (per bowl)":        (58,"bowls",180, 4.0, 32.0, 4.0, 2.5, 10,  20, 1.5),
    "Sev Puri (per piece)":        (60,"pieces",50, 1.0,  7.0, 2.0, 0.4,  1,   5, 0.3),
    "Kachori (per piece)":         (65,"pieces",160,3.5, 20.0, 7.0, 1.5,  1,  10, 0.8),
    "Dhokla (per piece)":          (49,"pieces",60, 3.0, 10.0, 1.0, 0.8,  1,  15, 0.5),
    "Biryani (per plate)":         (65,"plates",500,18.0,65.0,14.0, 2.5,  4,  50, 2.5),
    "Pulao (per plate)":           (62,"plates",350,8.0, 55.0, 8.0, 2.0,  3,  30, 1.5),
    "Curd Rice (per bowl)":        (50,"bowls",200, 6.0, 35.0, 4.0, 0.8,  0, 150, 0.5),
    "Lemon Rice (per bowl)":       (58,"bowls",220, 4.0, 40.0, 5.0, 1.0, 10,  15, 0.8),
    "Tamarind Rice (per bowl)":    (60,"bowls",230, 4.0, 42.0, 5.0, 1.5,  2,  15, 1.0),
    "Rasgulla (per piece)":        (65,"pieces",100,2.0, 22.0, 0.5, 0.0,  0,  40, 0.2),
    "Gulab Jamun (per piece)":     (76,"pieces",140,2.5, 25.0, 4.0, 0.2,  0,  30, 0.3),
    "Jalebi (per piece)":          (85,"pieces",60, 0.5, 14.0, 1.0, 0.0,  0,   5, 0.2),
    "Kheer (per bowl)":            (70,"bowls",250, 7.0, 42.0, 7.0, 0.2,  1, 200, 0.3),
    "Halwa (per bowl)":            (72,"bowls",300, 4.0, 50.0,10.0, 1.0,  0,  20, 1.0),
    "Ladoo (per piece)":           (74,"pieces",180,3.0, 30.0, 6.0, 0.5,  0,  20, 0.8),
    "Barfi (per piece)":           (71,"pieces",100,3.0, 16.0, 3.0, 0.2,  0,  50, 0.3),
    "Peda (per piece)":            (73,"pieces",90, 2.5, 15.0, 2.5, 0.0,  0,  60, 0.2),
    "Chai with sugar (per cup)":   (55,"cups",50,  1.5,  8.5, 1.5, 0.0,  0,  50, 0.2),
    "Lassi sweet (per glass)":     (62,"glasses",180,6.0,28.0, 5.0, 0.0,  1, 200, 0.2),
    "Nimbu Pani with sugar (per glass)":(58,"glasses",60,0.3,15.0,0.0,0.1,15,  5, 0.1),
    "Coconut Water (per glass)":   (54,"glasses",45, 0.5, 9.0, 0.5, 0.5,  3,  18, 0.1),
    "Sprouts Salad (per bowl)":    (22,"bowls",80,  7.0, 12.0, 0.5, 4.5, 10,  30, 2.0),
    "Raita (per bowl)":            (28,"bowls",70,  4.0,  8.0, 2.5, 0.3,  2, 130, 0.2),
    "Buttermilk plain (per glass)":(25,"glasses",40, 3.0, 5.0, 1.0, 0.0,  0, 100, 0.1),
    "Plain Curd (per bowl)":       (30,"bowls",100, 5.5,  8.0, 4.0, 0.0,  1, 180, 0.1),
}

FOOD_CATEGORIES = {
    "🍚 Grains & Rice":       ["White Rice (per cup)","Brown Rice (per cup)","Basmati Rice (per cup)","Roti (per piece)","Paratha (per piece)","Puri (per piece)","Naan (per piece)","Bhatura (per piece)"],
    "🌅 Breakfast":           ["Idli (per piece)","Dosa (per piece)","Uttapam (per piece)","Poha (per bowl)","Upma (per bowl)","Aloo Paratha (per piece)","Pongal (per bowl)","Sabudana Khichdi (per bowl)"],
    "🫘 Dal & Pulses":        ["Dal (per bowl)","Rajma (per bowl)","Chana Masala (per bowl)","Sambar (per bowl)","Moong Dal (per bowl)","Black Chana (per bowl)"],
    "🥦 Vegetables":          ["Aloo Sabzi (per bowl)","Mixed Veg Curry (per bowl)","Palak Paneer (per bowl)","Bhindi (per bowl)","Baingan Bharta (per bowl)"],
    "🥪 Snacks & Street Food":["Samosa (per piece)","Pakora (per piece)","Vada Pav (per piece)","Pav Bhaji (per plate)","Bhel Puri (per bowl)","Sev Puri (per piece)","Kachori (per piece)","Dhokla (per piece)"],
    "🍛 Rice Dishes":         ["Biryani (per plate)","Pulao (per plate)","Curd Rice (per bowl)","Lemon Rice (per bowl)","Tamarind Rice (per bowl)"],
    "🍮 Sweets":              ["Rasgulla (per piece)","Gulab Jamun (per piece)","Jalebi (per piece)","Kheer (per bowl)","Halwa (per bowl)","Ladoo (per piece)","Barfi (per piece)","Peda (per piece)"],
    "🥤 Drinks":              ["Chai with sugar (per cup)","Lassi sweet (per glass)","Nimbu Pani with sugar (per glass)","Coconut Water (per glass)"],
    "🥗 Healthy Options":     ["Sprouts Salad (per bowl)","Raita (per bowl)","Buttermilk plain (per glass)","Plain Curd (per bowl)"],
}

EXERCISES = {
    "🏃 Jogging":             {"cal_per_hr":500,"glucose_reduction":15,"intensity":"Medium"},
    "🚶 Walking":             {"cal_per_hr":280,"glucose_reduction":8, "intensity":"Low"},
    "🏊 Swimming":            {"cal_per_hr":600,"glucose_reduction":18,"intensity":"High"},
    "🏋️ Gym / Weightlifting": {"cal_per_hr":400,"glucose_reduction":10,"intensity":"Medium"},
    "🚴 Cycling":             {"cal_per_hr":550,"glucose_reduction":16,"intensity":"Medium"},
    "⚽ Football / Cricket":  {"cal_per_hr":480,"glucose_reduction":14,"intensity":"Medium"},
    "🧘 Yoga":                {"cal_per_hr":200,"glucose_reduction":5, "intensity":"Low"},
    "💃 Dancing / Zumba":     {"cal_per_hr":400,"glucose_reduction":12,"intensity":"Medium"},
    "🏸 Badminton":           {"cal_per_hr":450,"glucose_reduction":13,"intensity":"Medium"},
    "🤸 HIIT / Aerobics":     {"cal_per_hr":700,"glucose_reduction":20,"intensity":"High"},
    "🧗 Stair Climbing":      {"cal_per_hr":500,"glucose_reduction":15,"intensity":"Medium"},
    "🏇 Running (fast)":      {"cal_per_hr":750,"glucose_reduction":22,"intensity":"High"},
}

DIET_ADVICE = {
    "high": {
        "avoid":    ["White Rice","Maida items","Jalebi","Gulab Jamun","Biryani","Puri","Bhatura","Naan"],
        "reduce":   ["Paratha","Samosa","Vada Pav","Kheer","Chai with sugar"],
        "eat_more": ["Brown Rice","Roti","Dal","Rajma","Chana","Bhindi","Sprouts","Buttermilk","Moong Dal"],
        "tips": ["Eat small portions every 3-4 hours","Walk 30 min after meals","Replace white rice with brown rice",
                 "Drink water before meals","Avoid sweets and sugary drinks","Include methi seeds soaked overnight"],
    },
    "low": {
        "avoid":    ["Jalebi","Gulab Jamun","Ladoo","Barfi"],
        "reduce":   ["White Rice","Puri","Bhatura"],
        "eat_more": ["Dal","Rajma","Brown Rice","Vegetables","Sprouts","Curd"],
        "tips": ["Maintain your current healthy diet!","Exercise 30 min daily","Check glucose yearly",
                 "Reduce refined sugar gradually","Include more fiber — vegetables and pulses"],
    }
}

DAILY_NEEDS = {"calories":2000,"protein":50,"carbs":275,"fat":65,"fiber":28,"vit_c":90,"calcium":1000,"iron":18}

def get_nutrition(food_name, qty):
    gi, unit, cal, prot, carb, fat, fib, vitc, calc, iron = FOODS[food_name]
    return {"calories":cal*qty,"protein":prot*qty,"carbs":carb*qty,"fat":fat*qty,
            "fiber":fib*qty,"vit_c":vitc*qty,"calcium":calc*qty,"iron":iron*qty,"gi":gi,"unit":unit}

def sum_nutrition(food_list):
    total = {k:0.0 for k in ["calories","protein","carbs","fat","fiber","vit_c","calcium","iron"]}
    for fname, qty in food_list:
        n = get_nutrition(fname, qty)
        for k in total: total[k] += n[k]
    return total

# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Indian Diet Diabetes Risk", page_icon="🍛", layout="wide")
st.title("🍛 Indian Diet & Diabetes Risk Predictor")
st.markdown("*ML-powered • Nutrition tracking • Meal planner*")

df = load_data()
model, acc = train_model(df)

# ── Sidebar: Health Details ───────────────────────────────────────────────────
st.sidebar.header("🧑 Your Health Details")
name = st.sidebar.text_input("Name (optional)", placeholder="e.g. Rahul Sharma")
st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ BMI Calculator")
with st.sidebar.expander("Calculate BMI"):
    height_cm = st.number_input("Height (cm)", 100, 220, 165)
    weight_kg = st.number_input("Weight (kg)", 30, 200, 70)
    calc_bmi = weight_kg / ((height_cm/100)**2)
    bmi_cat = "Underweight" if calc_bmi<18.5 else "Normal" if calc_bmi<25 else "Overweight" if calc_bmi<30 else "Obese"
    st.metric("Your BMI", f"{calc_bmi:.1f}", bmi_cat)
st.sidebar.markdown("---")
age         = st.sidebar.slider("Age", 18, 80, 28)
bmi         = st.sidebar.slider("BMI", 15.0, 50.0, float(round(calc_bmi,1)))
glucose     = st.sidebar.slider("Glucose (mg/dL)", 50, 200, 100)
bp          = st.sidebar.slider("Blood Pressure (mmHg)", 40, 130, 72)
pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 1)

# ════════════════════════════════════════════════════════════════════════════
# THREE-COLUMN LAYOUT
# ════════════════════════════════════════════════════════════════════════════
left_col, mid_col, right_col = st.columns([2, 2, 2])

# ─────────────────────────────────────────────────────────────────────────────
# LEFT COLUMN — Food selector + Custom food entry
# ─────────────────────────────────────────────────────────────────────────────
with left_col:
    st.subheader("🥘 Food Selector")
    st.caption("Select foods and adjust quantity — results update instantly")

    # Custom food entry
    with st.expander("➕ Add Custom Food", expanded=False):
        cf_name  = st.text_input("Food name", placeholder="e.g. Appam, Puttu, Pesarattu")
        cf_cal   = st.number_input("Calories (per serving)", 0, 2000, 100)
        cf_prot  = st.number_input("Protein (g)", 0.0, 100.0, 3.0, step=0.5)
        cf_carb  = st.number_input("Carbs (g)", 0.0, 200.0, 20.0, step=1.0)
        cf_fat   = st.number_input("Fat (g)", 0.0, 100.0, 2.0, step=0.5)
        cf_fiber = st.number_input("Fiber (g)", 0.0, 50.0, 1.0, step=0.5)
        cf_gi    = st.slider("Glycemic Index", 0, 100, 50)
        if st.button("✅ Add to list") and cf_name.strip():
            key = f"{cf_name.strip()} (custom)"
            FOODS[key] = (cf_gi, "servings", cf_cal, cf_prot, cf_carb, cf_fat, cf_fiber, 0, 0, 0)
            if "🍽️ Custom Foods" not in FOOD_CATEGORIES:
                FOOD_CATEGORIES["🍽️ Custom Foods"] = []
            FOOD_CATEGORIES["🍽️ Custom Foods"].append(key)
            st.success(f"Added: {key}")

    selected_with_qty = []
    for category, items in FOOD_CATEGORIES.items():
        with st.expander(category):
            for item in items:
                if item not in FOODS:
                    continue
                gi, unit, cal, *_ = FOODS[item]
                c1, c2, c3 = st.columns([3, 2, 2])
                with c1:
                    checked = st.checkbox(item, key=f"chk_{item}")
                with c2:
                    qty = st.number_input("qty", min_value=0.5, max_value=20.0, value=1.0, step=0.5,
                                          key=f"qty_{item}", label_visibility="collapsed") if checked else 1.0
                with c3:
                    if checked:
                        gi_icon = "🔴" if gi>60 else "🟡" if gi>50 else "🟢"
                        st.caption(f"{gi_icon} GI:{gi}\n{int(cal*qty)} kcal")
                if checked:
                    selected_with_qty.append((item, qty))

# ─────────────────────────────────────────────────────────────────────────────
# MIDDLE COLUMN — Nutrition breakdown
# ─────────────────────────────────────────────────────────────────────────────
with mid_col:
    st.subheader("📊 Nutrition Breakdown")

    total_nut = sum_nutrition(selected_with_qty)

    # Macro cards
    st.markdown("**Macronutrients**")
    m1, m2, m3 = st.columns(3)
    m1.metric("🍗 Protein",  f"{total_nut['protein']:.1f}g",  f"{total_nut['protein']/DAILY_NEEDS['protein']*100:.0f}% DV")
    m2.metric("🍞 Carbs",    f"{total_nut['carbs']:.1f}g",    f"{total_nut['carbs']/DAILY_NEEDS['carbs']*100:.0f}% DV")
    m3.metric("🧈 Fat",      f"{total_nut['fat']:.1f}g",      f"{total_nut['fat']/DAILY_NEEDS['fat']*100:.0f}% DV")

    st.markdown("**Other Nutrients**")
    n1, n2 = st.columns(2)
    n1.metric("🌾 Fiber",    f"{total_nut['fiber']:.1f}g",    f"{total_nut['fiber']/DAILY_NEEDS['fiber']*100:.0f}% DV")
    n2.metric("🔥 Calories", f"{total_nut['calories']:.0f}",  f"{total_nut['calories']/DAILY_NEEDS['calories']*100:.0f}% DV")

    st.markdown("**Vitamins & Minerals**")
    v1, v2, v3 = st.columns(3)
    v1.metric("🍋 Vit C",   f"{total_nut['vit_c']:.1f}mg",   f"{total_nut['vit_c']/DAILY_NEEDS['vit_c']*100:.0f}% DV")
    v2.metric("🦴 Calcium", f"{total_nut['calcium']:.0f}mg",  f"{total_nut['calcium']/DAILY_NEEDS['calcium']*100:.0f}% DV")
    v3.metric("🩸 Iron",    f"{total_nut['iron']:.1f}mg",     f"{total_nut['iron']/DAILY_NEEDS['iron']*100:.0f}% DV")

    st.markdown("---")
    # Nutrition bar chart
    if selected_with_qty:
        st.markdown("**% of Daily Value**")
        nutrients = ["Calories","Protein","Carbs","Fat","Fiber","Vit C","Calcium","Iron"]
        keys      = ["calories","protein","carbs","fat","fiber","vit_c","calcium","iron"]
        pcts = [min(total_nut[k]/DAILY_NEEDS[k]*100, 150) for k in keys]
        colors_bar = ['#e74c3c' if p>100 else '#f39c12' if p>70 else '#27ae60' for p in pcts]

        fig_n, ax_n = plt.subplots(figsize=(5, 4))
        bars = ax_n.barh(nutrients, pcts, color=colors_bar)
        ax_n.axvline(x=100, color='gray', linestyle='--', alpha=0.6, label='100% DV')
        ax_n.set_xlabel('% of Daily Value')
        ax_n.set_title('Nutrition vs Daily Needs', fontsize=12, fontweight='bold')
        for bar, pct in zip(bars, pcts):
            ax_n.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                      f'{pct:.0f}%', va='center', fontsize=9)
        ax_n.set_xlim(0, 160)
        plt.tight_layout()
        st.pyplot(fig_n)
        plt.close()

        # Macro pie chart
        st.markdown("**Macro Split**")
        prot_cal = total_nut['protein'] * 4
        carb_cal = total_nut['carbs']   * 4
        fat_cal  = total_nut['fat']     * 9
        total_macro_cal = prot_cal + carb_cal + fat_cal
        if total_macro_cal > 0:
            fig_p, ax_p = plt.subplots(figsize=(4, 3))
            ax_p.pie(
                [prot_cal, carb_cal, fat_cal],
                labels=[f"Protein\n{prot_cal/total_macro_cal*100:.0f}%",
                        f"Carbs\n{carb_cal/total_macro_cal*100:.0f}%",
                        f"Fat\n{fat_cal/total_macro_cal*100:.0f}%"],
                colors=['#3498db','#e67e22','#e74c3c'],
                startangle=90, autopct='',
            )
            ax_p.set_title('Calorie Source', fontsize=11, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_p)
            plt.close()

        # Per-food breakdown table
        st.markdown("**Per-food details**")
        rows = []
        for fname, qty in selected_with_qty:
            n = get_nutrition(fname, qty)
            rows.append({
                "Food":     fname[:22],
                "Qty":      f"{qty} {n['unit']}",
                "Cal":      f"{n['calories']:.0f}",
                "Prot(g)":  f"{n['protein']:.1f}",
                "Carb(g)":  f"{n['carbs']:.1f}",
                "Fat(g)":   f"{n['fat']:.1f}",
                "Fiber(g)": f"{n['fiber']:.1f}",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Select foods on the left to see nutrition details here.")

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT COLUMN — Meal Planner
# ─────────────────────────────────────────────────────────────────────────────
with right_col:
    st.subheader("🗓️ Meal Planner")
    st.caption("Plan your day — add from the list or type custom foods")

    MEALS = ["🌅 Breakfast", "☀️ Lunch", "🌤️ Afternoon Snack", "🌙 Dinner"]

    # Init session state
    for meal in MEALS:
        key = f"meal_{meal}"
        if key not in st.session_state:
            st.session_state[key] = []  # list of (name, qty, cal)

    all_food_names = list(FOODS.keys())

    for meal in MEALS:
        skey = f"meal_{meal}"
        with st.expander(meal, expanded=True):
            # Add from dropdown — calories are automatic
            col_a, col_b, col_c = st.columns([3, 1, 1])
            with col_a:
                pick = st.selectbox("Pick food", ["— select —"] + all_food_names, key=f"pick_{meal}")
            with col_b:
                qty_pick = st.number_input("Qty", 0.5, 20.0, 1.0, 0.5, key=f"pickqty_{meal}")
            with col_c:
                st.markdown("<div style='margin-top:28px'>", unsafe_allow_html=True)
                if st.button("Add", key=f"addbtn_{meal}") and pick != "— select —":
                    cal = FOODS[pick][2] * qty_pick
                    unit = FOODS[pick][1]
                    st.session_state[skey].append((pick, qty_pick, round(cal, 1), unit))
                st.markdown("</div>", unsafe_allow_html=True)

            # Show auto-calculated calories preview before adding
            if pick != "— select —":
                preview_cal = FOODS[pick][2] * qty_pick
                preview_unit = FOODS[pick][1]
                st.caption(f"📊 Auto: {qty_pick} {preview_unit} = **{preview_cal:.0f} kcal** | "
                           f"Protein: {FOODS[pick][3]*qty_pick:.1f}g | "
                           f"Carbs: {FOODS[pick][4]*qty_pick:.1f}g | "
                           f"Fat: {FOODS[pick][5]*qty_pick:.1f}g")

            # Custom food — only name needed, calories come from left panel custom food entry
            st.markdown("**Or add a custom food** *(first add it via ➕ Add Custom Food on the left)*")
            cc1, cc2 = st.columns([4, 1])
            with cc1:
                custom_pick = st.selectbox(
                    "Custom foods", 
                    ["— select —"] + [k for k in FOODS if "(custom)" in k],
                    key=f"custompick_{meal}"
                )
            with cc2:
                cqty = st.number_input("Qty", 0.5, 20.0, 1.0, 0.5, key=f"cqty_{meal}")
            if st.button("Add custom", key=f"caddbtn_{meal}") and custom_pick != "— select —":
                ccal = FOODS[custom_pick][2] * cqty
                cunit = FOODS[custom_pick][1]
                st.session_state[skey].append((custom_pick, cqty, round(ccal, 1), cunit))
                st.success(f"Added {custom_pick} — {ccal:.0f} kcal (auto-calculated)")

            # Show meal items
            if st.session_state[skey]:
                meal_total_cal = 0
                to_remove = []
                for i, item in enumerate(st.session_state[skey]):
                    fname, qty, cal = item[0], item[1], item[2]
                    unit = item[3] if len(item) > 3 else "serving(s)"
                    r1, r2, r3 = st.columns([4, 2, 1])
                    with r1:
                        st.markdown(f"• **{fname[:28]}**")
                    with r2:
                        st.caption(f"{qty} {unit} = **{cal:.0f} kcal** ✅")
                    with r3:
                        if st.button("🗑️", key=f"del_{meal}_{i}"):
                            to_remove.append(i)
                    meal_total_cal += cal
                for i in reversed(to_remove):
                    st.session_state[skey].pop(i)
                st.markdown(f"**Meal total: {meal_total_cal:.0f} kcal**")
            else:
                st.caption("Nothing added yet.")

    # Daily planner summary
    st.markdown("---")
    st.markdown("**📋 Daily Plan Summary**")
    grand_total = 0
    for meal in MEALS:
        items = st.session_state.get(f"meal_{meal}", [])
        if items:
            meal_cal = sum(c for _, _, c in items)
            grand_total += meal_cal
            st.markdown(f"{meal}: **{meal_cal:.0f} kcal**")
    if grand_total > 0:
        pct = grand_total / DAILY_NEEDS['calories'] * 100
        color = "🔴" if pct > 110 else "🟡" if pct > 85 else "🟢"
        st.markdown(f"**Total: {grand_total:.0f} kcal** {color} ({pct:.0f}% of 2000 kcal goal)")
        st.progress(min(int(pct), 100))

    if st.button("🗑️ Clear all meals"):
        for meal in MEALS:
            st.session_state[f"meal_{meal}"] = []
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# DIABETES RISK PREDICTION (full width below)
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🏃 Exercise Tracker")
exercise_summary = []
total_cal_burned = 0
total_glucose_reduction = 0.0

ex_cols = st.columns(4)
for i, (ex_name, ex_data) in enumerate(EXERCISES.items()):
    with ex_cols[i % 4]:
        if st.checkbox(ex_name, key=f"ex_{ex_name}"):
            hrs = st.slider("Hours", 0.5, 5.0, 1.0, 0.5, key=f"hrs_{ex_name}")
            cal_b  = int(ex_data['cal_per_hr'] * hrs)
            gluc_r = round(ex_data['glucose_reduction'] * hrs, 1)
            ic = "🔴" if ex_data['intensity']=="High" else "🟡" if ex_data['intensity']=="Medium" else "🟢"
            st.caption(f"{ic} {cal_b} kcal | -{gluc_r} mg/dL")
            exercise_summary.append((ex_name, hrs))
            total_cal_burned    += cal_b
            total_glucose_reduction += gluc_r

if exercise_summary:
    st.success(f"💪 Total: **{total_cal_burned} kcal burned** | Glucose reduced by **{total_glucose_reduction:.1f} mg/dL**")

# Prediction
avg_gi = (sum(FOODS[f][0]*q for f,q in selected_with_qty)/sum(q for _,q in selected_with_qty)
          if selected_with_qty else 55.0)
sim_glucose = max(50, glucose + (avg_gi-55)*0.5 - total_glucose_reduction)
person = pd.DataFrame([{"Pregnancies":pregnancies,"Glucose":sim_glucose,"BloodPressure":bp,
                         "SkinThickness":29,"Insulin":125,"BMI":bmi,"DiabetesPedigreeFunction":0.37,"Age":age}])
prob   = model.predict_proba(person)[0][1] * 100
result = model.predict(person)[0]
advice_key = "high" if result==1 else "low"
risk_label = f"Higher Risk — {prob:.1f}% probability" if result==1 else f"Lower Risk — {prob:.1f}% probability"

st.markdown("---")
st.subheader("📊 Diabetes Risk Result")
rc1, rc2, rc3, rc4 = st.columns(4)
rc1.metric("Risk Score",      f"{prob:.1f}%",           "High" if result==1 else "Low")
rc2.metric("Diet Avg GI",     f"{avg_gi:.0f}",          "High" if avg_gi>60 else "Normal")
rc3.metric("Calories Burned", f"{total_cal_burned} kcal")
rc4.metric("Model Accuracy",  f"{acc*100:.1f}%")

if result==1: st.error(f"⚠️ {risk_label}")
else:         st.success(f"✅ {risk_label}")
st.progress(int(prob))
if total_glucose_reduction>0:
    st.info(f"🏃 Exercise lowered simulated glucose by {total_glucose_reduction:.1f} mg/dL — great work!")

# Diet advice
st.markdown("---")
st.subheader("🥗 Diet Recommendations")
advice = DIET_ADVICE[advice_key]
da1, da2, da3 = st.columns(3)
with da1:
    st.markdown("### ✅ Eat More")
    for item in advice['eat_more']: st.markdown(f"- {item}")
with da2:
    st.markdown("### ⚠️ Reduce")
    for item in advice['reduce']:   st.markdown(f"- {item}")
with da3:
    st.markdown("### ❌ Avoid")
    for item in advice['avoid']:    st.markdown(f"- {item}")
st.markdown("### 💡 Health Tips")
for tip in advice['tips']: st.info(tip)

# PDF Download
st.markdown("---")
st.subheader("📄 Download Report")
try:
    import reportlab
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.units import cm

    def make_pdf():
        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
        s_title = ParagraphStyle('t', fontSize=18, fontName='Helvetica-Bold', spaceAfter=6)
        s_sec   = ParagraphStyle('h', fontSize=12, fontName='Helvetica-Bold', spaceBefore=12, spaceAfter=4)
        s_body  = ParagraphStyle('b', fontSize=10, spaceAfter=3, leading=14)
        story   = []
        story.append(Paragraph("Indian Diet & Diabetes Risk Report", s_title))
        story.append(Paragraph(f"Name: {name if name else 'Patient'} | Age: {age} | BMI: {bmi:.1f}", s_body))
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph(f"Risk: {risk_label}", ParagraphStyle('r', fontSize=13, fontName='Helvetica-Bold',
            textColor=rl_colors.HexColor('#e74c3c' if advice_key=='high' else '#27ae60'), spaceAfter=8)))

        if selected_with_qty:
            story.append(Paragraph("Foods Selected", s_sec))
            rows = [['Food','Qty','Cal','Protein','Carbs','Fat','Fiber']]
            for fname, qty in selected_with_qty:
                n = get_nutrition(fname, qty)
                rows.append([fname[:30], f"{qty} {n['unit']}", f"{n['calories']:.0f}",
                             f"{n['protein']:.1f}g", f"{n['carbs']:.1f}g", f"{n['fat']:.1f}g", f"{n['fiber']:.1f}g"])
            t = Table(rows, colWidths=[5*cm,2*cm,1.5*cm,2*cm,2*cm,1.5*cm,1.5*cm])
            t.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),rl_colors.HexColor('#2c3e50')),
                ('TEXTCOLOR',(0,0),(-1,0),rl_colors.white),
                ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                ('FONTSIZE',(0,0),(-1,-1),8),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[rl_colors.HexColor('#f8f9fa'),rl_colors.white]),
                ('GRID',(0,0),(-1,-1),0.5,rl_colors.HexColor('#dddddd')),
                ('PADDING',(0,0),(-1,-1),4),
            ]))
            story.append(t)

        story.append(Paragraph("Nutrition Totals", s_sec))
        nut_rows = [['Nutrient','Amount','% Daily Value']]
        for label, k, unit_str in [("Calories","calories","kcal"),("Protein","protein","g"),
                                    ("Carbs","carbs","g"),("Fat","fat","g"),("Fiber","fiber","g"),
                                    ("Vitamin C","vit_c","mg"),("Calcium","calcium","mg"),("Iron","iron","mg")]:
            nut_rows.append([label, f"{total_nut[k]:.1f} {unit_str}", f"{total_nut[k]/DAILY_NEEDS[k]*100:.0f}%"])
        nt = Table(nut_rows, colWidths=[5*cm,5*cm,5*cm])
        nt.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),rl_colors.HexColor('#145a32')),
            ('TEXTCOLOR',(0,0),(-1,0),rl_colors.white),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,-1),10),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[rl_colors.HexColor('#eafaf1'),rl_colors.white]),
            ('GRID',(0,0),(-1,-1),0.5,rl_colors.HexColor('#dddddd')),
            ('PADDING',(0,0),(-1,-1),5),
        ]))
        story.append(nt)

        adv = DIET_ADVICE[advice_key]
        story.append(Paragraph("Diet Advice", s_sec))
        story.append(Paragraph("Eat More: " + ", ".join(adv['eat_more']), s_body))
        story.append(Paragraph("Reduce: "   + ", ".join(adv['reduce']),   s_body))
        story.append(Paragraph("Avoid: "    + ", ".join(adv['avoid']),    s_body))
        story.append(Spacer(1, 0.4*cm))
        story.append(Paragraph("Disclaimer: For educational purposes only. Consult a doctor.", ParagraphStyle('d', fontSize=8, textColor=rl_colors.HexColor('#888888'))))
        doc.build(story)
        buf.seek(0)
        return buf

    pdf_buf = make_pdf()
    st.download_button("⬇️ Download PDF Report", data=pdf_buf,
                       file_name=f"diabetes_risk_{name.replace(' ','_') if name else 'report'}.pdf",
                       mime="application/pdf")
except ImportError:
    st.warning("Add 'reportlab' to requirements.txt to enable PDF download.")

st.caption(f"Model accuracy: {acc*100:.1f}% | For educational purposes only. Consult a doctor for medical advice.")
