import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Mobile Price Predictor",
    page_icon="📱",
    layout="centered"
)

# Colorful page design
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e0f7fa, #f3e5f5, #fff3e0);
}

.main-title {
    text-align: center;
    color: #6a1b9a;
    font-size: 38px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #37474f;
    font-size: 17px;
}

.result-card {
    background: linear-gradient(120deg, #7b2ff7, #f107a3);
    padding: 28px;
    border-radius: 22px;
    color: white;
    text-align: center;
    box-shadow: 0 8px 20px #b39ddb;
}

.result-card h1 {
    color: white;
    font-size: 44px;
}

.result-card p {
    color: white;
    font-size: 20px;
}

div.stButton > button {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 12px 24px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    color: white;
    border: 2px solid white;
}

/* Only Mobile Specifications headings are black */
[data-testid="stNumberInput"] label,
[data-testid="stNumberInput"] label p {
    color: black !important;
}

/* Mobile Summary heading in black */
h3 {
    color: black !important;
}

/* Mobile Summary labels and values in black */
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] *,
[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] * {
    color: black !important;
}

[data-testid="stCaptionContainer"] {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)


# Page title
st.markdown(
    '<div class="main-title">📱 Mobile Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">✨ Smart predictions from mobile specifications ✨</div>',
    unsafe_allow_html=True
)

st.write("")

st.info(
    "💡 Enter your mobile details below and discover "
    "its predicted price category!"
)


# Train the model
@st.cache_resource
def train_model():
    df = pd.read_csv("mobile.csv")

    features = [
        "battery_power",
        "ram",
        "px_height",
        "px_width",
        "int_memory"
    ]

    X = df[features]
    y = df["price_range"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    return model, df, features


try:
    model, df, features = train_model()

    st.subheader("🎛️ Mobile Specifications")

    col1, col2 = st.columns(2)

    with col1:
        battery_power = st.number_input(
            "🔋 Battery Power",
            min_value=int(df["battery_power"].min()),
            max_value=int(df["battery_power"].max()),
            value=int(df["battery_power"].median())
        )

        ram = st.number_input(
            "🧠 RAM",
            min_value=int(df["ram"].min()),
            max_value=int(df["ram"].max()),
            value=int(df["ram"].median())
        )

        px_height = st.number_input(
            "📏 Pixel Height",
            min_value=int(df["px_height"].min()),
            max_value=int(df["px_height"].max()),
            value=int(df["px_height"].median())
        )

    with col2:
        px_width = st.number_input(
            "🖥️ Pixel Width",
            min_value=int(df["px_width"].min()),
            max_value=int(df["px_width"].max()),
            value=int(df["px_width"].median())
        )

        int_memory = st.number_input(
            "💾 Internal Memory",
            min_value=int(df["int_memory"].min()),
            max_value=int(df["int_memory"].max()),
            value=int(df["int_memory"].median())
        )

        st.markdown("""
        <div style="
            background:#ffffffb8;
            padding:14px;
            border-radius:12px;
            text-align:center;
        ">
            <span style="color:#6a1b9a;font-weight:bold;">
                🤖 AI Model
            </span><br>
            <span style="color:#37474f;">
                Random Forest Classifier
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Predict price range
    if st.button("🎉 Predict Price Range 🎉"):

        input_data = pd.DataFrame(
            [{
                "battery_power": battery_power,
                "ram": ram,
                "px_height": px_height,
                "px_width": px_width,
                "int_memory": int_memory
            }],
            columns=features
        )

        prediction = int(model.predict(input_data)[0])

        probabilities = model.predict_proba(input_data)[0]

        confidence = float(max(probabilities) * 100)

        meanings = {
            0: "Lowest price category",
            1: "Low-to-medium price category",
            2: "Medium-to-high price category",
            3: "Highest price category"
        }

        # Celebration animation
        st.balloons()

        st.markdown("---")

        st.markdown(
            f"""
            <div class="result-card">
                <p>🎊 YOUR PREDICTION RESULT 🎊</p>
                <h1>Class {prediction}</h1>
                <p>{meanings[prediction]}</p>
                <p>✨ Prediction completed successfully! ✨</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        # Mobile Summary
        st.subheader("📊 Your Mobile Summary")

        c1, c2 = st.columns(2)

        with c1:
            st.metric("🧠 RAM", f"{ram}")
            st.metric("🔋 Battery Power", f"{battery_power}")
            st.metric("💾 Internal Memory", f"{int_memory}")

        with c2:
            st.metric("📏 Pixel Height", f"{px_height}")
            st.metric("🖥️ Pixel Width", f"{px_width}")
            st.metric("🤖 Model Probability", f"{confidence:.1f}%")

        st.caption(
            "Price categories are defined by the dataset. "
            "Model probability is not a guarantee of prediction correctness."
        )

        st.success(
            "🌟 Thank you for using Mobile Price Predictor!"
        )

except FileNotFoundError:
    st.error(
        "⚠️ mobile.csv not found. Keep it beside app.py."
    )

except Exception as e:
    st.error(f"Something went wrong: {e}")