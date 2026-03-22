import streamlit as st
from model import LogisticRegressionMulti
from utils import load_data, accuracy, train_test_split

st.set_page_config(page_title="Placement Predictor", layout="wide")

# 🔥 Custom CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
    }
    .sub-text {
        text-align: center;
        color: grey;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">🎯 Student Placement Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">A data-driven system to predict student placement outcomes using academic metrics and practical experience</div>', unsafe_allow_html=True)

st.divider()

# Upload
uploaded_file = st.file_uploader("📂 Upload Dataset (CSV)", type=["csv"])

if uploaded_file:
    X, Y = load_data(uploaded_file)

    if len(X) == 0:
        st.error("❌ Invalid or empty dataset")
    else:
        # Split
        X_train, Y_train, X_test, Y_test = train_test_split(X, Y)

        # Train
        model = LogisticRegressionMulti(lr=0.1, epochs=3000)
        model.train(X_train, Y_train)

        # Accuracy
        train_acc = accuracy(model, X_train, Y_train)
        test_acc = accuracy(model, X_test, Y_test)

        # 📊 Metrics Row
        col1, col2 = st.columns(2)

        with col1:
            st.metric("📘 Train Accuracy", f"{train_acc:.2f}")

        with col2:
            st.metric("📊 Test Accuracy", f"{test_acc:.2f}")

        st.divider()

        st.subheader("🔍 Predict Placement")

        # 🎯 Input in columns
        col1, col2 = st.columns(2)

        with col1:
            hours = st.slider("📚 Study Hours", 0.0, 12.0, 5.0)
            projects = st.slider("💻 Projects", 0, 10, 2)

        with col2:
            internships = st.slider("🏢 Internships", 0, 5, 1)
            aptitude = st.slider("🧠 Aptitude Score", 0, 100, 60)

        features = [hours, projects, internships, aptitude]

        # Prediction
        prob = model.predict_proba(features)
        result = model.predict(features)

        st.divider()

        # 🎯 Result Section
        st.subheader("📈 Prediction Result")

        st.metric("Placement Probability", f"{prob:.2f}")

        # 🔥 Color based progress
        if prob > 0.7:
            st.success("🎉 High Placement Chance")
        elif prob > 0.4:
            st.warning("⚠️ Moderate Chance")
        else:
            st.error("❌ Low Placement Chance")

        st.progress(int(prob * 100))

        # 📊 Extra Insight
        st.caption("💡 Tip: Increase projects, internships, and aptitude score to improve placement chances.")