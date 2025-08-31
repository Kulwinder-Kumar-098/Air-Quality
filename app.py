import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Page setup
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

st.title("🌫️ Air Quality Analysis & AQI Prediction")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    return df

df = load_data()

# Define feature columns
features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'Holidays_Count', 'Days']
target = 'AQI'

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 Data Overview", "📊 Visualizations", "🤖 Model Training", "🔮 Predict AQI"])

# ---------------- TAB 1: Data Overview ----------------
with tab1:
    st.subheader("📄 Raw Data")
    st.dataframe(df)

    st.subheader("📈 Summary Statistics")
    st.write(df.describe())

# ---------------- TAB 2: Visualizations ----------------
with tab2:
    st.subheader("📌 Correlation Heatmap")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df[features + [target]].corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader("📉 AQI Trend Over Days")
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="Date", y="AQI", marker="o", ax=ax)
    ax.set_title("Daily AQI")
    st.pyplot(fig)

    st.subheader("🔎 Pairplot of Pollutants & AQI")
    sns_plot = sns.pairplot(df[features + [target]])
    st.pyplot(sns_plot.fig)

    st.subheader("📅 AQI by Holiday vs Non-Holiday")
    holiday_data = df.groupby("Holidays_Count")["AQI"].mean().reset_index()
    holiday_data["Holidays_Count"] = holiday_data["Holidays_Count"].map({0: "Non-Holiday", 1: "Holiday"})

    fig2, ax2 = plt.subplots()
    sns.barplot(data=holiday_data, x="Holidays_Count", y="AQI", ax=ax2)
    ax2.set_title("Average AQI: Holiday vs Non-Holiday")
    st.pyplot(fig2)

# ---------------- TAB 3: Model Training ----------------
with tab3:
    st.subheader("🤖 Train a Model to Predict AQI")

    # Split the data
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.markdown("### 🧪 Model Performance")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # Actual vs Predicted
    st.markdown("### 📉 Actual vs Predicted AQI")
    fig3, ax3 = plt.subplots()
    ax3.scatter(y_test, y_pred, alpha=0.7)
    ax3.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax3.set_xlabel("Actual AQI")
    ax3.set_ylabel("Predicted AQI")
    ax3.set_title("Actual vs Predicted AQI")
    st.pyplot(fig3)

# ---------------- TAB 4: Prediction ----------------
with tab4:
    st.subheader("🔮 Predict AQI from Custom Inputs")

    input_data = {}
    for col in features:
        default = float(df[col].mean())
        input_data[col] = st.number_input(f"{col}", value=round(default, 2))

    if st.button("Predict AQI"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.success(f"🌬️ Predicted AQI: {prediction:.2f}")
