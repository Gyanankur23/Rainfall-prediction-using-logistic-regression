# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Rainfall Prediction India", layout="wide")

# --- Regions and States mapping ---
regions = {
    "North": ["Delhi", "Punjab", "Haryana", "Uttar Pradesh", "Uttarakhand", "Himachal Pradesh"],
    "South": ["Kerala", "Tamil Nadu", "Karnataka", "Andhra Pradesh", "Telangana"],
    "East": ["West Bengal", "Odisha", "Bihar", "Jharkhand"],
    "West": ["Rajasthan", "Gujarat", "Maharashtra", "Goa"],
    "Central": ["Madhya Pradesh", "Chhattisgarh"],
    "Northeast": ["Assam", "Meghalaya", "Manipur", "Mizoram", "Nagaland", "Tripura", "Arunachal Pradesh", "Sikkim"],
    "Union Territories": ["Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
                          "Lakshadweep", "Puducherry", "Jammu and Kashmir", "Ladakh"]
}

# --- Synthetic dataset generator ---
@st.cache_data
def generate_data(samples_per_combo=60, seed=42):
    rng = np.random.default_rng(seed)
    circumstances = ["Monsoon", "Winter", "Summer"]

    data = []
    for region, states in regions.items():
        for state in states:
            for season in circumstances:
                n = samples_per_combo
                humidity = rng.uniform(30, 100, n)
                temperature = rng.uniform(10, 40, n)
                wind_speed = rng.uniform(0, 12, n)

                base = {"Monsoon": 50, "Winter": 20, "Summer": 10}[season]

                rainfall = (
                    0.6 * humidity
                    - 0.4 * temperature
                    - 0.3 * wind_speed
                    + base
                    + rng.normal(0, 10, n)
                )
                rainfall = np.clip(rainfall, 0, None)

                df = pd.DataFrame({
                    "region": region,
                    "state": state,
                    "season": season,
                    "humidity": humidity,
                    "temperature": temperature,
                    "wind_speed": wind_speed,
                    "rainfall_mm": rainfall
                })
                data.append(df)
    return pd.concat(data, ignore_index=True)

df = generate_data()

# --- Train model ---
@st.cache_resource
def train_model(df):
    df_encoded = pd.get_dummies(df, columns=["region", "state", "season"], drop_first=True)
    X = df_encoded.drop("rainfall_mm", axis=1)
    y = df_encoded["rainfall_mm"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test, df_encoded

model, X_train, X_test, y_train, y_test, df_encoded = train_model(df)

# --- Sidebar controls ---
st.sidebar.header("Controls")
region_choice = st.sidebar.selectbox("Select Region", list(regions.keys()))
state_choice = st.sidebar.selectbox("Select State", regions[region_choice])
season_choice = st.sidebar.selectbox("Select Season", ["Monsoon", "Winter", "Summer"])
hum = st.sidebar.slider("Humidity (%)", 30.0, 100.0, 70.0, 0.5)
temp = st.sidebar.slider("Temperature (°C)", 10.0, 40.0, 25.0, 0.5)
wind = st.sidebar.slider("Wind speed (m/s)", 0.0, 12.0, 3.0, 0.1)

# --- Prediction ---
user_df = pd.DataFrame({
    "humidity": [hum],
    "temperature": [temp],
    "wind_speed": [wind],
    "region": [region_choice],
    "state": [state_choice],
    "season": [season_choice]
})
user_encoded = pd.get_dummies(user_df, columns=["region", "state", "season"], drop_first=True)
for col in X_train.columns:
    if col not in user_encoded.columns:
        user_encoded[col] = 0
user_encoded = user_encoded[X_train.columns]
pred = model.predict(user_encoded)[0]

st.title("Rainfall Prediction Across Indian States")
st.metric(f"Predicted rainfall in {state_choice} ({season_choice})", f"{pred:.2f} mm")

# --- Model evaluation ---
st.subheader("Model Performance")
y_pred_test = model.predict(X_test)
st.write(f"R² Score: {r2_score(y_test, y_pred_test):.3f}")
st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_test):.2f} mm")

# --- Scatter plot ---
st.subheader("Humidity vs Rainfall")
fig1, ax1 = plt.subplots()
subset = df[(df["state"]==state_choice) & (df["season"]==season_choice)]
ax1.scatter(subset["humidity"], subset["rainfall_mm"], alpha=0.5)
ax1.set_xlabel("Humidity (%)")
ax1.set_ylabel("Rainfall (mm)")
st.pyplot(fig1)

# --- Residuals histogram ---
st.subheader("Residuals Distribution")
residuals = y_test - y_pred_test
fig2, ax2 = plt.subplots()
ax2.hist(residuals, bins=30, color="purple", alpha=0.7)
ax2.set_xlabel("Residual (Actual - Predicted)")
ax2.set_ylabel("Frequency")
st.pyplot(fig2)

# --- Correlation heatmap (matplotlib only) ---
st.subheader("Feature Correlation Heatmap")
corr = df[["humidity","temperature","wind_speed","rainfall_mm"]].corr()
fig3, ax3 = plt.subplots()
cax = ax3.matshow(corr, cmap="coolwarm")
fig3.colorbar(cax)
ax3.set_xticks(range(len(corr.columns)))
ax3.set_yticks(range(len(corr.columns)))
ax3.set_xticklabels(corr.columns, rotation=45)
ax3.set_yticklabels(corr.columns)
for (i, j), val in np.ndenumerate(corr.values):
    ax3.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")
st.pyplot(fig3)

# --- Region-wise bar chart ---
st.subheader(f"Average Rainfall in {region_choice}")
avg_rainfall = df[df["region"]==region_choice].groupby("state")["rainfall_mm"].mean().sort_values()
fig4, ax4 = plt.subplots()
avg_rainfall.plot(kind="bar", ax=ax4, color="skyblue")
ax4.set_ylabel("Average Rainfall (mm)")
st.pyplot(fig4)

# --- Pie chart of seasonal distribution ---
st.subheader("Seasonal Rainfall Share")
seasonal_avg = df.groupby("season")["rainfall_mm"].mean()
fig5, ax5 = plt.subplots()
ax5.pie(seasonal_avg, labels=seasonal_avg.index, autopct="%1.1f%%", startangle=90)
st.pyplot(fig5)

# --- Dataset preview ---
st.subheader("Dataset Sample")
st.dataframe(df.sample(15))
