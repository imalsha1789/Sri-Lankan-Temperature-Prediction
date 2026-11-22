
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Sri Lanka Weather AI",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "stacking_model_time_based.pkl"
DATA_PATH = BASE_DIR / "dataset" / "cleaned_weather.csv"


# ==================================================
# 2. BLUE AND WHITE THEME
# ==================================================

st.markdown("""
<style>

:root {
    --background: #F1F7FC;
    --white: #FFFFFF;
    --blue: #2563A6;
    --blue-dark: #1E5792;
    --text: #334155;
    --muted: #64748B;
    --light-blue: #DCEEFF;
    --border: #C7DCEC;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: var(--background) !important;
    color: var(--text) !important;
}

.block-container {
    max-width: 1380px;
    padding: 2rem 2rem 2rem 2rem;
}

header[data-testid="stHeader"] {
    background: var(--background) !important;
    height: 2.5rem;
}

html, body, [class*="css"] {
    color: var(--text);
}

p, li, label {
    color: var(--text) !important;
    line-height: 1.6;
}

h1, h2, h3, h4, h5 {
    color: var(--blue) !important;
    font-weight: 700 !important;
}

h1 {
    font-size: 2rem !important;
}

h2 {
    font-size: 1.55rem !important;
}

h3 {
    font-size: 1.15rem !important;
}

[data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
}

/* NAVIGATION */

div[data-testid="stRadio"] > div[role="radiogroup"] {
    background: var(--white) !important;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 9px;
    gap: 9px;
}

div[data-testid="stRadio"] label {
    background: #F1F7FC;
    border: 1px solid #D5E5F2;
    border-radius: 8px;
    padding: 7px 13px;
}

div[data-testid="stRadio"] label p {
    color: var(--blue) !important;
    font-weight: 600 !important;
}

div[data-testid="stRadio"] label:has(input:checked) {
    background: var(--light-blue) !important;
    border: 1px solid #7BAED8 !important;
}

div[data-testid="stRadio"] label:has(input:checked) p {
    color: var(--blue-dark) !important;
    font-weight: 800 !important;
}

/* INFO BOXES */

div[data-testid="stAlert"] {
    background: var(--light-blue) !important;
    border: 1px solid #B7D7F2 !important;
    border-radius: 11px !important;
}

div[data-testid="stAlert"] * {
    color: #245782 !important;
}

/* METRIC CARDS */

div[data-testid="stMetric"] {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 17px !important;
    box-shadow: 0 2px 7px rgba(37, 99, 166, 0.06);
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
    color: var(--muted) !important;
}

div[data-testid="stMetricValue"] {
    color: var(--blue) !important;
    font-weight: 800 !important;
}

/* CONTAINERS */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* BUTTONS */

.stButton button,
.stFormSubmitButton button {
    background: var(--blue) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--blue) !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    min-height: 43px;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: var(--blue-dark) !important;
    color: #FFFFFF !important;
    border-color: var(--blue-dark) !important;
}

/* INPUT FIELDS */

.stTextInput input,
.stNumberInput input,
.stDateInput input {
    background: var(--white) !important;
    color: var(--text) !important;
    border: 1px solid #A9C4DA !important;
    border-radius: 8px !important;
}

/* SELECT BOX */

div[data-baseweb="select"] > div {
    background: var(--white) !important;
    border-color: #A9C4DA !important;
}

div[data-baseweb="select"] * {
    color: var(--text) !important;
}

/* SELECT MENU */

div[data-baseweb="popover"],
div[data-baseweb="menu"] {
    background: var(--white) !important;
    color: var(--text) !important;
}

div[data-baseweb="menu"] li {
    color: var(--text) !important;
}

/* SLIDER */

div[data-testid="stSlider"] {
    color: var(--blue) !important;
}

/* TABS */

button[data-baseweb="tab"] {
    color: var(--text) !important;
    font-weight: 700 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--blue) !important;
}

/* DATAFRAME */

div[data-testid="stDataFrame"] {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* CHARTS */

div[data-testid="stVegaLiteChart"],
div[data-testid="stArrowVegaLiteChart"],
div[data-testid="stBarChart"],
div[data-testid="stLineChart"] {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

/* IMAGES */

div[data-testid="stImage"] {
    background: var(--white);
    border-radius: 10px;
    overflow: hidden;
}

/* DIVIDERS */

hr {
    border-color: var(--border) !important;
}

/* FOOTER */

.footer-text {
    color: var(--muted) !important;
    font-size: 13px;
    text-align: center;
    line-height: 1.8;
}

/* RESPONSIVE */

@media (max-width: 700px) {
    .block-container {
        padding: 1rem 1rem 1rem 1rem;
    }

    h1 {
        font-size: 1.6rem !important;
    }
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. LOAD MODEL AND DATA
# ==================================================

@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not DATA_PATH.exists():
    st.error(f"Dataset file not found: {DATA_PATH}")
    st.stop()

try:
    bundle = load_model(str(MODEL_PATH))

    preprocessor = bundle["preprocessor"]
    base_models = bundle["base_models"]
    meta_model = bundle["meta_model"]

    data = load_data(str(DATA_PATH))

except Exception as error:
    st.error("Unable to load the model or dataset.")
    st.exception(error)
    st.stop()


required_columns = [
    "city", "latitude", "longitude", "temp_max_c",
    "temp_min_c", "precipitation_mm",
    "wind_speed_max_kmh", "humidity_mean_pct"
]

missing_columns = [
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:
    st.error(f"Dataset is missing columns: {missing_columns}")
    st.stop()

if data.empty:
    st.error("The dataset is empty.")
    st.stop()


# ==================================================
# 4. TOP NAVIGATION
# ==================================================

st.markdown("### 🌤️ Weather AI")

page = st.radio(
    "Navigation",
    [
        "Overview",
        "Temperature Prediction",
        "Model Analysis",
        "About Project"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()


# ==================================================
# 5. MAIN HEADER
# ==================================================

st.markdown(
    "<div style='padding-top: 12px;'></div>",
    unsafe_allow_html=True
)

st.markdown("## 🌤️ Sri Lankan Weather AI")

st.caption(
    "Daily Maximum Temperature Prediction | "
    "Ensemble Machine Learning"
)

header1, header2, header3 = st.columns(3)

header1.info("📍 Sri Lanka")
header2.info("📊 Weather Data Analysis")
header3.info("🤖 Ensemble Machine Learning")

st.divider()


# ==================================================
# 6. OVERVIEW PAGE
# ==================================================

if page == "Overview":

    st.title("Project Overview")

    st.caption(
        "An overview of the dataset, prediction system "
        "and machine learning experiments."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Weather Records",
        f"{len(data):,}"
    )

    col2.metric(
        "Sri Lankan Cities",
        data["city"].nunique()
    )

    col3.metric(
        "Selected ML Model",
        "Stacking"
    )

    st.divider()

    st.subheader("Explore Dashboard")

    card1, card2 = st.columns(2)

    with card1:

        with st.container(border=True):
            st.subheader("🌡️ Temperature Prediction")
            st.write(
                "Estimate daily maximum temperature "
                "for a selected Sri Lankan city."
            )

        with st.container(border=True):
            st.subheader("📈 Feature Importance")
            st.write(
                "Explore important input features "
                "used by the machine learning models."
            )

    with card2:

        with st.container(border=True):
            st.subheader("📊 Model Comparison")
            st.write(
                "Compare six machine learning approaches "
                "using MAE, RMSE and R²."
            )

        with st.container(border=True):
            st.subheader("🔍 Model Evaluation")
            st.write(
                "Review model performance and "
                "experimental results."
            )

    st.divider()

    st.subheader("Dataset Insights")

    chart1, chart2 = st.columns(2)

    with chart1:

        with st.container(border=True):

            st.markdown("#### Weather Records by City")

            city_counts = (
                data["city"]
                .value_counts()
                .head(10)
                .rename_axis("City")
                .reset_index(name="Records")
                .set_index("City")
            )

            st.bar_chart(
                city_counts,
                color="#5799D2",
                height=320
            )

    with chart2:

        with st.container(border=True):

            st.markdown(
                "#### Maximum Temperature Distribution"
            )

            temperature_counts = (
                data["temp_max_c"]
                .value_counts(bins=15)
                .sort_index()
                .reset_index()
            )

            temperature_counts.columns = [
                "Temperature Range",
                "Records"
            ]

            temperature_counts["Temperature Range"] = [
                f"{interval.left:.1f}–{interval.right:.1f}"
                for interval in temperature_counts["Temperature Range"]
            ]

            temperature_counts = (
                temperature_counts
                .set_index("Temperature Range")
            )

            st.bar_chart(
                temperature_counts,
                color="#5799D2",
                height=320
            )

    st.info(
        "This dashboard uses same-day weather inputs to estimate "
        "daily maximum temperature. It is not a future-weather "
        "forecasting system."
    )


# ==================================================
# 7. TEMPERATURE PREDICTION PAGE
# ==================================================

elif page == "Temperature Prediction":

    st.title("🌡️ Maximum Temperature Prediction")

    st.caption(
        "Enter weather conditions to estimate "
        "the daily maximum temperature."
    )

    with st.container(border=True):

        with st.form("prediction_form"):

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("📍 Location Details")

                cities = sorted(data["city"].dropna().unique())

                if not cities:
                    st.error("No cities are available in the dataset.")
                    st.stop()

                city = st.selectbox(
                    "Select Sri Lankan City",
                    cities
                )

                selected_city = data[data["city"] == city]

                latitude = float(selected_city["latitude"].mean())
                longitude = float(selected_city["longitude"].mean())

                selected_date = st.date_input("Select Date")

                st.subheader("🌧️ Rainfall Information")

                precipitation = st.number_input(
                    "Precipitation (mm)",
                    min_value=0.0,
                    max_value=500.0,
                    value=0.0,
                    step=0.5
                )

            with col2:

                st.subheader("🌤️ Weather Conditions")

                temp_min = st.number_input(
                    "Minimum Temperature (°C)",
                    min_value=0.0,
                    max_value=50.0,
                    value=23.0,
                    step=0.5
                )

                wind_speed = st.number_input(
                    "Maximum Wind Speed (km/h)",
                    min_value=0.0,
                    max_value=200.0,
                    value=15.0,
                    step=0.5
                )

                humidity = st.slider(
                    "Mean Humidity (%)",
                    min_value=0,
                    max_value=100,
                    value=80
                )

            st.divider()

            submitted = st.form_submit_button(
                "🌡️ Predict Maximum Temperature",
                use_container_width=True
            )

    if submitted:

        input_data = pd.DataFrame([{
            "city": city,
            "latitude": latitude,
            "longitude": longitude,
            "temp_min_c": temp_min,
            "precipitation_mm": precipitation,
            "wind_speed_max_kmh": wind_speed,
            "humidity_mean_pct": humidity,
            "year": selected_date.year,
            "month": selected_date.month,
            "day": selected_date.day,
            "day_of_year": selected_date.timetuple().tm_yday
        }])

        try:

            processed_input = preprocessor.transform(input_data)

            base_predictions = np.column_stack([
                base_model.predict(processed_input)
                for _, base_model in base_models
            ])

            prediction = float(
                meta_model.predict(base_predictions)[0]
            )

            st.divider()
            st.subheader("Prediction Result")

            with st.container(border=True):

                st.metric(
                    "Estimated Daily Maximum Temperature",
                    f"{prediction:.2f} °C"
                )

                st.write(f"📍 **City:** {city}")
                st.write(f"📅 **Date:** {selected_date}")

            st.success(
                "Temperature estimation completed successfully!"
            )

        except Exception as error:

            st.error("Prediction could not be completed.")
            st.exception(error)


# ==================================================
# 8. MODEL ANALYSIS PAGE
# ==================================================

elif page == "Model Analysis":

    st.title("Model Performance Analysis")

    st.caption(
        "Compare the performance of the machine learning "
        "models evaluated in this project."
    )

    results = pd.DataFrame({
        "Model": [
            "Random Forest",
            "XGBoost",
            "LightGBM",
            "CatBoost",
            "Stacking",
            "Hybrid RF + ANN"
        ],
        "MAE": [
            0.839, 0.818, 0.819,
            0.858, 0.806, 0.872
        ],
        "RMSE": [
            1.093, 1.063, 1.062,
            1.109, 1.045, 1.131
        ],
        "R² Score": [
            0.892, 0.898, 0.898,
            0.889, 0.901, 0.884
        ]
    })

    # BEST BASELINE MODEL

    st.subheader("Best Baseline Model: Stacking")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Stacking MAE", "0.806 °C")
    metric2.metric("Stacking RMSE", "1.045 °C")
    metric3.metric("Stacking R²", "0.901")

    st.divider()

    # MODEL COMPARISON TABLE

    st.subheader("Model Comparison Table")

    styled_results = (
        results.sort_values("MAE")
        .reset_index(drop=True)
        .style
        .format({
            "MAE": "{:.3f}",
            "RMSE": "{:.3f}",
            "R² Score": "{:.3f}"
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#DCEEFF"),
                    ("color", "#2563A6"),
                    ("font-weight", "bold"),
                    ("border", "1px solid #C7DCEC"),
                    ("padding", "12px")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("background-color", "#FFFFFF"),
                    ("color", "#334155"),
                    ("border", "1px solid #E1EAF2"),
                    ("padding", "11px")
                ]
            }
        ])
    )

    st.dataframe(
        styled_results,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown(
        "<div style='padding-top: 12px;'></div>",
        unsafe_allow_html=True
    )

    st.title("Performance Comparison Dashboard")

    st.caption(
        "Select a performance metric to explore "
        "the results of each machine learning model."
    )

    # PERFORMANCE SUMMARY

    perf1, perf2, perf3 = st.columns(3)

    perf1.metric("Best MAE", "0.806 °C")
    perf2.metric("Best RMSE", "1.045 °C")
    perf3.metric("Best R² Score", "0.901")

    st.divider()

    # METRIC SELECTION

    st.subheader("Select Performance Metric")

    selected_metric = st.selectbox(
        "Choose a metric to compare",
        [
            "MAE",
            "RMSE",
            "R² Score"
        ],
        index=0,
        help="Select a metric to view its model comparison graph."
    )

    metric_details = {
        "MAE": {
            "column": "MAE",
            "title": "Mean Absolute Error (MAE)",
            "description": (
                "MAE measures the average absolute difference "
                "between actual and predicted temperatures."
            ),
            "guidance": "Lower values indicate better performance.",
            "best_model": "Stacking",
            "best_value": "0.806 °C",
            "color": "#5799D2"
        },
        "RMSE": {
            "column": "RMSE",
            "title": "Root Mean Squared Error (RMSE)",
            "description": (
                "RMSE measures prediction errors and gives "
                "more weight to larger errors."
            ),
            "guidance": "Lower values indicate better performance.",
            "best_model": "Stacking",
            "best_value": "1.045 °C",
            "color": "#397FC1"
        },
        "R² Score": {
            "column": "R² Score",
            "title": "R² Score",
            "description": (
                "R² indicates how well the model explains "
                "variation in the target variable."
            ),
            "guidance": "Higher values indicate better performance.",
            "best_model": "Stacking",
            "best_value": "0.901",
            "color": "#2563A6"
        }
    }

    metric_info = metric_details[selected_metric]

    selected_data = (
        results[["Model", metric_info["column"]]]
        .set_index("Model")
        .sort_values(
            by=metric_info["column"],
            ascending=(selected_metric != "R² Score")
        )
    )

    # SELECTED METRIC DISPLAY

    with st.container(border=True):

        st.markdown(f"### {metric_info['title']}")

        st.write(metric_info["description"])

        st.caption(metric_info["guidance"])

        result_col1, result_col2 = st.columns(2)

        result_col1.metric(
            "Best Performing Model",
            metric_info["best_model"]
        )

        result_col2.metric(
            f"Best {selected_metric}",
            metric_info["best_value"]
        )

        st.divider()

        st.markdown("#### Model Performance Graph")

        st.bar_chart(
            selected_data,
            color=metric_info["color"],
            height=420
        )

        st.markdown("#### Performance Values")

        display_data = (
            results[["Model", metric_info["column"]]]
            .sort_values(
                by=metric_info["column"],
                ascending=(selected_metric != "R² Score")
            )
            .reset_index(drop=True)
        )

        display_data.columns = [
            "Model",
            selected_metric
        ]

        st.dataframe(
            display_data.style.format({
                selected_metric: "{:.3f}"
            }),
            use_container_width=True,
            hide_index=True
        )

    st.info(
        "Stacking achieved the best baseline test performance "
        "among the six evaluated models."
    )

    st.divider()

    # FEATURE IMPORTANCE ANALYSIS

    st.markdown(
        "<div style='padding-top: 12px;'></div>",
        unsafe_allow_html=True
    )

    st.title("Feature Importance Analysis")

    st.caption(
        "Explore the contribution of different input features "
        "in the Random Forest and XGBoost models."
    )

    feature_col1, feature_col2 = st.columns(2)

    # RANDOM FOREST

    with feature_col1:

        with st.container(border=True):

            st.markdown("#### Random Forest")

            st.caption(
                "Feature importance generated by the Random Forest model."
            )

            rf_path = (
                BASE_DIR /
                "random_forest_feature_importance.png"
            )

            if rf_path.exists():

                st.image(
                    str(rf_path),
                    use_container_width=True
                )

            else:

                st.warning(
                    "Random Forest feature importance graph was not found."
                )

    # XGBOOST

    with feature_col2:

        with st.container(border=True):

            st.markdown("#### XGBoost")

            st.caption(
                "Feature importance generated by the XGBoost model."
            )

            xgb_path = (
                BASE_DIR /
                "xgboost_feature_importance.png"
            )

            if xgb_path.exists():

                st.image(
                    str(xgb_path),
                    use_container_width=True
                )

            else:

                st.warning(
                    "XGBoost feature importance graph was not found."
                )

    st.divider()

    # HYPERPARAMETER TUNING

    st.subheader("Hyperparameter Tuning")

    tuning_results = pd.DataFrame({
        "Model": [
            "Random Forest",
            "XGBoost",
            "LightGBM",
            "CatBoost"
        ],
        "Best CV MAE": [
            0.832,
            0.812,
            0.802,
            0.795
        ]
    })

    styled_tuning = (
        tuning_results.style
        .format({"Best CV MAE": "{:.3f}"})
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#DCEEFF"),
                    ("color", "#2563A6"),
                    ("font-weight", "bold"),
                    ("border", "1px solid #C7DCEC"),
                    ("padding", "12px")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("background-color", "#FFFFFF"),
                    ("color", "#334155"),
                    ("border", "1px solid #E1EAF2"),
                    ("padding", "11px")
                ]
            }
        ])
    )

    st.dataframe(
        styled_tuning,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The Stacking model achieved the lowest baseline test MAE "
        "among the six models. CatBoost achieved the lowest "
        "cross-validation MAE in the separate tuning experiment."
    )


# ==================================================
# 9. ABOUT PROJECT PAGE
# ==================================================

elif page == "About Project":

    st.title("About the Project")

    st.caption(
        "Project objectives, dataset information, machine learning "
        "methods and evaluation approach."
    )

    st.subheader("🎯 Project Objective")

    with st.container(border=True):

        st.write(
            "To estimate daily maximum temperatures in Sri Lanka "
            "using historical weather data and ensemble machine "
            "learning models."
        )

    st.divider()

    st.subheader("📂 Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", f"{len(data):,}")
    col2.metric("Total Cities", data["city"].nunique())
    col3.metric("Missing Values", int(data.isnull().sum().sum()))

    st.divider()

    st.subheader("🤖 Machine Learning Models")

    model_col1, model_col2 = st.columns(2)

    with model_col1:

        with st.container(border=True):
            st.markdown("### 🌳 Bagging")
            st.write("Random Forest")

        with st.container(border=True):
            st.markdown("### 🚀 Boosting")
            st.write("XGBoost, LightGBM and CatBoost")

    with model_col2:

        with st.container(border=True):
            st.markdown("### 🔗 Stacking Ensemble")
            st.write(
                "Combines multiple machine learning models."
            )

        with st.container(border=True):
            st.markdown("### 🧠 Hybrid Model")
            st.write("Random Forest combined with ANN.")

    st.divider()

    st.subheader("📏 Evaluation Metrics")

    eval1, eval2, eval3 = st.columns(3)

    with eval1:

        with st.container(border=True):
            st.markdown("### MAE")
            st.write(
                "Measures the average absolute difference "
                "between actual and predicted values."
            )

    with eval2:

        with st.container(border=True):
            st.markdown("### RMSE")
            st.write(
                "Measures prediction error while giving "
                "more weight to larger errors."
            )

    with eval3:

        with st.container(border=True):
            st.markdown("### R² Score")
            st.write(
                "Indicates how well the model explains "
                "variation in the target variable."
            )

    st.divider()

    st.subheader("🔄 Project Workflow")

    workflow = [
        "Dataset collection",
        "Data preprocessing and cleaning",
        "Exploratory data analysis",
        "Model development",
        "Ensemble and hybrid experiments",
        "Hyperparameter tuning",
        "Model evaluation and comparison",
        "Feature importance analysis",
        "Streamlit dashboard development"
    ]

    for number, step in enumerate(workflow, start=1):
        st.markdown(f"**{number}.** {step}")

    st.divider()

    st.subheader("⚠️ Important Note")

    st.info(
        "The current application estimates daily maximum temperature "
        "using same-day weather input variables. It should not be "
        "interpreted as a future-weather forecasting service."
    )


# ==================================================
# 10. FOOTER
# ==================================================

st.divider()

st.markdown(
    "<p class='footer-text'>"
    "🌤️ <b>Sri Lankan Weather AI</b><br>"
    "Daily Maximum Temperature Prediction<br>"
    "An Ensemble Machine Learning Project<br>"
    "Developed using Python, Streamlit, Scikit-learn, "
    "XGBoost, LightGBM and CatBoost."
    "</p>",
    unsafe_allow_html=True
)
