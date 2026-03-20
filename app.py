import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix,
    classification_report, log_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve

# page config
st.set_page_config(
    page_title="CTR Prediction Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@700;800&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }

    .stApp { background-color: #0e0e14; color: #e0ddd8; }
    section[data-testid="stSidebar"] { background: #13131c; border-right: 1px solid #2a2a3e; }

    .metric-card {
        background: #1a1a26;
        border: 1px solid #2a2a3e;
        border-radius: 8px;
        padding: 18px 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-card .val {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.04em;
    }
    .metric-card .lbl { font-size: 0.65rem; color: #6060a0; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 4px; }

    .pred-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #3a3a5e;
        border-radius: 10px;
        padding: 24px;
        text-align: center;
        margin: 16px 0;
    }
    .pred-pct { font-family: 'Syne', sans-serif; font-size: 3.5rem; font-weight: 800; letter-spacing: -0.05em; }
    .tag-chip {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        margin: 4px;
    }
    .section-divider {
        border: none;
        border-top: 1px solid #2a2a3e;
        margin: 24px 0;
    }
    div[data-testid="stMetric"] {
        background: #1a1a26;
        border: 1px solid #2a2a3e;
        border-radius: 8px;
        padding: 14px 18px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #e05c00, #c0392b);
        color: white;
        border: none;
        border-radius: 6px;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        letter-spacing: 0.04em;
        padding: 10px 24px;
        width: 100%;
        font-size: 1rem;
    }
    .stButton > button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)


# DATA GENERATION 

@st.cache_resource
def generate_and_train(n_samples=20000, noise=0.5, random_state=42):
    np.random.seed(random_state)
    N = n_samples

    df = pd.DataFrame({
        'user_age':       np.random.randint(18, 65, N),
        'device':         np.random.choice(['mobile', 'desktop', 'tablet'], N, p=[0.55, 0.35, 0.10]),
        'ad_category':    np.random.choice(['tech', 'fashion', 'sports', 'finance', 'travel'], N),
        'time_of_day':    np.random.choice(['morning', 'afternoon', 'evening', 'night'], N),
        'user_interests': np.random.choice(['low', 'medium', 'high'], N, p=[0.4, 0.4, 0.2]),
        'ad_quality':     np.clip(np.random.beta(2, 5, N), 0, 1),
        'bid_amount':     np.clip(np.random.exponential(2.5, N), 0.1, 20),
        'historical_ctr': np.clip(np.random.beta(1, 20, N), 0, 0.3),
    })


    prob = (
        0.02
        + 0.015 * (df['device'] == 'mobile')
        + 0.010 * (df['ad_category'] == 'tech')
        + 0.008 * (df['ad_category'] == 'finance')
        + 0.012 * (df['time_of_day'] == 'evening')
        + 0.006 * (df['time_of_day'] == 'morning')
        + 0.020 * (df['user_interests'] == 'high')
        + 0.010 * (df['user_interests'] == 'medium')
        + 0.040 * df['ad_quality']
        + 0.005 * np.log1p(df['bid_amount'])
        + 0.060 * df['historical_ctr']
        # Interaction: mobile + evening
        + 0.008 * ((df['device'] == 'mobile') & (df['time_of_day'] == 'evening'))
        # Age effect: younger users click more
        + 0.0002 * (40 - df['user_age']).clip(0)
    )
    prob = np.clip(prob + np.random.normal(0, noise * 0.01, N), 0.001, 0.5)
    df['clicked'] = np.random.binomial(1, prob)

    # ── Encode
    encoders = {}
    cat_cols = ['device', 'ad_category', 'time_of_day', 'user_interests']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    feature_cols = ['user_age', 'device', 'ad_category', 'time_of_day',
                    'user_interests', 'ad_quality', 'bid_amount', 'historical_ctr']
    X = df[feature_cols]
    y = df['clicked']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Three models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=300, random_state=42),
    }
    trained = {}
    metrics = {}

    for name, clf in models.items():
        if name == "Logistic Regression":
            clf.fit(X_train_sc, y_train)
            proba = clf.predict_proba(X_test_sc)[:, 1]
            trained[name] = (clf, scaler, True)
        else:
            clf.fit(X_train, y_train)
            proba = clf.predict_proba(X_test)[:, 1]
            trained[name] = (clf, scaler, False)

        preds = (proba >= 0.5).astype(int)
        auc   = roc_auc_score(y_test, proba)
        ap    = average_precision_score(y_test, proba)
        ll    = log_loss(y_test, proba)
        bs    = brier_score_loss(y_test, proba)
        acc   = (preds == y_test).mean()

        metrics[name] = {"AUC-ROC": auc, "Avg Precision": ap,
                         "Log Loss": ll, "Brier Score": bs,
                         "Accuracy": acc, "proba": proba}

    return trained, encoders, feature_cols, X_test, y_test, metrics, df


# SIDEBAR

with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")

    n_samples = st.select_slider("Training Samples", [5000, 10000, 20000, 50000], value=20000)
    noise_level = st.slider("Noise Level (σ)", 0.0, 3.0, 0.5, 0.1)
    model_choice = st.selectbox("Active Model", ["Random Forest", "Gradient Boosting", "Logistic Regression"])

    st.markdown("---")
    st.markdown("### Privacy")
    dp_enabled = st.toggle("Differential Privacy", value=False)
    if dp_enabled:
        dp_epsilon = st.slider("ε (privacy budget)", 0.1, 10.0, 1.0, 0.1)
        st.caption("Laplace noise injected into prediction scores.")

  

# LOAD

with st.spinner("Training models…"):
    trained_models, encoders, feature_cols, X_test, y_test, all_metrics, full_df = \
        generate_and_train(n_samples, noise_level)

active_clf, active_scaler, needs_scale = trained_models[model_choice]


# HEADER

st.markdown("""
<h1 style='margin-bottom:4px'>CTR Prediction Engine</h1>
<p style='color:#6060a0;font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:28px'>
Click-Through Rate · Ads Insights & Measurement
</p>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Model Evaluation", "Model Comparison", "Batch Predict"])

# TAB 1 — LIVE PREDICTION

with tab1:
    col_inp, col_out = st.columns([1, 1], gap="large")

    with col_inp:
        st.markdown("### Ad & User Features")

        user_age      = st.slider("User Age", 18, 65, 28)
        device        = st.selectbox("Device", ['mobile', 'desktop', 'tablet'])
        ad_category   = st.selectbox("Ad Category", ['tech', 'fashion', 'sports', 'finance', 'travel'])
        time_of_day   = st.selectbox("Time of Day", ['morning', 'afternoon', 'evening', 'night'])
        user_interest = st.selectbox("User Interest Level", ['low', 'medium', 'high'])
        ad_quality    = st.slider("Ad Quality Score", 0.0, 1.0, 0.65, 0.01)
        bid_amount    = st.slider("Bid Amount ($)", 0.1, 20.0, 2.5, 0.1)
        historical_ctr = st.slider("Historical CTR (%)", 0.0, 30.0, 3.5, 0.1) / 100

        predict_btn = st.button("PREDICT CTR PROBABILITY")

    with col_out:
        st.markdown("### Prediction Output")

        if predict_btn:
            row = {
                'user_age':       user_age,
                'device':         encoders['device'].transform([device])[0],
                'ad_category':    encoders['ad_category'].transform([ad_category])[0],
                'time_of_day':    encoders['time_of_day'].transform([time_of_day])[0],
                'user_interests': encoders['user_interests'].transform([user_interest])[0],
                'ad_quality':     ad_quality,
                'bid_amount':     bid_amount,
                'historical_ctr': historical_ctr,
            }
            input_df = pd.DataFrame([row])[feature_cols]
            X_in = active_scaler.transform(input_df) if needs_scale else input_df
            prob = active_clf.predict_proba(X_in)[0][1]

            # Differential Privacy noise
            if dp_enabled:
                sensitivity = 0.1
                b = sensitivity / dp_epsilon
                noise = np.random.laplace(0, b)
                prob = float(np.clip(prob + noise, 0, 1))

            color = "#e05c00" if prob < 0.05 else "#f59e0b" if prob < 0.10 else "#22c55e"
            verdict = "LOW" if prob < 0.05 else "MEDIUM" if prob < 0.10 else "HIGH"

            st.markdown(f"""
            <div class="pred-box">
                <div style='font-size:0.7rem;color:#6060a0;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px'>
                    Predicted Click Probability
                </div>
                <div class="pred-pct" style='color:{color}'>{prob*100:.2f}%</div>
                <div style='margin-top:10px'>
                    <span class='tag-chip' style='background:rgba(224,92,0,0.15);color:{color};border:1px solid {color}'>
                        {verdict} CLICK PROBABILITY
                    </span>
                </div>
                <div style='margin-top:12px;font-size:0.68rem;color:#6060a0'>
                    Model: {model_choice} {'· DP active (ε=' + str(dp_epsilon) + ')' if dp_enabled else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Feature contribution (manual SHAP-lite for tree models)
            st.markdown("#### Feature Contributions")
            if hasattr(active_clf, 'feature_importances_'):
                fi = active_clf.feature_importances_
                feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': fi})
                feat_df = feat_df.sort_values('Importance', ascending=True)

                fig, ax = plt.subplots(figsize=(5, 3.5))
                fig.patch.set_facecolor('#1a1a26')
                ax.set_facecolor('#1a1a26')
                bars = ax.barh(feat_df['Feature'], feat_df['Importance'],
                               color=['#e05c00' if v > feat_df['Importance'].median() else '#3a3a5e'
                                      for v in feat_df['Importance']], height=0.6)
                ax.tick_params(colors='#9090b0', labelsize=8)
                ax.spines[:].set_color('#2a2a3e')
                ax.set_xlabel('Importance', color='#6060a0', fontsize=8)
                ax.set_title('Feature Importance', color='#e0ddd8', fontsize=10, pad=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            elif hasattr(active_clf, 'coef_'):
                coefs = np.abs(active_clf.coef_[0])
                feat_df = pd.DataFrame({'Feature': feature_cols, 'Coef': coefs}).sort_values('Coef', ascending=True)
                fig, ax = plt.subplots(figsize=(5, 3.5))
                fig.patch.set_facecolor('#1a1a26'); ax.set_facecolor('#1a1a26')
                ax.barh(feat_df['Feature'], feat_df['Coef'], color='#3b82f6', height=0.6)
                ax.tick_params(colors='#9090b0', labelsize=8); ax.spines[:].set_color('#2a2a3e')
                ax.set_xlabel('|Coefficient|', color='#6060a0', fontsize=8)
                ax.set_title('LR Coefficients', color='#e0ddd8', fontsize=10)
                plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            st.info("Configure features on the left and click **PREDICT CTR PROBABILITY**.")


# TAB 2 — MODEL EVALUATION

with tab2:
    m = all_metrics[model_choice]
    proba_test = m["proba"]
    y_pred = (proba_test >= 0.5).astype(int)

    st.markdown(f"### {model_choice} — Evaluation Dashboard")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AUC-ROC",       f"{m['AUC-ROC']:.4f}")
    c2.metric("Avg Precision", f"{m['Avg Precision']:.4f}")
    c3.metric("Log Loss",      f"{m['Log Loss']:.4f}")
    c4.metric("Brier Score",   f"{m['Brier Score']:.4f}")
    c5.metric("Accuracy",      f"{m['Accuracy']*100:.1f}%")

    col_a, col_b = st.columns(2)

    with col_a:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, proba_test)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#1a1a26'); ax.set_facecolor('#1a1a26')
        ax.plot(fpr, tpr, color='#e05c00', lw=2, label=f"AUC = {m['AUC-ROC']:.4f}")
        ax.plot([0,1],[0,1], '--', color='#3a3a5e', lw=1)
        ax.fill_between(fpr, tpr, alpha=0.08, color='#e05c00')
        ax.set_xlabel("FPR", color='#6060a0', fontsize=9); ax.set_ylabel("TPR", color='#6060a0', fontsize=9)
        ax.set_title("ROC Curve", color='#e0ddd8', fontsize=11)
        ax.tick_params(colors='#6060a0', labelsize=8); ax.spines[:].set_color('#2a2a3e')
        ax.legend(fontsize=9, labelcolor='#e0ddd8', facecolor='#1a1a26')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_b:
        # Precision-Recall
        prec_c, rec_c, _ = precision_recall_curve(y_test, proba_test)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#1a1a26'); ax.set_facecolor('#1a1a26')
        ax.plot(rec_c, prec_c, color='#3b82f6', lw=2, label=f"AP = {m['Avg Precision']:.4f}")
        ax.fill_between(rec_c, prec_c, alpha=0.08, color='#3b82f6')
        ax.set_xlabel("Recall", color='#6060a0', fontsize=9); ax.set_ylabel("Precision", color='#6060a0', fontsize=9)
        ax.set_title("Precision-Recall Curve", color='#e0ddd8', fontsize=11)
        ax.tick_params(colors='#6060a0', labelsize=8); ax.spines[:].set_color('#2a2a3e')
        ax.legend(fontsize=9, labelcolor='#e0ddd8', facecolor='#1a1a26')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    col_c, col_d = st.columns(2)

    with col_c:
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        fig.patch.set_facecolor('#1a1a26'); ax.set_facecolor('#1a1a26')
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                    linewidths=1, linecolor='#0e0e14',
                    annot_kws={'size': 12, 'color': 'white', 'weight': 'bold'})
        ax.set_xlabel("Predicted", color='#6060a0', fontsize=9)
        ax.set_ylabel("Actual", color='#6060a0', fontsize=9)
        ax.set_title("Confusion Matrix", color='#e0ddd8', fontsize=11)
        ax.tick_params(colors='#9090b0', labelsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_d:
        # Calibration Curve
        prob_true, prob_pred = calibration_curve(y_test, proba_test, n_bins=10)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        fig.patch.set_facecolor('#1a1a26'); ax.set_facecolor('#1a1a26')
        ax.plot(prob_pred, prob_true, 's-', color='#22c55e', lw=2, ms=5, label='Model')
        ax.plot([0,1],[0,1],'--', color='#3a3a5e', lw=1, label='Perfect')
        ax.set_xlabel("Mean Predicted Prob", color='#6060a0', fontsize=9)
        ax.set_ylabel("Fraction Positives", color='#6060a0', fontsize=9)
        ax.set_title("Calibration Curve", color='#e0ddd8', fontsize=11)
        ax.tick_params(colors='#6060a0', labelsize=8); ax.spines[:].set_color('#2a2a3e')
        ax.legend(fontsize=9, labelcolor='#e0ddd8', facecolor='#1a1a26')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # CTR distribution by feature
    st.markdown("#### CTR by Feature (training data)")
    cat_feature = st.selectbox("Select feature to analyze", ['device', 'ad_category', 'time_of_day', 'user_interests'])
    plot_df = full_df.copy()

    # Decode back for display
    decoded = encoders[cat_feature].inverse_transform(plot_df[cat_feature])
    ctr_by_feat = pd.DataFrame({'feature': decoded, 'clicked': plot_df['clicked']})
    agg = ctr_by_feat.groupby('feature')['clicked'].agg(['mean', 'count']).reset_index()
    agg.columns = ['Category', 'CTR', 'Count']
    agg = agg.sort_values('CTR', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor('#1a1a26'); ax.set_facecolor('#1a1a26')
    colors = ['#e05c00', '#f59e0b', '#3b82f6', '#22c55e', '#a855f7']
    ax.bar(agg['Category'], agg['CTR']*100, color=colors[:len(agg)], width=0.5, edgecolor='#0e0e14')
    ax.set_ylabel("CTR (%)", color='#6060a0', fontsize=9)
    ax.set_title(f"CTR by {cat_feature}", color='#e0ddd8', fontsize=11)
    ax.tick_params(colors='#9090b0', labelsize=9); ax.spines[:].set_color('#2a2a3e')
    for i, (_, row) in enumerate(agg.iterrows()):
        ax.text(i, row['CTR']*100 + 0.01, f"{row['CTR']*100:.2f}%\n(n={row['Count']:,})",
                ha='center', va='bottom', fontsize=7.5, color='#c0c0d0')
    plt.tight_layout(); st.pyplot(fig); plt.close()


# TAB 3 — MODEL COMPARISON

with tab3:
    st.markdown("### All Models — Head-to-Head Comparison")

    comp_data = []
    for name, m in all_metrics.items():
        comp_data.append({
            "Model": name,
            "AUC-ROC":       round(m['AUC-ROC'], 4),
            "Avg Precision": round(m['Avg Precision'], 4),
            "Log Loss":      round(m['Log Loss'], 4),
            "Brier Score":   round(m['Brier Score'], 4),
            "Accuracy":      f"{m['Accuracy']*100:.2f}%",
        })
    comp_df = pd.DataFrame(comp_data).set_index("Model")
    st.dataframe(comp_df, use_container_width=True)

    # Overlaid ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax in axes: ax.set_facecolor('#1a1a26')
    fig.patch.set_facecolor('#1a1a26')

    colors_m = {'Random Forest': '#e05c00', 'Gradient Boosting': '#3b82f6', 'Logistic Regression': '#22c55e'}
    for name, m in all_metrics.items():
        fpr, tpr, _ = roc_curve(y_test, m['proba'])
        axes[0].plot(fpr, tpr, color=colors_m[name], lw=2, label=f"{name} ({m['AUC-ROC']:.3f})")
    axes[0].plot([0,1],[0,1],'--',color='#3a3a5e',lw=1)
    axes[0].set_xlabel("FPR", color='#6060a0', fontsize=9); axes[0].set_ylabel("TPR", color='#6060a0', fontsize=9)
    axes[0].set_title("ROC Curve Comparison", color='#e0ddd8', fontsize=11)
    axes[0].tick_params(colors='#6060a0', labelsize=8); axes[0].spines[:].set_color('#2a2a3e')
    axes[0].legend(fontsize=8, labelcolor='#e0ddd8', facecolor='#1a1a26')

    # Bar chart of metrics
    metric_names = ['AUC-ROC', 'Avg Precision']
    model_names  = list(all_metrics.keys())
    x = np.arange(len(metric_names)); width = 0.25
    for i, (name, m) in enumerate(all_metrics.items()):
        vals = [m['AUC-ROC'], m['Avg Precision']]
        axes[1].bar(x + i*width, vals, width, label=name, color=colors_m[name], alpha=0.85)
    axes[1].set_xticks(x + width); axes[1].set_xticklabels(metric_names, color='#9090b0', fontsize=9)
    axes[1].set_title("AUC & AP Comparison", color='#e0ddd8', fontsize=11)
    axes[1].tick_params(colors='#6060a0', labelsize=8); axes[1].spines[:].set_color('#2a2a3e')
    axes[1].set_facecolor('#1a1a26')
    axes[1].legend(fontsize=8, labelcolor='#e0ddd8', facecolor='#1a1a26')

    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Cross-validation scores
    st.markdown("#### 5-Fold Cross-Validation (AUC-ROC)")
    with st.spinner("Running cross-validation…"):
        X_full = full_df[feature_cols]
        y_full = full_df['clicked']
        cv_results = {}
        for name, (clf, scaler_, sc) in trained_models.items():
            Xc = scaler_.transform(X_full) if sc else X_full
            scores = cross_val_score(clf, Xc, y_full, cv=5, scoring='roc_auc', n_jobs=-1)
            cv_results[name] = scores

    cv_df = pd.DataFrame({k: v for k, v in cv_results.items()})
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('#1a1a26'); ax.set_facecolor('#1a1a26')
    bp = ax.boxplot(cv_df.values, labels=cv_df.columns, patch_artist=True,
                    medianprops=dict(color='white', linewidth=2))
    for patch, color in zip(bp['boxes'], colors_m.values()):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_ylabel("AUC-ROC", color='#6060a0', fontsize=9)
    ax.set_title("5-Fold CV AUC Distribution", color='#e0ddd8', fontsize=11)
    ax.tick_params(colors='#9090b0', labelsize=9); ax.spines[:].set_color('#2a2a3e')
    plt.tight_layout(); st.pyplot(fig); plt.close()

# TAB 4 — BATCH PREDICTION

with tab4:
    st.markdown("### Batch CTR Prediction")
    st.caption("Upload a CSV with columns: user_age, device, ad_category, time_of_day, user_interests, ad_quality, bid_amount, historical_ctr")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            for col in ['device', 'ad_category', 'time_of_day', 'user_interests']:
                if col in batch_df.columns:
                    batch_df[col] = encoders[col].transform(batch_df[col])
            X_batch = batch_df[feature_cols]
            if needs_scale: X_batch = active_scaler.transform(X_batch)
            probs = active_clf.predict_proba(X_batch)[:, 1]
            batch_df['predicted_ctr'] = probs
            batch_df['click_verdict'] = pd.cut(probs, bins=[0, 0.05, 0.10, 1.0],
                                                labels=['Low', 'Medium', 'High'])
            st.success(f"Predictions complete for {len(batch_df):,} rows.")
            st.dataframe(batch_df.head(50), use_container_width=True)

            # Distribution of predicted CTR
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor('#1a1a26'); ax.set_facecolor('#1a1a26')
            ax.hist(probs, bins=50, color='#e05c00', alpha=0.8, edgecolor='#0e0e14')
            ax.set_xlabel("Predicted CTR Probability", color='#6060a0', fontsize=9)
            ax.set_ylabel("Count", color='#6060a0', fontsize=9)
            ax.set_title("Batch Prediction Distribution", color='#e0ddd8', fontsize=11)
            ax.tick_params(colors='#9090b0', labelsize=8); ax.spines[:].set_color('#2a2a3e')
            plt.tight_layout(); st.pyplot(fig); plt.close()

            csv_out = batch_df.to_csv(index=False).encode()
            st.download_button("⬇ Download Predictions CSV", csv_out, "ctr_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.markdown("#### Or generate a sample batch:")
        if st.button("Generate 100 Random Samples & Predict"):
            np.random.seed(99)
            sample = pd.DataFrame({
                'user_age':       np.random.randint(18, 65, 100),
                'device':         np.random.choice(['mobile', 'desktop', 'tablet'], 100),
                'ad_category':    np.random.choice(['tech', 'fashion', 'sports', 'finance', 'travel'], 100),
                'time_of_day':    np.random.choice(['morning', 'afternoon', 'evening', 'night'], 100),
                'user_interests': np.random.choice(['low', 'medium', 'high'], 100),
                'ad_quality':     np.random.beta(2, 5, 100),
                'bid_amount':     np.random.exponential(2.5, 100),
                'historical_ctr': np.random.beta(1, 20, 100),
            })
            encoded = sample.copy()
            for col in ['device', 'ad_category', 'time_of_day', 'user_interests']:
                encoded[col] = encoders[col].transform(encoded[col])
            X_s = encoded[feature_cols]
            if needs_scale: X_s = active_scaler.transform(X_s)
            probs_s = active_clf.predict_proba(X_s)[:, 1]
            sample['predicted_ctr'] = probs_s
            sample['verdict'] = pd.cut(probs_s, bins=[0, 0.05, 0.10, 1.0], labels=['Low', 'Medium', 'High'])
            st.dataframe(sample, use_container_width=True)
            st.download_button("⬇ Download", sample.to_csv(index=False).encode(),
                               "sample_predictions.csv", "text/csv")

# FOOTER

st.markdown("---")
st.markdown("""
<div style='text-align:center;font-size:0.62rem;color:#3a3a5e;letter-spacing:0.1em;padding:12px 0'>
    CTR PREDICTION ENGINE · GOOGLE ADS · ADS INSIGHTS & MEASUREMENT · DATA SCIENTIST L3/L4
</div>
""", unsafe_allow_html=True)