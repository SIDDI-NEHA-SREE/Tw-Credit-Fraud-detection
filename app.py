import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import kagglehub
import os

st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("💳 Deep Learning Fraud Detection System")
st.markdown("**Sequential Financial Transactions + Attention + Positional Encoding**")

# ─── Positional Encoding ─────────────────────────────────────────────────────
def positional_encoding(max_len, d_model):
    PE = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = math.sin(pos / (10000 ** (2*i/d_model)))
            if i+1 < d_model:
                PE[pos, i+1] = math.cos(pos / (10000 ** (2*i/d_model)))
    return PE

# ─── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith('.csv'):
                    df = pd.read_csv(os.path.join(root, f))
                    return df
    except Exception:
        pass
    # Synthetic fallback
    np.random.seed(42)
    n = 10000
    df = pd.DataFrame(np.random.randn(n, 28), columns=[f'V{i}' for i in range(1, 29)])
    df['Amount'] = np.abs(np.random.exponential(100, n))
    df['Time'] = np.arange(n)
    df['Class'] = (np.random.rand(n) < 0.02).astype(int)
    return df

@st.cache_data
def make_sequences(df, seq_len=5):
    feature_cols = [c for c in df.columns if c not in ['Class', 'Time']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    labels = df['Class'].values
    sequences, targets = [], []
    for i in range(len(df) - seq_len):
        sequences.append(X_scaled[i:i+seq_len])
        targets.append(labels[i+seq_len])
    return np.array(sequences), np.array(targets), scaler, feature_cols

task = st.sidebar.radio("📌 Select Task", [
    "Task 1: Business Understanding",
    "Task 2: Exploratory Analysis",
    "Task 3: Sequence Generation",
    "Task 4: Model Comparison",
    "Task 5: Positional Encoding",
    "Task 6: Attention Investigation",
    "Task 7: Fraud Dashboard"
])

with st.spinner("Loading dataset..."):
    df = load_data()

# ════════════════════════════════════════════════════════════════════════════════
if task == "Task 1: Business Understanding":
    st.header("💡 Task 1: Business Understanding")

    st.subheader("Why is Fraud Detection Difficult?")
    st.markdown("""
    1. **Extreme Class Imbalance**: Fraud accounts for ~0.17% of transactions. A naive model predicting 'Not Fraud' always achieves 99.8% accuracy but catches zero fraud.
    2. **Concept Drift**: Fraudsters constantly adapt their behavior, making static models obsolete.
    3. **Low Latency Requirements**: Decisions must be made in milliseconds during real-time transactions.
    4. **Feature Camouflage**: Fraudulent transactions are designed to mimic legitimate behavior.
    5. **High Cost of Errors**: False negatives (missed fraud) cost money; false positives (blocking legitimate users) erode trust.
    6. **Data Privacy**: Full transaction details may not be available due to regulations (PCI-DSS, GDPR).
    """)

    st.subheader("Why Accuracy Alone is Misleading?")
    fraud_pct = df['Class'].mean() * 100
    legit_pct = 100 - fraud_pct

    col1, col2, col3 = st.columns(3)
    col1.metric("Fraud Transactions", f"{fraud_pct:.2f}%")
    col2.metric("Legitimate Transactions", f"{legit_pct:.2f}%")
    col3.metric("Naive Model Accuracy", f"{legit_pct:.2f}%")

    st.error(f"""
    ⚠️ A model that **always predicts 'Not Fraud'** achieves **{legit_pct:.2f}% accuracy** but detects **0 fraud cases**.

    Better metrics: **Precision, Recall, F1-Score, ROC-AUC, and PR-AUC**
    - **Recall (Sensitivity)**: Of all actual fraud, how much did we catch? (Critical!)
    - **Precision**: Of all we flagged as fraud, how many were actually fraud?
    - **F1 Score**: Harmonic mean balancing both
    """)

# ════════════════════════════════════════════════════════════════════════════════
elif task == "Task 2: Exploratory Analysis":
    st.header("📊 Task 2: Exploratory Analysis")

    fraud = df[df['Class'] == 1]
    legit = df[df['Class'] == 0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", len(df))
    col2.metric("Fraud Count", len(fraud))
    col3.metric("Legitimate Count", len(legit))
    col4.metric("Imbalance Ratio", f"1 : {len(legit)//max(len(fraud),1)}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fraud vs Non-Fraud Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Legitimate', 'Fraud'], [len(legit), len(fraud)], color=['#2ecc71','#e74c3c'])
        ax.set_ylabel("Count"); ax.set_title("Class Distribution")
        for i, v in enumerate([len(legit), len(fraud)]):
            ax.text(i, v + 50, f'{v:,}', ha='center', fontweight='bold')
        st.pyplot(fig)

    with col2:
        st.subheader("Transaction Amount Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(legit['Amount'].clip(upper=500), bins=50, alpha=0.7, label='Legitimate', color='#2ecc71')
        ax.hist(fraud['Amount'].clip(upper=500), bins=50, alpha=0.7, label='Fraud', color='#e74c3c')
        ax.set_xlabel("Amount (clipped at 500)"); ax.legend()
        st.pyplot(fig)

    st.subheader("Correlation Heatmap (Top Features)")
    top_cols = ['V1','V2','V3','V4','V5','V6','V7','Amount','Class']
    top_cols = [c for c in top_cols if c in df.columns]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[top_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# ════════════════════════════════════════════════════════════════════════════════
elif task == "Task 3: Sequence Generation":
    st.header("🔢 Task 3: Sequence Generation")

    seq_len = st.slider("Sequence Length", 3, 10, 5)
    X_seq, y_seq, scaler, feat_cols = make_sequences(df, seq_len)

    st.subheader("How Sequences Are Created")
    st.markdown(f"""
    Each sample consists of **{seq_len} consecutive transactions** from the same dataset order (proxy for customer history).  
    The model predicts whether the **next transaction** (Txn{seq_len+1}) is fraudulent.
    """)

    st.code(f"""
    Txn1 → features: {feat_cols[:3]} ...
    Txn2 → features: ...
    ...
    Txn{seq_len} → features: ...
    ↓
    Predict Txn{seq_len+1}: Fraud? (0 or 1)
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Sequences Created", len(X_seq))
    col2.metric("Fraud Sequences", int(y_seq.sum()))
    col3.metric("Sequence Shape", f"{X_seq.shape}")

    st.subheader("Sample Sequence Visualization (First Sequence)")
    fig, ax = plt.subplots(figsize=(12, 4))
    for i in range(min(5, X_seq.shape[2])):
        ax.plot(X_seq[0, :, i], marker='o', label=feat_cols[i])
    ax.set_xlabel("Time Step"); ax.set_ylabel("Normalized Value")
    ax.set_title(f"Transaction Sequence (Label: {'Fraud' if y_seq[0] else 'Legit'})"); ax.legend()
    st.pyplot(fig)

# ════════════════════════════════════════════════════════════════════════════════
elif task == "Task 4: Model Comparison":
    st.header("⚖️ Task 4: Model Comparison")

    seq_len = 5
    X_seq, y_seq, scaler, feat_cols = make_sequences(df, seq_len)
    n_features = X_seq.shape[2]

    # Subsample for speed
    idx = np.random.choice(len(X_seq), min(5000, len(X_seq)), replace=False)
    X_s, y_s = X_seq[idx], y_seq[idx]
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y_s, test_size=0.2, random_state=42, stratify=y_s)

    class_weight = {0: 1.0, 1: float(len(y_tr[y_tr==0])) / max(len(y_tr[y_tr==1]), 1)}

    results = {}

    def eval_model(model, name):
        model.fit(X_tr, y_tr, epochs=5, batch_size=64, verbose=0, class_weight=class_weight)
        y_prob = model.predict(X_te, verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        acc = accuracy_score(y_te, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_te, y_pred, average='binary', zero_division=0)
        try: auc = roc_auc_score(y_te, y_prob)
        except: auc = 0.0
        results[name] = {'Accuracy': acc, 'Precision': p, 'Recall': r, 'F1': f1, 'ROC-AUC': auc}
        return model

    # Model A: Dense
    inp_flat = keras.Input(shape=(seq_len * n_features,))
    x = layers.Dense(64, activation='relu')(inp_flat)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    out_a = layers.Dense(1, activation='sigmoid')(x)
    model_a = keras.Model(inp_flat, out_a)
    model_a.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    X_tr_flat = X_tr.reshape(len(X_tr), -1)
    X_te_flat = X_te.reshape(len(X_te), -1)

    # Model B: LSTM
    inp_seq = keras.Input(shape=(seq_len, n_features))
    x_b = layers.LSTM(64)(inp_seq)
    x_b = layers.Dropout(0.3)(x_b)
    out_b = layers.Dense(1, activation='sigmoid')(x_b)
    model_b = keras.Model(inp_seq, out_b)
    model_b.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Model C: LSTM + Attention
    inp_c = keras.Input(shape=(seq_len, n_features))
    x_c = layers.LSTM(64, return_sequences=True)(inp_c)
    x_c = layers.MultiHeadAttention(num_heads=2, key_dim=16)(x_c, x_c)
    x_c = layers.GlobalAveragePooling1D()(x_c)
    x_c = layers.Dropout(0.3)(x_c)
    out_c = layers.Dense(1, activation='sigmoid')(x_c)
    model_c = keras.Model(inp_c, out_c)
    model_c.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    with st.spinner("Training all 3 models..."):
        model_a.fit(X_tr_flat, y_tr, epochs=5, batch_size=64, verbose=0, class_weight=class_weight)
        y_prob_a = model_a.predict(X_te_flat, verbose=0).flatten()
        y_pred_a = (y_prob_a > 0.5).astype(int)
        acc = accuracy_score(y_te, y_pred_a)
        p, r, f1, _ = precision_recall_fscore_support(y_te, y_pred_a, average='binary', zero_division=0)
        try: auc = roc_auc_score(y_te, y_prob_a)
        except: auc = 0.0
        results['Model A: Dense'] = {'Accuracy': acc, 'Precision': p, 'Recall': r, 'F1': f1, 'ROC-AUC': auc}

        model_b = eval_model(model_b, 'Model B: LSTM')
        model_c = eval_model(model_c, 'Model C: LSTM + Attention')

    res_df = pd.DataFrame(results).T
    st.subheader("Model Comparison Results")
    st.dataframe(res_df.style.highlight_max(axis=0, color='lightgreen').format("{:.4f}"))

    st.subheader("Performance Comparison Chart")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(res_df.columns))
    width = 0.25
    for i, (model_name, row) in enumerate(res_df.iterrows()):
        ax.bar(x + i*width, row.values, width, label=model_name)
    ax.set_xticks(x + width)
    ax.set_xticklabels(res_df.columns)
    ax.legend(); ax.set_ylim(0, 1.1)
    st.pyplot(fig)

# ════════════════════════════════════════════════════════════════════════════════
elif task == "Task 5: Positional Encoding":
    st.header("📍 Task 5: Positional Encoding for Transaction Order")

    seq_len = st.slider("Transaction Sequence Length", 3, 10, 5)
    d_model = st.slider("Feature Dimension", 16, 64, 32, step=8)

    PE = positional_encoding(seq_len, d_model)

    st.subheader("Transaction Order Encoding Heatmap")
    txn_labels = [f'Txn {i+1}' for i in range(seq_len)]
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(PE, cmap='viridis', ax=ax, yticklabels=txn_labels)
    ax.set_xlabel("Encoding Dimension"); ax.set_title("Positional Encoding per Transaction")
    st.pyplot(fig)

    st.info("""
    **Why transaction order matters?**
    - A sequence: Normal → Normal → Large Amount → Foreign Country → Fraud follows a **specific pattern**
    - Reversing the order breaks the temporal context the model relies on
    - Positional encoding injects **position-aware information** into each transaction embedding
    - Without it, an attention model treats all transactions as a **bag (order-free)**
    - Example: A small test transaction *before* a large fraudulent one is a common fraud pattern — order matters!
    """)

    st.subheader("Position-wise Encoding Vectors")
    fig, axes = plt.subplots(1, seq_len, figsize=(14, 3))
    if seq_len == 1: axes = [axes]
    colors = plt.cm.plasma(np.linspace(0, 1, seq_len))
    for i in range(seq_len):
        axes[i].plot(PE[i], color=colors[i])
        axes[i].set_title(f'Txn {i+1}')
        axes[i].set_xlabel('Dim')
    plt.tight_layout()
    st.pyplot(fig)

# ════════════════════════════════════════════════════════════════════════════════
elif task == "Task 6: Attention Investigation":
    st.header("🔍 Task 6: Attention Investigation")

    seq_len = 5
    X_seq, y_seq, scaler, feat_cols = make_sequences(df, seq_len)
    n_features = X_seq.shape[2]

    idx = np.random.choice(len(X_seq), min(3000, len(X_seq)), replace=False)
    X_s, y_s = X_seq[idx], y_seq[idx]
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y_s, test_size=0.2, random_state=42)

    inp = keras.Input(shape=(seq_len, n_features))
    x = layers.LSTM(64, return_sequences=True)(inp)
    attn_out, attn_scores = layers.MultiHeadAttention(num_heads=2, key_dim=16, return_attention_scores=True)(x, x)
    pool = layers.GlobalAveragePooling1D()(attn_out)
    out = layers.Dense(1, activation='sigmoid')(pool)
    model = keras.Model(inp, out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    class_weight = {0: 1.0, 1: float(len(y_tr[y_tr==0])) / max(len(y_tr[y_tr==1]),1)}

    with st.spinner("Training LSTM+Attention model..."):
        model.fit(X_tr, y_tr, epochs=5, batch_size=64, verbose=0, class_weight=class_weight)

    attn_extractor = keras.Model(inputs=model.input,
                                  outputs=[model.output, model.layers[2].output[1]])

    # Find a fraud sample
    fraud_indices = np.where(y_te == 1)[0]
    if len(fraud_indices) == 0:
        fraud_indices = np.arange(len(y_te))

    sample_idx = fraud_indices[0]
    sample = X_te[sample_idx:sample_idx+1]
    pred, attn = attn_extractor.predict(sample, verbose=0)

    fraud_prob = pred[0][0]
    st.metric("Fraud Probability", f"{fraud_prob*100:.1f}%",
              delta="HIGH RISK" if fraud_prob > 0.5 else "LOW RISK")

    # Average attention across heads
    avg_attn = np.mean(attn[0], axis=0)  # (seq_len, seq_len)
    txn_importance = np.mean(avg_attn, axis=0)

    st.subheader("Which Transaction Influenced Fraud Prediction Most?")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    txn_labels = [f'Txn {i+1}' for i in range(seq_len)]
    ax1.bar(txn_labels, txn_importance, color=['#e74c3c' if v == max(txn_importance) else '#3498db' for v in txn_importance])
    ax1.set_title("Transaction Attention Importance"); ax1.set_ylabel("Attention Weight")

    sns.heatmap(avg_attn, annot=True, fmt='.2f', ax=ax2,
                xticklabels=txn_labels, yticklabels=txn_labels, cmap='YlOrRd')
    ax2.set_title("Attention Score Matrix")
    plt.tight_layout()
    st.pyplot(fig)

    most_important = txn_labels[np.argmax(txn_importance)]
    st.success(f"✅ **{most_important}** had the highest attention weight ({txn_importance.max():.4f}) — most influential for fraud prediction.")

# ════════════════════════════════════════════════════════════════════════════════
elif task == "Task 7: Fraud Dashboard":
    st.header("🚨 Task 7: Fraud Intelligence Dashboard")

    @st.cache_resource
    def build_fraud_model():
        seq_len = 5
        X_seq, y_seq, scaler, feat_cols = make_sequences(df, seq_len)
        n_features = X_seq.shape[2]
        idx = np.random.choice(len(X_seq), min(5000, len(X_seq)), replace=False)
        X_s, y_s = X_seq[idx], y_seq[idx]
        X_tr, _, y_tr, _ = train_test_split(X_s, y_s, test_size=0.2, random_state=42)
        inp = keras.Input(shape=(seq_len, n_features))
        x = layers.LSTM(64, return_sequences=True)(inp)
        attn_out, _ = layers.MultiHeadAttention(num_heads=2, key_dim=16, return_attention_scores=True)(x, x)
        pool = layers.GlobalAveragePooling1D()(attn_out)
        out = layers.Dense(1, activation='sigmoid')(pool)
        m = keras.Model(inp, out)
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        cw = {0: 1.0, 1: float(len(y_tr[y_tr==0])) / max(len(y_tr[y_tr==1]),1)}
        m.fit(X_tr, y_tr, epochs=5, batch_size=64, verbose=0, class_weight=cw)
        ae = keras.Model(inputs=m.input, outputs=[m.output, m.layers[2].output[1]])
        return ae, scaler, feat_cols, seq_len

    with st.spinner("Building fraud model..."):
        attn_model, scaler, feat_cols, seq_len = build_fraud_model()

    uploaded = st.file_uploader("📤 Upload Transaction CSV", type=['csv'])

    if uploaded:
        user_df = pd.read_csv(uploaded)
        st.subheader("Uploaded Data Preview")
        st.dataframe(user_df.head())

        available_feats = [c for c in feat_cols if c in user_df.columns]
        if len(available_feats) < 3:
            st.warning("CSV doesn't have matching features. Using demo analysis.")
        else:
            scaled = scaler.transform(user_df[feat_cols].fillna(0).values[:, :len(feat_cols)])
            if len(scaled) >= seq_len:
                seq = scaled[:seq_len].reshape(1, seq_len, len(feat_cols))
                pred, attn = attn_model.predict(seq, verbose=0)
                fraud_prob = pred[0][0]

                col1, col2 = st.columns(2)
                col1.metric("Fraud Probability", f"{fraud_prob*100:.1f}%")
                col2.metric("Risk Level", "🔴 HIGH" if fraud_prob > 0.5 else "🟢 LOW")

                avg_attn = np.mean(attn[0], axis=0)
                txn_imp = np.mean(avg_attn, axis=0)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar([f'Txn {i+1}' for i in range(seq_len)], txn_imp, color='#e74c3c')
                ax.set_title("Attention: Transaction Risk Contribution")
                st.pyplot(fig)
    else:
        st.info("No file uploaded. Running simulation on dataset sample.")

        # Real-time simulation
        st.subheader("🔴 Real-time Fraud Detection Simulation (Bonus)")
        sim_size = st.slider("Number of transactions to simulate", 10, 100, 20)
        feature_cols_available = [c for c in feat_cols if c in df.columns]
        sample_txns = df[feature_cols_available].sample(sim_size + seq_len).values
        scaled_txns = scaler.transform(sample_txns[:, :len(feat_cols)])

        fraud_probs = []
        for i in range(sim_size):
            seq = scaled_txns[i:i+seq_len].reshape(1, seq_len, len(feat_cols))
            pred, _ = attn_model.predict(seq, verbose=0)
            fraud_probs.append(pred[0][0])

        sim_df = pd.DataFrame({
            'Transaction': range(1, sim_size+1),
            'Fraud Probability': fraud_probs,
            'Risk': ['🔴 HIGH' if p > 0.5 else '🟡 MEDIUM' if p > 0.3 else '🟢 LOW' for p in fraud_probs]
        })

        st.dataframe(sim_df)

        fig, ax = plt.subplots(figsize=(12, 4))
        colors = ['red' if p > 0.5 else 'orange' if p > 0.3 else 'green' for p in fraud_probs]
        ax.bar(range(1, sim_size+1), fraud_probs, color=colors)
        ax.axhline(0.5, color='red', linestyle='--', label='Threshold')
        ax.set_xlabel("Transaction #"); ax.set_ylabel("Fraud Probability")
        ax.set_title("Real-time Fraud Probability per Transaction"); ax.legend()
        st.pyplot(fig)

        high_risk = sim_df[sim_df['Fraud Probability'] > 0.5]
        if len(high_risk) > 0:
            st.error(f"⚠️ {len(high_risk)} HIGH RISK transactions detected!")
            st.dataframe(high_risk)
        else:
            st.success("✅ No high-risk transactions detected in this batch.")
