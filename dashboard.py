import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Counterfeit Medicine — Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Consistent, publication-style defaults (avoid hard-coded style names that may be missing)
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("ggplot")
    except OSError:
        pass
sns.set_theme(style="whitegrid", palette="deep")

st.title("Counterfeit medicine detection")
st.caption("Dataset overview, sample images, and reported model metrics.")

# --- Paths (adjust in sidebar if needed) ---
with st.sidebar:
    st.header("Paths")
    genuine_path = st.text_input("Genuine folder", value="dataset/genuine")
    fake_path = st.text_input("Fake folder", value="dataset/fake")
    st.divider()
    st.markdown("**Tips**")
    st.markdown(
        "- Pie/bar charts need at least one image in the folders.\n"
        "- Use `app.py` for inference; this page is analytics only."
    )


def count_images(folder: str) -> int:
    if not folder or not os.path.isdir(folder):
        return 0
    total = 0
    for root, _dirs, files in os.walk(folder):
        total += sum(1 for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    return total


def get_sample_image(folder: str) -> str | None:
    if not folder or not os.path.isdir(folder):
        return None
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                return os.path.join(root, f)
    return None


genuine_count = count_images(genuine_path)
fake_count = count_images(fake_path)
total_count = genuine_count + fake_count

# --- Dataset summary ---
st.subheader("Dataset summary")
c1, c2, c3 = st.columns(3)
c1.metric("Genuine images", f"{genuine_count:,}")
c2.metric("Counterfeit images", f"{fake_count:,}")
c3.metric("Total", f"{total_count:,}")

if not os.path.isdir(genuine_path) and not os.path.isdir(fake_path):
    st.warning(
        f"Neither folder exists yet. Create **`{genuine_path}`** and **`{fake_path}`** "
        "or update the paths in the sidebar."
    )
elif total_count == 0:
    st.info("No images found under the configured folders. Add images to enable charts.")

# --- Charts (skip pie when total is 0 — avoids NaN in matplotlib wedges) ---
st.subheader("Class distribution")

if total_count > 0:
    labels = ["Genuine", "Counterfeit"]
    values = [float(genuine_count), float(fake_count)]
    # Extra safety: never pass NaN to matplotlib
    values = [0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in values]

    col_bar, col_pie = st.columns((1, 1))

    with col_bar:
        fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
        bars = ax_bar.bar(labels, values, color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=1)
        ax_bar.set_ylabel("Image count")
        ax_bar.set_title("Counts by class")
        for bar, v in zip(bars, values, strict=True):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(v):,}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        fig_bar.tight_layout()
        st.pyplot(fig_bar, clear_figure=True)
        plt.close(fig_bar)

    with col_pie:
        fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
        ax_pie.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=["#2ecc71", "#e74c3c"],
            wedgeprops={"edgecolor": "white", "linewidth": 1},
        )
        ax_pie.set_title("Share of dataset")
        fig_pie.tight_layout()
        st.pyplot(fig_pie, clear_figure=True)
        plt.close(fig_pie)
else:
    st.caption("Distribution charts appear once there is at least one image in the dataset folders.")

# --- Sample images ---
st.subheader("Sample images")
genuine_sample = get_sample_image(genuine_path)
fake_sample = get_sample_image(fake_path)
img_a, img_b = st.columns(2)
with img_a:
    if genuine_sample:
        st.image(genuine_sample, caption="Genuine sample", use_container_width=True)
    else:
        st.caption("No genuine sample available.")
with img_b:
    if fake_sample:
        st.image(fake_sample, caption="Counterfeit sample", use_container_width=True)
    else:
        st.caption("No counterfeit sample available.")

# --- Model metrics (static placeholders — replace when you log real eval) ---
st.subheader("Model performance (reported)")
accuracy = 0.769
precision = 0.812
recall = 0.933
f1 = 0.868

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{accuracy:.1%}")
m2.metric("Precision", f"{precision:.1%}")
m3.metric("Recall", f"{recall:.1%}")
m4.metric("F1 score", f"{f1:.1%}")

st.subheader("Confusion matrix")
cm = np.array([[5, 155], [48, 672]], dtype=float)
fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt=".0f",
    cmap="Blues",
    ax=ax_cm,
    xticklabels=["Pred. genuine", "Pred. counterfeit"],
    yticklabels=["Actual genuine", "Actual counterfeit"],
    cbar_kws={"label": "Count"},
)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
fig_cm.tight_layout()
st.pyplot(fig_cm, clear_figure=True)
plt.close(fig_cm)
