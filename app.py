import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import auth
from database import clear_history, get_history, save_prediction, set_feedback
from utils import explain, red_flags, trust_score, clean_text, extract_text_from_url, is_hindi, translate_to_english

st.set_page_config(page_title="TruthLens AI", page_icon="🔍", layout="wide")

MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
PAGES = ["🏠 Welcome", "🔎 Detect News", "📊 Dashboard", "🕘 History", "🏆 Model Performance", "ℹ️ About"]
FAKE_COLOR, REAL_COLOR = "#ff4d6d", "#2ee6a6"

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#141b34,#0b1020); border-right: 1px solid #232c4d;}
.hero {background: linear-gradient(135deg,#5b3df5 0%,#8a4dff 50%,#d946ef 100%);
       padding: 3.2rem 2.5rem; border-radius: 24px; margin-bottom: 1.5rem;}
.hero h1 {font-size: 2.8rem; margin: 0 0 .6rem 0; color: white;}
.hero p {font-size: 1.1rem; color: #f1e9ff; max-width: 640px; line-height: 1.6;}
.card {background: #141b34; border: 1px solid #232c4d; border-radius: 18px; padding: 1.3rem; height: 100%;}
.card h4 {margin: 0 0 .4rem 0;} .card p {color: #9aa6c9; margin: 0; font-size: .95rem;}
.result {border-radius: 20px; padding: 1.6rem; text-align: center; margin: 1rem 0;}
.result.real {background: rgba(46,230,166,.12); border: 1px solid #2ee6a6;}
.result.fake {background: rgba(255,77,109,.12); border: 1px solid #ff4d6d;}
.result h2 {margin: 0; font-size: 2.2rem;} .result span {color: #c3cbe6;}
.stat {background:#141b34; border:1px solid #232c4d; border-radius:16px; padding:1rem 1.2rem;}
.stat b {font-size: 1.9rem;} .stat div {color:#9aa6c9; font-size:.85rem;}
.stButton > button {border-radius: 12px; font-weight: 700; padding: .6rem 1.4rem;}
</style>""", unsafe_allow_html=True)


# ---------- Load models ----------
@st.cache_resource
def load_everything():
    vec = joblib.load("models/vectorizer.joblib")
    models = {n: joblib.load(f"models/{n.replace(' ', '_')}.joblib") for n in MODEL_NAMES}
    info = json.load(open("models/metrics.json"))
    return vec, models, info


if not Path("models/metrics.json").exists():
    st.error("Models not found. Please run:  python train_models.py")
    st.stop()
vectorizer, models, info = load_everything()
metrics, dataset = info["models"], info["dataset"]


def predict(model_name, text):
    """Returns (label, confidence %) for one model."""
    X = vectorizer.transform([clean_text(text)])
    p_real = float(models[model_name].predict_proba(X)[0][1])
    return ("REAL", p_real * 100) if p_real >= 0.5 else ("FAKE", (1 - p_real) * 100)


def stat_box(label, value):
    st.markdown(f'<div class="stat"><div>{label}</div><b>{value}</b></div>', unsafe_allow_html=True)


def dark(fig, h=340):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ---------- Login / Register ----------
if not st.session_state.get("user"):
    st.markdown('<div class="hero"><h1>🔍 TruthLens AI</h1><p>Login or create an account to start detecting fake news.</p></div>',
                unsafe_allow_html=True)
    tab_login, tab_reg = st.tabs(["Login", "Create account"])
    with tab_login:
        u = st.text_input("Username", key="lu")
        p = st.text_input("Password", type="password", key="lp")
        if st.button("Login", type="primary"):
            if auth.login(u, p):
                st.session_state.user = u.strip().lower()
                st.rerun()
            st.error("Wrong username or password.")
    with tab_reg:
        u2 = st.text_input("Choose username", key="ru")
        p2 = st.text_input("Choose password", type="password", key="rp")
        if st.button("Create account"):
            ok, msg = auth.register(u2, p2)
            (st.success if ok else st.error)(msg)
    st.stop()
user = st.session_state.user

# ---------- Sidebar navigation ----------
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]
st.sidebar.markdown("## 🔍 TruthLens AI")
st.sidebar.caption("Fake News Detection System")
st.sidebar.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption(f"👤 Logged in as **{user}**")
if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.rerun()
st.sidebar.caption(f"Trained on {dataset['total']:,} news articles")
page = st.session_state.page


def go_detect():
    st.session_state.page = PAGES[1]


# ---------- Welcome ----------
if page == PAGES[0]:
    st.markdown("""<div class="hero"><h1>Detect Fake News with AI</h1>
    <p>TruthLens uses four Machine Learning models and TF-IDF text analysis to tell whether a news
    article looks REAL or FAKE, in seconds.</p></div>""", unsafe_allow_html=True)
    st.button("🚀 Get Started", on_click=go_detect, type="primary")
    st.write("")
    cols = st.columns(4)
    feats = [("⚡ Instant Results", "Paste any news and get a verdict with confidence score."),
             ("🤖 4 ML Models", "Logistic Regression, Decision Tree, Random Forest, Gradient Boosting."),
             ("📊 Live Dashboard", "Charts and statistics of all your predictions."),
             ("🕘 Saved History", "Every prediction is stored in a local SQLite database.")]
    for c, (t, d) in zip(cols, feats):
        c.markdown(f'<div class="card"><h4>{t}</h4><p>{d}</p></div>', unsafe_allow_html=True)
    st.write("")
    a, b, c, d = st.columns(4)
    with a: stat_box("Articles used", f"{dataset['total']:,}")
    with b: stat_box("Models", "4")
    with c: stat_box("Best accuracy", f"{max(m['accuracy'] for m in metrics.values())}%")
    with d: stat_box("Test articles", f"{dataset['test_size']:,}")

# ---------- Detect ----------
elif page == PAGES[1]:
    st.title("🔎 Detect News")
    source = st.radio("Input", ["Paste text", "From URL"], horizontal=True)
    if source == "Paste text":
        text = st.text_area("Paste a news headline or article (English or Hindi)", height=200,
                            placeholder="Paste the full news text here (at least 20 words)...")
    else:
        url = st.text_input("News article URL", placeholder="https://example.com/news/article")
        text = ""
    mode = st.radio("Mode", ["Single model", "Compare all 4 models"], horizontal=True)
    chosen = None
    if mode == "Single model":
        chosen = st.selectbox("Choose model", MODEL_NAMES, index=2)

    if st.button("Analyze News", type="primary"):
        if source == "From URL":
            try:
                with st.spinner("Fetching article..."):
                    text = extract_text_from_url(url)
                st.caption(f"Fetched {len(text.split())} words from the URL.")
            except Exception as e:
                st.error(f"Could not read this URL: {e}")
                st.stop()
        if is_hindi(text):
            try:
                with st.spinner("Hindi detected, translating to English..."):
                    text = translate_to_english(text)
                st.info("🇮🇳 Hindi text was translated to English before analysis. "
                        "Accuracy is lower than for English news.")
            except Exception:
                st.error("Hindi translation needs internet and the `deep-translator` package "
                         "(pip install deep-translator).")
                st.stop()
        if len(text.split()) < 20:
            st.warning("Please enter at least 20 words for a reliable result.")
        else:
            res = {n: predict(n, text) for n in (MODEL_NAMES if mode != "Single model" else [chosen])}
            real_votes = sum(1 for l, _ in res.values() if l == "REAL")
            if mode == "Single model":
                final, conf, used, label_mode = res[chosen][0], res[chosen][1], chosen, "Single"
            else:
                final = "REAL" if real_votes > 2 else "FAKE" if real_votes < 2 else \
                    res[max(res, key=lambda n: res[n][1])][0]
                same = [c for l, c in res.values() if l == final]
                conf, used, label_mode = sum(same) / max(1, len(same)), "All 4 (vote)", "Compare"
            real_p = [c if l == "REAL" else 100 - c for l, c in res.values()]
            flags = red_flags(text)
            score, level = trust_score(sum(real_p) / len(real_p), len(flags))
            rid = save_prediction(user, text, label_mode, used, final, round(conf, 1))
            fake_w, real_w = explain(vectorizer, models["Logistic Regression"], text)
            st.session_state.result = dict(id=rid, mode=mode, chosen=chosen, res=res, final=final, conf=conf,
                                           votes=real_votes, flags=flags, score=score, level=level,
                                           fake_w=fake_w, real_w=real_w, fb=None)

    r = st.session_state.get("result")
    if r:
        label = r["final"]
        sub = (f'{r["chosen"]} · Confidence {r["conf"]:.1f}%' if r["mode"] == "Single model"
               else f'Final verdict by majority vote · {r["votes"]}/4 models say REAL')
        st.markdown(f'<div class="result {label.lower()}"><h2>{"✅" if label == "REAL" else "🚨"} {label} NEWS</h2>'
                    f'<span>{sub}</span></div>', unsafe_allow_html=True)
        if r["mode"] != "Single model":
            cols = st.columns(4)
            for col, (n, (l, c)) in zip(cols, r["res"].items()):
                color = REAL_COLOR if l == "REAL" else FAKE_COLOR
                col.markdown(f'<div class="card"><h4>{n}</h4><b style="color:{color};font-size:1.5rem">{l}</b>'
                             f'<p>Confidence {c:.1f}%</p></div>', unsafe_allow_html=True)
        st.write("")
        g, f = st.columns(2)
        tcolor = REAL_COLOR if r["score"] >= 70 else "#ffc857" if r["score"] >= 40 else FAKE_COLOR
        g.markdown(f'<div class="card"><h4>🛡 Trust Score</h4><b style="font-size:2.6rem;color:{tcolor}">{r["score"]}</b>'
                   f'<span style="color:#9aa6c9"> / 100 · {r["level"]}</span></div>', unsafe_allow_html=True)
        g.progress(r["score"])
        flag_html = "".join(f"<p>⚠ {x}</p>" for x in r["flags"]) or "<p>✅ No red flags found</p>"
        f.markdown(f'<div class="card"><h4>🚩 Red Flags</h4>{flag_html}</div>', unsafe_allow_html=True)
        st.write("")
        fw = ", ".join(r["fake_w"]) or "-"
        rw = ", ".join(r["real_w"]) or "-"
        st.markdown(f'<div class="card"><h4>🧠 Why this verdict? (Logistic Regression)</h4>'
                    f'<p>Words pushing towards <b style="color:{FAKE_COLOR}">FAKE</b>: {fw}</p>'
                    f'<p>Words pushing towards <b style="color:{REAL_COLOR}">REAL</b>: {rw}</p></div>',
                    unsafe_allow_html=True)
        st.write("")
        if r["fb"]:
            st.success("Thanks for your feedback!")
        else:
            st.caption("Was this verdict correct?")
            b1, b2, _ = st.columns([1, 1, 5])
            for btn, val, txt in ((b1, "correct", "👍 Correct"), (b2, "wrong", "👎 Wrong")):
                if btn.button(txt, key=val):
                    set_feedback(r["id"], val)
                    st.session_state.result["fb"] = val
                    st.rerun()

# ---------- Dashboard ----------
elif page == PAGES[2]:
    st.title("📊 Dashboard")
    df = get_history(user)
    if df.empty:
        st.info("No predictions yet. Go to **Detect News** and analyze something first.")
    else:
        a, b, c, d = st.columns(4)
        with a: stat_box("Total predictions", len(df))
        with b: stat_box("Real news", int((df.prediction == "REAL").sum()))
        with c: stat_box("Fake news", int((df.prediction == "FAKE").sum()))
        with d:
            fb = df[df.feedback.notna()]
            stat_box("User-verified accuracy", f"{(fb.feedback == 'correct').mean() * 100:.0f}%" if len(fb) else "No feedback yet")
        st.write("")
        l, r = st.columns(2)
        pie = px.pie(df, names="prediction", hole=0.55, color="prediction", title="Real vs Fake",
                     color_discrete_map={"REAL": REAL_COLOR, "FAKE": FAKE_COLOR})
        l.plotly_chart(dark(pie), use_container_width=True)
        bymodel = df.groupby(["model", "prediction"]).size().reset_index(name="count")
        bar = px.bar(bymodel, x="model", y="count", color="prediction", barmode="group", title="Predictions by model",
                     color_discrete_map={"REAL": REAL_COLOR, "FAKE": FAKE_COLOR})
        r.plotly_chart(dark(bar), use_container_width=True)
        df["day"] = pd.to_datetime(df.created_at).dt.date
        daily = df.groupby("day").size().reset_index(name="count")
        st.plotly_chart(dark(px.line(daily, x="day", y="count", markers=True, title="Predictions per day")),
                        use_container_width=True)

# ---------- History ----------
elif page == PAGES[3]:
    st.title("🕘 Prediction History")
    df = get_history(user)
    if df.empty:
        st.info("History is empty.")
    else:
        pick = st.selectbox("Filter", ["All", "REAL", "FAKE"])
        if pick != "All":
            df = df[df.prediction == pick]
        df["news"] = df["news"].str.slice(0, 120) + "..."
        st.dataframe(df[["created_at", "news", "mode", "model", "prediction", "confidence", "feedback"]],
                     use_container_width=True, hide_index=True)
        c1, c2, _ = st.columns([1, 1, 4])
        c1.download_button("⬇ Download CSV", df.to_csv(index=False), "history.csv", "text/csv")
        if c2.button("🗑 Clear history"):
            clear_history(user)
            st.rerun()

# ---------- Model Performance ----------
elif page == PAGES[4]:
    st.title("🏆 Model Performance")
    st.caption(f"Evaluated on {dataset['test_size']:,} unseen test articles (20% of the dataset).")
    table = pd.DataFrame(metrics).T[["accuracy", "precision", "recall", "f1", "train_seconds"]]
    table.columns = ["Accuracy %", "Precision %", "Recall %", "F1 %", "Train time (s)"]
    st.dataframe(table, use_container_width=True)
    long = table.drop(columns="Train time (s)").reset_index().melt(id_vars="index")
    fig = px.bar(long, x="index", y="value", color="variable", barmode="group",
                 labels={"index": "", "value": "Score %", "variable": "Metric"}, title="Metrics comparison")
    fig.update_yaxes(range=[min(95, long.value.min() - 1), 100.5])
    st.plotly_chart(dark(fig, 380), use_container_width=True)
    name = st.selectbox("Confusion matrix for", MODEL_NAMES)
    cm = metrics[name]["confusion_matrix"]
    heat = go.Figure(go.Heatmap(z=cm, x=["Pred FAKE", "Pred REAL"], y=["Actual FAKE", "Actual REAL"],
                                text=cm, texttemplate="%{text}", colorscale="Purples", showscale=False))
    st.plotly_chart(dark(heat, 340), use_container_width=True)

# ---------- About ----------
else:
    st.title("ℹ️ About Project")
    st.markdown(f"""
**TruthLens AI** is a beginner-friendly Machine Learning project that classifies news as REAL or FAKE.

**How it works**
1. News title + text are cleaned (lowercase, remove links/symbols).
2. **TF-IDF** converts the text into 5,000 numeric features.
3. Four models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) predict the label.
4. In compare mode, the final answer is a majority vote of the 4 models.

**Dataset:** Fake & Real News dataset (ISOT) — {dataset['fake']:,} fake and {dataset['real']:,} real articles after cleaning.

**Tech stack:** Python · Pandas · Scikit-learn · Streamlit · Plotly · SQLite

**⚠ Limitations**
- Hindi news is translated to English first, so accuracy is lower. The models were trained on English news only.
- The dataset is mostly US political news from 2016–2017, so results on other topics or languages may be less reliable.
- The model learns *writing style*, not facts. It cannot verify whether a claim is actually true.
- Use it as a helper, not as a final judge — always cross-check with trusted sources.
""")
