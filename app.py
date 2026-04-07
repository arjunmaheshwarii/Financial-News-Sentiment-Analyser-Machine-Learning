import streamlit as st
import pickle
import numpy as np
import time

# Load models
mnb_model = pickle.load(open("mnb_model.pkl", "rb"))
lr_model = pickle.load(open("lr_model.pkl", "rb"))
svm_model = pickle.load(open("svm_model.pkl", "rb"))

# Load vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🤖", layout="centered")

# -------- PREMIUM THEME --------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Apply Outfit font everywhere */
html, body, [class*="css"], [data-testid="stAppViewContainer"] * {
    font-family: 'Outfit', sans-serif !important;
}

/* Animated Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg, #0f172a, #020617, #1e1b4b, #312e81);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: white;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Glassmorphic Main Layout */
[data-testid="stMainBlockContainer"] {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 20px;
    padding: 2rem 3rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-top: 2rem;
    margin-bottom: 2rem;
}

/* typing animation */
.typing {
    overflow: hidden;
    border-right: .15em solid #8b5cf6;
    white-space: nowrap;
    animation: typing 2.5s steps(30, end);
    margin-bottom: 0px !important;
}

@keyframes typing {
    from { width: 0 }
    to { width: 100% }
}

/* Cards */
.card {
    padding: 24px;
    border-radius: 16px;
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    text-align: center;
    animation: fadeIn 0.8s ease-in;
    transition: transform 0.3s ease, box-shadow 0.3s ease, border 0.3s ease;
    border: 1px solid rgba(255, 255, 255, 0.05);
    display: flex;
    flex-direction: column;
    justify-content: center;
    height: 100%;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(139, 92, 246, 0.4);
    border: 1px solid rgba(139, 92, 246, 0.4);
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Tooltip Design */
.card-header {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
}

.tooltip-container {
    position: relative;
    display: inline-block;
    cursor: help;
}

.tooltip-icon {
    background: rgba(255,255,255,0.1);
    border-radius: 50%;
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    color: rgba(255,255,255,0.8);
    transition: all 0.3s ease;
}

.tooltip-container:hover .tooltip-icon {
    background: #8b5cf6;
    color: white;
    box-shadow: 0 0 10px #8b5cf6;
}

.tooltip-text {
    visibility: hidden;
    width: 220px;
    background-color: rgba(15, 23, 42, 0.95);
    color: #fff;
    text-align: center;
    border-radius: 10px;
    padding: 12px;
    position: absolute;
    z-index: 100;
    bottom: 135%;
    left: 50%;
    margin-left: -110px;
    opacity: 0;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(139, 92, 246, 0.5);
    font-size: 13px;
    font-weight: 400;
    line-height: 1.5;
    box-shadow: 0 8px 30px rgba(0,0,0,0.6);
    transform: translateY(10px);
}

.tooltip-text::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -6px;
    border-width: 6px;
    border-style: solid;
    border-color: rgba(139, 92, 246, 0.5) transparent transparent transparent;
}

.tooltip-container:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
    transform: translateY(0);
}


/* Progress Bars */
.progress-container {
    width: 100%;
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    margin-top: 15px;
    overflow: hidden;
    height: 8px;
}

.progress-bar {
    height: 100%;
    border-radius: 8px;
    width: 0%;
    animation: fillBar 1.5s ease-out forwards;
}

@keyframes fillBar {
    from { width: 0%; }
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 12px;
    font-weight: 600;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    border: none;
    width: 100%;
    height: 52px;
    letter-spacing: 0.5px;
}

.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.6), 0 0 40px rgba(139, 92, 246, 0.4);
    background: linear-gradient(90deg, #8b5cf6, #6366f1);
    border: none;
    color: white;
}

.stButton>button:active {
    transform: scale(0.95) !important;
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.8), inset 0 4px 10px rgba(0,0,0,0.2) !important;
}

/* Textarea */
textarea {
    border-radius: 12px !important;
    background-color: rgba(17, 24, 39, 0.7) !important;
    color: white !important;
    transition: all 0.3s ease !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding: 16px !important;
    font-size: 16px !important;
}

textarea:hover {
    border-color: rgba(139, 92, 246, 0.5) !important;
    box-shadow: 0 0 15px rgba(139, 92, 246, 0.2) !important;
}

textarea:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.6) !important;
}

/* Character Counter */
.stTextArea p {
    color: rgba(255,255,255,0.6) !important;
    font-weight: 300 !important;
}

/* Skeleton Loader */
.skeleton-card {
    padding: 24px;
    border-radius: 16px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    height: 170px;
    animation: pulse 1.5s infinite ease-in-out;
}
.skeleton-title {
    height: 15px; width: 60%; background: rgba(255,255,255,0.1); border-radius: 4px; margin: 0 auto 20px auto;
}
.skeleton-text {
    height: 30px; width: 80%; background: rgba(255,255,255,0.1); border-radius: 8px; margin: 0 auto 15px auto;
}
.skeleton-bar {
    height: 8px; width: 100%; background: rgba(255,255,255,0.1); border-radius: 4px; margin: 20px auto 0 auto;
}

@keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
}

/* Final Verdict Banner */
.verdict-banner {
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    margin-top: 20px;
    animation: popIn 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    backdrop-filter: blur(10px);
}
@keyframes popIn {
    0% { opacity: 0; transform: scale(0.9) translateY(20px); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
}

</style>
""", unsafe_allow_html=True)

# -------- HEADER --------
st.markdown("""
<div style="text-align: center; margin-bottom: 25px;">
    <h1 class="typing" style="font-size:46px; font-weight: 700; margin:0;">
        🤖 Sentiment Analyzer
    </h1>
    <p style='opacity:0.7; font-size: 18px; margin-top: 5px; font-weight: 300;'>Real-time AI prediction system</p>
</div>
""", unsafe_allow_html=True)

# -------- INPUT --------
# max_chars=500 adds the built-in clean character counter
user_input = st.text_area("✍️ Enter your text below:", height=140, max_chars=500, placeholder="Type something like 'I absolutely loved this!'...")

# -------- UI COMPONENTS --------
def skeleton_card():
    return """<div class="skeleton-card">
    <div class="skeleton-title"></div>
    <div class="skeleton-text"></div>
    <div class="skeleton-bar"></div>
</div>"""

def card(title, pred, tooltip, prob=None):
    if "pos" in pred:
        color = "#22c55e" # Green
    elif "neg" in pred:
        color = "#ef4444" # Red
    else:
        color = "#facc15" # Yellow

    prob_text = f"{prob:.1f}%" if prob else "N/A"
    
    progress_bar = f"""<div class="progress-container">
    <div class="progress-bar" style="--target-width: {prob}%; width: {prob}%; background-color: {color}; box-shadow: 0 0 10px {color};"></div>
</div>""" if prob else ""

    return f"""<div class="card">
    <div class="card-header">
        <h4 style="opacity:0.8; margin: 0; font-weight: 500;">{title}</h4>
        <div class="tooltip-container">
            <div class="tooltip-icon">?</div>
            <div class="tooltip-text">{tooltip}</div>
        </div>
    </div>
    <h2 style="color:{color}; margin: 10px 0 0 0; text-transform: uppercase; font-weight: 700; text-shadow: 0 0 15px {color}66;">{pred}</h2>
    <p style="opacity:0.6; margin-top: 8px; font-size: 14px; font-weight: 300;">Confidence: {prob_text}</p>
    {progress_bar}
</div>"""

def final_verdict_banner(sentiment):
    if "pos" in sentiment:
        color = "#22c55e"
        bg = "rgba(34, 197, 94, 0.1)"
        glow = "rgba(34, 197, 94, 0.4)"
        icon = "✨"
        title = "Positive"
    elif "neg" in sentiment:
        color = "#ef4444"
        bg = "rgba(239, 68, 68, 0.1)"
        glow = "rgba(239, 68, 68, 0.4)"
        icon = "🛑"
        title = "Negative"
    else:
        color = "#facc15"
        bg = "rgba(250, 204, 21, 0.1)"
        glow = "rgba(250, 204, 21, 0.4)"
        icon = "⚖️"
        title = "Neutral"
        
    return f"""<div class="verdict-banner" style="background: {bg}; border: 1px solid {color}; box-shadow: 0 0 30px {glow};">
    <h2 style="color: {color}; margin: 0; font-size: 36px; font-weight: 700; text-shadow: 0 0 15px {color};">
        {icon} Overall Sentiment: {title} {icon}
    </h2>
</div>"""

# Define Tooltips
tooltip_nb = "A probabilistic classifier based on applying Bayes' theorem. Great for fast, baseline text classification."
tooltip_lr = "A statistical model that predicts probabilities. Highly reliable and calibrated for text analysis."
tooltip_svm = "A robust algorithm that finds the optimal boundary to divide positive and negative sentiments."

# -------- MAIN LOGIC --------
if st.button("🚀 Analyze Sentiment"):
    if user_input.strip() != "":
        
        # 1. SHOW SKELETON LOADERS
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        placeholder1 = col1.empty()
        placeholder2 = col2.empty()
        placeholder3 = col3.empty()
        verdict_placeholder = st.empty()
        
        placeholder1.markdown(skeleton_card(), unsafe_allow_html=True)
        placeholder2.markdown(skeleton_card(), unsafe_allow_html=True)
        placeholder3.markdown(skeleton_card(), unsafe_allow_html=True)
        
        # 2. FAKE DELAY & INFERENCE
        time.sleep(1.2) 
        
        transformed = vectorizer.transform([user_input])

        pred_nb = str(mnb_model.predict(transformed)[0]).lower()
        prob_nb = np.max(mnb_model.predict_proba(transformed)) * 100

        pred_lr = str(lr_model.predict(transformed)[0]).lower()
        prob_lr = np.max(lr_model.predict_proba(transformed)) * 100

        pred_svm = str(svm_model.predict(transformed)[0]).lower()

        # 3. REVEAL ACTUAL CARDS
        placeholder1.markdown(card("Naive Bayes", pred_nb, tooltip_nb, prob_nb), unsafe_allow_html=True)
        time.sleep(0.15)
        placeholder2.markdown(card("Log. Regression", pred_lr, tooltip_lr, prob_lr), unsafe_allow_html=True)
        time.sleep(0.15)
        placeholder3.markdown(card("SVM Model", pred_svm, tooltip_svm), unsafe_allow_html=True)
        
        # 4. SHOW FINAL VERDICT
        time.sleep(0.4)
        results = [pred_nb, pred_lr, pred_svm]
        final = max(set(results), key=results.count)
        
        verdict_placeholder.markdown(final_verdict_banner(final), unsafe_allow_html=True)

        # 5. MICRO-INTERACTIONS (Confetti / Snow)
        if "pos" in final:
            st.balloons()
        elif "neg" in final:
            st.snow()

    else:
        st.warning("⚠️ Please enter some text to analyze.")

# -------- FOOTER --------
st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("<div style='text-align: center; opacity: 0.5; font-weight: 300;'>✨ Ultra-Premium UI • Built with Streamlit</div>", unsafe_allow_html=True)