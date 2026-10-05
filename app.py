import os
import re
import requests
import numpy as np
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SpeakWise AI",
    page_icon="🎀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# COLORFUL / CUTE / ELEGANT CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       🌈 GLOBAL BACKGROUND
       ===================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 5% 5%,
                rgba(255, 105, 180, 0.30),
                transparent 22%
            ),
            radial-gradient(
                circle at 95% 5%,
                rgba(0, 191, 255, 0.28),
                transparent 22%
            ),
            radial-gradient(
                circle at 10% 90%,
                rgba(255, 215, 0, 0.22),
                transparent 22%
            ),
            radial-gradient(
                circle at 90% 90%,
                rgba(138, 43, 226, 0.25),
                transparent 25%
            ),
            linear-gradient(
                135deg,
                #fff0f8 0%,
                #f4efff 30%,
                #eef9ff 60%,
                #fff8e8 100%
            );

        min-height: 100vh;
    }


    .block-container {
        max-width: 1350px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* =====================================================
       ✨ HERO HEADER
       ===================================================== */

    .hero {
        position: relative;
        overflow: hidden;

        background:
            linear-gradient(
                120deg,
                #ff4fa3,
                #b45cff,
                #6c63ff,
                #00bfff,
                #00d4a8
            );

        background-size: 300% 300%;

        animation: gradientMove 8s ease infinite;

        padding: 45px 30px;

        border-radius: 35px;

        text-align: center;

        color: white;

        box-shadow:
            0 20px 50px rgba(120, 70, 180, 0.30);

        margin-bottom: 30px;
    }


    @keyframes gradientMove {

        0% {
            background-position: 0% 50%;
        }

        50% {
            background-position: 100% 50%;
        }

        100% {
            background-position: 0% 50%;
        }

    }


    .hero::before {
        content: "✨";

        position: absolute;

        font-size: 55px;

        left: 35px;

        top: 20px;

        animation: float 3s ease-in-out infinite;
    }


    .hero::after {
        content: "🌸";

        position: absolute;

        font-size: 55px;

        right: 35px;

        bottom: 15px;

        animation: float 3s ease-in-out infinite;
    }


    @keyframes float {

        0%, 100% {
            transform: translateY(0px) rotate(0deg);
        }

        50% {
            transform: translateY(-10px) rotate(8deg);
        }

    }


    .hero-title {

        font-size: 48px;

        font-weight: 900;

        letter-spacing: -1px;

        text-shadow:
            0 4px 15px rgba(0, 0, 0, 0.20);
    }


    .hero-subtitle {

        font-size: 20px;

        font-weight: 500;

        margin-top: 10px;
    }


    .hero-mini {

        margin-top: 18px;

        font-size: 16px;

        font-weight: 700;
    }


    /* =====================================================
       🎀 SECTION TITLES
       ===================================================== */

    .section-title {

        font-size: 29px;

        font-weight: 900;

        background:
            linear-gradient(
                90deg,
                #ff3f9f,
                #8e44ff,
                #007cff
            );

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;

        margin-top: 25px;

        margin-bottom: 18px;
    }


    /* =====================================================
       💎 GENERAL CARD
       ===================================================== */

    .cute-card {

        background:
            rgba(255, 255, 255, 0.80);

        backdrop-filter: blur(15px);

        border-radius: 28px;

        padding: 25px;

        border:
            2px solid rgba(255, 255, 255, 0.90);

        box-shadow:
            0 12px 35px rgba(100, 70, 150, 0.12);

        margin-bottom: 20px;
    }


    /* =====================================================
       🌈 SCORE CARDS
       ===================================================== */

    .score-card {

        position: relative;

        padding: 25px 15px;

        border-radius: 25px;

        text-align: center;

        color: white;

        overflow: hidden;

        box-shadow:
            0 12px 30px rgba(100, 80, 150, 0.20);

        transition:
            transform 0.25s ease,
            box-shadow 0.25s ease;
    }


    .score-card:hover {

        transform:
            translateY(-7px)
            scale(1.02);

        box-shadow:
            0 18px 40px rgba(100, 80, 150, 0.25);
    }


    .score-card-pink {

        background:
            linear-gradient(
                135deg,
                #ff4fa3,
                #ff75b9,
                #ff9ed2
            );
    }


    .score-card-purple {

        background:
            linear-gradient(
                135deg,
                #7b2cff,
                #a855f7,
                #c084fc
            );
    }


    .score-card-blue {

        background:
            linear-gradient(
                135deg,
                #007cff,
                #00bfff,
                #67e8f9
            );
    }


    .score-card-orange {

        background:
            linear-gradient(
                135deg,
                #ff7a18,
                #ffb347,
                #ffd166
            );
    }


    .score-big {

        font-size: 46px;

        font-weight: 900;

        text-shadow:
            0 3px 10px rgba(0, 0, 0, 0.15);
    }


    .score-label {

        font-size: 15px;

        font-weight: 700;

        margin-top: 5px;
    }


    /* =====================================================
       📊 SKILL BOX
       ===================================================== */

    .skill-box {

        padding: 20px;

        border-radius: 22px;

        background:
            linear-gradient(
                135deg,
                rgba(255, 255, 255, 0.90),
                rgba(248, 240, 255, 0.90)
            );

        border:
            2px solid rgba(255, 255, 255, 0.90);

        box-shadow:
            0 8px 25px rgba(100, 80, 150, 0.10);

        margin-bottom: 15px;
    }


    /* =====================================================
       💬 CHAT INTRO
       ===================================================== */

    .chat-intro {

        background:
            linear-gradient(
                135deg,
                #ffedf8,
                #f1eaff,
                #e5f7ff
            );

        border-radius: 30px;

        padding: 28px;

        border:
            2px solid white;

        box-shadow:
            0 12px 35px rgba(120, 80, 170, 0.12);

        margin-bottom: 20px;
    }


    .coach-title {

        font-size: 30px;

        font-weight: 900;

        background:
            linear-gradient(
                90deg,
                #ff3f9f,
                #8e44ff,
                #008cff
            );

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;
    }


    /* =====================================================
       🌸 TIP CARD
       ===================================================== */

    .tip-card {

        background:
            linear-gradient(
                135deg,
                #fff1f8,
                #fff8dc,
                #eaf8ff
            );

        border-radius: 20px;

        padding: 18px;

        margin: 10px 0;

        border-left:
            6px solid #ff65b5;

        box-shadow:
            0 6px 20px rgba(120, 80, 150, 0.08);
    }


    /* =====================================================
       🎯 BUTTONS
       ===================================================== */

    .stButton > button {

        background:
            linear-gradient(
                90deg,
                #ff3f9f,
                #a855f7,
                #5b6cff
            );

        color: white;

        border: none;

        border-radius: 18px;

        font-size: 17px;

        font-weight: 800;

        padding:
            0.75rem 1.2rem;

        box-shadow:
            0 8px 22px rgba(150, 70, 180, 0.25);

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }


    .stButton > button:hover {

        transform:
            translateY(-3px)
            scale(1.02);

        box-shadow:
            0 12px 30px rgba(150, 70, 180, 0.35);
    }


    /* =====================================================
       📝 TEXT AREA
       ===================================================== */

    textarea {

        border-radius:
            22px !important;

        border:
            2px solid #e4d4ff !important;

        background:
            rgba(255, 255, 255, 0.88) !important;

        box-shadow:
            0 8px 25px rgba(120, 80, 160, 0.08);

        font-size:
            16px !important;
    }


    textarea:focus {

        border:
            2px solid #b45cff !important;

        box-shadow:
            0 0 0 4px
            rgba(180, 92, 255, 0.12) !important;
    }


    /* =====================================================
       🎀 SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                #ffe8f5 0%,
                #f0e7ff 35%,
                #e4f7ff 70%,
                #fff5dc 100%
            );

        border-right:
            2px solid rgba(255, 255, 255, 0.8);
    }


    /* =====================================================
       📈 PROGRESS BARS
       ===================================================== */

    div[data-testid="stProgress"] > div {

        background-color:
            rgba(255, 255, 255, 0.8);

        border-radius: 20px;
    }


    /* =====================================================
       💬 CHAT MESSAGES
       ===================================================== */

    div[data-testid="stChatMessage"] {

        border-radius: 22px;

        margin-bottom: 10px;

        box-shadow:
            0 5px 18px rgba(100, 80, 140, 0.08);
    }


    /* =====================================================
       📋 DATAFRAME
       ===================================================== */

    div[data-testid="stDataFrame"] {

        border-radius: 20px;

        overflow: hidden;

        box-shadow:
            0 8px 25px rgba(100, 80, 140, 0.10);
    }


    /* =====================================================
       🌟 ALERTS
       ===================================================== */

    div[data-testid="stAlert"] {

        border-radius: 18px;

        border: none;

        box-shadow:
            0 6px 20px rgba(100, 80, 140, 0.08);
    }


    /* =====================================================
       🌈 FOOTER
       ===================================================== */

    .footer {

        margin-top: 45px;

        padding: 25px;

        border-radius: 25px;

        text-align: center;

        background:
            linear-gradient(
                90deg,
                #ffe4f3,
                #eee4ff,
                #def5ff
            );

        color: #765b82;

        font-weight: 700;

        box-shadow:
            0 8px 25px rgba(100, 80, 140, 0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🎀 SpeakWise AI 🎀
        </div>

        <div class="hero-subtitle">
            Your Personal AI Presentation & Communication Coach
        </div>

        <div class="hero-mini">
            🌸 Speak Clearly
            &nbsp; • &nbsp;
            🎤 Present Confidently
            &nbsp; • &nbsp;
            ✨ Improve Continuously
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# 🎀 SpeakWise"
    )

    st.markdown(
        "### Your presentation companion 🌸"
    )

    st.markdown("---")

    duration = st.number_input(
        "⏱️ Presentation Duration (seconds)",
        min_value=1,
        value=120,
        step=10
    )

    presentation_type = st.selectbox(
        "📚 Presentation Type",
        [
            "General",
            "Academic",
            "Technical",
            "Business",
            "Project Presentation",
            "Seminar"
        ]
    )

    st.markdown("---")

    st.markdown(
        """
        ### 🌷 Quick Tips

        💗 **Speak naturally**

        🌸 **Take small pauses**

        ✨ **Avoid unnecessary filler words**

        🎤 **Maintain a comfortable speed**

        🧠 **Understand your topic**

        😊 **Practice with confidence**

        👀 **Connect with your audience**
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ### 🌟 Goal

        Your goal isn't to be perfect.

        Your goal is to become a
        **clearer and more confident speaker.** 💕
        """
    )


# ============================================================
# PRESENTATION INPUT
# ============================================================

st.markdown(
    '<div class="section-title">📝 Your Presentation</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="cute-card">

        <b>🌸 Paste your speech below</b>

        <br><br>

        SpeakWise will analyze your presentation
        and generate a personalized performance report.

        <br><br>

        🎤 Speaking speed
        &nbsp;&nbsp; • &nbsp;&nbsp;
        💬 Filler words
        &nbsp;&nbsp; • &nbsp;&nbsp;
        📝 Content
        &nbsp;&nbsp; • &nbsp;&nbsp;
        🧩 Structure

    </div>
    """,
    unsafe_allow_html=True
)


presentation_text = st.text_area(
    "Speech",
    height=280,
    label_visibility="collapsed",
    placeholder=(
        "Good morning everyone...\n\n"
        "Today I am going to talk about artificial intelligence..."
    )
)


# ============================================================
# PRESENTATION ANALYSIS FUNCTION
# ============================================================

def analyze_presentation(text, duration):

    words = re.findall(
        r"\b[\w']+\b",
        text.lower()
    )

    word_count = len(words)

    duration = max(
        float(duration),
        1
    )

    wpm = round(
        (word_count / duration) * 60
    )

    # --------------------------------------------------------
    # FILLER WORDS
    # --------------------------------------------------------

    filler_words = [
        "um",
        "uh",
        "like",
        "actually",
        "basically",
        "you know",
        "so",
        "literally",
        "okay",
        "right"
    ]

    lower_text = text.lower()

    filler_count = 0

    for filler in filler_words:

        if " " in filler:

            filler_count += lower_text.count(
                filler
            )

        else:

            filler_count += len(
                re.findall(
                    rf"\b{re.escape(filler)}\b",
                    lower_text
                )
            )

    # --------------------------------------------------------
    # SENTENCES
    # --------------------------------------------------------

    sentences = re.split(
        r"[.!?]+",
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    if sentences:

        average_sentence_length = round(
            word_count / len(sentences),
            2
        )

    else:

        average_sentence_length = 0

    # --------------------------------------------------------
    # SPEAKING SPEED SCORE
    # --------------------------------------------------------

    if 110 <= wpm <= 160:

        speed_score = 100

        speed_feedback = (
            "🌟 Excellent! Your speaking speed is "
            "within a comfortable range for most listeners."
        )

    elif 100 <= wpm < 110:

        speed_score = 85

        speed_feedback = (
            "🌸 Your speaking speed is slightly slow. "
            "You can add a little more energy."
        )

    elif 160 < wpm <= 175:

        speed_score = 85

        speed_feedback = (
            "🌸 Your speaking speed is slightly fast. "
            "Try adding short pauses after important points."
        )

    elif wpm < 100:

        speed_score = 70

        speed_feedback = (
            "🐢 Your speaking speed is slow. "
            "Try increasing your pace slightly while "
            "keeping your pronunciation clear."
        )

    else:

        speed_score = 65

        speed_feedback = (
            "⚡ Your speaking speed is quite fast. "
            "Try slowing down and pausing between ideas."
        )

    # --------------------------------------------------------
    # FILLER SCORE
    # --------------------------------------------------------

    if filler_count == 0:

        filler_score = 100

    else:

        filler_score = max(
            0,
            100 - filler_count * 5
        )

    # --------------------------------------------------------
    # CONTENT SCORE
    # --------------------------------------------------------

    if word_count >= 250:

        content_score = 100

    elif word_count >= 180:

        content_score = 92

    elif word_count >= 120:

        content_score = 85

    elif word_count >= 70:

        content_score = 72

    else:

        content_score = 55

    # --------------------------------------------------------
    # STRUCTURE SCORE
    # --------------------------------------------------------

    structure_score = 50

    lower = text.lower()

    structure_words = [
        "today",
        "introduction",
        "first",
        "second",
        "third",
        "next",
        "finally",
        "conclusion",
        "in conclusion",
        "to conclude"
    ]

    found_structure_words = sum(
        1
        for word in structure_words
        if word in lower
    )

    if found_structure_words >= 1:

        structure_score += 20

    if found_structure_words >= 3:

        structure_score += 15

    if len(sentences) >= 5:

        structure_score += 15

    structure_score = min(
        structure_score,
        100
    )

    # --------------------------------------------------------
    # OVERALL SCORE
    # --------------------------------------------------------

    overall_score = round(
        np.average(
            [
                speed_score,
                filler_score,
                content_score,
                structure_score
            ],
            weights=[
                0.30,
                0.20,
                0.30,
                0.20
            ]
        )
    )

    return {

        "Word Count": word_count,

        "Words Per Minute": wpm,

        "Filler Words": filler_count,

        "Average Sentence Length":
            average_sentence_length,

        "Speaking Speed Score":
            speed_score,

        "Filler Word Score":
            filler_score,

        "Content Score":
            content_score,

        "Structure Score":
            structure_score,

        "Overall Score":
            overall_score,

        "Speed Feedback":
            speed_feedback
    }


# ============================================================
# LOCAL AI COACH
# ============================================================

def local_coach_response(
    message,
    analysis=None
):

    msg = message.lower().strip()

    if analysis:

        score = analysis["Overall Score"]

        wpm = analysis["Words Per Minute"]

        fillers = analysis["Filler Words"]

    else:

        score = None

        wpm = None

        fillers = None

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    if "confidence" in msg:

        return (
            "Absolutely! 🌷\n\n"

            "Confidence usually improves when you focus "
            "less on memorizing every sentence and more "
            "on understanding your main ideas.\n\n"

            "Try this before presenting:\n\n"

            "1. 🌬️ Take two slow breaths.\n"
            "2. 👀 Look at one friendly person in the audience.\n"
            "3. 🎤 Start your first sentence slowly.\n"
            "4. ⏸️ Pause after important points.\n"
            "5. 😊 Keep your shoulders relaxed.\n\n"

            "Remember: you don't need to be perfect. "
            "You need to communicate your ideas clearly."
        )

    # --------------------------------------------------------
    # FILLER WORDS
    # --------------------------------------------------------

    if (
        "filler" in msg
        or "um" in msg
        or "uh" in msg
    ):

        if fillers is not None:

            return (
                f"I checked your presentation and detected "
                f"approximately **{fillers} filler words**. 🌸\n\n"

                "Instead of saying 'um', 'uh', or 'like', "
                "try using a short silent pause.\n\n"

                "A useful exercise is to record yourself "
                "for one minute and count your filler words. "
                "Then repeat the speech and try to reduce "
                "the number.\n\n"

                "Silence can actually make you sound "
                "more confident. ✨"
            )

        return (
            "To reduce filler words such as 'um', 'uh', "
            "and 'like', replace them with short pauses. 🌷\n\n"

            "Practice speaking one sentence at a time "
            "and deliberately pause before your next idea."
        )

    # --------------------------------------------------------
    # SPEAKING SPEED
    # --------------------------------------------------------

    if (
        "speed" in msg
        or "fast" in msg
        or "slow" in msg
    ):

        if wpm is not None:

            if wpm < 100:

                advice = (
                    "Your pace is a little slow. "
                    "Try adding a little energy and "
                    "reducing unnecessary pauses."
                )

            elif wpm <= 160:

                advice = (
                    "Your pace is in a comfortable range. "
                    "Keep it natural and pause around "
                    "important ideas."
                )

            else:

                advice = (
                    "Your pace is quite fast. "
                    "Slow down, breathe between ideas, "
                    "and emphasize important words."
                )

            return (
                f"Your measured speaking speed is "
                f"approximately **{wpm} words per minute**. 🎤\n\n"

                f"{advice}\n\n"

                "Try pausing for about one second after "
                "an important idea. This gives the audience "
                "time to understand your message."
            )

        return (
            "A comfortable presentation pace is more "
            "important than speaking quickly. 🌸\n\n"

            "Pause after important ideas and give your "
            "audience time to understand you."
        )

    # --------------------------------------------------------
    # INTRODUCTION
    # --------------------------------------------------------

    if (
        "introduction" in msg
        or "start" in msg
        or "opening" in msg
    ):

        return (
            "Here's a simple structure for a strong "
            "presentation opening: ✨\n\n"

            "**1. Greeting**\n"
            "\"Good morning everyone.\"\n\n"

            "**2. Introduce the topic**\n"
            "\"Today I am going to talk about...\"\n\n"

            "**3. Explain why it matters**\n"
            "\"This topic is important because...\"\n\n"

            "**4. Give the audience a roadmap**\n"
            "\"First I will explain..., then..., and finally...\"\n\n"

            "Keep your opening short and confident. "
            "The goal is to make the audience interested "
            "in what comes next."
        )

    # --------------------------------------------------------
    # CONCLUSION
    # --------------------------------------------------------

    if (
        "conclusion" in msg
        or "ending" in msg
    ):

        return (
            "A strong conclusion should feel like a "
            "clear finish rather than simply stopping. 🌸\n\n"

            "Try this structure:\n\n"

            "• Summarize your main idea.\n"
            "• Mention the most important takeaway.\n"
            "• End with a memorable final sentence.\n\n"

            "For example:\n\n"

            "\"To conclude, Artificial Intelligence has "
            "enormous potential to improve education. "
            "If we use it responsibly, it can support "
            "both students and teachers. Thank you.\""
        )

    # --------------------------------------------------------
    # NERVOUSNESS
    # --------------------------------------------------------

    if (
        "nervous" in msg
        or "anxiety" in msg
        or "scared" in msg
        or "fear" in msg
    ):

        return (
            "Feeling nervous before presenting is "
            "completely normal. 💗\n\n"

            "Try this 30-second routine:\n\n"

            "🌬️ Take two slow breaths.\n"
            "👀 Look at the audience instead of the floor.\n"
            "😊 Relax your shoulders.\n"
            "🎤 Say your first sentence slowly.\n"
            "⏸️ Pause instead of rushing if you forget something.\n\n"

            "Most importantly, don't interpret nervousness "
            "as failure. It simply means that the presentation "
            "matters to you."
        )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    if (
        "score" in msg
        or "performance" in msg
        or "how did i do" in msg
    ):

        if score is not None:

            return (
                f"Your current overall score is "
                f"**{score}/100**. 🌟\n\n"

                f"🎤 Speaking speed: "
                f"**{analysis['Speaking Speed Score']}/100**\n\n"

                f"💬 Filler control: "
                f"**{analysis['Filler Word Score']}/100**\n\n"

                f"📝 Content: "
                f"**{analysis['Content Score']}/100**\n\n"

                f"🧩 Structure: "
                f"**{analysis['Structure Score']}/100**\n\n"

                "The best strategy is to focus on your "
                "lowest score first. Improving one weak "
                "area can make your overall presentation "
                "noticeably stronger."
            )

        return (
            "Analyze your presentation first and I can "
            "give you a detailed explanation of your "
            "performance score. 🌷"
        )

    # --------------------------------------------------------
    # IMPROVEMENT
    # --------------------------------------------------------

    if (
        "improve" in msg
        or "better" in msg
        or "advice" in msg
        or "tips" in msg
    ):

        return (
            "Here are my top recommendations for becoming "
            "a stronger presenter: ✨\n\n"

            "### 🌸 1. Improve your opening\n"
            "Start with a clear topic and explain why "
            "it matters.\n\n"

            "### 🎤 2. Control your speed\n"
            "Don't rush. Use pauses to separate ideas.\n\n"

            "### 💬 3. Reduce filler words\n"
            "Replace 'um' and 'uh' with silence.\n\n"

            "### 👀 4. Improve audience connection\n"
            "Look up regularly instead of continuously "
            "reading your slides.\n\n"

            "### 💗 5. Practice out loud\n"
            "Speaking aloud is much more effective than "
            "only reading your presentation silently."
        )

    # --------------------------------------------------------
    # GENERAL
    # --------------------------------------------------------

    return (
        "Of course! 🌷 I'm your presentation coach.\n\n"

        "Ask me a question naturally. For example:\n\n"

        "💗 **How can I become more confident?**\n\n"

        "💬 **How can I reduce filler words?**\n\n"

        "🎤 **Is my speaking speed good?**\n\n"

        "🌸 **How should I start my presentation?**\n\n"

        "✨ **How should I end my presentation?**\n\n"

        "😊 **Why am I nervous when presenting?**\n\n"

        "📊 **How can I improve my score?**"
    )


# ============================================================
# REAL AI RESPONSE
# ============================================================

def ai_coach_response(
    message,
    analysis=None
):

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    # --------------------------------------------------------
    # NO API KEY
    # --------------------------------------------------------

    if not api_key:

        return local_coach_response(
            message,
            analysis
        )

    # --------------------------------------------------------
    # ANALYSIS CONTEXT
    # --------------------------------------------------------

    if analysis:

        context = f"""
The student's presentation analysis is:

Overall score:
{analysis['Overall Score']}/100

Word count:
{analysis['Word Count']}

Words per minute:
{analysis['Words Per Minute']}

Filler words:
{analysis['Filler Words']}

Average sentence length:
{analysis['Average Sentence Length']}

Speaking speed score:
{analysis['Speaking Speed Score']}/100

Filler word control:
{analysis['Filler Word Score']}/100

Content score:
{analysis['Content Score']}/100

Structure score:
{analysis['Structure Score']}/100
"""

    else:

        context = (
            "The student has not analyzed a presentation yet."
        )

    # --------------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------------

    system_prompt = """
You are SpeakWise AI, a friendly and intelligent
presentation and communication coach.

Answer the student's questions naturally,
similar to a helpful AI tutor.

Do NOT simply list the topics you can help with.

Actually answer the student's question.

Use the presentation analysis when available.

Be encouraging, practical and honest.

The student is a college student, so use clear,
easy-to-understand language.

When appropriate, include:

- examples
- sample sentences
- step-by-step advice
- practice exercises
- personalized recommendations

Do not claim that you performed analysis that is
not present in the provided data.

Keep responses useful and reasonably concise.
"""

    # --------------------------------------------------------
    # API REQUEST
    # --------------------------------------------------------

    payload = {

        "model": "gpt-4o-mini",

        "messages": [

            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content":
                    context
                    + "\n\nStudent's question:\n"
                    + message
            }

        ],

        "temperature": 0.7
    }

    try:

        response = requests.post(

            "https://api.openai.com/v1/chat/completions",

            headers={
                "Authorization":
                    f"Bearer {api_key}",

                "Content-Type":
                    "application/json"
            },

            json=payload,

            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        return (
            data[
                "choices"
            ][
                0
            ][
                "message"
            ][
                "content"
            ]
        )

    except Exception:

        return (
            "🌸 I couldn't connect to the AI service "
            "right now, so I'll answer using my built-in "
            "presentation coach.\n\n"
            + local_coach_response(
                message,
                analysis
            )
        )


# ============================================================
# ANALYZE PRESENTATION BUTTON
# ============================================================

if st.button(
    "✨ Analyze My Presentation",
    type="primary",
    use_container_width=True
):

    if not presentation_text.strip():

        st.warning(
            "🌸 Please enter your presentation speech first."
        )

    else:

        result = analyze_presentation(
            presentation_text,
            duration
        )

        st.session_state[
            "analysis"
        ] = result

        # Reset chat for new presentation

        st.session_state[
            "chat_history"
        ] = []

        st.success(
            "✨ Your presentation has been analyzed successfully!"
        )


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "analysis" in st.session_state:

    result = st.session_state[
        "analysis"
    ]

    # --------------------------------------------------------
    # PERFORMANCE TITLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '📊 Your Performance Report'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SCORE CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="score-card score-card-pink">

                <div class="score-big">
                    🌟 {result['Overall Score']}
                </div>

                <div class="score-label">
                    Overall Score / 100
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="score-card score-card-purple">

                <div class="score-big">
                    🎤 {result['Words Per Minute']}
                </div>

                <div class="score-label">
                    Words / Minute
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="score-card score-card-blue">

                <div class="score-big">
                    💬 {result['Filler Words']}
                </div>

                <div class="score-label">
                    Filler Words
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            f"""
            <div class="score-card score-card-orange">

                <div class="score-big">
                    📝 {result['Word Count']}
                </div>

                <div class="score-label">
                    Total Words
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # COMMUNICATION SKILLS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '🌈 Communication Skills'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="skill-box">
                <b>🎤 Speaking Speed</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            result[
                "Speaking Speed Score"
            ] / 100
        )

        st.caption(
            f"{result['Speaking Speed Score']}/100"
        )

        st.markdown(
            """
            <div class="skill-box">
                <b>💬 Filler Word Control</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            result[
                "Filler Word Score"
            ] / 100
        )

        st.caption(
            f"{result['Filler Word Score']}/100"
        )

    with col2:

        st.markdown(
            """
            <div class="skill-box">
                <b>📝 Content Quality</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            result[
                "Content Score"
            ] / 100
        )

        st.caption(
            f"{result['Content Score']}/100"
        )

        st.markdown(
            """
            <div class="skill-box">
                <b>🧩 Presentation Structure</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            result[
                "Structure Score"
            ] / 100
        )

        st.caption(
            f"{result['Structure Score']}/100"
        )

    # --------------------------------------------------------
    # FEEDBACK
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '💡 Personalized Feedback'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        result[
            "Speed Feedback"
        ]
    )

    # Filler feedback

    if result["Filler Words"] == 0:

        st.success(
            "🌟 Amazing! No common filler words were detected."
        )

    elif result["Filler Words"] <= 5:

        st.warning(
            "🌸 You used a few filler words. "
            "Try replacing them with short pauses."
        )

    else:

        st.error(
            "💬 You are using quite a few filler words. "
            "Practice pausing instead of filling silence."
        )

    # Content feedback

    if result["Content Score"] >= 85:

        st.success(
            "✨ Your presentation contains a good amount of content."
        )

    else:

        st.warning(
            "📝 Consider adding more explanations, examples "
            "and supporting points."
        )

    # Structure feedback

    if result["Structure Score"] >= 75:

        st.success(
            "🧩 Your presentation has a reasonably clear structure."
        )

    else:

        st.warning(
            "🧩 Try using a stronger introduction, organized "
            "main points and a clear conclusion."
        )

    # --------------------------------------------------------
    # DETAILED TABLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '📋 Detailed Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    df = pd.DataFrame(
        {
            "Metric": [

                "Presentation Type",

                "Word Count",

                "Words Per Minute",

                "Filler Words",

                "Average Sentence Length",

                "Speaking Speed",

                "Filler Word Control",

                "Content",

                "Structure",

                "Overall Score"
            ],

            "Value": [

                presentation_type,

                result[
                    "Word Count"
                ],

                result[
                    "Words Per Minute"
                ],

                result[
                    "Filler Words"
                ],

                result[
                    "Average Sentence Length"
                ],

                f"{result['Speaking Speed Score']}/100",

                f"{result['Filler Word Score']}/100",

                f"{result['Content Score']}/100",

                f"{result['Structure Score']}/100",

                f"{result['Overall Score']}/100"
            ]
        }
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# AI PRESENTATION COACH
# ============================================================

st.markdown(
    '<div class="section-title">'
    '💗 AI Presentation Coach'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="chat-intro">

        <div class="coach-title">
            🌸 Talk to your personal AI presentation coach
        </div>

        <br>

        <div>
            Ask questions naturally about your presentation,
            confidence, speaking style, nervousness,
            structure, or communication skills.
        </div>

        <br>

        <b>💬 Try asking:</b>

        <br><br>

        🌷 How can I become more confident?

        <br>

        🎤 Is my speaking speed good?

        <br>

        💬 How can I reduce filler words?

        <br>

        📝 How should I improve my introduction?

        <br>

        🌟 How did I perform?

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHAT HISTORY
# ============================================================

if "chat_history" not in st.session_state:

    st.session_state[
        "chat_history"
    ] = []


# Display history

for message in st.session_state[
    "chat_history"
]:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

user_message = st.chat_input(
    "💬 Ask SpeakWise anything..."
)


if user_message:

    # Add user message

    st.session_state[
        "chat_history"
    ].append(
        {
            "role":
                "user",

            "content":
                user_message
        }
    )

    # Current analysis

    analysis = st.session_state.get(
        "analysis",
        None
    )

    # Generate response

    response = ai_coach_response(
        user_message,
        analysis
    )

    # Add AI response

    st.session_state[
        "chat_history"
    ].append(
        {
            "role":
                "assistant",

            "content":
                response
        }
    )

    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        🎀 <b>SpeakWise AI</b>

        <br><br>

        🌸 Practice today
        &nbsp; • &nbsp;
        🎤 Present confidently
        &nbsp; • &nbsp;
        ✨ Improve tomorrow

        <br><br>

        Made with 💗 using Python + Streamlit + AI

    </div>
    """,
    unsafe_allow_html=True
)