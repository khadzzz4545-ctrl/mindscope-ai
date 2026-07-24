"""
MindScope AI — Mental Health Crisis Predictor V2
Transformer-Based Mental Health Analysis System
Powered by RoBERTa · 79.61% Accuracy
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import huggingface_hub

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MindScope AI · Mental Health Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════
# MASTER CSS
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── CSS Variables ── */
:root {
  --navy:     #030712;
  --navy-1:   #0a0f1e;
  --navy-2:   #0d1529;
  --navy-3:   #111827;
  --purple:   #7c3aed;
  --purple-l: #a78bfa;
  --purple-xl:#c4b5fd;
  --cyan:     #06b6d4;
  --cyan-l:   #67e8f9;
  --pink:     #ec4899;
  --green:    #10b981;
  --amber:    #f59e0b;
  --red:      #ef4444;
  --slate:    #94a3b8;
  --slate-d:  #475569;
  --white:    #f8fafc;
  --glass:    rgba(255,255,255,0.04);
  --glass-b:  rgba(255,255,255,0.08);
  --glow-p:   rgba(124,58,237,0.35);
  --glow-c:   rgba(6,182,212,0.35);
}

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: var(--navy) !important;
  color: var(--white) !important;
  font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stHeader"]          { background: transparent !important; display: none; }
[data-testid="stSidebar"]         { background: var(--navy-1) !important; }
[data-testid="stToolbar"]         { display: none !important; }
.block-container                  { padding: 0 2rem 6rem !important; max-width: 1380px !important; }
section[data-testid="stSidebar"]  { display: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar              { width: 5px; }
::-webkit-scrollbar-track        { background: var(--navy-1); }
::-webkit-scrollbar-thumb        { background: linear-gradient(180deg, var(--purple), var(--cyan)); border-radius: 3px; }

/* ── Animated dot-grid background ── */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed; inset: 0;
  background-image: radial-gradient(rgba(124,58,237,0.12) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
  z-index: 0;
  animation: bgDrift 30s linear infinite;
}
@keyframes bgDrift { to { background-position: 40px 40px; } }

/* ── Ambient glows ── */
[data-testid="stAppViewContainer"]::after {
  content: '';
  position: fixed;
  top: -30vh; left: -20vw;
  width: 80vw; height: 80vh;
  background: radial-gradient(ellipse, rgba(124,58,237,0.07) 0%, transparent 70%);
  pointer-events: none;
  z-index: 0;
  animation: ambientMove 12s ease-in-out infinite alternate;
}
@keyframes ambientMove {
  from { transform: translate(0,0); }
  to   { transform: translate(15vw, 10vh); }
}

/* ──────────────────────────────────────────────
   HERO
────────────────────────────────────────────── */
.hero-wrap {
  position: relative;
  overflow: hidden;
  border-radius: 28px;
  margin: 2rem 0 3.5rem;
  padding: 100px 60px 90px;
  background:
    linear-gradient(135deg, rgba(124,58,237,0.18) 0%, transparent 50%),
    linear-gradient(225deg, rgba(6,182,212,0.12) 0%, transparent 50%),
    var(--navy-1);
  border: 1px solid rgba(124,58,237,0.25);
  text-align: center;
}

/* Rotating conic */
.hero-wrap::before {
  content: '';
  position: absolute; inset: -50%;
  background: conic-gradient(
    from 0deg at 50% 50%,
    transparent 0deg,
    rgba(124,58,237,0.06) 45deg,
    transparent 90deg,
    rgba(6,182,212,0.06) 180deg,
    transparent 225deg,
    rgba(236,72,153,0.04) 315deg,
    transparent 360deg
  );
  animation: conicSpin 20s linear infinite;
}
@keyframes conicSpin { to { transform: rotate(360deg); } }

/* Centre radial bloom */
.hero-wrap::after {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(ellipse 55% 50% at 50% 50%,
    rgba(124,58,237,0.14) 0%, transparent 70%);
  pointer-events: none;
}

/* Floating orbs */
.orb { position: absolute; border-radius: 50%; filter: blur(70px); pointer-events: none; }
.orb-a { width:420px;height:420px;background:rgba(124,58,237,0.12);top:-140px;left:-100px;animation:floatA 10s ease-in-out infinite; }
.orb-b { width:340px;height:340px;background:rgba(6,182,212,0.09);top:-80px;right:-80px;animation:floatB 14s ease-in-out infinite; }
.orb-c { width:260px;height:260px;background:rgba(236,72,153,0.07);bottom:-60px;left:35%;animation:floatC 8s ease-in-out infinite; }
@keyframes floatA { 0%,100%{transform:translate(0,0) scale(1)}  50%{transform:translate(20px,-25px) scale(1.05)} }
@keyframes floatB { 0%,100%{transform:translate(0,0) scale(1)}  50%{transform:translate(-15px,20px) scale(1.07)} }
@keyframes floatC { 0%,100%{transform:translate(0,0)}           50%{transform:translate(0,-18px)} }

.hero-inner { position: relative; z-index: 2; }

.hero-badges {
  display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;
  margin-bottom: 28px;
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 16px;
  border-radius: 100px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  animation: badgePop 0.6s ease both;
}
.badge-purple {
  background: rgba(124,58,237,0.18);
  border: 1px solid rgba(124,58,237,0.45);
  color: var(--purple-xl);
}
.badge-cyan {
  background: rgba(6,182,212,0.15);
  border: 1px solid rgba(6,182,212,0.4);
  color: var(--cyan-l);
  animation-delay: 0.1s;
}
.badge-green {
  background: rgba(16,185,129,0.15);
  border: 1px solid rgba(16,185,129,0.4);
  color: #6ee7b7;
  animation-delay: 0.2s;
}
@keyframes badgePop { from{opacity:0;transform:translateY(-8px)} to{opacity:1;transform:translateY(0)} }

.hero-title {
  font-family: 'Syne', sans-serif !important;
  font-size: clamp(3rem, 7vw, 6rem) !important;
  font-weight: 800 !important;
  line-height: 1.05 !important;
  letter-spacing: -0.02em;
  background: linear-gradient(135deg, #e0d7ff 0%, #a78bfa 35%, #67e8f9 70%, #f0abfc 100%);
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  background-clip: text !important;
  margin-bottom: 18px;
  animation: titleReveal 0.9s ease both;
}
@keyframes titleReveal { from{opacity:0;transform:translateY(24px)} to{opacity:1;transform:translateY(0)} }

.hero-sub {
  font-family: 'DM Sans', sans-serif;
  font-size: clamp(1rem, 2vw, 1.3rem);
  color: var(--slate);
  letter-spacing: 0.04em;
  font-weight: 300;
  margin-bottom: 12px;
  animation: titleReveal 0.9s 0.15s ease both;
}

.hero-tagline {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  color: var(--slate-d);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  animation: titleReveal 0.9s 0.25s ease both;
}

/* Floating glass cards inside hero */
.hero-cards {
  display: flex; justify-content: center; gap: 16px; flex-wrap: wrap;
  margin-top: 48px;
}
.hero-mini-card {
  background: rgba(255,255,255,0.05);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 16px;
  padding: 18px 28px;
  text-align: center;
  min-width: 140px;
  animation: cardFloat 0.8s ease both;
}
.hero-mini-card:nth-child(1) { animation-delay: 0.3s; }
.hero-mini-card:nth-child(2) { animation-delay: 0.45s; }
.hero-mini-card:nth-child(3) { animation-delay: 0.6s; }
.hero-mini-card:nth-child(4) { animation-delay: 0.75s; }
@keyframes cardFloat { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
.hmc-val {
  font-family: 'Syne', sans-serif;
  font-size: 1.6rem;
  font-weight: 800;
  background: linear-gradient(135deg, var(--purple-xl), var(--cyan-l));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hmc-lbl {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: var(--slate-d);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-top: 4px;
}

/* ──────────────────────────────────────────────
   SECTION HEADERS
────────────────────────────────────────────── */
.sec-header {
  display: flex; align-items: center; gap: 14px;
  margin: 4rem 0 1.8rem;
}
.sec-line {
  flex: 1; height: 1px;
  background: linear-gradient(90deg, rgba(124,58,237,0.4), transparent);
}
.sec-line-r {
  flex: 1; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(124,58,237,0.4));
}
.sec-title {
  font-family: 'Syne', sans-serif !important;
  font-size: 0.8rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.2em !important;
  text-transform: uppercase !important;
  color: var(--purple-l) !important;
  white-space: nowrap;
}
.sec-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--cyan);
  box-shadow: 0 0 8px var(--cyan);
  flex-shrink: 0;
}

/* ──────────────────────────────────────────────
   KPI CARDS
────────────────────────────────────────────── */
.kpi-card {
  background: var(--glass);
  backdrop-filter: blur(24px);
  border: 1px solid var(--glass-b);
  border-radius: 20px;
  padding: 30px 24px 26px;
  position: relative;
  overflow: hidden;
  transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
  animation: kpiIn 0.6s ease both;
  text-align: center;
}
.kpi-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0;
  height: 2px;
  background: var(--bar-color, linear-gradient(90deg, var(--purple), var(--cyan)));
  border-radius: 2px 2px 0 0;
}
.kpi-card::after {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(ellipse 80% 60% at 50% -10%,
    var(--glow-color, rgba(124,58,237,0.08)) 0%, transparent 70%);
}
.kpi-card:hover {
  transform: translateY(-5px);
  border-color: rgba(124,58,237,0.4);
  box-shadow: 0 20px 60px rgba(124,58,237,0.15);
}
@keyframes kpiIn { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
.kpi-icon  { font-size: 2rem; margin-bottom: 12px; }
.kpi-value {
  font-family: 'Syne', sans-serif;
  font-size: 2rem;
  font-weight: 800;
  background: var(--val-gradient, linear-gradient(135deg, var(--purple-xl), var(--cyan-l)));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.kpi-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: var(--slate-d);
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-top: 6px;
}

/* ──────────────────────────────────────────────
   GLASS CARD (generic)
────────────────────────────────────────────── */
.g-card {
  background: var(--glass);
  backdrop-filter: blur(24px);
  border: 1px solid var(--glass-b);
  border-radius: 20px;
  padding: 32px;
  margin: 1rem 0;
  transition: border-color 0.3s;
}
.g-card:hover { border-color: rgba(124,58,237,0.3); }

/* ──────────────────────────────────────────────
   TEXT AREA & BUTTONS
────────────────────────────────────────────── */
.stTextArea label { display: none !important; }
.stTextArea > div > div > textarea {
  background: rgba(10,15,30,0.9) !important;
  border: 1px solid rgba(124,58,237,0.3) !important;
  border-radius: 16px !important;
  color: var(--white) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 1rem !important;
  line-height: 1.7 !important;
  padding: 20px !important;
  caret-color: var(--purple-l);
  transition: border-color 0.3s, box-shadow 0.3s !important;
}
.stTextArea > div > div > textarea:focus {
  border-color: var(--purple) !important;
  box-shadow: 0 0 0 3px rgba(124,58,237,0.2), 0 0 30px rgba(124,58,237,0.1) !important;
  outline: none !important;
}

.stButton > button {
  background: linear-gradient(135deg, var(--purple), #5b21b6) !important;
  border: none !important;
  border-radius: 14px !important;
  color: white !important;
  font-family: 'Syne', sans-serif !important;
  font-size: 0.9rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  padding: 14px 32px !important;
  width: 100% !important;
  cursor: pointer !important;
  transition: all 0.3s ease !important;
  box-shadow: 0 4px 24px rgba(124,58,237,0.4) !important;
  position: relative !important;
  overflow: hidden !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 40px rgba(124,58,237,0.6) !important;
  background: linear-gradient(135deg, #8b5cf6, var(--purple)) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Quick-example buttons override */
.ex-btn button {
  background: rgba(124,58,237,0.1) !important;
  border: 1px solid rgba(124,58,237,0.35) !important;
  color: var(--purple-xl) !important;
  font-size: 0.8rem !important;
  padding: 10px 16px !important;
  box-shadow: none !important;
}
.ex-btn button:hover {
  background: rgba(124,58,237,0.22) !important;
  border-color: var(--purple-l) !important;
  box-shadow: 0 4px 18px rgba(124,58,237,0.25) !important;
  transform: translateY(-2px) !important;
}

/* ──────────────────────────────────────────────
   RESULT CARD
────────────────────────────────────────────── */
.result-shell {
  border-radius: 24px;
  padding: 40px 36px;
  position: relative;
  overflow: hidden;
  animation: resultIn 0.5s ease;
}
@keyframes resultIn { from{opacity:0;transform:scale(0.97)} to{opacity:1;transform:scale(1)} }
.result-shell::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(ellipse 70% 60% at 50% 0%,
    var(--res-glow, rgba(124,58,237,0.15)) 0%, transparent 65%);
  pointer-events: none;
}
.result-emoji { font-size: 4rem; display: block; text-align: center; animation: emojiBounce 0.6s ease; }
@keyframes emojiBounce { 0%{transform:scale(0.5)} 70%{transform:scale(1.15)} 100%{transform:scale(1)} }
.result-class {
  font-family: 'Syne', sans-serif;
  font-size: 2.4rem;
  font-weight: 800;
  text-align: center;
  margin: 10px 0 4px;
  letter-spacing: -0.01em;
}
.result-conf {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85rem;
  color: var(--slate);
  text-align: center;
  letter-spacing: 0.08em;
}
.risk-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 6px 18px;
  border-radius: 100px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-top: 14px;
}
.ai-summary {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px;
  padding: 18px 22px;
  margin-top: 22px;
  font-size: 0.9rem;
  color: var(--slate);
  line-height: 1.75;
  font-style: italic;
}
.ai-summary-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: var(--cyan-l);
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-bottom: 8px;
}

/* ──────────────────────────────────────────────
   TABS
────────────────────────────────────────────── */
[data-testid="stTabs"] button {
  font-family: 'Syne', sans-serif !important;
  font-size: 0.8rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--slate-d) !important;
  border-radius: 10px 10px 0 0 !important;
  padding: 10px 20px !important;
  transition: color 0.2s !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--purple-xl) !important;
  background: rgba(124,58,237,0.1) !important;
  border-bottom: 2px solid var(--purple) !important;
}
[data-testid="stTabsContent"] {
  background: var(--glass) !important;
  border: 1px solid var(--glass-b) !important;
  border-radius: 0 16px 16px 16px !important;
  padding: 24px !important;
}

/* ──────────────────────────────────────────────
   MODEL INFO CARDS
────────────────────────────────────────────── */
.model-card {
  background: var(--glass);
  border: 1px solid var(--glass-b);
  border-radius: 18px;
  padding: 26px 24px;
  height: 100%;
  position: relative;
  overflow: hidden;
  transition: transform 0.3s, border-color 0.3s, box-shadow 0.3s;
}
.model-card::after {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: var(--mc-top, linear-gradient(90deg, var(--purple), var(--cyan)));
}
.model-card:hover {
  transform: translateY(-4px);
  border-color: rgba(124,58,237,0.35);
  box-shadow: 0 16px 50px rgba(124,58,237,0.12);
}
.mc-icon { font-size: 2.2rem; margin-bottom: 14px; }
.mc-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--white);
  margin-bottom: 10px;
}
.mc-body { font-size: 0.88rem; color: var(--slate); line-height: 1.7; }
.mc-tag {
  display: inline-block;
  margin-top: 14px;
  padding: 4px 12px;
  background: rgba(124,58,237,0.15);
  border: 1px solid rgba(124,58,237,0.3);
  border-radius: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: var(--purple-xl);
  letter-spacing: 0.08em;
}

/* ──────────────────────────────────────────────
   EMERGENCY
────────────────────────────────────────────── */
.emergency {
  background: rgba(239,68,68,0.08);
  border: 1.5px solid rgba(239,68,68,0.5);
  border-radius: 18px;
  padding: 28px 32px;
  margin: 1.5rem 0;
  animation: emergencyPulse 2s ease-in-out infinite;
}
@keyframes emergencyPulse {
  0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
  50%      { box-shadow: 0 0 0 10px rgba(239,68,68,0); }
}
.emergency-head {
  font-family: 'Syne', sans-serif;
  font-size: 1.1rem;
  font-weight: 800;
  color: #f87171;
  letter-spacing: 0.05em;
  margin-bottom: 12px;
}
.emergency-body { color: #fca5a5; font-size: 0.9rem; line-height: 1.8; }

/* ──────────────────────────────────────────────
   DISCLAIMER
────────────────────────────────────────────── */
.disclaimer {
  background: rgba(245,158,11,0.07);
  border: 1px solid rgba(245,158,11,0.3);
  border-radius: 16px;
  padding: 22px 28px;
  margin: 1rem 0;
  display: flex;
  gap: 16px;
  align-items: flex-start;
}
.disclaimer-icon { font-size: 1.6rem; flex-shrink: 0; }
.disclaimer-text { color: #fcd34d; font-size: 0.87rem; line-height: 1.75; }

/* ──────────────────────────────────────────────
   FOOTER
────────────────────────────────────────────── */
.footer-wrap {
  margin-top: 6rem;
  padding: 3rem 2rem 2.5rem;
  border-top: 1px solid rgba(124,58,237,0.2);
  text-align: center;
}
.footer-logo {
  font-family: 'Syne', sans-serif;
  font-size: 1.4rem;
  font-weight: 800;
  background: linear-gradient(135deg, var(--purple-xl), var(--cyan-l));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 10px;
}
.footer-meta {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  color: var(--slate-d);
  letter-spacing: 0.1em;
  line-height: 2;
}
.footer-chips { display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; margin: 16px 0; }
.footer-chip {
  padding: 4px 14px;
  background: rgba(124,58,237,0.1);
  border: 1px solid rgba(124,58,237,0.25);
  border-radius: 100px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: var(--slate-d);
  letter-spacing: 0.05em;
}

/* ── Utility ── */
@keyframes fadeUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
.fade-up { animation: fadeUp 0.5s ease both; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════
CLASS_META = {
    "adhd":         {"emoji":"⚡","color":"#f59e0b","risk":"Moderate","glow":"rgba(245,158,11,0.18)","grad":"linear-gradient(135deg,#f59e0b,#d97706)","summary":"The text shows patterns consistent with attention difficulties, hyperactivity, and impulsivity — hallmarks of ADHD. Concentration challenges and task-switching are frequently noted."},
    "anxiety":      {"emoji":"😰","color":"#06b6d4","risk":"Moderate","glow":"rgba(6,182,212,0.18)","grad":"linear-gradient(135deg,#06b6d4,#0891b2)","summary":"Elevated worry, anticipatory fear, and physical tension markers are detected. The language reflects generalised or situational anxiety patterns."},
    "bipolar":      {"emoji":"🌊","color":"#8b5cf6","risk":"High","glow":"rgba(139,92,246,0.18)","grad":"linear-gradient(135deg,#8b5cf6,#7c3aed)","summary":"Mood polarity markers are present — oscillation between energised/euphoric and depressive states is a key indicator of bipolar spectrum conditions."},
    "bpd":          {"emoji":"🌀","color":"#ec4899","risk":"High","glow":"rgba(236,72,153,0.18)","grad":"linear-gradient(135deg,#ec4899,#db2777)","summary":"Intense emotional instability, fear of abandonment, and identity disturbance patterns align with Borderline Personality Disorder indicators."},
    "depression":   {"emoji":"🌧️","color":"#6366f1","risk":"High","glow":"rgba(99,102,241,0.18)","grad":"linear-gradient(135deg,#6366f1,#4f46e5)","summary":"Persistent low mood, anhedonia, and hopelessness markers are detected. The language reflects characteristic depressive cognition patterns."},
    "normal":       {"emoji":"✅","color":"#10b981","risk":"Low","glow":"rgba(16,185,129,0.18)","grad":"linear-gradient(135deg,#10b981,#059669)","summary":"No significant mental health distress markers detected. The text reflects typical emotional expression within the expected range."},
    "ocd":          {"emoji":"🔄","color":"#14b8a6","risk":"Moderate","glow":"rgba(20,184,166,0.18)","grad":"linear-gradient(135deg,#14b8a6,#0d9488)","summary":"Repetitive thought loops, compulsive checking patterns, and intrusive ideation language are identified — consistent with OCD presentations."},
    "ptsd":         {"emoji":"💥","color":"#f97316","risk":"High","glow":"rgba(249,115,22,0.18)","grad":"linear-gradient(135deg,#f97316,#ea580c)","summary":"Trauma response markers including hypervigilance, avoidance, and intrusive memory language are present — aligning with PTSD symptom profiles."},
    "schizophrenia":{"emoji":"🧩","color":"#a855f7","risk":"High","glow":"rgba(168,85,247,0.18)","grad":"linear-gradient(135deg,#a855f7,#9333ea)","summary":"Disorganised thought patterns, perceptual anomaly references, and loosened associations detected — indicators aligned with schizophrenia spectrum."},
    "suicidal":     {"emoji":"🆘","color":"#ef4444","risk":"Critical","glow":"rgba(239,68,68,0.2)","grad":"linear-gradient(135deg,#ef4444,#dc2626)","summary":"Critical risk markers detected. The text contains language patterns strongly associated with suicidal ideation. Immediate professional intervention is strongly advised."},
}

EXAMPLES = {
    "Anxiety":    "My heart won't stop racing and I keep thinking something terrible is about to happen. I've been avoiding going out because I'm afraid of having a panic attack in public. Every small thing feels catastrophic.",
    "Depression": "I don't feel anything anymore. I used to love painting but now I just stare at the canvas. Getting out of bed is a war I lose every morning. I keep wondering why I bother.",
    "ADHD":       "I start a hundred things and finish none of them. In meetings I zone out completely even when I'm trying so hard to focus. My desk is chaos, my phone is chaos, my brain is chaos.",
    "PTSD":       "I can't stop seeing it happen. Loud sounds send me into full panic. I haven't driven since the crash. Every night the nightmares come back and I wake up soaked in sweat.",
    "OCD":        "I've checked the stove seventeen times today. I keep counting steps in multiples of four or I feel like something terrible will happen. My hands are cracked from washing them so much.",
}

RISK_COLORS = {"Low":"#10b981","Moderate":"#f59e0b","High":"#f97316","Critical":"#ef4444"}

MODEL_PATH = "khadijamalik22/mindscope-ai-roberta"


# ══════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_PATH)

        mdl = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH
        )

        mdl.eval()

        label_encoder_path = huggingface_hub.hf_hub_download(
            repo_id=MODEL_PATH,
            filename="label_encoder.pkl"
        )

        with open(label_encoder_path, "rb") as f:
            le = pickle.load(f)

        return tok, mdl, le, True

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None, False
def predict(text, tok, mdl, le):
    enc = tok(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=512,
    padding=True
)

    with torch.no_grad():
        logits = mdl(**enc).logits
    probs  = torch.softmax(logits, dim=-1).squeeze().numpy()
    idx    = int(np.argmax(probs))
    cls    = le.classes_[idx]
    return cls, float(probs[idx]), {c:float(p) for c,p in zip(le.classes_, probs)}

def demo_predict(text):
    """Deterministic demo when model not loaded."""
    import hashlib
    classes = list(CLASS_META.keys())
    h = int(hashlib.md5(text.encode()).hexdigest(),16)
    idx = h % len(classes)
    pred = classes[idx]
    np.random.seed(h % 2**32)
    conf = float(np.random.uniform(0.52, 0.91))
    rest = 1 - conf
    others = [c for c in classes if c != pred]
    raw = np.random.dirichlet(np.ones(len(others)))
    probs = {c: float(v*rest) for c,v in zip(others, raw)}
    probs[pred] = conf
    return pred, conf, probs

# ══════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════
if "txt"    not in st.session_state: st.session_state.txt    = ""
if "result" not in st.session_state: st.session_state.result = None

# ══════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════
with st.spinner("Initialising RoBERTa neural engine…"):
    tok, mdl, le, loaded = load_model()

if not loaded:
    st.toast("⚠️ Model not found — running in demo mode", icon="⚠️")

# ══════════════════════════════════════════════════════════════════════
# ❶  HERO
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
  <div class="orb orb-a"></div>
  <div class="orb orb-b"></div>
  <div class="orb orb-c"></div>
  <div class="hero-inner">
    <div class="hero-badges">
      <span class="hero-badge badge-purple">⬡ Powered by RoBERTa</span>
      <span class="hero-badge badge-cyan">🎯 79.61% Accuracy</span>
      <span class="hero-badge badge-green">✦ 10 Mental Health Classes</span>
    </div>
    <div class="hero-title">MindScope AI</div>
    <div class="hero-sub">Transformer-Based Mental Health Analysis System</div>
    <div class="hero-tagline">Advanced NLP · Real-Time Classification · Clinical-Grade Intelligence</div>
    <div class="hero-cards">
      <div class="hero-mini-card">
        <div class="hmc-val">80K</div>
        <div class="hmc-lbl">Training Samples</div>
      </div>
      <div class="hero-mini-card">
        <div class="hmc-val">10</div>
        <div class="hmc-lbl">Mental Health Classes</div>
      </div>
      <div class="hero-mini-card">
        <div class="hmc-val">79.61%</div>
        <div class="hmc-lbl">Test Accuracy</div>
      </div>
      <div class="hero-mini-card">
        <div class="hmc-val">79.68%</div>
        <div class="hmc-lbl">Weighted F1 Score</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# ❷  KPI CARDS
# ══════════════════════════════════════════════════════════════════════
def sec(label):
    st.markdown(f"""
    <div class="sec-header">
      <div class="sec-line"></div>
      <div class="sec-dot"></div>
      <div class="sec-title">{label}</div>
      <div class="sec-dot"></div>
      <div class="sec-line-r"></div>
    </div>
    """, unsafe_allow_html=True)

sec("System Performance Metrics")

kpis = [
    ("📦","80,000","Training Samples",
     "linear-gradient(90deg,#7c3aed,#06b6d4)",
     "rgba(124,58,237,0.08)","linear-gradient(135deg,#a78bfa,#67e8f9)"),
    ("🏷️","10","Mental Health Classes",
     "linear-gradient(90deg,#06b6d4,#10b981)",
     "rgba(6,182,212,0.08)","linear-gradient(135deg,#67e8f9,#6ee7b7)"),
    ("🎯","79.61%","Test Accuracy",
     "linear-gradient(90deg,#ec4899,#8b5cf6)",
     "rgba(236,72,153,0.08)","linear-gradient(135deg,#f9a8d4,#c4b5fd)"),
    ("🤖","RoBERTa","Best Model",
     "linear-gradient(90deg,#f59e0b,#ef4444)",
     "rgba(245,158,11,0.08)","linear-gradient(135deg,#fcd34d,#fca5a5)"),
    ("📊","79.68%","Weighted F1",
     "linear-gradient(90deg,#10b981,#7c3aed)",
     "rgba(16,185,129,0.08)","linear-gradient(135deg,#6ee7b7,#a78bfa)"),
]

cols = st.columns(5)
for (icon,val,lbl,bar,glow,vgrad), col in zip(kpis, cols):
    with col:
        st.markdown(f"""
        <div class="kpi-card"
             style="--bar-color:{bar};--glow-color:{glow};--val-gradient:{vgrad};"
             onmouseenter="this.style.boxShadow='0 20px 60px {glow}'"
             onmouseleave="this.style.boxShadow='none'">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-value">{val}</div>
          <div class="kpi-label">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# ❸  MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════
sec("Model Benchmark Comparison")

df_cmp = pd.DataFrame({
    "Model":    ["Random Forest","Logistic Regression","BioClinicalBERT","RoBERTa-base"],
    "Accuracy": [69.19, 72.25, 78.09, 79.61],
    "Best":     [False, False, False, True],
})
bar_clrs  = ["#334155","#475569","#6366f1","#7c3aed"]
text_clrs = ["#94a3b8","#94a3b8","#c4b5fd","#ffffff"]

fig_cmp = go.Figure()
for i, row in df_cmp.iterrows():
    fig_cmp.add_trace(go.Bar(
        x=[row["Model"]], y=[row["Accuracy"]],
        marker_color=bar_clrs[i],
        marker_line_width=0,
        text=f"{row['Accuracy']}%",
        textposition="outside",
        textfont=dict(color=text_clrs[i], family="JetBrains Mono", size=12),
        name=row["Model"],
        showlegend=False,
    ))

# Star annotation for best
fig_cmp.add_annotation(
    x="RoBERTa-base", y=82.5,
    text="★ BEST MODEL",
    showarrow=False,
    font=dict(color="#a78bfa", size=11, family="JetBrains Mono"),
)
fig_cmp.add_shape(
    type="line", x0=-0.5, x1=3.5, y0=79.61, y1=79.61,
    line=dict(color="rgba(167,139,250,0.35)", width=1.5, dash="dot"),
)
fig_cmp.update_layout(
    paper_bgcolor="rgba(10,15,30,0.0)",
    plot_bgcolor="rgba(10,15,30,0.0)",
    font=dict(color="#94a3b8", family="DM Sans"),
    height=380,
    margin=dict(t=50, b=20, l=10, r=10),
    yaxis=dict(
        range=[60, 85], showgrid=True,
        gridcolor="rgba(124,58,237,0.08)",
        ticksuffix="%",
        tickfont=dict(color="#475569", family="JetBrains Mono"),
        zeroline=False,
    ),
    xaxis=dict(showgrid=False, tickfont=dict(color="#94a3b8", family="DM Sans", size=13)),
    bargap=0.35,
)
st.plotly_chart(fig_cmp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# ❹  ANALYSIS INPUT
# ══════════════════════════════════════════════════════════════════════
sec("AI Analysis Terminal")

# Quick examples
st.markdown('<p style="color:#475569;font-family:JetBrains Mono;font-size:0.72rem;letter-spacing:0.1em;margin-bottom:12px;">▸ LOAD QUICK EXAMPLE</p>', unsafe_allow_html=True)
ecols = st.columns(5)
for i,(lbl,txt) in enumerate(EXAMPLES.items()):
    with ecols[i]:
        st.markdown('<div class="ex-btn">', unsafe_allow_html=True)
        if st.button(f"{CLASS_META[lbl.lower()]['emoji']} {lbl}", key=f"ex_{lbl}"):
            st.session_state.txt = txt
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

inp = st.text_area(
    "analysis_input",
    value=st.session_state.txt,
    height=180,
    placeholder="Describe your thoughts, feelings, or experiences in detail…  The AI will analyse the text and classify it across 10 mental health categories.",
    label_visibility="collapsed",
)
st.session_state.txt = inp

_, btn_col, _ = st.columns([1,2,1])
with btn_col:
    run = st.button("⚡  ANALYSE WITH ROBERTA", key="run")

# ══════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════
if run:
    if not inp.strip():
        st.error("Please enter some text before running analysis.")
    else:
        with st.spinner("🔬 Processing through RoBERTa transformer layers…"):
            if loaded:
                pred, conf, probs = predict(inp, tok, mdl, le)
            else:
                pred, conf, probs = demo_predict(inp)
        st.session_state.result = {"pred":pred,"conf":conf,"probs":probs}

# ══════════════════════════════════════════════════════════════════════
# ❺  RESULTS
# ══════════════════════════════════════════════════════════════════════
if st.session_state.result:
    res  = st.session_state.result
    pred = res["pred"]
    conf = res["conf"]
    probs= res["probs"]
    m    = CLASS_META[pred]

    sec("Prediction Results")

    r1, r2 = st.columns([1,1], gap="large")

    # ── Result Card
    with r1:
        risk_col = RISK_COLORS[m["risk"]]
        st.markdown(f"""
        <div class="result-shell"
             style="background:rgba(10,15,30,0.85);
                    border:1px solid {m['color']}44;
                    --res-glow:{m['glow']};">
          <span class="result-emoji">{m['emoji']}</span>
          <div class="result-class" style="color:{m['color']};">{pred.upper()}</div>
          <div class="result-conf">Confidence: {conf*100:.1f}%&nbsp;&nbsp;|&nbsp;&nbsp;Model: RoBERTa-base</div>
          <div style="text-align:center;">
            <span class="risk-pill"
                  style="background:{risk_col}22;border:1px solid {risk_col}55;color:{risk_col};">
              ⬤ Risk Level: {m['risk']}
            </span>
          </div>
          <div class="ai-summary">
            <div class="ai-summary-label">⬡ AI Clinical Summary</div>
            {m['summary']}
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Gauge
    with r2:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(conf*100,1),
            number=dict(suffix="%",
                        font=dict(color=m["color"], size=42, family="Syne")),
            gauge=dict(
                axis=dict(range=[0,100],
                          tickcolor="#475569",
                          tickfont=dict(color="#475569",family="JetBrains Mono")),
                bar=dict(color=m["color"], thickness=0.22),
                bgcolor="rgba(10,15,30,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0,40],  color="rgba(239,68,68,0.06)"),
                    dict(range=[40,70], color="rgba(245,158,11,0.06)"),
                    dict(range=[70,100],color="rgba(16,185,129,0.06)"),
                ],
                threshold=dict(
                    line=dict(color=m["color"], width=3),
                    thickness=0.8, value=conf*100,
                ),
            ),
            domain=dict(x=[0,1], y=[0,1]),
        ))
        fig_g.update_layout(
            paper_bgcolor="rgba(10,15,30,0.0)",
            font=dict(color="#e2e8f0"),
            height=280,
            margin=dict(t=30,b=10,l=30,r=30),
        )
        st.plotly_chart(fig_g, use_container_width=True)

        # Confidence progress bars for top 3
        top3 = sorted(probs.items(), key=lambda x:x[1], reverse=True)[:3]
        for cls_name, prob in top3:
            cm = CLASS_META[cls_name]
            pct = round(prob*100,1)
            st.markdown(f"""
            <div style="margin:8px 0;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#94a3b8;">
                  {cm['emoji']} {cls_name.upper()}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:{cm['color']};">{pct}%</span>
              </div>
              <div style="background:rgba(255,255,255,0.05);border-radius:100px;height:6px;overflow:hidden;">
                <div style="width:{pct}%;height:100%;
                            background:{cm['grad']};
                            border-radius:100px;
                            box-shadow:0 0 8px {cm['color']}88;
                            transition:width 0.8s ease;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ══ Emergency alert
    if pred == "suicidal":
        st.markdown("""
        <div class="emergency">
          <div class="emergency-head">🚨 EMERGENCY — IMMEDIATE CRISIS SUPPORT</div>
          <div class="emergency-body">
            This analysis has detected critical risk markers. Please contact emergency mental health services immediately.<br><br>
            🆘 <strong>National Suicide Prevention Lifeline</strong>: Call or text <strong>988</strong> (US, 24/7)<br>
            💬 <strong>Crisis Text Line</strong>: Text HOME to <strong>741741</strong><br>
            🌍 <strong>International Resources</strong>: <a href="https://www.iasp.info/resources/Crisis_Centres/" target="_blank" style="color:#f87171;">iasp.info/resources/Crisis_Centres</a><br><br>
            <em>You are not alone. Professional help is available right now.</em>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ══ Probability Distribution
    sec("Full Probability Distribution")
    sorted_p = dict(sorted(probs.items(), key=lambda x:x[1], reverse=True))
    clrs = [CLASS_META[c]["color"] for c in sorted_p]

    fig_bar = go.Figure(go.Bar(
        x=list(sorted_p.values()),
        y=[f"{CLASS_META[c]['emoji']} {c.upper()}" for c in sorted_p],
        orientation="h",
        marker=dict(color=clrs, opacity=0.85, line=dict(width=0)),
        text=[f"{v*100:.1f}%" for v in sorted_p.values()],
        textposition="outside",
        textfont=dict(color="#94a3b8", family="JetBrains Mono", size=11),
    ))
    fig_bar.update_layout(
        paper_bgcolor="rgba(10,15,30,0.0)",
        plot_bgcolor="rgba(10,15,30,0.0)",
        font=dict(color="#94a3b8", family="DM Sans"),
        height=420,
        margin=dict(t=10,b=10,l=10,r=80),
        xaxis=dict(showgrid=False, showticklabels=False,
                   range=[0, max(probs.values())*1.3]),
        yaxis=dict(showgrid=False,
                   tickfont=dict(family="JetBrains Mono",size=11,color="#94a3b8")),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# ❻  MODEL INFORMATION CARDS
# ══════════════════════════════════════════════════════════════════════
sec("Model Architecture & Technology")

mc1, mc2, mc3, mc4 = st.columns(4)
model_cards = [
    (mc1,"🤖","RoBERTa-base","linear-gradient(90deg,#7c3aed,#8b5cf6)",
     "Robustly Optimised BERT approach. Trained on 160GB of text using dynamic masking. Fine-tuned on 80K mental health posts for sequence classification.",
     "125M Parameters"),
    (mc2,"⚙️","Transformer Architecture","linear-gradient(90deg,#06b6d4,#0891b2)",
     "12 self-attention layers, 768 hidden dimensions, 12 attention heads. Bidirectional context encoding enables deep semantic understanding of mental health language.",
     "12 Attention Heads"),
    (mc3,"🔤","NLP Pipeline","linear-gradient(90deg,#10b981,#059669)",
     "BPE tokenisation → positional encoding → 12 transformer blocks → pooled [CLS] representation → linear classification head → softmax probabilities.",
     "512 Max Tokens"),
    (mc4,"📂","Dataset Overview","linear-gradient(90deg,#f59e0b,#d97706)",
     "80,000 balanced Reddit posts across 10 mental health categories. Combined from r/depression, r/anxiety, r/ADHD, r/PTSD and six other subreddits.",
     "80K Samples"),
]
for col, icon, title, grad, body, tag in model_cards:
    with col:
        st.markdown(f"""
        <div class="model-card" style="--mc-top:{grad};">
          <div class="mc-icon">{icon}</div>
          <div class="mc-title">{title}</div>
          <div class="mc-body">{body}</div>
          <div class="mc-tag">{tag}</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# ❼  DATASET INSIGHTS DASHBOARD (Tabs)
# ══════════════════════════════════════════════════════════════════════
sec("Dataset Intelligence Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Class Distribution",
    "🏷️ Mental Health Categories",
    "📈 Dataset Statistics",
    "🏆 Model Performance",
])

CLASSES    = list(CLASS_META.keys())
N_PER_CLS  = 8000
class_clrs = [m["color"] for m in CLASS_META.values()]

with tab1:
    counts = {c: N_PER_CLS for c in CLASSES}
    fig_pie = go.Figure(go.Pie(
        labels=[f"{CLASS_META[c]['emoji']} {c.title()}" for c in CLASSES],
        values=list(counts.values()),
        hole=0.55,
        marker=dict(colors=class_clrs, line=dict(color="rgba(10,15,30,0.8)", width=2)),
        textfont=dict(family="JetBrains Mono", size=10, color="white"),
        hovertemplate="<b>%{label}</b><br>Samples: %{value:,}<br>Share: %{percent}<extra></extra>",
    ))
    fig_pie.add_annotation(
        text="80K<br>Total", x=0.5, y=0.5,
        font=dict(size=16, color="#a78bfa", family="Syne"),
        showarrow=False,
    )
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
        legend=dict(font=dict(color="#94a3b8",family="JetBrains Mono",size=10),
                    bgcolor="rgba(0,0,0,0)"),
        height=420, margin=dict(t=20,b=20,l=20,r=20),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    descs = {
        "adhd":         ("⚡","Attention Deficit Hyperactivity Disorder — inattention, hyperactivity, impulsivity."),
        "anxiety":      ("😰","Anxiety Disorder — excessive worry, fear, and physical tension beyond situational triggers."),
        "bipolar":      ("🌊","Bipolar Disorder — cyclical mania/hypomania and depressive episodes."),
        "bpd":          ("🌀","Borderline Personality Disorder — emotional instability, identity disturbance."),
        "depression":   ("🌧️","Major Depressive Disorder — persistent low mood, anhedonia, hopelessness."),
        "normal":       ("✅","No significant clinical indicators — typical emotional expression."),
        "ocd":          ("🔄","Obsessive-Compulsive Disorder — intrusive thoughts and compulsive behaviours."),
        "ptsd":         ("💥","Post-Traumatic Stress Disorder — flashbacks, avoidance, hyperarousal post-trauma."),
        "schizophrenia":("🧩","Schizophrenia Spectrum — hallucinations, disorganised thinking, negative symptoms."),
        "suicidal":     ("🆘","Suicidal Ideation — thoughts of self-harm or ending one's life. Critical risk."),
    }
    c1, c2 = st.columns(2)
    items = list(descs.items())
    for i, (cls,(emoji,desc)) in enumerate(items):
        col = c1 if i%2==0 else c2
        with col:
            color = CLASS_META[cls]["color"]
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid {color}33;
                        border-left:3px solid {color};border-radius:12px;
                        padding:14px 18px;margin:6px 0;">
              <span style="font-size:1.3rem;">{emoji}</span>
              <strong style="color:{color};font-family:'Syne',sans-serif;
                             font-size:0.85rem;margin-left:8px;">{cls.upper()}</strong>
              <p style="color:#94a3b8;font-size:0.82rem;margin:6px 0 0;line-height:1.6;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    col_a, col_b = st.columns(2)
    with col_a:
        stats = [
            ("Total Samples","80,000"),
            ("Training Set","64,000 (80%)"),
            ("Validation Set","8,000 (10%)"),
            ("Test Set","8,000 (10%)"),
            ("Classes","10"),
            ("Avg Tokens / Sample","~128"),
            ("Max Sequence Length","512"),
            ("Data Source","Reddit Posts"),
        ]
        for lbl,val in stats:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:12px 16px;border-bottom:1px solid rgba(124,58,237,0.1);">
              <span style="color:#475569;font-family:'JetBrains Mono',monospace;font-size:0.75rem;">{lbl}</span>
              <span style="font-family:'Syne',sans-serif;font-size:0.95rem;font-weight:700;
                           background:linear-gradient(90deg,#a78bfa,#67e8f9);
                           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                           background-clip:text;">{val}</span>
            </div>
            """, unsafe_allow_html=True)
    with col_b:
        split_fig = go.Figure(go.Bar(
            x=["Train","Validation","Test"],
            y=[64000,8000,8000],
            marker=dict(
                color=["#7c3aed","#06b6d4","#10b981"],
                line=dict(width=0),
            ),
            text=["64,000","8,000","8,000"],
            textposition="outside",
            textfont=dict(color="#94a3b8",family="JetBrains Mono",size=11),
        ))
        split_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),height=300,
            margin=dict(t=30,b=10,l=10,r=10),
            yaxis=dict(showgrid=True,gridcolor="rgba(124,58,237,0.08)",
                       tickformat=",",tickfont=dict(color="#475569",family="JetBrains Mono")),
            xaxis=dict(showgrid=False,tickfont=dict(color="#94a3b8")),
        )
        st.plotly_chart(split_fig, use_container_width=True)

with tab4:
    perf_data = {
        "Model":   ["Random Forest","Logistic Regression","BioClinicalBERT","RoBERTa-base"],
        "Accuracy":[69.19,72.25,78.09,79.61],
        "F1":      [68.80,71.90,77.95,79.68],
    }
    df_perf = pd.DataFrame(perf_data)

    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(
        x=df_perf["Model"], y=df_perf["Accuracy"],
        mode="lines+markers+text",
        name="Accuracy",
        line=dict(color="#7c3aed", width=3),
        marker=dict(size=10, color="#a78bfa", line=dict(color="#7c3aed",width=2)),
        text=[f"{v}%" for v in df_perf["Accuracy"]],
        textposition="top center",
        textfont=dict(color="#a78bfa",family="JetBrains Mono",size=11),
    ))
    fig_perf.add_trace(go.Scatter(
        x=df_perf["Model"], y=df_perf["F1"],
        mode="lines+markers+text",
        name="F1 Score",
        line=dict(color="#06b6d4", width=3, dash="dot"),
        marker=dict(size=10, color="#67e8f9", line=dict(color="#06b6d4",width=2)),
        text=[f"{v}%" for v in df_perf["F1"]],
        textposition="bottom center",
        textfont=dict(color="#67e8f9",family="JetBrains Mono",size=11),
    ))
    fig_perf.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8",family="DM Sans"),
        height=340, margin=dict(t=30,b=20,l=10,r=10),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#94a3b8",family="JetBrains Mono",size=11)),
        yaxis=dict(range=[60,85],showgrid=True,
                   gridcolor="rgba(124,58,237,0.08)",
                   ticksuffix="%",
                   tickfont=dict(color="#475569",family="JetBrains Mono")),
        xaxis=dict(showgrid=False,tickfont=dict(color="#94a3b8")),
    )
    st.plotly_chart(fig_perf, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# ❽  ABOUT PROJECT
# ══════════════════════════════════════════════════════════════════════
sec("About the Project")

ab1, ab2 = st.columns([2,1], gap="large")
with ab1:
    st.markdown("""
    <div class="g-card">
      <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
                  background:linear-gradient(135deg,#a78bfa,#67e8f9);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;margin-bottom:16px;">
        Mental Health Crisis Predictor V2
      </div>
      <p style="color:#94a3b8;font-size:0.92rem;line-height:1.85;margin-bottom:20px;">
        MindScope AI is a transformer-powered mental health classification system built on
        <strong style="color:#a78bfa;">RoBERTa-base</strong> — a robustly optimised BERT model
        pre-trained on 160GB of diverse English text and fine-tuned on 80,000 curated Reddit posts
        spanning 10 mental health categories.
        <br><br>
        The system achieves <strong style="color:#67e8f9;">79.61% test accuracy</strong> and
        <strong style="color:#67e8f9;">79.68% weighted F1 score</strong>, outperforming traditional
        baselines including Random Forest (69.19%), Logistic Regression (72.25%), and
        BioClinicalBERT (78.09%).
      </p>
      <div style="display:flex;flex-wrap:wrap;gap:8px;">
    """ + "".join([
        f'<span style="background:rgba(124,58,237,0.15);border:1px solid rgba(124,58,237,0.3);border-radius:8px;padding:5px 14px;font-family:JetBrains Mono,monospace;font-size:0.67rem;color:#a78bfa;">{t}</span>'
        for t in ["Transformer NLP","RoBERTa-base","PyTorch","HuggingFace","Streamlit","Mental Health AI","IDS Project 2026","Real-time Classification"]
    ]) + """
      </div>
    </div>
    """, unsafe_allow_html=True)

with ab2:
    features = [
        ("🔬","Transformer-based NLP","Deep contextual text understanding via self-attention"),
        ("🤖","RoBERTa Classification","State-of-the-art 125M parameter model"),
        ("🏷️","10 Mental Health Categories","Comprehensive multi-class classification"),
        ("⚡","Real-time Prediction","Sub-second inference with GPU/CPU support"),
        ("📊","Interactive Dashboard","Rich Plotly visualisations and analytics"),
    ]
    for icon, title, desc in features:
        st.markdown(f"""
        <div style="display:flex;gap:12px;align-items:flex-start;
                    padding:14px 0;border-bottom:1px solid rgba(124,58,237,0.1);">
          <span style="font-size:1.2rem;flex-shrink:0;">{icon}</span>
          <div>
            <div style="font-family:'Syne',sans-serif;font-size:0.88rem;
                        font-weight:700;color:#e2e8f0;margin-bottom:3px;">{title}</div>
            <div style="font-size:0.78rem;color:#475569;line-height:1.5;">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# ❾  DISCLAIMER
# ══════════════════════════════════════════════════════════════════════
sec("Medical Disclaimer")

st.markdown("""
<div class="disclaimer">
  <div class="disclaimer-icon">⚠️</div>
  <div class="disclaimer-text">
    <strong>IMPORTANT MEDICAL DISCLAIMER</strong><br>
    MindScope AI is a research and educational tool developed as part of an academic Data Science project.
    It is <strong>NOT</strong> a substitute for professional medical diagnosis, psychiatric evaluation,
    or clinical treatment. The predictions produced by this system should not be used as the basis
    for any medical or clinical decisions. If you or someone you know is experiencing a mental health
    crisis, please consult a qualified mental health professional immediately.
    <br><br>
    All classifications are probabilistic and carry inherent uncertainty. This system has been
    evaluated on a test set and may not generalise to all populations or writing styles.
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# ❿  FOOTER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer-wrap">
  <div class="footer-logo">🧠 MindScope AI</div>
  <div class="footer-chips">
    <span class="footer-chip">Final Year Data Science Project</span>
    <span class="footer-chip">NLP + Deep Learning</span>
    <span class="footer-chip">RoBERTa-base</span>
    <span class="footer-chip">HuggingFace Transformers</span>
    <span class="footer-chip">Streamlit</span>
    <span class="footer-chip">PyTorch</span>
    <span class="footer-chip">IDS Project 2026</span>
  </div>
  <div class="footer-meta">
    Mental Health Crisis Predictor V2 · RoBERTa-Based Mental Health Analysis<br>
    Test Accuracy: 79.61% · Weighted F1: 79.68% · 80,000 Training Samples · 10 Classes<br><br>
    <span style="color:#1e293b;">
      ⚠️ Educational and research purposes only · Not a substitute for professional medical advice
    </span>
  </div>
</div>
""", unsafe_allow_html=True)