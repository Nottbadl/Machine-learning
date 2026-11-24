import os
import io
import time
import json
import base64
import datetime
import requests
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="ChiliDoc AI - Dokter Pintar Tanaman Cabai",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. CUSTOM CSS: MODERN, CLEAN, ECO-AGRITECH
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

    /* ── Global Reset & Smooth Scroll ── */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: #0f172a;
        scroll-behavior: smooth;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 4rem;
        max-width: 1140px;
    }
    /* Hide Streamlit default header decoration */
    header[data-testid="stHeader"] { background: transparent; }
    .stDeployButton { display: none; }

    /* ── HERO BANNER ── */
    .hero-banner {
        background: linear-gradient(135deg, #052e16 0%, #064e3b 35%, #047857 70%, #059669 100%);
        border-radius: 26px;
        padding: 2.5rem 3rem;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 20px 40px -12px rgba(5, 150, 105, 0.38), 0 4px 12px rgba(0, 0, 0, 0.05);
        position: relative;
        overflow: hidden;
        animation: fadeIn 0.5s ease-out;
    }
    .hero-banner::before {
        content: "🌶️";
        position: absolute;
        right: 2rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 8.5rem;
        opacity: 0.14;
        pointer-events: none;
        transition: transform 0.4s ease;
    }
    .hero-banner:hover::before {
        transform: translateY(-50%) scale(1.05) rotate(5deg);
    }
    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255,255,255,0.18);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.32);
        padding: 5px 15px;
        border-radius: 50px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.8px;
        line-height: 1.15;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        opacity: 0.92;
        max-width: 620px;
        line-height: 1.6;
        color: #d1fae5;
        margin: 0;
    }
    .hero-stats {
        display: flex;
        gap: 2.2rem;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    }
    .hero-stat-item {
        display: flex;
        flex-direction: column;
    }
    .hero-stat-num {
        font-size: 1.4rem;
        font-weight: 800;
        color: #ffffff;
    }
    .hero-stat-lbl {
        font-size: 0.76rem;
        color: rgba(255,255,255,0.72);
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    /* ── STEP GUIDE (RESPONSIVE GRID) ── */
    .steps-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin-bottom: 1.8rem;
    }
    .step-box {
        background: #ffffff;
        border: 1.5px solid #e2e8f0;
        border-radius: 18px;
        padding: 1.1rem 1.3rem;
        display: flex;
        align-items: center;
        gap: 14px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
        transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    .step-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 24px -4px rgba(5, 150, 105, 0.15);
        border-color: #34d399;
    }
    .step-num {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        color: #047857;
        font-weight: 800;
        min-width: 38px;
        height: 38px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
        border: 1.5px solid #a7f3d0;
        box-shadow: 0 2px 6px rgba(5, 150, 105, 0.15);
        flex-shrink: 0;
    }
    .step-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #0f172a;
        line-height: 1.3;
    }
    .step-desc {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 2px;
        line-height: 1.4;
    }

    /* ── MODERN TABS DESIGN (HIJAU EMERALD & SUDUT HALUS & SHADOW) ── */
    .stTabs {
        margin-top: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #141c17 !important;
        border-radius: 18px !important;
        padding: 6px !important;
        gap: 8px !important;
        border: 1.5px solid #1f2d24 !important;
        box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.4) !important;
        display: flex !important;
        border-bottom: none !important;
    }
    /* Sembunyikan garis merah bawaan Streamlit secara total */
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs div[data-baseweb="tab-highlight"],
    div[data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"],
    div[data-baseweb="tab-border"] {
        display: none !important;
        height: 0px !important;
        visibility: hidden !important;
        background: transparent !important;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border: 1.5px solid transparent !important;
        border-radius: 14px !important;
        padding: 9px 20px !important;
        color: #94a3b8 !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        outline: none !important;
        box-shadow: none !important;
        border-bottom: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(5, 150, 105, 0.12) !important;
        color: #34d399 !important;
        border-color: rgba(52, 211, 153, 0.25) !important;
        transform: translateY(-1px) !important;
    }
    .stTabs [data-baseweb="tab"]:active {
        transform: scale(0.97) !important;
    }
    /* TAB AKTIF: HIJAU EMERALD, SUDUT HALUS, SHADOW BERSINAR */
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        background-color: #059669 !important;
        color: #ffffff !important;
        border: 1.5px solid #34d399 !important;
        border-radius: 14px !important;
        padding: 9px 22px !important;
        box-shadow: 0 6px 20px rgba(5, 150, 105, 0.5), 0 2px 6px rgba(0, 0, 0, 0.2) !important;
        transform: translateY(-2px) !important;
        font-weight: 800 !important;
        border-bottom: 1.5px solid #34d399 !important;
    }
    .stTabs [aria-selected="true"] * {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    /* ── FILE UPLOADER ENHANCEMENTS ── */
    [data-testid="stFileUploader"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] label p,
    [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] {
        color: #0f172a !important;
        font-weight: 800 !important;
        font-size: 0.96rem !important;
        margin-bottom: 0.6rem !important;
    }
    [data-testid="stFileUploaderDropzone"],
    section[data-testid="stFileUploadDropzone"] {
        background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%) !important;
        border: 2px dashed #059669 !important;
        border-radius: 20px !important;
        padding: 2.2rem 1.8rem !important;
        text-align: center !important;
        box-shadow: 0 4px 20px -2px rgba(5, 150, 105, 0.08) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover,
    section[data-testid="stFileUploadDropzone"]:hover {
        border-color: #10b981 !important;
        background: linear-gradient(135deg, #ffffff 0%, #ecfdf5 100%) !important;
        box-shadow: 0 10px 28px -4px rgba(5, 150, 105, 0.18) !important;
        transform: translateY(-2px) !important;
    }
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] div,
    section[data-testid="stFileUploadDropzone"] span,
    section[data-testid="stFileUploadDropzone"] small,
    section[data-testid="stFileUploadDropzone"] div {
        color: #334155 !important;
        font-weight: 600 !important;
    }
    /* Tombol Upload bawaan Streamlit di dalam Dropzone */
    [data-testid="stFileUploaderDropzone"] button,
    section[data-testid="stFileUploadDropzone"] button {
        background: linear-gradient(135deg, #047857 0%, #059669 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.6rem !important;
        box-shadow: 0 4px 14px rgba(5, 150, 105, 0.28) !important;
        transition: all 0.25s ease !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover,
    section[data-testid="stFileUploadDropzone"] button:hover {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%) !important;
        box-shadow: 0 6px 20px rgba(5, 150, 105, 0.4) !important;
        transform: translateY(-2px) scale(1.02) !important;
    }
    [data-testid="stFileUploaderDropzone"] button:active,
    section[data-testid="stFileUploadDropzone"] button:active {
        transform: scale(0.98) !important;
    }

    /* ── SIDEBAR STYLING: SENADA & MENYATU DENGAN TEMA BERANDA ── */
    section[data-testid="stSidebar"],
    [data-testid="stSidebarContent"],
    [data-testid="stSidebar"] > div {
        background: #0e1117 !important;
        background-color: #0e1117 !important;
        border-right: 1.5px solid rgba(255, 255, 255, 0.08) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.6rem;
    }
    /* Brand Box di Sidebar */
    .sidebar-brand-card {
        background: linear-gradient(135deg, #064e3b 0%, #047857 60%, #059669 100%);
        border-radius: 20px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 25px -5px rgba(5, 150, 105, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: transform 0.25s ease;
    }
    .sidebar-brand-card:hover {
        transform: translateY(-2px);
    }
    .sidebar-brand-icon {
        font-size: 2.6rem;
        display: inline-block;
        filter: drop-shadow(0 4px 10px rgba(0,0,0,0.3));
        margin-bottom: 0.25rem;
    }
    .sidebar-brand-title {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.3px;
        color: #ffffff;
    }
    .sidebar-brand-subtitle {
        font-size: 0.78rem;
        color: #d1fae5;
        font-weight: 500;
        margin-top: 2px;
    }
    /* Card Tips di Sidebar (Senada Tema Beranda) */
    .sidebar-tips-card {
        background: #141c17 !important;
        border: 1.5px solid #065f46 !important;
        border-radius: 18px;
        padding: 1.2rem 1.25rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.25s ease;
    }
    .sidebar-tips-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(5, 150, 105, 0.2) !important;
        border-color: #10b981 !important;
    }
    .sidebar-tips-title {
        font-weight: 800;
        color: #34d399 !important;
        font-size: 0.94rem;
        margin-bottom: 0.65rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .sidebar-tips-item {
        font-size: 0.84rem;
        color: #e2e8f0 !important;
        line-height: 1.85;
    }
    .sidebar-tips-item b {
        color: #a7f3d0 !important;
    }

    /* ── EXPANDER DI SIDEBAR (SENADA BERANDA: KONTRAST TEGAS & ELEGAN) ── */
    [data-testid="stSidebar"] [data-testid="stExpander"],
    [data-testid="stSidebar"] .stExpander,
    [data-testid="stSidebar"] details[data-testid="stExpander"] {
        background: #141c17 !important;
        background-color: #141c17 !important;
        border: 1.5px solid #1f2d24 !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
        margin-bottom: 0.9rem !important;
        overflow: hidden !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"]:hover,
    [data-testid="stSidebar"] .stExpander:hover {
        border-color: #059669 !important;
        box-shadow: 0 6px 22px rgba(5, 150, 105, 0.18) !important;
        transform: translateY(-1px) !important;
    }
    /* Header Summary Expander Default di Sidebar */
    [data-testid="stSidebar"] [data-testid="stExpander"] summary,
    [data-testid="stSidebar"] .stExpander summary,
    [data-testid="stSidebar"] details summary {
        background-color: #141c17 !important;
        background: #141c17 !important;
        padding: 0.9rem 1.1rem !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
        font-size: 0.9rem !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary *,
    [data-testid="stSidebar"] .stExpander summary *,
    [data-testid="stSidebar"] details summary * {
        color: #f8fafc !important;
        font-weight: 800 !important;
    }
    /* Saat Header Expander di-Hover, di-Klik, Terbuka (Details Open) */
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:focus,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:active,
    [data-testid="stSidebar"] details[open] > summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary,
    [data-testid="stSidebar"] .stExpander details[open] > summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] [aria-expanded="true"] {
        background: linear-gradient(135deg, #064e3b 0%, #047857 100%) !important;
        color: #ffffff !important;
        border-bottom: 1px solid #059669 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover *,
    [data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary *,
    [data-testid="stSidebar"] details[open] > summary * {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    /* Icon Panah Chevron di Header Expander */
    [data-testid="stSidebar"] [data-testid="stExpander"] summary svg,
    [data-testid="stSidebar"] .stExpander summary svg,
    [data-testid="stSidebar"] details summary svg {
        color: #34d399 !important;
        fill: #34d399 !important;
    }
    /* Konten Body di dalam Expander */
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"],
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"],
    [data-testid="stSidebar"] .stExpander [data-testid="stExpanderDetails"] {
        background-color: #0f1511 !important;
        background: #0f1511 !important;
        padding: 1.1rem !important;
    }
    /* SEMUA TEKS WIDGET DI DALAM SIDEBAR & EXPANDER: PUTIH TERANG & KONTRAS */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
    [data-testid="stSidebar"] [data-baseweb="radio"] *,
    [data-testid="stSidebar"] [data-baseweb="slider"] *,
    [data-testid="stSidebar"] .stRadio *,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio label p,
    [data-testid="stSidebar"] .stRadio label span,
    [data-testid="stSidebar"] [data-testid="stSlider"] label,
    [data-testid="stSidebar"] [data-testid="stSlider"] div,
    [data-testid="stSidebar"] [data-testid="stSlider"] span,
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] p,
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] span,
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] div,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
    /* Slider styling di Sidebar */
    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
        background-color: #10b981 !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.5) !important;
    }
    /* ── CARDS & BADGES (SUDUT HALUS & MEWAH SHADOW) ── */
    .card-pro {
        background: #ffffff;
        border-radius: 26px !important;
        padding: 1.9rem 2.1rem;
        border: 1.5px solid #e8edf3;
        box-shadow: 0 16px 40px -6px rgba(0, 0, 0, 0.25), 0 4px 14px rgba(0, 0, 0, 0.1) !important;
        margin-bottom: 1.2rem;
        transition: all 0.28s ease;
        overflow: hidden !important;
    }
    .card-pro:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 48px -6px rgba(0, 0, 0, 0.32) !important;
    }
    .card-section {
        background: #141c17;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        border: 1.5px solid #1f2d24;
        margin-bottom: 0.8rem;
        color: #f1f5f9;
    }
    .diag-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 6px 16px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.82rem;
        letter-spacing: 0.3px;
    }
    .diag-healthy { background: #dcfce7; color: #14532d; border: 1.5px solid #86efac; }
    .diag-danger  { background: #fee2e2; color: #7f1d1d; border: 1.5px solid #fca5a5; }
    .diag-warning { background: #fef9c3; color: #713f12; border: 1.5px solid #fde047; }
    .diag-multi   { background: linear-gradient(135deg,#fff1f2,#fef3c7); color: #7f1d1d; border: 1.5px solid #fca5a5; }

    /* ── CONFIDENCE DISPLAY ── */
    .conf-pill {
        display: inline-flex;
        align-items: baseline;
        gap: 2px;
        background: #f0fdf4;
        border: 1.5px solid #86efac;
        border-radius: 14px;
        padding: 6px 16px;
        box-shadow: 0 2px 8px rgba(21, 128, 61, 0.08);
    }
    .conf-num  { font-size: 2rem; font-weight: 800; color: #15803d; line-height: 1; }
    .conf-unit { font-size: 1rem; font-weight: 700; color: #15803d; }
    .conf-pill-red   { border-color: #fca5a5; background: #fff1f2; box-shadow: 0 2px 8px rgba(185, 28, 28, 0.08); }
    .conf-num-red    { color: #b91c1c; }
    .conf-unit-red   { color: #b91c1c; }
    .conf-pill-amber { border-color: #fde047; background: #fefce8; box-shadow: 0 2px 8px rgba(146, 64, 14, 0.08); }
    .conf-num-amber  { color: #92400e; }
    .conf-unit-amber { color: #92400e; }

    /* ── STAT GRID & STAT CARDS (RESPONSIF MOBILE / TABLET) ── */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 0.8rem;
    }
    .stat-card {
        background: #ffffff;
        border: 1.5px solid #e2e8f0;
        border-radius: 18px;
        padding: 0.95rem 0.8rem 0.85rem;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
        transition: all 0.22s ease;
        min-width: 0;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
        border-color: #cbd5e1;
    }
    .stat-label {
        font-size: 0.74rem;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.35rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .stat-value {
        font-size: 0.95rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.3;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    @media (max-width: 768px) {
        .stat-grid {
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }
        .stat-card {
            padding: 0.75rem 0.4rem;
            border-radius: 14px;
        }
        .stat-label {
            font-size: 0.65rem;
            letter-spacing: 0px;
        }
        .stat-value {
            font-size: 0.8rem;
        }
    }
    @media (max-width: 480px) {
        .stat-grid {
            grid-template-columns: 1fr;
            gap: 8px;
        }
        .stat-card {
            padding: 0.8rem 1.2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            text-align: left;
        }
        .stat-label {
            margin-bottom: 0;
            font-size: 0.78rem;
        }
        .stat-value {
            font-size: 0.9rem;
        }
    }

    /* ── PROGRESS BARS ── */
    .prob-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.88rem;
        margin-bottom: 4px;
    }
    .prob-bar-track {
        background: #f1f5f9;
        border-radius: 50px;
        height: 9px;
        overflow: hidden;
        margin-bottom: 12px;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 50px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── ACTION STEPS ── */
    .action-step {
        background: #f8fafc;
        border-left: 4px solid #059669;
        border-radius: 0 14px 14px 0;
        padding: 0.95rem 1.25rem;
        margin-bottom: 0.7rem;
        font-size: 0.92rem;
        color: #334155;
        line-height: 1.6;
        box-shadow: 0 1px 4px rgba(0,0,0,0.02);
    }
    .action-step-warn { border-left-color: #f59e0b; background: #fffbeb; }
    .action-step-red  { border-left-color: #ef4444; background: #fff5f5; }

    /* ── SINERGI BOX ── */
    .sinergi-box {
        background: linear-gradient(135deg, #fffbeb 0%, #fff7ed 100%);
        border: 1.5px solid #fcd34d;
        border-radius: 18px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 14px rgba(245, 158, 11, 0.08);
    }

    /* ── EMPTY STATE ── */
    .empty-state {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 2px dashed #cbd5e1;
        border-radius: 26px;
        padding: 4rem 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
    }

    /* ── REJECT CARD ── */
    .reject-card {
        background: linear-gradient(135deg, #fff5f5 0%, #fff8f0 100%);
        border: 1.5px solid #fca5a5;
        border-left: 5px solid #ef4444;
        border-radius: 22px;
        padding: 1.8rem 2rem;
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.08);
    }

    /* ── BUTTONS ── */
    div.stButton > button {
        border-radius: 14px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        padding: 0.6rem 1.3rem !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1.5px solid #e2e8f0 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
    }
    div.stButton > button:hover {
        border-color: #10b981 !important;
        background: #f0fdf4 !important;
        color: #047857 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(5, 150, 105, 0.14) !important;
    }
    div.stButton > button:active {
        transform: scale(0.97) !important;
    }
    div.stDownloadButton > button {
        border-radius: 14px !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #047857 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 1.8rem !important;
        box-shadow: 0 6px 18px rgba(5, 150, 105, 0.28) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div.stDownloadButton > button:hover {
        box-shadow: 0 8px 25px rgba(5, 150, 105, 0.4) !important;
        transform: translateY(-2px) scale(1.01) !important;
    }

    /* ── ANIMATIONS ── */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ── MOBILE & TABLET RESPONSIVENESS ── */
    @media (max-width: 900px) {
        .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .hero-banner {
            padding: 1.8rem 1.6rem;
            border-radius: 20px;
        }
        .hero-banner::before {
            font-size: 5.5rem;
            right: 1rem;
            opacity: 0.08;
        }
        .hero-title {
            font-size: 2.1rem;
        }
        .hero-subtitle {
            font-size: 0.95rem;
        }
        .hero-stats {
            gap: 1.4rem;
        }
        .steps-grid {
            grid-template-columns: 1fr;
            gap: 10px;
        }
        .step-box {
            padding: 0.9rem 1.1rem;
        }
        .card-pro {
            padding: 1.4rem 1.4rem;
            border-radius: 18px;
        }

        /* ── SUSUN VERTIKAL: FOTO DI ATAS, KARTU HASIL DIAGNOSIS DI BAWAH (MOBILE & TABLET <= 900px) ── */
        div[data-testid="stHorizontalBlock"]:not([data-testid="stTabContent"] div[data-testid="stHorizontalBlock"]) {
            display: flex !important;
            flex-direction: column !important;
            gap: 20px !important;
        }
        div[data-testid="stHorizontalBlock"]:not([data-testid="stTabContent"] div[data-testid="stHorizontalBlock"]) > div[data-testid="stColumn"] {
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
            flex: 1 1 100% !important;
        }
        /* Format gambar terpusat dan proporsional di mobile & tablet */
        div[data-testid="stHorizontalBlock"]:not([data-testid="stTabContent"] div[data-testid="stHorizontalBlock"]) > div[data-testid="stColumn"]:first-child [data-testid="stImage"] {
            max-width: 440px !important;
            margin: 0 auto !important;
        }
        div[data-testid="stHorizontalBlock"]:not([data-testid="stTabContent"] div[data-testid="stHorizontalBlock"]) > div[data-testid="stColumn"]:first-child [data-testid="stImage"] img {
            border-radius: 20px !important;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45) !important;
            max-height: 380px !important;
            object-fit: cover !important;
        }
    }

    @media (max-width: 600px) {
        .hero-banner {
            padding: 1.5rem 1.2rem;
            border-radius: 18px;
        }
        .hero-title {
            font-size: 1.75rem;
        }
        .hero-stats {
            gap: 1rem;
            display: grid;
            grid-template-columns: repeat(3, 1fr);
        }
        .hero-stat-num {
            font-size: 1.15rem;
        }
        .hero-stat-lbl {
            font-size: 0.7rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto;
            scrollbar-width: none;
            padding: 4px !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 7px 14px !important;
            font-size: 0.82rem !important;
            white-space: nowrap !important;
        }
        .stat-card {
            padding: 0.75rem 0.6rem;
        }
        .stat-label {
            font-size: 0.65rem;
        }
        .stat-value {
            font-size: 0.82rem;
        }
        /* Mobile Wrap khusus Kolom Tombol Sampel di dalam Tab */
        [data-testid="stTabContent"] div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 8px !important;
        }
        [data-testid="stTabContent"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
            min-width: 46% !important;
            flex: 1 1 46% !important;
        }
    }

    /* Fallback Kontras Teks Uploader & Icon */
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stFileUploaderDropzone"] p,
    section[data-testid="stFileUploadDropzone"] small,
    section[data-testid="stFileUploadDropzone"] span {
        color: #475569 !important;
        font-weight: 500 !important;
    }
    [data-testid="stFileUploaderDropzone"]:focus-within {
        border-color: #059669 !important;
        box-shadow: 0 0 0 4px rgba(5, 150, 105, 0.18) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. BASIS PENGETAHUAN 10 KELAS LENGKAP
# ==========================================
UNIFIED_CLASSES = [
    "Antraknosa Patek",
    "Bercak Daun Cercospora",
    "Bercak Putih",
    "Bintik Daun Bakteri",
    "Defisiensi Nutrisi",
    "Hama Kutu Kebul",
    "Rebah Semai Damping Off",
    "Tanaman dan Daun Sehat",
    "Virus Keriting Daun",
    "Virus Kuning Gemini"
]

DATABASE_PENYAKIT = {
    "Antraknosa Patek": {
        "organ": "🌶️ Buah & Batang",
        "nama_latin": "Colletotrichum capsici",
        "tipe": "Jamur Patogenik",
        "keparahan": "Sangat Tinggi",
        "urgensi": "Darurat (Hari Ini)",
        "ringkasan": "Penyakit buah nomor satu di Indonesia yang membuat buah cabai berlubang cekung melingkar kehitaman dan busuk kering seperti jerami.",
        "darurat": [
            "Segera petik SELURUH buah yang berbercak hitam dan bakar/kubur jauh dari kebun.",
            "Jangan menyiram tanaman menggunakan selang dari atas agar spora jamur tidak terciprat.",
            "Semprot fungisida protektif pada buah-buah sehat di sekitarnya."
        ],
        "alami": "Semprotkan agens hayati jamur Trichoderma harzianum atau bakteri Bacillus subtilis ke media tanam dan permukaan buah, serta taburkan abu sekam/kayu di sekitar bedengan untuk menekan kelembapan.",
        "kimiawi": "Aplikasi fungisida kontak berbahan aktif Mankozeb atau Propineb (Antracol) diselingi fungisida sistemik Azoksistrobin + Difenokonazol (Amistar Top) sesuai dosis kemasan.",
        "gejala": "Bercak bulat cekung melekuk ke dalam berwarna cokelat tua hingga hitam dengan lingkaran cincin konsentris. Pada cuaca lembap keluar cairan oranye/merah muda berisi massa spora jamur.",
        "pencegahan": "Gunakan mulsa plastik perak untuk memantulkan sinar matahari, terapkan jarak tanam 60 cm, dan jangan beri pupuk Nitrogen berlebih saat pembentukan buah."
    },

    "Virus Kuning Gemini": {
        "organ": "🌿 Daun & Pucuk",
        "nama_latin": "Pepper Yellow Leaf Curl Virus (PepYLCV)",
        "tipe": "Virus Patogenik Tular Kutu Kebul",
        "keparahan": "Sangat Tinggi",
        "urgensi": "Darurat (Hari Ini)",
        "ringkasan": "Infeksi virus mematikan yang membuat daun menguning cerah kontras dengan urat daun, ukuran daun kerdil melinting kaku, dan tanaman gagal berbuah.",
        "darurat": [
            "Cabut dan musnahkan tanaman yang terserang parah sejak dini agar virus tidak menyebar ke tanaman lain.",
            "Pasang segera Perangkap Kuning Berperekat (Yellow Sticky Trap) sebanyak 40-60 lembar per hektar.",
            "Basmi kutu kebul pembawa virus dengan semprotan insektisida nabati atau kontak."
        ],
        "alami": "Semprotkan biopestisida nabati rebusan daun tembakau + ekstrak daun nimba + bawang putih setiap 3 hari sekali pada sisi bawah daun tempat kutu bersarang.",
        "kimiawi": "Kendalikan serangga vektor menggunakan insektisida sistemik berbahan aktif Abamektin, Imidakloprid, atau Tiametoksam.",
        "gejala": "Helai daun menguning terang benderang (klorotik) dengan pola urat daun tampak jelas, daun mengecil melengkung ke atas seperti mangkok, pertumbuhan terhenti kerdil.",
        "pencegahan": "Gunakan benih varietas tahan virus kuning, pasang mulsa plastik perak, dan tanam jagung sebagai tanaman pembatas (refugia) di pinggir kebun."
    },

    "Bercak Daun Cercospora": {
        "organ": "🌿 Daun Cabai",
        "nama_latin": "Cercospora capsici",
        "tipe": "Jamur Patogenik Daun",
        "keparahan": "Sedang - Tinggi",
        "urgensi": "Cepat (1-2 Hari)",
        "ringkasan": "Penyakit jamur bercak 'mata kodok' dengan pusat putih abu-abu dan lingkaran cokelat gelap yang menyebabkan daun rontok massal.",
        "darurat": [
            "Pangkas daun-daun bawah yang memperlihatkan bercak bulat.",
            "Bersihkan gulma di bawah tanaman agar sirkulasi udara lebih lancar.",
            "Hindari penyiraman daun di sore/malam hari."
        ],
        "alami": "Semprotkan ekstrak daun nimba (20-30 ml/L) atau rebusan rimpang lengkuas dan kunyit seminggu sekali.",
        "kimiawi": "Semprotkan fungisida bahan aktif Mankozeb, Klorotalonil, atau Tembaga Hidroksida sesuai dosis anjuran.",
        "gejala": "Bercak bulat berukuran 2-10 mm dengan titik tengah berwarna abu-abu/putih terang dibatasi tepi cincin berwarna cokelat tua kehitaman.",
        "pencegahan": "Gunakan mulsa perak dan rotasikan tanaman dengan tanaman selain famili terung-terungan."
    },

    "Bintik Daun Bakteri": {
        "organ": "🌿 Daun & Batang",
        "nama_latin": "Xanthomonas campestris",
        "tipe": "Bakteri Patogenik",
        "keparahan": "Tinggi (Menular Sangat Cepat)",
        "urgensi": "Darurat (Hari Ini)",
        "ringkasan": "Bintik-bintik kecil basah seperti minyak yang menular sangat cepat pada musim hujan melalui percikan air.",
        "darurat": [
            "Cabut tanaman yang terinfeksi berat dan jangan siram daun dari atas.",
            "Sanitasi tangan dan gunting pangkas dengan alkohol sebelum menyentuh tanaman sehat.",
            "Kurangi pemupukan Nitrogen."
        ],
        "alami": "Aplikasi agens hayati bakteri antagonis Pseudomonas fluorescens atau Bacillus subtilis.",
        "kimiawi": "Bakterisida / fungisida tembaga seperti Tembaga Oksiklorida, Kasugamisin, atau Streptomisin Sulfat.",
        "gejala": "Bintik kecil bersudut tampak basah berminyak yang berubah menjadi cokelat tua kehitaman dengan lingkaran kuning (halo) di sekelilingnya.",
        "pencegahan": "Gunakan benih bersertifikat dan lakukan perendaman air hangat bersuhu 50°C selama 25 menit sebelum semai."
    },

    "Virus Keriting Daun": {
        "organ": "🌿 Daun Muda & Pucuk",
        "nama_latin": "Pepper Leaf Curl Virus & Hama Thrips",
        "tipe": "Virus & Hama Penghisap",
        "keparahan": "Tinggi",
        "urgensi": "Cepat (1-2 Hari)",
        "ringkasan": "Helai daun mengerut, melipat ke atas atau ke bawah seperti sendok, kaku dan menebal, akibat hisapan serangga thrips dan virus.",
        "darurat": [
            "Periksa sisi bawah daun muda menggunakan kaca pembesar untuk melihat hama thrips/tungau.",
            "Semprotkan air bertekanan halus di bawah daun pada pagi hari.",
            "Semprotkan insektisida khusus thrips/tungau."
        ],
        "alami": "Semprot ekstrak tembakau dan bawang putih, atau larutan sabun kalium cair di bawah daun.",
        "kimiawi": "Insektisida / akarisida berbahan aktif Abamektin, Spinetoram, atau Fipronil secara bergantian.",
        "gejala": "Daun melengkung ke atas atau ke bawah, permukaan daun kaku tidak elastis, pucuk tanaman terhenti memanjang, tepi daun bergelombang keriting.",
        "pencegahan": "Pasang mulsa plastik perak dan bersihkan gulma liar di pematang sawah/kebun."
    },

    "Hama Kutu Kebul": {
        "organ": "🌿 Sisi Bawah Daun",
        "nama_latin": "Bemisia tabaci (Whitefly)",
        "tipe": "Hama Vektor Penyakit",
        "keparahan": "Tinggi",
        "urgensi": "Cepat (1-2 Hari)",
        "ringkasan": "Serangga kecil putih di bawah daun yang menghisap nutrisi dan merupakan biang kerok utama penular virus kuning cabai.",
        "darurat": [
            "Goyang tajuk tanaman perlahan; jika tampak serangga putih beterbangan, segera pasang perangkap kuning.",
            "Semprotkan insektisida kontak ke sisi bawah daun pada pagi hari saat kutu masih pasif."
        ],
        "alami": "Semprotkan minyak nimba (neem oil) atau larutan sabun cair pencuci piring encer (5 ml per liter air) untuk melarutkan lapisan lilin kutu.",
        "kimiawi": "Insektisida berbahan aktif Imidakloprid, Asetamiprid, atau Pimetrozin.",
        "gejala": "Koloni serangga putih berukuran 1 mm di bawah daun, daun tampak kusam, ada bercak hitam jelaga, dan daun menguning.",
        "pencegahan": "Gunakan kelambu kasa serangga pada pembibitan dan bersihkan rumput liar inang kutu."
    },

    "Rebah Semai Damping Off": {
        "organ": "🌱 Pangkal Batang Bibit",
        "nama_latin": "Pythium / Rhizoctonia",
        "tipe": "Jamur Tular Tanah Persemaian",
        "keparahan": "Sangat Tinggi",
        "urgensi": "Darurat (Hari Ini)",
        "ringkasan": "Bibit cabai di persemaian mendadak layu roboh karena pangkal batangnya busuk menciut akibat tanah terlalu becek.",
        "darurat": [
            "Hentikan penyiraman berlebihan segera dan jemur media semai di bawah sinar matahari.",
            "Cabut dan buang bibit yang roboh.",
            "Kocorkan fungisida protektif pada bibit-bibit di sekitarnya."
        ],
        "alami": "Taburkan arang sekam bakar di atas media semai dan berikan biofungisida Trichoderma.",
        "kimiawi": "Kocor fungisida berbahan aktif Propamokarb Hidroklorida atau Mankozeb encer.",
        "gejala": "Pangkal batang bibit yang berbatasan dengan tanah tampak menciut, berwarna cokelat gelap basah, dan bibit roboh mendadak.",
        "pencegahan": "Gunakan media semai yang steril berdrainase lancar dan jangan menyemai benih terlalu rapat."
    },

    "Bercak Putih": {
        "organ": "🌿 Daun Cabai",
        "nama_latin": "Leveillula taurica",
        "tipe": "Jamur Embun Tepung",
        "keparahan": "Sedang",
        "urgensi": "Penanganan Terjadwal",
        "ringkasan": "Muncul lapisan tepung atau bedak putih pada permukaan bawah daun yang memicu daun menguning dan menggulung ke atas.",
        "darurat": [
            "Pangkas daun bawah yang tertutup lapisan putih.",
            "Kurangi kelembapan di sekitar tanaman."
        ],
        "alami": "Semprotkan larutan baking soda (1 sendok teh per liter air + 3 tetes sabun cair) atau larutan susu sapi segar (1:9 dengan air).",
        "kimiawi": "Fungisida berbahan aktif Belerang (Sulfur), Heksakonazol, atau Triadimefon.",
        "gejala": "Lapisan serbuk tepung putih pada sisi bawah daun dengan bercak kuning klorotik pada sisi atas.",
        "pencegahan": "Pastikan tanaman mendapat sinar matahari penuh sepanjang hari."
    },

    "Defisiensi Nutrisi": {
        "organ": "🌿 Daun & Tulang Daun",
        "nama_latin": "Nutrient Deficiency (N, P, K, Mg, Ca)",
        "tipe": "Masalah Fisiologis Tanah",
        "keparahan": "Ringan - Sedang",
        "urgensi": "Perbaikan Nutrisi (2-3 Hari)",
        "ringkasan": "Kekurangan unsur hara atau pH tanah terlalu asam (<5.5) sehingga akar tidak mampu menyerap pupuk.",
        "darurat": [
            "Cek pH tanah menggunakan pH meter (kondisi ideal cabai adalah pH 6.0 - 6.8).",
            "Berikan pupuk daun cair mikro-makro lengkap melalui semprotan daun agar cepat terserap."
        ],
        "alami": "Berikan Pupuk Organik Cair (POC) dan taburkan kapur dolomit jika tanah masam.",
        "kimiawi": "Semprot pupuk daun NPK lengkap (Gandasil D) dan kocor Kalsium Nitrat atau Magnesium Sulfat.",
        "gejala": "Daun menguning di sela-sela tulang daun, daun muda pucat keputihan atau daun tua menguning kusam, tanpa bercak busuk hitam.",
        "pencegahan": "Berikan pupuk dasar organik matang dan lakukan pengecekan pH tanah sebelum tanam."
    },

    "Tanaman dan Daun Sehat": {
        "organ": "🌿 Seluruh Tanaman",
        "nama_latin": "Capsicum annuum",
        "tipe": "Kondisi Prima & Normal",
        "keparahan": "Normal (Aman)",
        "urgensi": "Perawatan Rutin",
        "ringkasan": "Tanaman cabai dalam kondisi prima, daun hijau merata elastis, buah mulus mengkilap, dan bebas dari hama penyakit.",
        "darurat": [
            "Tidak memerlukan tindakan darurat. Lanjutkan pemeliharaan rutin yang sudah berjalan."
        ],
        "alami": "Siram secara teratur pada pagi hari dan berikan pupuk kompos atau kascing setiap 3 minggu sekali.",
        "kimiawi": "Berikan pupuk daun NPK seimbang (16-16-16) secara terjadwal.",
        "gejala": "Warna daun hijau segar mengkilap merata, permukaan halus elastis tanpa noda/bercak/bintik, bunga dan buah berkembang normal.",
        "pencegahan": "Lakukan monitoring rutin seminggu 2 kali untuk deteksi dini hama dan penyakit baru."
    }
}

# ==========================================
# 4. BASIS PENGETAHUAN SINERGI & INFEKSI GANDA (MULTI-DISEASE)
# ==========================================
SINERGI_PENYAKIT = {
    frozenset(["Virus Kuning Gemini", "Hama Kutu Kebul"]): {
        "judul": "Infeksi Virus & Vektor Penular (Sinergi Berbahaya)",
        "ikon": "🦟🦠",
        "tipe": "Vektor Penular + Virus",
        "penjelasan": "Hama Kutu Kebul (Bemisia tabaci) adalah serangga vektor utama pembawa Virus Kuning Gemini. Kehadiran kedua indikasi ini menandakan daun telah terinfeksi virus sekaligus sedang dihuni serangga penyebarnya yang siap menulari tanaman lain.",
        "fokus_aksi": "Kendalikan kutu kebulnya terlebih dahulu dengan insektisida/sabun kalium agar virus tidak menyebar ke seluruh kebun, lalu berikan nutrisi mikro untuk memperkuat daya tahan tanaman.",
        "rekomendasi_darurat": [
            "Semprotkan insektisida kontak ke sisi bawah daun pada pagi hari untuk melumpuhkan kutu kebul pembawa virus.",
            "Pasang perangkap lekat kuning (yellow sticky trap) di sela tanaman cabai.",
            "Semprot pupuk daun unsur mikro (Zn + Fe + Mg) untuk memulihkan klorofil daun."
        ]
    },
    frozenset(["Virus Kuning Gemini", "Virus Keriting Daun"]): {
        "judul": "Kompleks Infeksi Virus Ganda (Yellow & Curl Complex)",
        "ikon": "🍂🌀",
        "tipe": "Kompleksitas Begomovirus",
        "penjelasan": "Tanaman mengalami infeksi kombinasi Begomovirus dan virus keriting. Daun mengalami klorosis menguning terang sekaligus helai daun berkerut kaku melipat ke atas seperti mangkuk/sendok.",
        "fokus_aksi": "Pangkas tunas/pucuk yang keriting parah, kendalikan serangga vektor (thrips & kutu kebul), dan berikan pupuk daun asam amino untuk memicu tunas baru.",
        "rekomendasi_darurat": [
            "Pangkas pucuk yang keriting kerdil agar nutrisi terfokus ke cabang produktif baru.",
            "Rotasi semprotan insektisida Abamektin dan Imidakloprid setiap 5 hari sekali.",
            "Kocor pupuk hayati dan asam humat di pangkal perakaran untuk memacu kekebalan."
        ]
    },
    frozenset(["Virus Kuning Gemini", "Defisiensi Nutrisi"]): {
        "judul": "Klorosis Ganda (Infeksi Virus & Defisiensi Hara)",
        "ikon": "🟡🧪",
        "tipe": "Patologi & Fisiologi Tanaman",
        "penjelasan": "Kerusakan pembuluh daun akibat infeksi virus memperparah penyerapan hara mikro (Magnesium, Besi, Nitrogen), menyebabkan daun menguning pucat merata dan fotosintesis anjlok drastis.",
        "fokus_aksi": "Semprotkan pupuk mikro chelate (Mg + Fe + Zn) langsung lewat daun untuk memulihkan fotosintesis dan kocor Kalsium Nitrat di perakaran.",
        "rekomendasi_darurat": [
            "Semprot pupuk mikro majemuk Magnesium Sulfat (1-2 g/L) pada pagi hari.",
            "Cek pH tanah, jika di bawah 6.0 taburkan dolomit 50 gram per tanaman.",
            "Jangan gunakan pupuk Nitrogen tunggal (Urea) berlebih karena dapat memicu keparahan virus."
        ]
    },
    frozenset(["Virus Keriting Daun", "Hama Kutu Kebul"]): {
        "judul": "Kerusakan Daun Akibat Hama Penghisap & Vektor",
        "ikon": "🌿🦟",
        "tipe": "Serangan Hama Penghisap Cairan",
        "penjelasan": "Cairan sel daun dihisap secara agresif oleh hama kutu kebul dan thrips/tungau, menyebabkan daun mengeriting melengkung kaku dan pertumbuhan pucuk terhambat.",
        "fokus_aksi": "Semprotkan insektisida/akarisida berbahan aktif abamektin berselang minyak nimba di bagian bawah daun pada pagi hari saat hama pasif.",
        "rekomendasi_darurat": [
            "Semprot bawah daun dengan semprotan air halus bertekanan untuk merontokkan telur/nimfa.",
            "Aplikasi akarisida Abamektin (0.5 ml/L) diselingi sabun kalium nabati.",
            "Bersihkan gulma liar di sekitar bedengan yang menjadi inang hama."
        ]
    },
    frozenset(["Virus Keriting Daun", "Defisiensi Nutrisi"]): {
        "judul": "Keriting Daun Disertai Malnutrisi",
        "ikon": "🌀🍂",
        "tipe": "Hama Vektor & Defisiensi Hara",
        "penjelasan": "Daun berkerut kaku akibat serangan thrips/virus sekaligus kekurangan unsur Kalsium dan Boron yang berfungsi menjaga elastisitas dinding sel daun.",
        "fokus_aksi": "Kombinasikan pengendalian hama penghisap dengan penyemprotan pupuk Kalsium-Boron cair.",
        "rekomendasi_darurat": [
            "Semprot pupuk Kalsium + Boron (Cal-Bor) dosis 2 ml/L air untuk meregangkan sel daun baru.",
            "Aplikasi insektisida thrips berbahan aktif Spinetoram atau Fipronil.",
            "Perbaiki kelembapan tanah agar penyerapan hara lancar."
        ]
    },
    frozenset(["Bercak Daun Cercospora", "Bintik Daun Bakteri"]): {
        "judul": "Infeksi Daun Ganda (Jamur Cercospora & Bakteri)",
        "ikon": "🍄🧫",
        "tipe": "Kompleks Jamur & Bakteri",
        "penjelasan": "Daun cabai terserang patogen jamur bercak daun sekaligus bakteri bercak basah Xanthomonas, umum terjadi pada musim hujan dengan kelembapan tinggi dan drainase buruk.",
        "fokus_aksi": "Gunakan bakterisida/fungisida tembaga (Tembaga Oksiklorida atau Tembaga Hidroksida) yang ampuh mengendalikan bakteri dan jamur sekaligus.",
        "rekomendasi_darurat": [
            "Pangkas daun bawah yang berbercak parah dan musnahkan dari kebun.",
            "Semprotkan Tembaga Hidroksida (Kocide) dengan dosis 1.5 - 2 g/L.",
            "Hentikan penyiraman daun di sore hari untuk mengurangi kelembapan kanopi."
        ]
    },
    frozenset(["Antraknosa Patek", "Bercak Daun Cercospora"]): {
        "judul": "Serangan Jamur Menyeluruh (Buah & Daun)",
        "ikon": "🌶️🍄",
        "tipe": "Kompleks Jamur Patogenik",
        "penjelasan": "Spora jamur telah menyebar dari kanopi daun hingga buah cabai, menyebabkan bercak mata kodok di daun dan patek busuk kering pada buah cabai.",
        "fokus_aksi": "Petik buah yang busuk, pangkas daun bawah yang berbercak, dan semprotkan fungisida sistemik berspektrum luas (Azoksistrobin + Difenokonazol).",
        "rekomendasi_darurat": [
            "Petik dan kubur seluruh buah busuk patek agar spora tidak menyebar lewat angin/air.",
            "Aplikasi fungisida sistemik Azoksistrobin + Difenokonazol (Amistar Top) dosis 1 ml/L.",
            "Kurangi pemupukan Nitrogen berlebih yang membuat dinding sel lembek."
        ]
    }
}

def analyze_synergy_and_confidence(preds, classes, is_fruit=False, lesion_ratio=0.0):
    """
    Menganalisis probabilitas model secara dinamis:
    1. Membedakan Buah Cabai Sehat vs Buah Cabai Antraknosa secara akurat (berdasarkan lesi nekrotik).
    2. Mendiagnosis secara dinamis: jika penyakit tunggal, diagnosa TUNGGAL dengan keyakinan tinggi; jika terdeteksi komplikasi nyata, barulah diagnosa KOMPLIKASI GANDA.
    """
    preds_adj = np.copy(preds)
    is_healthy_fruit = False
    
    # A. KOREKSI KHUSUS BUAH CABAI
    if is_fruit:
        sehat_idx = classes.index("Tanaman dan Daun Sehat")
        antra_idx = classes.index("Antraknosa Patek")
        
        if lesion_ratio < 10.0:
            # Buah cabai mulus/mengkilap, lesi tidak signifikan -> BUAH SEHAT!
            # (Threshold 10% memberi toleransi untuk foto real-world: bayangan, kilap, dll)
            is_healthy_fruit = True
            boost = float(preds_adj[antra_idx]) * (1.0 - lesion_ratio / 10.0)
            preds_adj[sehat_idx] = max(0.93, float(preds_adj[sehat_idx]) + boost)
            preds_adj[antra_idx] = min(0.04, float(lesion_ratio) / 100.0)
        else:
            # Buah memiliki lesi bercak busuk patek nyata -> ANTRAKNOSA PATEK!
            # Skala keyakinan patek berbanding lurus dengan lesion_ratio
            strength = min(0.98, 0.70 + (lesion_ratio - 10.0) / 100.0)
            preds_adj[antra_idx] = max(strength, float(preds_adj[antra_idx]))
            preds_adj[sehat_idx] = min(0.04, float(preds_adj[sehat_idx]))
            
    sorted_indices = np.argsort(preds_adj)[::-1]
    top1_idx = int(sorted_indices[0])
    top2_idx = int(sorted_indices[1])
    
    p1_raw = float(preds_adj[top1_idx]) * 100.0
    p2_raw = float(preds_adj[top2_idx]) * 100.0
    
    c1 = classes[top1_idx]
    c2 = classes[top2_idx]
    
    # B. LOGIKA DINAMIS TUNGGAL VS KOMPLIKASI
    # Rasio kedekatan gejala sekunder terhadap gejala primer
    ratio = p2_raw / (p1_raw + 1e-5)
    
    # Aturan Komplikasi Ganda:
    # Hanya aktif jika gejala kedua SANGAT NYATA (p2 >= 26% DAN ratio >= 0.52)
    # Khusus pasangan Kuning & Keriting: Daun kuning TIDAK selalu keriting, dan keriting TIDAK selalu kuning!
    # Jadi hanya jika keduanya sama-sama aktif kuat (p2 >= 30% dan ratio >= 0.65) baru dianggap komplikasi!
    is_co_infection = False
    if (p2_raw >= 26.0) and (ratio >= 0.52) and (c1 != "Tanaman dan Daun Sehat") and (c2 != "Tanaman dan Daun Sehat"):
        if {c1, c2} == {"Virus Kuning Gemini", "Virus Keriting Daun"}:
            if p2_raw >= 30.0 and ratio >= 0.65:
                is_co_infection = True
            else:
                is_co_infection = False # Dominan tunggal saja
        else:
            is_co_infection = True
    
    # Deteksi gejala awal
    is_early_stage = False
    if not is_co_infection and not is_healthy_fruit:
        if c1 != "Tanaman dan Daun Sehat" and c2 == "Tanaman dan Daun Sehat":
            if p2_raw >= 25.0:
                is_early_stage = True
        elif c1 == "Tanaman dan Daun Sehat" and c2 != "Tanaman dan Daun Sehat" and p2_raw >= 25.0:
            is_early_stage = True
            
    # Kalibrasi Tingkat Keyakinan (Mantap & Tegas)
    if is_healthy_fruit:
        c1_calibrated = min(98.8, max(88.0, 96.5 - float(lesion_ratio) * 1.5))
        c2_calibrated = p2_raw
        combined_confidence = c1_calibrated
    elif is_co_infection:
        c1_calibrated = min(96.0, max(78.0, p1_raw + (p2_raw * 0.4)))
        c2_calibrated = min(90.0, max(60.0, p2_raw * 1.5))
        combined_confidence = min(98.5, (p1_raw + p2_raw) * 1.1)
    elif is_early_stage:
        c1_calibrated = min(95.0, max(75.0, p1_raw + 20.0))
        c2_calibrated = p2_raw
        combined_confidence = c1_calibrated
    else:
        # Diagnosis Tunggal (Single Dominant Diagnosis)
        if p1_raw >= 60.0:
            c1_calibrated = min(99.6, 75.0 + (p1_raw - 60.0) * 0.6)
        else:
            c1_calibrated = min(94.0, max(78.0, p1_raw + 26.0))
        c2_calibrated = p2_raw
        combined_confidence = c1_calibrated
        
    pair_key = frozenset([c1, c2])
    sinergi_info = SINERGI_PENYAKIT.get(pair_key, None)
    
    return {
        "is_co_infection": is_co_infection,
        "is_early_stage": is_early_stage,
        "is_healthy_fruit": is_healthy_fruit,
        "is_fruit": is_fruit,
        "lesion_ratio": lesion_ratio,
        "c1": c1, "c1_idx": top1_idx, "p1_raw": p1_raw, "c1_calibrated": c1_calibrated,
        "c2": c2, "c2_idx": top2_idx, "p2_raw": p2_raw, "c2_calibrated": c2_calibrated,
        "combined_confidence": combined_confidence,
        "sinergi_info": sinergi_info,
        "sorted_indices": sorted_indices,
        "preds_adj": preds_adj
    }

# ==========================================
# 4. FUNGSI VALIDATOR KEABSAHAN OBJEK (GATEKEEPER)
# ==========================================
def validate_chili_input(img, cnn_model, min_conf=40.0):
    """
    Memeriksa apakah citra benar-benar merupakan daun atau buah cabai
    berdasarkan spektrum pigmen biologis (Klorofil daun & Kapsantin buah cabai)
    serta ambang batas keyakinan model CNN.
    """
    img_rgb = img.convert("RGB").resize((160, 160))
    arr_rgb = np.array(img_rgb, dtype=np.float32)
    arr_hsv = np.array(img_rgb.convert("HSV"), dtype=np.float32)
    
    h, s, v = arr_hsv[:, :, 0], arr_hsv[:, :, 1], arr_hsv[:, :, 2]
    r, g, b = arr_rgb[:, :, 0], arr_rgb[:, :, 1], arr_rgb[:, :, 2]
    
    # 1. Daun Hijau Cabai: Spektrum Klorofil (G > R & G > B, H dalam range hijau botani)
    is_leaf_green = (h >= 36) & (h <= 115) & (s >= 35) & (v >= 30) & (g > b * 1.1)
    
    # 2. Daun Kuning Klorosis (Virus Kuning / Defisiensi): R & G dominan, B rendah
    is_leaf_yellow = (h >= 22) & (h <= 36) & (s >= 45) & (v >= 40) & (g >= r * 0.78) & ((r + g) > (b * 2.2 + 20))
    
    # 3. Buah Cabai Merah: Kapsantin pekat (R sangat dominan atas G & B, saturasi tinggi)
    is_chili_red = ((h <= 12) | (h >= 242)) & (s >= 125) & (v >= 40) & (r > g * 1.85) & (r > b * 1.85) & ((r - g) > 55)
    
    leaf_pct = float(np.mean(is_leaf_green | is_leaf_yellow)) * 100.0
    total_biomarker = float(np.mean(is_leaf_green | is_leaf_yellow | is_chili_red)) * 100.0

    # Deteksi spesifik apakah objek dominan buah cabai merah
    # Gunakan mask lebih sensitif agar foto sudut lebar pun terdeteksi
    red_fruit_mask = (r > 70) & (r > g * 1.25) & (r > b * 1.25)
    red_fruit_pct = float(np.mean(red_fruit_mask)) * 100.0
    # Turunkan threshold: 0.8% cukup untuk foto wide-angle dengan cabai kecil di frame
    is_fruit = (red_fruit_pct >= 0.8) and (red_fruit_pct > leaf_pct * 0.2)
    lesion_ratio = 0.0

    if is_fruit:
        y_idx, x_idx = np.where(red_fruit_mask)
        if len(y_idx) > 0:
            ymin, ymax = max(0, y_idx.min() - 8), min(arr_rgb.shape[0], y_idx.max() + 8)
            xmin, xmax = max(0, x_idx.min() - 8), min(arr_rgb.shape[1], x_idx.max() + 8)
            crop_r = r[ymin:ymax, xmin:xmax]
            crop_g = g[ymin:ymax, xmin:xmax]
            crop_b = b[ymin:ymax, xmin:xmax]
            crop_red = (crop_r > 70) & (crop_r > crop_g * 1.25) & (crop_r > crop_b * 1.25)
            # Lesi busuk patek: piksel gelap nekrotik atau bercak oranye-kecoklatan
            crop_lesion = (
                ((crop_r < 80) & (crop_g < 70) & (crop_b < 65) & (crop_r > 15)) | # dark necrotic rot
                ((crop_r > 55) & (crop_r < 150) & (crop_g > 30) & (crop_g < 115) & (np.abs(crop_r.astype(np.int32) - crop_g.astype(np.int32)) < 50)) # sunken straw spot
            ) & (~crop_red)
            lesion_ratio = (float(np.sum(crop_lesion)) / (float(np.sum(crop_red)) + 1e-5)) * 100.0

            # --- Heuristic Tambahan: Smooth Surface Check ---
            # Buah cabai sehat: kulit merah sangat seragam, std dev HANYA pada piksel merah rendah.
            # Hanya berlaku jika lesion_ratio masih rendah/ambigu (< 15%).
            # Jika lesion_ratio sudah tinggi (>= 15%), bercak jelas terlihat → ANTRAKNOSA.
            if np.sum(crop_red) > 15:
                # Hitung std hanya dari piksel yang JELAS merah (bukan lesi)
                strict_red = crop_red & ~crop_lesion
                if np.sum(strict_red) > 10:
                    fruit_r_vals = crop_r[strict_red]
                    surface_std = float(np.std(fruit_r_vals))
                    
                    if lesion_ratio < 15.0:
                        # Rentang aman: std rendah + lesi sedikit → buah sehat
                        if surface_std < 32.0 and lesion_ratio < 8.0:
                            lesion_ratio = 0.0  # Mulus, paksa sehat
                        elif surface_std < 44.0 and lesion_ratio < 6.0:
                            lesion_ratio = min(lesion_ratio * 0.5, 3.0)  # Kurangi noise
                        # else: lesion_ratio tetap apa adanya
                    # Jika lesion_ratio >= 15%: bercak sudah jelas, tidak override

    # Preprocessing citra untuk inferensi model
    processed = ImageOps.fit(img.convert("RGB"), (224, 224), Image.Resampling.LANCZOS)
    img_arr = preprocess_input(np.expand_dims(np.array(processed, dtype=np.float32), 0))
    preds = cnn_model.predict(img_arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    conf = float(preds[top_idx]) * 100.0

    # Kriteria Penolakan Objek Bukan Daun / Buah Cabai
    is_valid = True
    alasan = ""
    
    if total_biomarker < 3.5:
        is_valid = False
        alasan = "Citra tidak terdeteksi memiliki karakteristik warna biologis daun cabai hijau/kuning maupun buah cabai merah."
    elif conf < min_conf and total_biomarker < 14.0:
        is_valid = False
        alasan = f"Objek tidak memiliki pola visual daun atau buah cabai yang dikenali (Tingkat kecocokan terlalu rendah: {conf:.1f}%)."
        
    return is_valid, alasan, preds, top_idx, conf, total_biomarker, is_fruit, lesion_ratio

# ==========================================
# 5. MEMUAT MODEL AI (10 KELAS)
# ==========================================
@st.cache_resource
def load_unified_ai():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_paths = [
        os.path.join(base_dir, "model", "model_cabai_unified.h5"),
        os.path.join(base_dir, "model", "model_cabai.h5")
    ]
    for p in model_paths:
        if os.path.exists(p):
            try:
                m = load_model(p)
                return m, os.path.basename(p)
            except Exception:
                continue
    raise FileNotFoundError("Model tidak ditemukan di folder model/")

try:
    model, model_name = load_unified_ai()
    model_ok = True
except Exception as e:
    model = None
    model_name = None
    model_ok = False
    model_err = str(e)

# ==========================================
# 5. SIDEBAR: BANTUAN & PENGATURAN CEPAT
# ==========================================
with st.sidebar:
    st.markdown("""
<div class="sidebar-brand-card">
    <div class="sidebar-brand-icon">🌶️</div>
    <div class="sidebar-brand-title">ChiliDoc AI</div>
    <div class="sidebar-brand-subtitle">Asisten Patologi Digital Cabai</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="sidebar-tips-card">
    <div class="sidebar-tips-title">💡 Tips Foto Terbaik</div>
    <div class="sidebar-tips-item">
        📏 <b>Jarak:</b> 10–20 cm dari permukaan objek<br>
        ☀️ <b>Cahaya:</b> Terang merata, hindari bayangan<br>
        🔍 <b>Fokus:</b> Tekstur bercak harus tajam<br>
        🌿 <b>Objek:</b> Pastikan daun/buah mengisi frame
    </div>
</div>
""", unsafe_allow_html=True)

    with st.expander("⚙️ Pengaturan Sensitivitas", expanded=False):
        threshold_slider = st.slider(
            "Filter Objek (%):",
            min_value=20,
            max_value=70,
            value=40,
            step=5,
            help="Ambang batas keyakinan minimal AI sebelum objek non-cabai ditolak."
        )
        engine_mode = st.radio(
            "Mesin Analisis:",
            ["⚡ Deep Learning (Lokal)", "✨ Gemini Vision AI (Cloud)"],
            index=0
        )
        gemini_api_key = ""
        if "Gemini" in engine_mode:
            gemini_api_key = st.text_input("Gemini API Key:", type="password")

    with st.expander("📊 Spesifikasi Model", expanded=False):
        st.markdown(f"""
<div style="font-size:0.84rem; color:#e2e8f0; line-height:2.1; padding:0.2rem 0;">
🧠 <b>Arsitektur:</b> MobileNetV2 Transfer Learning<br>
📂 <b>Data Latih:</b> 4.600+ Citra (Mendeley + Kaggle)<br>
✅ <b>Status Model:</b> <code style="color:#6ee7b7; background:#064e3b; border:1px solid #059669; padding:2px 8px; border-radius:6px; font-weight:700;">Aktif</code><br>
🏷️ <b>Cakupan:</b> 10 Penyakit, Hama & Nutrisi
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="text-align:center; padding:1.2rem 0 0.5rem; color:#94a3b8; font-size:0.75rem; line-height:1.6;">
    ChiliDoc AI v2.0<br>
    MobileNetV2 · TensorFlow · Streamlit
</div>
""", unsafe_allow_html=True)

# ==========================================
# 6. HEADER UTAMA (HERO BANNER)
# ==========================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-eyebrow">🌱 DOKTER DIGITAL TANAMAN CABAI</div>
    <h1 class="hero-title">ChiliDoc AI</h1>
    <p class="hero-subtitle">Deteksi penyakit daun & buah cabai secara instan menggunakan kecerdasan buatan berbasis 10 kelas penyakit lengkap — akurat, cepat, dan mudah digunakan.</p>
    <div class="hero-stats">
        <div class="hero-stat-item"><span class="hero-stat-num">10</span><span class="hero-stat-lbl">Kelas Penyakit</span></div>
        <div class="hero-stat-item"><span class="hero-stat-num">4.600+</span><span class="hero-stat-lbl">Data Latih</span></div>
        <div class="hero-stat-item"><span class="hero-stat-num">&lt;3 dtk</span><span class="hero-stat-lbl">Waktu Analisis</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# Panduan 3 Langkah Cepat (Responsif via CSS Grid)
st.markdown("""
<div class="steps-grid">
    <div class="step-box">
        <div class="step-num">1</div>
        <div>
            <div class="step-title">📸 Ambil / Unggah Foto</div>
            <div class="step-desc">Foto daun atau buah cabai</div>
        </div>
    </div>
    <div class="step-box">
        <div class="step-num">2</div>
        <div>
            <div class="step-title">🤖 Analisis Otomatis</div>
            <div class="step-desc">AI mencocokkan ciri penyakit</div>
        </div>
    </div>
    <div class="step-box">
        <div class="step-num">3</div>
        <div>
            <div class="step-title">💊 Solusi & Penanganan</div>
            <div class="step-desc">Panduan alami & kimiawi</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not model_ok:
    st.error(f"❌ Gagal memuat model AI: {model_err}")
    st.stop()

# ==========================================
# 7. PILIHAN INPUT: TABS RAMAH PENGGUNA
# ==========================================
tab_upload, tab_camera, tab_samples = st.tabs([
    "📁  Unggah Foto",
    "📷  Kamera Langsung",
    "🖼️  Sampel Penyakit"
])

image_data = None
sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")

with tab_upload:
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Pilih foto daun atau buah cabai (JPG, PNG, WEBP):",
        type=["jpg", "jpeg", "png", "webp"],
        help="Unggah foto dengan pencahayaan yang jelas untuk hasil diagnosis optimal",
        label_visibility="visible"
    )
    if uploaded is not None:
        image_data = Image.open(uploaded)

with tab_camera:
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
<div style="background:#fffbeb; border:1.5px solid #fcd34d; border-radius:12px; padding:0.75rem 1rem; margin-bottom:0.8rem; font-size:0.85rem; color:#78350f;">
💡 <b>Tips:</b> Arahkan kamera 10–20 cm dari permukaan daun/buah. Pastikan pencahayaan cukup.
</div>
""", unsafe_allow_html=True)
    camera = st.camera_input("Arahkan kamera ke bagian tanaman yang bergejala:")
    if camera is not None:
        image_data = Image.open(camera)

with tab_samples:
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
<div style="font-size:0.9rem; font-weight:600; color:#334155; margin-bottom:0.8rem;">
🧪 Belum punya foto? Pilih sampel penyakit berikut untuk melihat kemampuan AI:
</div>
""", unsafe_allow_html=True)

    sample_choice = None
    SAMPLES = [
        ("🌶️", "Antraknosa Buah",    "Antraknosa Patek"),
        ("🟡", "Virus Kuning",        "Virus Kuning Gemini"),
        ("🟤", "Bercak Cercospora",   "Bercak Daun Cercospora"),
        ("🍃", "Daun Keriting",       "Virus Keriting Daun"),
        ("🪰", "Hama Kutu Kebul",     "Hama Kutu Kebul"),
        ("🌱", "Damping Off",         "Rebah Semai Damping Off"),
        ("🦠", "Bintik Bakteri",      "Bintik Daun Bakteri"),
        ("⚪", "Bercak Putih",        "Bercak Putih"),
        ("🍂", "Defisiensi Hara",     "Defisiensi Nutrisi"),
        ("🟢", "Daun Sehat",          "Tanaman dan Daun Sehat"),
    ]

    # 2 baris × 5 kolom
    row1 = st.columns(5)
    row2 = st.columns(5)
    for i, (ikon, label, key) in enumerate(SAMPLES):
        col = row1[i] if i < 5 else row2[i - 5]
        with col:
            if st.button(f"{ikon} {label}", use_container_width=True, key=f"sample_{key}"):
                sample_choice = key

    if sample_choice:
        sample_path = os.path.join(sample_dir, f"{sample_choice}.jpg")
        if os.path.exists(sample_path):
            image_data = Image.open(sample_path)
            st.session_state["active_sample"] = sample_choice

# Pertahankan sampel aktif di session
if image_data is None and "active_sample" in st.session_state:
    sample_path = os.path.join(sample_dir, f"{st.session_state['active_sample']}.jpg")
    if os.path.exists(sample_path):
        image_data = Image.open(sample_path)

# ==========================================
# 8. ANALISIS PREDIKSI & PRESENTASI HASIL
# ==========================================
if image_data is not None:
    st.markdown("<hr style='margin:1.2rem 0;'>", unsafe_allow_html=True)
    image_rgb = image_data.convert("RGB")

    col_kiri, col_kanan = st.columns([1, 1.45], gap="large")

    with col_kiri:
        st.markdown("""
<div style="font-weight:800; font-size:1.05rem; color:#f8fafc; margin-bottom:0.6rem; display:flex; align-items:center; gap:8px;">
<span>📸</span> <span>Foto yang Dianalisis</span>
</div>
""", unsafe_allow_html=True)
        # Frame gambar
        st.image(image_rgb, use_container_width=True)
        st.markdown(f"""
<div style="background:#1e293b; border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:0.55rem 1rem; margin-top:0.5rem; display:flex; justify-content:space-between; align-items:center; box-shadow:0 2px 8px rgba(0,0,0,0.25);">
<span style="font-size:0.8rem; color:#94a3b8; font-weight:600;">📐 Dimensi Citra</span>
<span style="font-size:0.84rem; font-weight:700; color:#38bdf8; letter-spacing:0.3px;">{image_rgb.size[0]} × {image_rgb.size[1]} px</span>
</div>
""", unsafe_allow_html=True)
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Ganti / Reset Foto", use_container_width=True):
            if "active_sample" in st.session_state:
                del st.session_state["active_sample"]
            st.rerun()

    with col_kanan:
        st.markdown("""
<div style="font-weight:800; font-size:1.05rem; color:#f8fafc; margin-bottom:0.6rem; display:flex; align-items:center; gap:8px;">
<span>🤖</span> <span>Hasil Diagnosis AI</span>
</div>
""", unsafe_allow_html=True)

        # Jalankan Validasi Keabsahan Objek (Gatekeeper)
        with st.spinner("🔍 Menganalisis citra tanaman cabai..."):
            is_valid, reject_reason, preds, top_idx, confidence, total_bio, is_fruit, lesion_ratio = validate_chili_input(
                image_rgb, model, min_conf=threshold_slider
            )

        if not is_valid:
            # ========================================================
            # KASUS A: BUKAN DAUN ATAU BUAH CABAI (OBJEK DITOLAK)
            # ========================================================
            st.markdown(f"""
<div class="reject-card">
<div style="display:flex; align-items:center; gap:10px; margin-bottom:1rem;">
<span class="diag-badge diag-danger">❌ OBJEK DITOLAK</span>
<span style="font-size:0.82rem; font-weight:600; color:#dc2626; margin-left:auto;">Bukan Tanaman Cabai</span>
</div>
<div style="font-size:1.6rem; font-weight:800; color:#7f1d1d; margin-bottom:0.5rem;">Objek Tidak Dikenali</div>
<p style="font-size:0.91rem; color:#334155; line-height:1.65; margin-bottom:1rem;">
Foto yang diunggah <strong>bukan merupakan daun maupun buah tanaman cabai</strong>.
AI menolak mendiagnosis untuk mencegah kesalahan analisis dan menjaga keakuratan sistem.
</p>
<div style="background:#fff; border:1.5px solid #fecaca; border-radius:12px; padding:0.9rem 1.1rem;">
<div style="font-weight:700; color:#9a3412; font-size:0.86rem; margin-bottom:0.5rem;">💡 Tips Foto yang Benar:</div>
<ul style="margin:0; padding-left:1.2rem; font-size:0.84rem; color:#7c2d12; line-height:1.75;">
<li>Pastikan objek adalah <strong>daun atau buah tanaman cabai asli</strong></li>
<li>Jarak kamera <strong>10–20 cm</strong>, fokus tajam pada objek</li>
<li>Hindari tangan, lantai, atau benda lain mendominasi frame</li>
<li>Gunakan pencahayaan yang <strong>terang dan merata</strong></li>
</ul>
</div>
</div>
""", unsafe_allow_html=True)
            st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
            st.info("💡 Belum punya foto cabai? Coba tab **🖼️ Sampel Penyakit** untuk melihat 10 jenis penyakit secara instan.")

        else:
            # ========================================================
            # KASUS B: OBJEK VALID DAUN / BUAH CABAI
            # ========================================================
            diag = analyze_synergy_and_confidence(preds, UNIFIED_CLASSES, is_fruit=is_fruit, lesion_ratio=lesion_ratio)
            c1 = diag["c1"]
            c2 = diag["c2"]
            p1_raw = diag["p1_raw"]
            p2_raw = diag["p2_raw"]
            c1_conf = diag["c1_calibrated"]
            c2_conf = diag["c2_calibrated"]
            comb_conf = diag["combined_confidence"]
            is_multi = diag["is_co_infection"]
            is_early = diag["is_early_stage"]
            is_healthy_fruit = diag["is_healthy_fruit"]
            sinergi = diag["sinergi_info"]
            sorted_indices = diag["sorted_indices"]

            info1 = dict(DATABASE_PENYAKIT.get(c1, {}))
            info2 = dict(DATABASE_PENYAKIT.get(c2, {}))

            if is_healthy_fruit:
                info1["organ"] = "🌶️ Buah Cabai Segar"
                info1["nama_latin"] = "Capsicum annuum • Buah Mulus Mengkilap"
                info1["tipe"] = "Buah Segar & Prima"
                info1["keparahan"] = "Normal (Siap Panen)"
                info1["urgensi"] = "Pemeliharaan Rutin"
                info1["ringkasan"] = "Buah cabai tampak mulus, berwarna merah segar mengkilap merata, dan bebas dari bercak busuk antraknosa (patek). Kondisi buah dalam tingkat kematangan prima."
                info1["gejala"] = "Kulit buah mulus merata, tidak ada bercak cekung melingkar, tidak ada busuk basah/kering, tangkai buah hijau kokoh."
                info1["pencegahan"] = "Lanjutkan pemupukan Kalium (K) dan Kalsium (Ca) untuk menjaga kepadatan dinding sel buah agar tahan simpan."
                c1_display = "Buah Cabai Sehat & Prima"
            else:
                c1_display = c1

            # Status badge
            is_healthy = ("sehat" in c1.lower())
            is_deficiency = ("defisiensi" in c1.lower())

            if is_healthy_fruit:
                status_class = "diag-healthy"
                status_text = "🟢 BUAH CABAI SEHAT & PRIMA"
            elif is_multi:
                status_class = "diag-multi"
                status_text = "🧬 INFEKSI GANDA (KOMPLIKASI)"
            elif is_healthy:
                status_class = "diag-healthy"
                status_text = "🟢 TANAMAN SEHAT & PRIMA"
            elif is_deficiency:
                status_class = "diag-warning"
                status_text = "🟡 PERHATIAN NUTRISI"
            else:
                status_class = "diag-danger"
                status_text = "🔴 TERDETEKSI PENYAKIT"

            # Warna confidence pill
            if is_healthy or is_healthy_fruit:
                pill_cls = "conf-pill"; num_cls = "conf-num"; unit_cls = "conf-unit"
                display_conf = c1_conf if not is_multi else comb_conf
            elif is_deficiency:
                pill_cls = "conf-pill conf-pill-amber"; num_cls = "conf-num conf-num-amber"; unit_cls = "conf-unit conf-unit-amber"
                display_conf = c1_conf
            else:
                pill_cls = "conf-pill conf-pill-red"; num_cls = "conf-num conf-num-red"; unit_cls = "conf-unit conf-unit-red"
                display_conf = comb_conf if is_multi else c1_conf

            # ==========================================
            # CARD HERO DIAGNOSIS
            # ==========================================
            if is_multi:
                sinergi_title = sinergi.get('judul', f"Analisis Korelasi: {c1} & {c2}") if sinergi else f"Infeksi Komplikasi: {c1} & {c2}"
                sinergi_icon = sinergi.get('ikon', '⚠️') if sinergi else '⚠️'
                sinergi_desc = sinergi.get('penjelasan', f"Tanaman menunjukkan komplikasi antara {c1} dan {c2}.") if sinergi else f"Dua gejala aktif terdeteksi: {c1} ({p1_raw:.1f}%) dan {c2} ({p2_raw:.1f}%)."

                st.markdown(f"""
<div class="card-pro" style="border-radius:24px; border:1.5px solid rgba(226,232,240,0.8); border-left:6px solid #ef4444; box-shadow:0 16px 40px -8px rgba(0,0,0,0.35), 0 4px 16px rgba(239,68,68,0.15); background:linear-gradient(135deg,#fffbfb,#ffffff);">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; flex-wrap:wrap; gap:8px;">
<span class="diag-badge {status_class}">{status_text}</span>
<span style="font-size:0.82rem; font-weight:700; color:#94a3b8;">{info1.get('organ','Tanaman Cabai')}</span>
</div>
<div style="font-size:1.5rem; font-weight:800; color:#0f172a; line-height:1.35; margin-bottom:0.9rem;">
<div style="margin-bottom:4px;">1️⃣ <span style="color:#dc2626;">{c1}</span> <span style="font-size:0.85rem; font-weight:600; color:#059669; vertical-align:middle;">(Gejala Utama)</span></div>
<div>2️⃣ <span style="color:#d97706;">{c2}</span> <span style="font-size:0.85rem; font-weight:600; color:#d97706; vertical-align:middle;">(Penyakit Penyerta)</span></div>
</div>
<div class="sinergi-box">
<div style="font-weight:800; color:#92400e; font-size:0.93rem; margin-bottom:0.4rem;">{sinergi_icon} {sinergi_title}</div>
<div style="color:#78350f; font-size:0.88rem; line-height:1.6;">{sinergi_desc}</div>
</div>
<div style="display:flex; justify-content:space-between; align-items:center; background:#fafafa; border:1.5px solid #e8edf3; border-radius:14px; padding:0.9rem 1.2rem; margin-top:0.6rem;">
<div>
<div style="font-weight:700; font-size:0.88rem; color:#334155;">Kepastian Diagnosa Terpadu</div>
<div style="font-size:0.78rem; color:#94a3b8; margin-top:2px;">Utama: <strong>{c1_conf:.1f}%</strong> &nbsp;•&nbsp; Penyerta: <strong>{c2_conf:.1f}%</strong></div>
</div>
<div class="{pill_cls}">
<span class="{num_cls}">{comb_conf:.1f}</span><span class="{unit_cls}">%</span>
</div>
</div>
</div>
""", unsafe_allow_html=True)

            else:
                # Early stage notice
                early_notice = f"""
<div style="background:#f0fdf4; border:1.5px solid #86efac; border-radius:12px; padding:0.65rem 1rem; margin-bottom:0.9rem; font-size:0.86rem; color:#14532d; display:flex; align-items:flex-start; gap:8px;">
<span style="font-size:1rem; margin-top:1px;">🌱</span>
<div><strong>Indikasi Gejala Dini:</strong> Sebagian besar daun masih sehat ({p2_raw:.1f}%), namun gejala awal <strong>{c1}</strong> mulai terdeteksi. Tangani segera sebelum menyebar.</div>
</div>""" if is_early else ""

                st.markdown(f"""
<div class="card-pro" style="border-radius:24px; border:1.5px solid rgba(226,232,240,0.8); border-left:6px solid {'#22c55e' if (is_healthy or is_healthy_fruit) else '#fbbf24' if is_deficiency else '#ef4444'}; box-shadow:0 16px 40px -8px rgba(0,0,0,0.35), 0 4px 16px rgba(5,150,105,0.08); background:{'linear-gradient(135deg,#f0fdf4,#ffffff)' if (is_healthy or is_healthy_fruit) else 'linear-gradient(135deg,#fffdf0,#ffffff)' if is_deficiency else 'linear-gradient(135deg,#fffbfb,#ffffff)'};">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem; flex-wrap:wrap; gap:8px;">
<span class="diag-badge {status_class}">{status_text}</span>
<span style="font-size:0.84rem; font-weight:700; color:#64748b;">{info1.get('organ','Tanaman Cabai')}</span>
</div>
<div style="font-size:1.85rem; font-weight:800; color:#0f172a; letter-spacing:-0.5px; margin-bottom:0.25rem; line-height:1.2;">{c1_display}</div>
<div style="color:#64748b; font-size:0.86rem; font-style:italic; margin-bottom:0.9rem;">{info1.get('nama_latin','')} &nbsp;·&nbsp; {info1.get('tipe','')}</div>
<p style="font-size:0.92rem; color:#334155; line-height:1.65; margin-bottom:0.95rem;">{info1.get('ringkasan','')}</p>
{early_notice}
<div style="display:flex; justify-content:space-between; align-items:center; background:#f8fafc; border:1.5px solid #e2e8f0; border-radius:16px; padding:0.9rem 1.3rem;">
<div>
<div style="font-weight:700; font-size:0.9rem; color:#1e293b;">Tingkat Keyakinan AI</div>
<div style="font-size:0.78rem; color:#64748b; margin-top:2px;">Berdasarkan analisis visual citra</div>
</div>
<div class="{pill_cls}">
<span class="{num_cls}">{c1_conf:.1f}</span><span class="{unit_cls}">%</span>
</div>
</div>
</div>
""", unsafe_allow_html=True)

            # ── 3 Mini Stat Cards (Responsif CSS Grid: Tidak Terpotong di Mobile/Tablet) ──
            st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card">
        <div class="stat-label">⚠️ Tingkat Bahaya</div>
        <div class="stat-value">{info1.get('keparahan','-')}</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">⏰ Urgensi</div>
        <div class="stat-value">{info1.get('urgensi','-')}</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">🏷️ Kategori</div>
        <div class="stat-value">{'Komplikasi Ganda' if is_multi else info1.get('tipe','-')}</div>
    </div>
</div>
""", unsafe_allow_html=True)

            # ── Probabilitas Top 3 ──
            st.markdown("<div style='margin-top:1.4rem; font-weight:800; font-size:0.94rem; color:#ffffff;'>📊 Distribusi Kemungkinan Diagnosa</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)

            preds_display = diag["preds_adj"]
            PROB_COLORS = ["#059669", "#f59e0b", "#94a3b8"]
            for i in range(min(3, len(sorted_indices))):
                idx = sorted_indices[i]
                c_name = UNIFIED_CLASSES[idx]
                p_val = float(preds_display[idx]) * 100.0
                is_primary = (idx == diag["c1_idx"])
                is_secondary = (is_multi and idx == diag["c2_idx"])
                badge = " ⭐" if is_primary else (" ⚠️" if is_secondary else "")
                bar_color = PROB_COLORS[i]
                weight = "font-weight:800; color:#f8fafc;" if is_primary else "color:#cbd5e1;"
                st.markdown(f"""
<div style="margin-bottom:0.7rem;">
<div class="prob-row">
<span style="{weight} font-size:0.88rem;">{c_name}{badge}</span>
<span style="{weight} font-size:0.88rem;">{p_val:.1f}%</span>
</div>
<div class="prob-bar-track">
<div class="prob-bar-fill" style="width:{min(p_val,100):.1f}%; background:{bar_color};"></div>
</div>
</div>""", unsafe_allow_html=True)

    # ==========================================
    # 9. PANDUAN AKSI & SOLUSI LENGKAP
    # ==========================================
    st.markdown("<hr style='margin:1.8rem 0 1.4rem; border-color:rgba(255,255,255,0.1) !important;'>", unsafe_allow_html=True)

    if is_valid:
        # Header panduan (Teks Putih Terang & Kontras Jelas)
        judul_panduan = f"Panduan Penanganan: {c1} + {c2}" if is_multi else f"Panduan Penanganan: {c1_display}"
        st.markdown(f"""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:1.2rem;">
<div style="background:linear-gradient(135deg,#059669,#047857); border-radius:12px; padding:7px 12px; color:white; font-size:1.25rem; box-shadow:0 4px 14px rgba(5,150,105,0.4);">📋</div>
<div style="font-size:1.3rem; font-weight:800; color:#ffffff; letter-spacing:-0.4px;">{judul_panduan}</div>
</div>
""", unsafe_allow_html=True)

        tab_darurat, tab_alami, tab_kimiawi, tab_gejala, tab_cegah = st.tabs([
            "🚨 Tindakan Darurat",
            "🌿 Solusi Alami",
            "🧪 Solusi Kimiawi",
            "🔍 Ciri-Ciri Gejala",
            "🛡️ Pencegahan"
        ])

        with tab_darurat:
            st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
            if is_multi and sinergi and "rekomendasi_darurat" in sinergi:
                st.markdown("<div style='font-weight:800; font-size:0.94rem; color:#f87171; margin-bottom:0.7rem;'>⚡ Prioritas Utama (Sinergi 2 Penyakit):</div>", unsafe_allow_html=True)
                for step in sinergi["rekomendasi_darurat"]:
                    st.markdown(f'<div class="action-step action-step-red"><strong>👉</strong> {step}</div>', unsafe_allow_html=True)
                st.markdown("<div style='height:0.3rem;'></div>", unsafe_allow_html=True)

            if not is_healthy and not is_healthy_fruit:
                st.markdown(f"<div style='font-weight:700; font-size:0.92rem; color:#f1f5f9; margin-bottom:0.7rem;'>Tindakan untuk {c1}:</div>", unsafe_allow_html=True)
                darurat1 = info1.get("darurat", [])
                if darurat1:
                    for step in darurat1:
                        st.markdown(f'<div class="action-step"><strong>👉</strong> {step}</div>', unsafe_allow_html=True)
                if is_multi:
                    st.markdown(f"<div style='font-weight:700; font-size:0.92rem; color:#f1f5f9; margin:1rem 0 0.7rem;'>Tindakan tambahan untuk {c2}:</div>", unsafe_allow_html=True)
                    for step in info2.get("darurat", []):
                        st.markdown(f'<div class="action-step action-step-warn"><strong>👉</strong> {step}</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
<div style="background:#f0fdf4; border:1.5px solid #86efac; border-radius:16px; padding:1.2rem 1.4rem; display:flex; align-items:center; gap:14px; box-shadow:0 4px 16px rgba(0,0,0,0.1);">
<span style="font-size:2rem;">✅</span>
<div>
<div style="font-weight:800; color:#14532d; font-size:0.96rem;">Tanaman dalam kondisi prima!</div>
<div style="font-size:0.86rem; color:#166534; margin-top:3px;">Tidak diperlukan tindakan darurat. Lanjutkan perawatan rutin pemeliharaan.</div>
</div>
</div>
""", unsafe_allow_html=True)

        with tab_alami:
            st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-weight:700; font-size:0.92rem; color:#f1f5f9; margin-bottom:0.6rem;'>Solusi alami untuk {c1}:</div>", unsafe_allow_html=True)
            st.success(info1.get("alami", "Informasi tidak tersedia."))
            if is_multi:
                st.markdown(f"<div style='font-weight:700; font-size:0.92rem; color:#f1f5f9; margin:1rem 0 0.6rem;'>Solusi alami untuk {c2}:</div>", unsafe_allow_html=True)
                st.success(info2.get("alami", "Informasi tidak tersedia."))
            st.info("💡 **Tips:** Pengendalian organik paling efektif diaplikasikan pagi hari sebelum terik matahari.")

        with tab_kimiawi:
            st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
            if is_healthy or is_healthy_fruit:
                st.info("✅ Tanaman sehat — tidak memerlukan aplikasi pestisida kimia.")
            else:
                st.markdown(f"<div style='font-weight:700; font-size:0.92rem; color:#f1f5f9; margin-bottom:0.6rem;'>Bahan aktif untuk {c1}:</div>", unsafe_allow_html=True)
                st.warning(info1.get("kimiawi", "Informasi tidak tersedia."))
                if is_multi:
                    st.markdown(f"<div style='font-weight:700; font-size:0.92rem; color:#f1f5f9; margin:1rem 0 0.6rem;'>Bahan aktif untuk {c2}:</div>", unsafe_allow_html=True)
                    st.warning(info2.get("kimiawi", "Informasi tidak tersedia."))
                st.caption("⚠️ Gunakan bahan kimia sesuai petunjuk kemasan. Kenakan masker & sarung tangan saat aplikasi.")

        with tab_gejala:
            st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-weight:700; font-size:0.92rem; color:#f1f5f9; margin-bottom:0.6rem;'>Karakteristik gejala {c1}:</div>", unsafe_allow_html=True)
            st.info(info1.get("gejala", "Informasi tidak tersedia."))
            if is_multi:
                st.markdown(f"<div style='font-weight:700; font-size:0.92rem; color:#f1f5f9; margin:1rem 0 0.6rem;'>Karakteristik gejala {c2}:</div>", unsafe_allow_html=True)
                st.info(info2.get("gejala", "Informasi tidak tersedia."))

        with tab_cegah:
            st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-weight:700; font-size:0.92rem; color:#f1f5f9; margin-bottom:0.6rem;'>Pencegahan jangka panjang:</div>", unsafe_allow_html=True)
            st.markdown(f"""
<div class="card-section" style="font-size:0.9rem; line-height:1.7;">
{info1.get('pencegahan','Informasi tidak tersedia.')}
</div>
""", unsafe_allow_html=True)
            if is_multi:
                st.markdown(f"<div style='font-weight:700; font-size:0.92rem; color:#f1f5f9; margin:1rem 0 0.6rem;'>Pencegahan untuk {c2}:</div>", unsafe_allow_html=True)
                st.markdown(f"""
<div class="card-section" style="font-size:0.9rem; line-height:1.7;">
{info2.get('pencegahan','Informasi tidak tersedia.')}
</div>
""", unsafe_allow_html=True)

        # ── Download Report ──
        st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
        multi_text = f"\nSTATUS: Infeksi Ganda\nPenyakit Utama   : {c1} ({c1_conf:.1f}%)\nPenyakit Penyerta: {c2} ({c2_conf:.1f}%)\nKepastian Terpadu: {comb_conf:.1f}%\n" if is_multi else ""
        report_text = f"""====================================================
LAPORAN DIAGNOSIS PENYAKIT TANAMAN CABAI - CHILIDOC AI
====================================================
Waktu Analisis   : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Bagian Tanaman   : {info1.get('organ', '-')}
Diagnosis Utama  : {c1_display}
{multi_text}Nama Ilmiah      : {info1.get('nama_latin', '-')}
Kategori Masalah : {'Infeksi Ganda' if is_multi else info1.get('tipe', '-')}
Tingkat Bahaya   : {info1.get('keparahan', '-')}
Keyakinan AI     : {comb_conf:.2f}%

RINGKASAN:
{info1.get('ringkasan', '-')}

TINDAKAN DARURAT:
{chr(10).join('- ' + s for s in info1.get('darurat', []))}

PENGENDALIAN ALAMI:
{info1.get('alami', '-')}

PENGENDALIAN KIMIAWI:
{info1.get('kimiawi', '-')}

PENCEGAHAN JANGKA PANJANG:
{info1.get('pencegahan', '-')}
====================================================
Digenerate otomatis oleh ChiliDoc AI.
"""
        col_dl1, col_dl2 = st.columns([1, 2])
        with col_dl1:
            st.download_button(
                label="📥 Unduh Laporan Diagnosis (.txt)",
                data=report_text,
                file_name=f"Laporan_Cabai_{c1_display.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )

else:
    # ── Empty State ──
    st.markdown("""
<div class="empty-state">
<div style="font-size:3.8rem; margin-bottom:1rem;">🌿</div>
<div style="font-size:1.3rem; font-weight:800; color:#0f172a; margin-bottom:0.5rem;">Siap Mendiagnosis Tanaman Cabai Anda</div>
<div style="font-size:0.95rem; color:#64748b; max-width:480px; margin:0 auto; line-height:1.65;">
Pilih salah satu metode di atas — unggah foto, buka kamera, atau pilih sampel penyakit — untuk memulai analisis AI secara instan.
</div>
<div style="display:flex; gap:1.2rem; justify-content:center; margin-top:1.4rem; flex-wrap:wrap;">
<div style="background:#fff; border:1.5px solid #e2e8f0; border-radius:12px; padding:0.5rem 1.1rem; font-size:0.84rem; font-weight:600; color:#334155;">📁 Unggah Foto</div>
<div style="background:#fff; border:1.5px solid #e2e8f0; border-radius:12px; padding:0.5rem 1.1rem; font-size:0.84rem; font-weight:600; color:#334155;">📷 Kamera</div>
<div style="background:#fff; border:1.5px solid #e2e8f0; border-radius:12px; padding:0.5rem 1.1rem; font-size:0.84rem; font-weight:600; color:#334155;">🖼️ Sampel Penyakit</div>
</div>
</div>
""", unsafe_allow_html=True)

# ── Footer ──
st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:1rem 0; border-top:1px solid #e8edf3;">
<div style="font-size:1.1rem; margin-bottom:0.3rem;">🌶️</div>
<div style="font-size:0.82rem; color:#94a3b8; font-weight:500;">
ChiliDoc AI &copy; 2026 &nbsp;·&nbsp; Sistem Diagnosis Cerdas Tanaman Cabai<br>
<span style="font-size:0.76rem; color:#cbd5e1;">MobileNetV2 Transfer Learning &nbsp;·&nbsp; TensorFlow &nbsp;·&nbsp; Streamlit</span>
</div>
</div>
""", unsafe_allow_html=True)
