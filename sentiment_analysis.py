import os
import json
import re
import base64
import html
import textwrap
from io import BytesIO

import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from langdetect import detect
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit as st

from scrapper import TikTokExtractor
from llm_analyzer import OllamaCommentAnalyzer
from llm_comment_classifier import OllamaCommentClassifier
from xquik_export import build_xquik_metadata, normalize_xquik_export


nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("vader_lexicon")


class TikTokSentimentAnalyzer:
    def __init__(self):
        st.set_page_config(
            page_title="TikTok Yorum Analiz Sistemi",
            layout="wide",
            page_icon="analysis.ico"
        )

    def inject_styles(self):
        st.markdown(
            """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Space+Grotesk:wght@400;500;600&display=swap');

.stApp {
    background: radial-gradient(circle at 20% 10%, #f7f4ee 0%, #f1eee6 55%, #ece9e1 100%);
    color: #1b1c20;
}

.block-container {
    padding-top: 2.2rem;
    max-width: 1200px;
}

h1, h2, h3, h4 {
    font-family: 'Playfair Display', serif;
    color: #111318;
    letter-spacing: 0.3px;
}

p, label, div, span, input, button {
    font-family: 'Space Grotesk', sans-serif;
}

.stTextInput input {
    background: #ffffff;
    border: 1px solid #d6d2c8 !important;
    color: #1b1c20;
    height: 44px;
    border-radius: 10px;
    box-shadow: none !important;
    outline: none !important;
}

[data-testid="stTextInput"] > div {
    border: 1px solid #d6d2c8 !important;
    border-radius: 10px !important;
    background: #ffffff !important;
    box-shadow: none !important;
}

[data-testid="stTextInput"] div[data-baseweb="input"] {
    border: 1px solid #d6d2c8 !important;
    background: #ffffff !important;
    box-shadow: none !important;
}

[data-testid="stTextInput"] div[data-baseweb="input"] > div {
    border: none !important;
    background: #ffffff !important;
    box-shadow: none !important;
}

.stTextInput input:focus {
    border-color: #a9a39a !important;
    box-shadow: 0 0 0 2px rgba(169, 163, 154, 0.25) !important;
}

.stTextInput input:focus-visible {
    outline: none !important;
}

[data-testid="stTextInput"] small,
[data-testid="stTextInput"] [data-testid="inputInstructions"],
[data-testid="stTextInput"] [data-testid="InputInstructions"],
[data-testid="stTextInput"] [aria-live="polite"],
[data-testid="stTextInput"] div[role="alert"],
[data-testid="stForm"] [data-testid="inputInstructions"],
[data-testid="stForm"] [data-testid="InputInstructions"],
[data-testid="stForm"] [aria-live="polite"],
[data-testid="stForm"] div[role="alert"] {
    display: none !important;
}

.input-label {
    font-size: 14px;
    color: #111318;
    font-weight: 600;
    margin: 6px 0 6px 0;
}

[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid #e1ddd4;
    border-radius: 18px;
    padding: 18px 18px 16px 18px;
    box-shadow: 0 10px 26px rgba(23, 24, 28, 0.12);
}

[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(180deg, #f7f7f7 0%, #dfe3ea 100%);
    color: #0e1117;
    border-radius: 12px;
    height: 44px;
    width: 100%;
    font-weight: 600;
    border: 1px solid #dfe3ea;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
    font-size: 14px;
    padding: 0 16px;
    white-space: nowrap;
    text-overflow: ellipsis;
    overflow: hidden;
}

[data-testid="stFormSubmitButton"] button:hover {
    background: #ffffff;
    border-color: #ffffff;
    transform: translateY(-1px);
}

.status-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 14px;
    margin-bottom: 18px;
}

.pill {
    background: #f6f2ea;
    border: 1px solid #e2ddd4;
    color: #5b5f68;
    padding: 10px 14px;
    border-radius: 12px;
    font-size: 14px;
    text-align: center;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
}

.pill .spinner {
    width: 14px;
    height: 14px;
    border: 2px solid rgba(120, 170, 235, 0.25);
    border-top-color: #5f9ce6;
    border-radius: 50%;
    display: inline-block;
    animation: spin 0.8s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.pill.active {
    border-color: #9cc7ff;
    color: #2b5fa6;
    background: #e8f2ff;
}

.pill.success {
    border-color: #9fdeb6;
    color: #2f6f4f;
    background: #e2f5e7;
}

.card-group {
    background: #ffffff;
    border: 1px solid #e1ddd4;
    border-radius: 22px;
    padding: 26px;
    box-shadow: 0 12px 28px rgba(23, 24, 28, 0.12);
    margin-top: 8px;
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 22px;
}

.card-group-title {
    grid-column: 1 / -1;
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: #1b1c20;
    margin: 0 0 6px 0;
}

.card {
    background: rgba(15, 16, 20, 0.88);
    border: 1px solid #20222b;
    border-radius: 16px;
    padding: 18px 18px 16px 18px;
    box-shadow: 0 24px 40px rgba(0, 0, 0, 0.28);
    min-height: 300px;
    color: #f0f0f0;
    overflow: hidden;
}

.card h3 {
    margin: 4px 0 16px 0;
    font-size: 20px;
    white-space: nowrap;
    color: #f0f0f0;
}

.info-row {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 6px 0;
    border-bottom: 1px solid #1d2029;
    font-size: 13px;
}

.info-row span:first-child {
    color: #a7b0bf;
    flex: 0 0 42%;
}

.info-row span:last-child {
    color: #cfd4df;
    text-align: right;
    max-width: 58%;
    word-break: break-word;
}

.info-row:last-child {
    border-bottom: none;
}

.metric-block {
    margin-bottom: 16px;
}

.metric-label {
    font-size: 11px;
    color: #a7b0bf;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: #f0f0f0;
}

.metric-positive {
    color: #38e183;
}

.metric-negative {
    color: #ff6b6b;
}

.metric-neutral {
    color: #c3c7d1;
}

.chart-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 8px;
    min-height: 240px;
}

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 24px;
    font-weight: 700;
    margin: 28px 0 12px 0;
    color: #1b1c20;
}

.analysis-table-card {
    background: #ffffff;
    border: 1px solid #e1ddd4;
    border-radius: 22px;
    padding: 26px;
    box-shadow: 0 12px 28px rgba(23, 24, 28, 0.12);
    margin-top: 24px;
    margin-bottom: 24px;
}

.analysis-table-title {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: #1b1c20;
    margin: 0 0 18px 0;
}

.analysis-table-wrap {
    background: #101318;
    border: 1px solid #2b3038;
    border-radius: 12px;
    overflow: hidden;
}

.analysis-table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
    color: #f3f4f6;
    font-family: 'Space Grotesk', sans-serif;
}

.analysis-table th {
    background: #1a1d24;
    color: #aeb4bf;
    text-align: left;
    font-size: 14px;
    padding: 12px 10px;
    border-right: 1px solid #30343c;
    border-bottom: 1px solid #30343c;
}

.analysis-table td {
    background: #101318;
    color: #f3f4f6;
    font-size: 13px;
    padding: 10px;
    border-right: 1px solid #2b3038;
    border-bottom: 1px solid #2b3038;
    vertical-align: middle;
    word-break: break-word;
}

.analysis-table th:last-child,
.analysis-table td:last-child {
    border-right: none;
}

.analysis-table tr:last-child td {
    border-bottom: none;
}

.analysis-table .num {
    text-align: right;
    font-variant-numeric: tabular-nums;
}

.pagination-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin-top: 16px;
}

.pager-btn {
    width: 42px;
    height: 38px;
    border-radius: 999px;
    border: 1px solid #d8dde6;
    background: #f2f4f7;
    color: #1b1c20 !important;
    text-decoration: none !important;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    font-weight: 700;
    line-height: 1;
    box-shadow: 0 6px 14px rgba(23, 24, 28, 0.12);
}

.pager-btn:hover {
    background: #ffffff;
    border-color: #cfd4df;
}

.pager-btn.disabled {
    opacity: 0.55;
    color: #8a909b !important;
    background: #edf0f4;
    pointer-events: none;
    box-shadow: none;
}

.pagination-info {
    color: #5b5f68;
    font-size: 15px;
    font-weight: 600;
    text-align: center;
    min-width: 225px;
}

.llm-card {
    background: #ffffff;
    border: 1px solid #e1ddd4;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 12px 28px rgba(23, 24, 28, 0.12);
    margin-top: 14px;
    color: #1b1c20;
}

.llm-card p,
.llm-card li {
    font-size: 15px;
    line-height: 1.65;
    color: #1b1c20;
}

@media (max-width: 900px) {
    .status-row {
        grid-template-columns: 1fr;
    }

    .card-group {
        grid-template-columns: 1fr;
    }

    .analysis-table {
        min-width: 900px;
    }

    .analysis-table-wrap {
        overflow-x: auto;
    }
}

/* SAFE TABLE UI PATCH 2026-06-03 */

/* Detaylı Yorum Analizi başlığı */
.table-card-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    color: #111318 !important;
    letter-spacing: 0.3px !important;
    margin: 0 0 18px 0 !important;
}

/* Tabloyu gerçekten beyaz card / border içine al */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.table-card-anchor),
div[data-testid="stVerticalBlockBorderWrapper"]:has(.table-card-title),
div[data-testid="stVerticalBlock"]:has(.table-card-anchor),
div[data-testid="stVerticalBlock"]:has(.table-card-title),
div[data-testid="stContainer"]:has(.table-card-anchor),
div[data-testid="stContainer"]:has(.table-card-title) {
    background: #ffffff !important;
    border: 1px solid #e1ddd4 !important;
    border-radius: 22px !important;
    padding: 26px !important;
    box-shadow: 0 12px 28px rgba(23, 24, 28, 0.12) !important;
    margin-top: 24px !important;
    margin-bottom: 24px !important;
}

/* Dataframe köşeleri */
div[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Pagination yazısı */
.pagination-info {
    color: #5b5f68 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    text-align: center !important;
    padding-top: 9px !important;
    white-space: nowrap !important;
}

/* Pagination butonları siyah olmasın */
div[data-testid="stButton"] button {
    width: 42px !important;
    min-width: 42px !important;
    height: 38px !important;
    padding: 0 !important;
    border-radius: 999px !important;
    font-size: 22px !important;
    line-height: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: #f2f4f7 !important;
    color: #1b1c20 !important;
    border: 1px solid #d8dde6 !important;
    box-shadow: 0 6px 14px rgba(23, 24, 28, 0.12) !important;
}

/* Hover */
div[data-testid="stButton"] button:hover {
    background: #ffffff !important;
    color: #1b1c20 !important;
    border-color: #cfd4df !important;
}

/* Disabled pagination butonu boş/siyah görünmesin */
div[data-testid="stButton"] button:disabled,
div[data-testid="stButton"] button[disabled] {
    opacity: 0.55 !important;
    color: #8a909b !important;
    background: #edf0f4 !important;
    border-color: #d7dbe2 !important;
    box-shadow: none !important;
}



/* SAFE LLM REPORT CARD AND HEADER PATCH */

/* En üst Streamlit header alanı */
header[data-testid="stHeader"],
[data-testid="stHeader"] {
    background: #3b82f6 !important;
}

/* Sağ üst toolbar alanı varsa onun da koyu görünmesini engelle */
[data-testid="stToolbar"] {
    background: transparent !important;
}

/* LLM raporunu beyaz card içine al */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.llm-report-card-marker),
div[data-testid="stVerticalBlock"]:has(.llm-report-card-marker),
div[data-testid="stContainer"]:has(.llm-report-card-marker) {
    background: #ffffff !important;
    border: 1px solid #e1ddd4 !important;
    border-radius: 22px !important;
    padding: 28px !important;
    box-shadow: 0 12px 28px rgba(23, 24, 28, 0.12) !important;
    margin-top: 18px !important;
    margin-bottom: 28px !important;
}

/* LLM rapor iç metni */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.llm-report-card-marker) p,
div[data-testid="stVerticalBlock"]:has(.llm-report-card-marker) p,
div[data-testid="stContainer"]:has(.llm-report-card-marker) p {
    font-size: 17px !important;
    line-height: 1.75 !important;
    color: #1b1c20 !important;
}

/* LLM rapor başlıkları */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.llm-report-card-marker) h1,
div[data-testid="stVerticalBlockBorderWrapper"]:has(.llm-report-card-marker) h2,
div[data-testid="stVerticalBlockBorderWrapper"]:has(.llm-report-card-marker) h3,
div[data-testid="stVerticalBlock"]:has(.llm-report-card-marker) h1,
div[data-testid="stVerticalBlock"]:has(.llm-report-card-marker) h2,
div[data-testid="stVerticalBlock"]:has(.llm-report-card-marker) h3,
div[data-testid="stContainer"]:has(.llm-report-card-marker) h1,
div[data-testid="stContainer"]:has(.llm-report-card-marker) h2,
div[data-testid="stContainer"]:has(.llm-report-card-marker) h3 {
    color: #111318 !important;
}


/* SAFE CARD TITLE COLOR FIX */

/* Analiz Sonucu içindeki koyu kart başlıklarını tekrar açık renge al */
.card h1,
.card h2,
.card h3,
.card h4 {
    color: #f0f0f0 !important;
}

/* Kart içindeki label ve metinleri de koyu zeminde okunur tut */
.card .metric-label,
.card .info-row span:first-child {
    color: #a7b0bf !important;
}

.card .info-row span:last-child {
    color: #cfd4df !important;
}

.card .metric-value {
    color: #f0f0f0 !important;
}

.card .metric-positive {
    color: #38e183 !important;
}

.card .metric-negative {
    color: #ff6b6b !important;
}

.card .metric-neutral {
    color: #c3c7d1 !important;
}


/* SAFE DARK CARD TITLE FIX */

.dark-card-title {
    color: #f0f0f0 !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    line-height: 1.25 !important;
    margin: 4px 0 16px 0 !important;
    letter-spacing: 0.2px !important;
}

.card .dark-card-title {
    color: #f0f0f0 !important;
}


/* SAFE LLM REPORT FONT SIZE FIX */

/* LLM rapor kartı içindeki ana rapor başlığı */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.llm-report-card-marker) h1,
div[data-testid="stVerticalBlock"]:has(.llm-report-card-marker) h1,
div[data-testid="stContainer"]:has(.llm-report-card-marker) h1 {
    font-size: 30px !important;
    line-height: 1.25 !important;
    margin-top: 18px !important;
    margin-bottom: 20px !important;
}

/* LLM rapor kartı içindeki bölüm başlıkları */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.llm-report-card-marker) h2,
div[data-testid="stVerticalBlock"]:has(.llm-report-card-marker) h2,
div[data-testid="stContainer"]:has(.llm-report-card-marker) h2 {
    font-size: 24px !important;
    line-height: 1.3 !important;
    margin-top: 26px !important;
    margin-bottom: 12px !important;
}

/* LLM rapor kartı içindeki daha küçük başlıklar */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.llm-report-card-marker) h3,
div[data-testid="stVerticalBlock"]:has(.llm-report-card-marker) h3,
div[data-testid="stContainer"]:has(.llm-report-card-marker) h3 {
    font-size: 20px !important;
    line-height: 1.3 !important;
    margin-top: 22px !important;
    margin-bottom: 10px !important;
}

/* LLM rapor paragraf metni */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.llm-report-card-marker) p,
div[data-testid="stVerticalBlock"]:has(.llm-report-card-marker) p,
div[data-testid="stContainer"]:has(.llm-report-card-marker) p {
    font-size: 15.5px !important;
    line-height: 1.65 !important;
    margin-bottom: 12px !important;
}

/* LLM rapor liste maddeleri */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.llm-report-card-marker) li,
div[data-testid="stVerticalBlock"]:has(.llm-report-card-marker) li,
div[data-testid="stContainer"]:has(.llm-report-card-marker) li {
    font-size: 15.5px !important;
    line-height: 1.6 !important;
    margin-bottom: 6px !important;
}

/* En üstteki LLM Genel Analiz Raporu başlığı */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.llm-report-card-marker) .section-title,
div[data-testid="stVerticalBlock"]:has(.llm-report-card-marker) .section-title,
div[data-testid="stContainer"]:has(.llm-report-card-marker) .section-title {
    font-size: 24px !important;
    line-height: 1.3 !important;
    margin-bottom: 18px !important;
}


/* SAFE STREAMLIT HEADER ACTIONS HIDE */

/* Streamlit sağ üst Deploy ve üç nokta alanını görünmez yap */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
}

/* Header mavi kalsın */
header[data-testid="stHeader"],
[data-testid="stHeader"] {
    background: #3b82f6 !important;
}

/* Sağ üst buton varyasyonlarını da hedefle */
button[title="Deploy"],
button[aria-label="Deploy"],
button[aria-label="More options"],
button[title="More options"],
button[kind="header"] {
    display: none !important;
    visibility: hidden !important;
}


/* SAFE HEADER TITLE PATCH */

/* Üst mavi bar yüksekliği */
header[data-testid="stHeader"],
[data-testid="stHeader"] {
    background: #3b82f6 !important;
    height: 64px !important;
}

/* Üst bara ortalanmış başlık ekle */
header[data-testid="stHeader"]::before,
[data-testid="stHeader"]::before {
    content: "";
    position: fixed;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    letter-spacing: 0.2px !important;
    z-index: 999999;
    pointer-events: none;
    white-space: nowrap;
}

/* İçerikte eski başlığın yer kaplamasını engelle */
.app-title-placeholder {
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* İçeriği üst bardan biraz aşağı al */
.block-container {
    padding-top: 5rem !important;
}


/* SAFE REAL TIKTOK LOGO HEADER PATCH */

.custom-top-header-title {
    position: fixed;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 999999;
    pointer-events: none;
    white-space: nowrap;
}

.custom-top-header-title span {
    color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    letter-spacing: 0.2px !important;
}


/* FORCE SINGLE ANALYSIS RESULT CARD */

.single-analysis-card {
    background: #ffffff !important;
    border: 1px solid #e1ddd4 !important;
    border-radius: 24px !important;
    padding: 28px !important;
    box-shadow: 0 12px 28px rgba(23, 24, 28, 0.12) !important;
    margin-top: 12px !important;
    margin-bottom: 24px !important;
}

.single-analysis-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #111318 !important;
    margin-bottom: 22px !important;
}

.single-layout {
    display: grid !important;
    grid-template-columns: 1.45fr 0.85fr 0.9fr !important;
    gap: 18px !important;
    align-items: stretch !important;
}

.single-left,
.single-middle,
.single-right {
    background: #2c2d31 !important;
    border: 1px solid #20222b !important;
    border-radius: 18px !important;
    padding: 20px !important;
    color: #f0f0f0 !important;
    box-shadow: 0 18px 30px rgba(0, 0, 0, 0.22) !important;
}

.single-panel-title {
    color: #f0f0f0 !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 21px !important;
    font-weight: 700 !important;
    margin-bottom: 18px !important;
}

.single-meta-grid {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 10px 14px !important;
}

.single-meta-item {
    border-bottom: 1px solid #1d2029 !important;
    padding-bottom: 8px !important;
    min-width: 0 !important;
}

.single-meta-item.wide {
    grid-column: 1 / -1 !important;
}

.single-meta-item span {
    display: block !important;
    color: #a7b0bf !important;
    font-size: 12px !important;
    margin-bottom: 4px !important;
}

.single-meta-item strong {
    display: block !important;
    color: #d7dce6 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    line-height: 1.35 !important;
    word-break: break-word !important;
}

.single-description-box {
    margin-top: 16px !important;
    border-top: 1px solid #1d2029 !important;
    padding-top: 14px !important;
}

.single-description-box span {
    color: #a7b0bf !important;
    font-size: 12px !important;
    display: block !important;
    margin-bottom: 6px !important;
}

.single-description-box p {
    color: #d7dce6 !important;
    font-size: 14px !important;
    line-height: 1.55 !important;
    margin: 0 !important;
    max-height: 130px !important;
    overflow-y: auto !important;
    padding-right: 6px !important;
    text-align: left !important;
}

.single-metric-grid {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 12px !important;
}

.single-metric-box {
    background: rgba(15, 16, 20, 0.55) !important;
    border: 1px solid #20222b !important;
    border-radius: 14px !important;
    padding: 16px !important;
}

.single-metric-box span {
    display: block !important;
    color: #a7b0bf !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.7px !important;
    margin-bottom: 8px !important;
}

.single-metric-box strong {
    display: block !important;
    color: #f0f0f0 !important;
    font-size: 34px !important;
    font-weight: 800 !important;
    line-height: 1 !important;
}

.single-metric-box.positive strong {
    color: #38e183 !important;
}

.single-metric-box.negative strong {
    color: #ff6b6b !important;
}

.single-metric-box.neutral strong {
    color: #c3c7d1 !important;
}

.single-chart-box {
    min-height: 250px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.single-chart-box img {
    max-width: 100% !important;
    height: auto !important;
}

@media (max-width: 1000px) {
    .single-layout {
        grid-template-columns: 1fr !important;
    }

    .single-meta-grid {
        grid-template-columns: 1fr !important;
    }
}


/* FINAL SINGLE RESULT CARD OVERRIDE */

/* Analiz Sonucu artık tek koyu kart */
.single-analysis-card {
    background: #2c2d31 !important;
    border: 1px solid #20222b !important;
    border-radius: 24px !important;
    padding: 28px !important;
    box-shadow: 0 18px 34px rgba(0, 0, 0, 0.22) !important;
    margin-top: 12px !important;
    margin-bottom: 24px !important;
}

/* Analiz Sonucu başlığı da aynı kartın içinde beyaz */
.single-analysis-title {
    color: #f0f0f0 !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    margin-bottom: 24px !important;
}

/* İçteki 3 ayrı kart görünümünü kaldır */
.single-left,
.single-middle,
.single-right {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Tek kart içinde 3 bölüm gibi dursun */
.single-layout {
    display: grid !important;
    grid-template-columns: 1.45fr 0.85fr 0.9fr !important;
    gap: 32px !important;
    align-items: start !important;
}

/* Bölüm başlıkları */
.single-panel-title {
    color: #f0f0f0 !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    margin-bottom: 18px !important;
}

/* Video bilgileri */
.single-meta-grid {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 12px 18px !important;
}

.single-meta-item {
    border-bottom: 1px solid #1d2029 !important;
    padding-bottom: 10px !important;
    min-width: 0 !important;
}

.single-meta-item.wide {
    grid-column: 1 / -1 !important;
}

.single-meta-item span {
    color: #a7b0bf !important;
    font-size: 12px !important;
    display: block !important;
    margin-bottom: 4px !important;
}

.single-meta-item strong {
    color: #d7dce6 !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    display: block !important;
    line-height: 1.35 !important;
    word-break: break-word !important;
}

/* Açıklama tek kart içinde okunur olsun */
.single-description-box {
    margin-top: 18px !important;
    border-top: 1px solid #1d2029 !important;
    padding-top: 14px !important;
}

.single-description-box span {
    color: #a7b0bf !important;
    font-size: 12px !important;
    display: block !important;
    margin-bottom: 8px !important;
}

.single-description-box p {
    color: #d7dce6 !important;
    font-size: 14px !important;
    line-height: 1.55 !important;
    margin: 0 !important;
    max-height: 120px !important;
    overflow-y: auto !important;
    padding-right: 6px !important;
    text-align: left !important;
}

/* Analiz özeti kutuları tek kart içinde küçük kutular olarak kalsın */
.single-metric-grid {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 12px !important;
}

.single-metric-box {
    background: rgba(15, 16, 20, 0.55) !important;
    border: 1px solid #20222b !important;
    border-radius: 14px !important;
    padding: 16px !important;
}

.single-metric-box span {
    color: #a7b0bf !important;
    display: block !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.7px !important;
    margin-bottom: 8px !important;
}

.single-metric-box strong {
    color: #f0f0f0 !important;
    display: block !important;
    font-size: 34px !important;
    font-weight: 800 !important;
    line-height: 1 !important;
}

.single-metric-box.positive strong {
    color: #38e183 !important;
}

.single-metric-box.negative strong {
    color: #ff6b6b !important;
}

.single-metric-box.neutral strong {
    color: #c3c7d1 !important;
}

/* Grafik */
.single-chart-box {
    min-height: 250px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.single-chart-box img {
    max-width: 100% !important;
    height: auto !important;
}

@media (max-width: 1000px) {
    .single-layout {
        grid-template-columns: 1fr !important;
    }

    .single-meta-grid {
        grid-template-columns: 1fr !important;
    }
}


/* FINAL TOP VIDEO BOTTOM SUMMARY LAYOUT */

.final-analysis-card {
    background: #2c2d31 !important;
    border: 1px solid #20222b !important;
    border-radius: 24px !important;
    padding: 30px !important;
    box-shadow: 0 18px 34px rgba(0, 0, 0, 0.22) !important;
    margin-top: 12px !important;
    margin-bottom: 24px !important;
}

.final-card-title {
    color: #f0f0f0 !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 30px !important;
    font-weight: 700 !important;
    margin-bottom: 26px !important;
}

.final-section-title {
    color: #f0f0f0 !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 23px !important;
    font-weight: 700 !important;
    margin-bottom: 18px !important;
}

.final-video-section {
    width: 100% !important;
    padding-bottom: 24px !important;
    border-bottom: 1px solid #1d2029 !important;
    margin-bottom: 26px !important;
}

.final-info-grid {
    display: grid !important;
    grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
    gap: 14px 20px !important;
}

.final-info-item {
    border-bottom: 1px solid #1d2029 !important;
    padding-bottom: 10px !important;
    min-width: 0 !important;
}

.final-info-item.final-wide {
    grid-column: span 2 !important;
}

.final-info-item span {
    display: block !important;
    color: #a7b0bf !important;
    font-size: 13px !important;
    margin-bottom: 5px !important;
}

.final-info-item strong {
    display: block !important;
    color: #e3e7ef !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    line-height: 1.35 !important;
    word-break: break-word !important;
}

.final-description-box {
    margin-top: 20px !important;
    background: rgba(15, 16, 20, 0.42) !important;
    border: 1px solid #20222b !important;
    border-radius: 16px !important;
    padding: 16px !important;
}

.final-description-box span {
    display: block !important;
    color: #a7b0bf !important;
    font-size: 13px !important;
    margin-bottom: 8px !important;
}

.final-description-box p {
    color: #e3e7ef !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
    margin: 0 !important;
    max-height: 95px !important;
    overflow-y: auto !important;
    padding-right: 8px !important;
    text-align: left !important;
    opacity: 1 !important;
}

.final-bottom-grid {
    display: grid !important;
    grid-template-columns: 0.9fr 1.1fr !important;
    gap: 28px !important;
    align-items: start !important;
}

.final-summary-section,
.final-chart-section {
    min-width: 0 !important;
}

.final-metric-grid {
    display: grid !important;
    grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
    gap: 14px !important;
}

.final-metric-box {
    background: rgba(15, 16, 20, 0.55) !important;
    border: 1px solid #20222b !important;
    border-radius: 16px !important;
    padding: 16px !important;
    min-height: 120px !important;
}

.final-metric-box span {
    display: block !important;
    color: #a7b0bf !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.7px !important;
    margin-bottom: 10px !important;
}

.final-metric-box strong {
    display: block !important;
    color: #f0f0f0 !important;
    font-size: 36px !important;
    font-weight: 800 !important;
    line-height: 1 !important;
}

.final-metric-box.positive strong {
    color: #38e183 !important;
}

.final-metric-box.negative strong {
    color: #ff6b6b !important;
}

.final-metric-box.neutral strong {
    color: #c3c7d1 !important;
}

.final-chart-box {
    background: rgba(15, 16, 20, 0.25) !important;
    border: 1px solid #20222b !important;
    border-radius: 16px !important;
    min-height: 260px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 14px !important;
}

.final-chart-box img {
    max-width: 100% !important;
    height: auto !important;
}

@media (max-width: 1100px) {
    .final-info-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    }

    .final-bottom-grid {
        grid-template-columns: 1fr !important;
    }

    .final-metric-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    }
}

@media (max-width: 700px) {
    .final-info-grid {
        grid-template-columns: 1fr !important;
    }

    .final-info-item.final-wide {
        grid-column: span 1 !important;
    }

    .final-metric-grid {
        grid-template-columns: 1fr !important;
    }
}


/* FINAL DESCRIPTION AND METRIC FIX */

/* Açıklama metni kesin okunur olsun */
.final-analysis-card .final-description-box p,
.final-description-box p {
    color: #e3e7ef !important;
    opacity: 1 !important;
    font-weight: 500 !important;
}

/* Açıklama kutusunun arka planını biraz aç */
.final-analysis-card .final-description-box,
.final-description-box {
    background: rgba(15, 16, 20, 0.28) !important;
}

/* Analiz özeti: üstte 2, altta 2 */
.final-analysis-card .final-metric-grid,
.final-metric-grid {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 16px !important;
    width: 100% !important;
}

/* Kutular alanı doldursun */
.final-analysis-card .final-metric-box,
.final-metric-box {
    min-height: 150px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
}

/* Analiz özeti bölümü boş kalmasın */
.final-summary-section {
    width: 100% !important;
}

/* Büyük ekranda da 2 kolon kalsın */
@media (min-width: 1001px) {
    .final-analysis-card .final-metric-grid,
    .final-metric-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    }
}


/* FINAL COMPACT SUMMARY AND CHART FIX */

/* Alt bölüm çok büyük durmasın */
.final-bottom-grid {
    grid-template-columns: 0.85fr 1fr !important;
    gap: 22px !important;
    align-items: start !important;
}

/* Analiz Özeti başlığı biraz küçülsün */
.final-summary-section .final-section-title,
.final-chart-section .final-section-title {
    font-size: 21px !important;
    margin-bottom: 14px !important;
}

/* Analiz özeti kutuları daha kompakt */
.final-metric-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 12px !important;
}

.final-metric-box {
    min-height: 105px !important;
    padding: 14px !important;
    border-radius: 14px !important;
}

.final-metric-box span {
    font-size: 11px !important;
    letter-spacing: 0.6px !important;
    margin-bottom: 8px !important;
}

.final-metric-box strong {
    font-size: 30px !important;
}

/* Grafik kutusu küçülsün */
.final-chart-box {
    min-height: 220px !important;
    max-height: 280px !important;
    padding: 10px !important;
    border-radius: 14px !important;
}

/* Grafik görselinin aşırı büyümesini engelle */
.final-chart-box img {
    max-width: 78% !important;
    max-height: 240px !important;
    width: auto !important;
    height: auto !important;
}

/* Büyük ekranda kartın alt kısmı daha dengeli dursun */
@media (min-width: 1001px) {
    .final-bottom-grid {
        grid-template-columns: 0.78fr 1fr !important;
    }

    .final-metric-box {
        min-height: 105px !important;
    }

    .final-chart-box {
        min-height: 240px !important;
    }
}

/* Orta/küçük ekranda bozulmasın */
@media (max-width: 1100px) {
    .final-bottom-grid {
        grid-template-columns: 1fr !important;
    }

    .final-chart-box img {
        max-width: 70% !important;
    }
}


/* FINAL EQUAL SUMMARY CHART HEIGHT FIX */

/* Alt bölümde sol analiz özeti ve sağ grafik aynı boyda dursun */
.final-bottom-grid {
    display: grid !important;
    grid-template-columns: 0.85fr 1fr !important;
    gap: 24px !important;
    align-items: stretch !important;
}

/* Sol ve sağ ana blokların yüksekliğini eşitle */
.final-summary-section,
.final-chart-section {
    height: 360px !important;
    min-height: 360px !important;
    max-height: 360px !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Başlıklar aynı hizada dursun */
.final-summary-section .final-section-title,
.final-chart-section .final-section-title {
    font-size: 22px !important;
    line-height: 1.25 !important;
    margin: 0 0 16px 0 !important;
    min-height: 58px !important;
    display: flex !important;
    align-items: flex-start !important;
}

/* Soldaki 4 metrik kart 2x2 ve daha küçük */
.final-metric-grid {
    flex: 1 !important;
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    grid-template-rows: repeat(2, 1fr) !important;
    gap: 14px !important;
    height: 286px !important;
}

/* Küçük kartların boyu küçülsün */
.final-metric-box {
    min-height: 0 !important;
    height: 136px !important;
    padding: 14px 16px !important;
    border-radius: 14px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
}

/* Metrik label ve sayı daha dengeli */
.final-metric-box span {
    font-size: 11px !important;
    letter-spacing: 0.6px !important;
    margin-bottom: 10px !important;
}

.final-metric-box strong {
    font-size: 32px !important;
    line-height: 1 !important;
}

/* Sağ grafik kutusu sol tarafla aynı toplam yüksekliğe otursun */
.final-chart-box {
    flex: 1 !important;
    height: 286px !important;
    min-height: 286px !important;
    max-height: 286px !important;
    padding: 12px !important;
    border-radius: 14px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Grafik kendi kutusu içinde büyüyüp taşmasın */
.final-chart-box img {
    max-width: 72% !important;
    max-height: 250px !important;
    width: auto !important;
    height: auto !important;
}

/* Mobilde bozulmasın */
@media (max-width: 1100px) {
    .final-bottom-grid {
        grid-template-columns: 1fr !important;
    }

    .final-summary-section,
    .final-chart-section {
        height: auto !important;
        min-height: auto !important;
        max-height: none !important;
    }

    .final-metric-grid {
        height: auto !important;
    }

    .final-metric-box {
        height: 120px !important;
    }

    .final-chart-box {
        height: 260px !important;
        min-height: 260px !important;
        max-height: 260px !important;
    }
}


/* ABSOLUTE FINAL SMALL METRIC CARDS FIX */

/* Ana kartın altına iç boşluk ver, yapışmayı kes */
.final-analysis-card {
    padding-bottom: 46px !important;
}

/* Alt bölüm çok aşağıya yapışmasın ve içeride dengeli dursun */
.final-bottom-grid {
    grid-template-columns: 0.75fr 1fr !important;
    gap: 28px !important;
    align-items: start !important;
    margin-bottom: 18px !important;
}

/* Sol analiz bölümü sabit dev yükseklik almasın */
.final-summary-section {
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
    display: block !important;
}

/* Sağ grafik bölümü de dev blok gibi davranmasın */
.final-chart-section {
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
    display: block !important;
}

/* Soldaki 4 kutunun toplam alanını küçült */
.final-metric-grid {
    display: grid !important;
    grid-template-columns: repeat(2, 150px) !important;
    grid-template-rows: repeat(2, 112px) !important;
    gap: 14px !important;
    height: auto !important;
    width: max-content !important;
    align-items: start !important;
}

/* Her bir metrik kartı gerçekten küçük yap */
.final-metric-box {
    width: 150px !important;
    height: 112px !important;
    min-height: 112px !important;
    max-height: 112px !important;
    padding: 14px !important;
    border-radius: 14px !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
}

/* Label küçük */
.final-metric-box span {
    font-size: 10.5px !important;
    line-height: 1.25 !important;
    letter-spacing: 0.5px !important;
    margin-bottom: 8px !important;
}

/* Sayılar küçük */
.final-metric-box strong {
    font-size: 28px !important;
    line-height: 1 !important;
}

/* Grafik kutusu soldaki alanla daha dengeli olsun */
.final-chart-box {
    height: 252px !important;
    min-height: 252px !important;
    max-height: 252px !important;
    padding: 10px !important;
    border-radius: 14px !important;
}

/* Grafik görseli fazla büyümesin */
.final-chart-box img {
    max-width: 62% !important;
    max-height: 215px !important;
    width: auto !important;
    height: auto !important;
}

/* Başlıkların altında fazla boşluk olmasın */
.final-summary-section .final-section-title,
.final-chart-section .final-section-title {
    min-height: 0 !important;
    margin-bottom: 14px !important;
}

/* Mobilde yine düzgün aksın */
@media (max-width: 1100px) {
    .final-bottom-grid {
        grid-template-columns: 1fr !important;
    }

    .final-metric-grid {
        grid-template-columns: repeat(2, 150px) !important;
        grid-template-rows: repeat(2, 112px) !important;
        width: max-content !important;
    }

    .final-chart-box {
        height: 240px !important;
        min-height: 240px !important;
        max-height: 240px !important;
    }
}


/* ABSOLUTE INLINE METRIC SAFETY OVERRIDE */

.final-analysis-card {
    padding-bottom: 52px !important;
}

.final-metric-box span {
    font-size: 10.5px !important;
    line-height: 1.25 !important;
    margin-bottom: 8px !important;
}

.final-metric-box strong {
    font-size: 27px !important;
    line-height: 1 !important;
}

.final-chart-box img {
    max-width: 54% !important;
    max-height: 190px !important;
    width: auto !important;
    height: auto !important;
}


/* SAFE REMOVE HEADER LOGO AND RENAME TITLE */

.custom-top-header-title img {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
}

.custom-top-header-title span {
    color: #ffffff !important;
}


/* FINAL HEADER TITLE ONLY FIX */

.custom-top-header-title {
    position: fixed !important;
    top: 0 !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    height: 64px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    z-index: 999999 !important;
    pointer-events: none !important;
    white-space: nowrap !important;
}

.custom-top-header-title img {
    display: none !important;
}

.custom-top-header-title span {
    color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    letter-spacing: 0.2px !important;
}

</style>
            """,
            unsafe_allow_html=True
        )

    def render_app_header(self):
        st.markdown(
            """
            <div class="custom-top-header-title">
                <span>Campaign Performance Analyzer</span>
            </div>
            """,
            unsafe_allow_html=True
        )


    def preprocess_comment_with_language_detection(self, comment):
        try:
            lang = detect(comment)
            stop_words = stopwords.words(lang) if lang in stopwords.fileids() else stopwords.words("english")
        except Exception:
            stop_words = stopwords.words("english")

        comment = comment.lower()
        comment = re.sub(r"[^a-zA-ZığüşöçİĞÜŞÖÇ\s]", "", comment)

        tokens = word_tokenize(comment)
        filtered_tokens = [word for word in tokens if word not in stop_words]

        return filtered_tokens if filtered_tokens else ["unknown"]

    def analyze_sentiment(self, comment):
        blob_analysis = TextBlob(comment)
        polarity = blob_analysis.sentiment.polarity
        subjectivity = blob_analysis.sentiment.subjectivity

        sia = SentimentIntensityAnalyzer()
        vader_scores = sia.polarity_scores(comment)

        if polarity > 0 and vader_scores["compound"] > 0.05:
            overall_sentiment = "Olumlu"
        elif polarity < 0 and vader_scores["compound"] < -0.05:
            overall_sentiment = "Olumsuz"
        else:
            overall_sentiment = "Nötr"

        intensity = "Orta"

        if abs(vader_scores["compound"]) > 0.6:
            intensity = "Güçlü"
        elif abs(vader_scores["compound"]) < 0.3:
            intensity = "Zayıf"

        return {
            "overall_sentiment": overall_sentiment,
            "polarity": polarity,
            "subjectivity": subjectivity,
            "vader_scores": vader_scores,
            "intensity": intensity
        }

    def generate_llm_report(self, comments):
        try:
            llm_analyzer = OllamaCommentAnalyzer(model="qwen2.5:3b")
            return llm_analyzer.analyze_comments(
                comments=comments,
                max_comment_count=50
            )
        except Exception as e:
            return (
                "LLM analiz raporu oluşturulamadı.\n\n"
                f"Hata detayı: {e}\n\n"
                "Ollama uygulamasının açık olduğundan ve qwen2.5:3b modelinin yüklü olduğundan emin olun."
            )

    def render_status(self, status_placeholder, stage):
        if stage == "scraping":
            statuses = ["active", "", ""]
            labels = [
                "<span class='spinner'></span>Yorumlar çekiliyor...",
                "Yorum analizi bekleniyor...",
                "LLM analizi bekleniyor..."
            ]
        elif stage == "sentiment_analyzing":
            statuses = ["success", "active", ""]
            labels = [
                "Yorumlar çekildi.",
                "<span class='spinner'></span>Yorumlar analiz ediliyor...",
                "LLM analizi bekleniyor..."
            ]
        elif stage == "llm_analyzing":
            statuses = ["success", "success", "active"]
            labels = [
                "Yorumlar çekildi.",
                "Yorumlar analiz edildi.",
                "<span class='spinner'></span>LLM analizi yapılıyor..."
            ]
        else:
            statuses = ["success", "success", "success"]
            labels = [
                "Yorumlar çekildi.",
                "Yorumlar analiz edildi.",
                "LLM analizi tamamlandı."
            ]

        status_placeholder.markdown(
            (
                "<div class='status-row'>"
                f"<div class='pill {statuses[0]}'>{labels[0]}</div>"
                f"<div class='pill {statuses[1]}'>{labels[1]}</div>"
                f"<div class='pill {statuses[2]}'>{labels[2]}</div>"
                "</div>"
            ),
            unsafe_allow_html=True
        )

    def is_valid_tiktok_url(self, url):
        if not url:
            return False

        pattern = r"https?://(www\.)?tiktok\.com/@[^/]+/video/\d+"
        return re.search(pattern, url) is not None

    def render_table(self, analyzed_comments):
        try:
            table_container = st.container(border=True)
        except TypeError:
            table_container = st.container()

        with table_container:
            st.markdown(
                f"<div class='table-card-anchor'></div>",
                unsafe_allow_html=True
            )
            st.markdown("<div class='table-card-title'>Detaylı Yorum Analizi</div>", unsafe_allow_html=True)

            if analyzed_comments:
                page_size = 20
                total_pages = max(1, (len(analyzed_comments) + page_size - 1) // page_size)

                current_page = st.session_state.get("table_page", 1)

                if current_page < 1:
                    current_page = 1
                    st.session_state["table_page"] = 1

                if current_page > total_pages:
                    current_page = total_pages
                    st.session_state["table_page"] = total_pages

                start_index = (current_page - 1) * page_size
                end_index = start_index + page_size

                st.dataframe(
                    analyzed_comments[start_index:end_index],
                    use_container_width=True,
                    hide_index=True
                )

                left_space, prev_col, info_col, next_col, right_space = st.columns([2.0, 0.45, 2.8, 0.45, 2.0])

                with prev_col:
                    if st.button("‹", disabled=current_page <= 1, key="prev_page"):
                        st.session_state["table_page"] = current_page - 1
                        st.rerun()

                with info_col:
                    st.markdown(
                        f"<div class='pagination-info'>Toplam {len(analyzed_comments)} yorum | Sayfa {current_page}/{total_pages}</div>",
                        unsafe_allow_html=True
                    )

                with next_col:
                    if st.button("›", disabled=current_page >= total_pages, key="next_page"):
                        st.session_state["table_page"] = current_page + 1
                        st.rerun()
            else:
                st.warning("Analiz edilecek yorum bulunamadı.")


    def create_chart_base64(self, sentiment_counts, total_analyzed):
        values = list(sentiment_counts.values())
        colors = ["#38e183", "#ff6b6b", "#7a8190"]

        fig, ax = plt.subplots(figsize=(4, 4))

        ax.pie(
            values,
            labels=None,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.3}
        )

        ax.text(
            0,
            0,
            f"{total_analyzed}\nToplam",
            ha="center",
            va="center",
            fontsize=14,
            color="#e6e6e6"
        )

        ax.set_aspect("equal")
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=140, transparent=True, bbox_inches="tight")
        plt.close(fig)

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def render_result_cards(self, metadata, sentiment_counts, total_analyzed, chart_base64):
        positive_count = sentiment_counts["Olumlu"]
        negative_count = sentiment_counts["Olumsuz"]
        neutral_count = sentiment_counts["Nötr"]

        def safe(value):
            if value in [None, ""]:
                return "Bilinmiyor"
            return html.escape(str(value))

        video_id = safe(metadata.get("idVideo", "Bilinmiyor"))
        unique_id = safe(metadata.get("uniqueId", "Bilinmiyor"))
        nickname = safe(metadata.get("nickname", "Bilinmiyor"))
        description = safe(metadata.get("description", "Açıklama yok"))
        total_like = safe(metadata.get("totalLike", "Bilinmiyor"))
        video_comment_count = safe(metadata.get("totalComment", total_analyzed))
        total_share = safe(metadata.get("totalShare", "Bilinmiyor"))
        create_time = safe(metadata.get("createTime", "Bilinmiyor"))
        duration = safe(metadata.get("duration", "Bilinmiyor"))
        fetch_status = safe(metadata.get("fetchStatus", metadata.get("metadataStatus", "")))
        last_cursor = safe(metadata.get("lastCursor", ""))

        status_html = ""

        if fetch_status:
            status_html += f"""
                <div class="final-info-item final-wide">
                    <span>Veri Çekme Durumu</span>
                    <strong>{fetch_status}</strong>
                </div>
            """

        if last_cursor:
            status_html += f"""
                <div class="final-info-item">
                    <span>Son Cursor</span>
                    <strong>{last_cursor}</strong>
                </div>
            """

        cards_html = textwrap.dedent(f"""
        <div class="final-analysis-card">
            <div class="final-card-title">Analiz Sonucu</div>

            <div class="final-video-section">
                <div class="final-section-title">Video Bilgileri</div>

                <div class="final-info-grid">
                    <div class="final-info-item">
                        <span>Video ID</span>
                        <strong>{video_id}</strong>
                    </div>

                    <div class="final-info-item">
                        <span>Kullanıcı</span>
                        <strong>{unique_id} ({nickname})</strong>
                    </div>

                    <div class="final-info-item">
                        <span>Toplam Beğeni</span>
                        <strong>{total_like}</strong>
                    </div>

                    <div class="final-info-item">
                        <span>Çekilen Yorum</span>
                        <strong>{total_analyzed}</strong>
                    </div>

                    <div class="final-info-item">
                        <span>Video Yorum Sayısı</span>
                        <strong>{video_comment_count}</strong>
                    </div>

                    <div class="final-info-item">
                        <span>Toplam Paylaşım</span>
                        <strong>{total_share}</strong>
                    </div>

                    <div class="final-info-item">
                        <span>Oluşturulma Tarihi</span>
                        <strong>{create_time}</strong>
                    </div>

                    <div class="final-info-item">
                        <span>Video Süresi</span>
                        <strong>{duration}</strong>
                    </div>

                    {status_html}
                </div>

                <div class="final-description-box">
                    <span>Açıklama</span>
                    <p style="color:#e3e7ef !important; opacity:1 !important;">{description}</p>
                </div>
            </div>

            <div class="final-bottom-grid" style="display:grid !important; grid-template-columns:380px 1fr !important; gap:28px !important; align-items:start !important; margin-bottom:34px !important;">
                <div class="final-summary-section">
                    <div class="final-section-title">Analiz Özeti</div>

                    <div class="final-metric-grid" style="display:grid !important; grid-template-columns:repeat(2, 170px) !important; grid-template-rows:repeat(2, 105px) !important; gap:14px !important; width:354px !important; height:224px !important;">
                        <div class="final-metric-box" style="width:170px !important; height:105px !important; min-height:105px !important; max-height:105px !important; padding:14px !important; box-sizing:border-box !important; display:flex !important; flex-direction:column !important; justify-content:center !important;">
                            <span>Analiz Edilen</span>
                            <strong>{total_analyzed}</strong>
                        </div>

                        <div class="final-metric-box positive" style="width:170px !important; height:105px !important; min-height:105px !important; max-height:105px !important; padding:14px !important; box-sizing:border-box !important; display:flex !important; flex-direction:column !important; justify-content:center !important;">
                            <span>Olumlu</span>
                            <strong>{positive_count}</strong>
                        </div>

                        <div class="final-metric-box negative" style="width:170px !important; height:105px !important; min-height:105px !important; max-height:105px !important; padding:14px !important; box-sizing:border-box !important; display:flex !important; flex-direction:column !important; justify-content:center !important;">
                            <span>Olumsuz</span>
                            <strong>{negative_count}</strong>
                        </div>

                        <div class="final-metric-box neutral" style="width:170px !important; height:105px !important; min-height:105px !important; max-height:105px !important; padding:14px !important; box-sizing:border-box !important; display:flex !important; flex-direction:column !important; justify-content:center !important;">
                            <span>Nötr</span>
                            <strong>{neutral_count}</strong>
                        </div>
                    </div>
                </div>

                <div class="final-chart-section">
                    <div class="final-section-title">Yorum Sınıflandırma Özeti</div>

                    <div class="final-chart-box" style="height:224px !important; min-height:224px !important; max-height:224px !important; display:flex !important; align-items:center !important; justify-content:center !important; padding:10px !important; box-sizing:border-box !important;">
                        <img src="data:image/png;base64,{chart_base64}" />
                    </div>
                </div>
            </div>
        </div>
        """).strip()

        if hasattr(st, "html"):
            st.html(cards_html)
        else:
            st.markdown(cards_html, unsafe_allow_html=True)


    def store_analysis_result(self, comments, metadata, status_placeholder):
        self.render_status(status_placeholder, "sentiment_analyzing")

        original_comment_texts = [comment_data["comment"] for comment_data in comments]

        classifier = OllamaCommentClassifier(model="qwen2.5:3b")

        comment_texts_for_classification = [
            comment_data["comment"]
            for comment_data in comments
        ]

        llm_classifications = classifier.classify_comments(
            comments=comment_texts_for_classification,
            batch_size=10,
            max_comment_count=30
        )

        classification_map = {
            item["index"]: item
            for item in llm_classifications
        }

        analyzed_comments = []

        sentiment_counts = {
            "Olumlu": 0,
            "Olumsuz": 0,
            "Nötr": 0
        }

        for index, comment_data in enumerate(comments[:100], start=1):
            comment = comment_data["comment"]
            username = comment_data.get("username", "Anonim")
            source_id = comment_data.get("source_id", "")

            classification = classification_map.get(index, {
                "sentiment": "Nötr",
                "category": "Diğer",
                "confidence": 0.3,
                "reason": "Bu yorum için güvenilir LLM sınıflandırması alınamadı."
            })

            sentiment_label = classification.get("sentiment", "Nötr")

            # Özet kartları ve grafik şimdilik 3 ana duygu üzerinden çalışıyor.
            # Anlamsız / Spam / Toksik yorumları tabloda kendi etiketiyle gösteriyoruz,
            # ama özet dağılımda Nötr altında topluyoruz.
            summary_label = sentiment_label if sentiment_label in ["Olumlu", "Olumsuz", "Nötr"] else "Nötr"
            sentiment_counts[summary_label] += 1

            analyzed_comments.append({
                "Kullanıcı": username,
                "Kaynak ID": source_id,
                "Yorum": comment,
                "Duygu": sentiment_label,
                "Kategori": classification.get("category", "Diğer"),
                "Güven": round(float(classification.get("confidence", 0.5)), 2),
                "Gerekçe": classification.get("reason", "Kısa gerekçe üretilemedi.")
            })

        total_analyzed = len(analyzed_comments)
        chart_base64 = self.create_chart_base64(sentiment_counts, total_analyzed)

        self.render_status(status_placeholder, "llm_analyzing")

        llm_report = self.generate_llm_report(original_comment_texts)

        self.render_status(status_placeholder, "done")

        st.session_state["analysis_result"] = {
            "metadata": metadata,
            "sentiment_counts": sentiment_counts,
            "total_analyzed": total_analyzed,
            "chart_base64": chart_base64,
            "analyzed_comments": analyzed_comments,
            "llm_report": llm_report
        }

        st.session_state["table_page"] = 1

    def run_analysis(self, video_url, status_placeholder):
        output_file = "output.json"

        self.render_status(status_placeholder, "scraping")

        scraper = TikTokExtractor(
            url=video_url,
            output=output_file,
            comment_count=2000,
            file_type="json"
        )

        scraper.run()

        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        comments = [c for c in data.get("comments", []) if c.get("comment")]
        metadata = data["metadata"]
        self.store_analysis_result(comments, metadata, status_placeholder)

        if os.path.exists(output_file):
            os.remove(output_file)

    def run_xquik_analysis(self, uploaded_file, status_placeholder):
        comments = normalize_xquik_export(uploaded_file.getvalue())
        if not comments:
            st.warning("CSV içinde analiz edilecek metin bulunamadı.")
            return

        metadata = build_xquik_metadata(comments)
        self.store_analysis_result(comments, metadata, status_placeholder)

    def render_saved_result(self):
        result = st.session_state.get("analysis_result")

        if not result:
            return

        self.render_result_cards(
            metadata=result["metadata"],
            sentiment_counts=result["sentiment_counts"],
            total_analyzed=result["total_analyzed"],
            chart_base64=result["chart_base64"]
        )

        self.render_table(result["analyzed_comments"])

        try:
            llm_report_container = st.container(border=True)
        except TypeError:
            llm_report_container = st.container()

        with llm_report_container:
            st.markdown("<div class='llm-report-card-marker'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-title' style='margin-top: 0;'>LLM Genel Analiz Raporu</div>", unsafe_allow_html=True)
            st.markdown(result["llm_report"])

    def run(self):
        self.inject_styles()
        self.render_app_header()

        st.markdown("<div class='app-title-placeholder'></div>", unsafe_allow_html=True)

        with st.form("video_input_form"):
            st.write("Analiz etmek istediğiniz sosyal medya içeriğinin bağlantısını girin.")
            st.write("Sistem, kullanıcı yorumlarını yapay zekâ destekli olarak analiz ederek duygu dağılımını, öne çıkan temaları ve kampanya performansına ilişkin içgörüleri sunar.")

            st.markdown(
                "<div class='input-label'>Sosyal Medya İçerik Bağlantısı:</div>",
                unsafe_allow_html=True
            )

            col_url, col_btn = st.columns([5, 1.4])

            with col_url:
                video_url = st.text_input(
                    "Sosyal Medya İçerik Bağlantısı:",
                    placeholder="Örnek: https://www.tiktok.com/@kullanici/video/7452354083213775",
                    label_visibility="collapsed"
                )

            with col_btn:
                analyze_button = st.form_submit_button("Yorumları Analiz Et")

            st.markdown(
                "<div class='input-label'>Xquik CSV Dışa Aktarımı:</div>",
                unsafe_allow_html=True
            )
            xquik_file = st.file_uploader(
                "Xquik CSV Dışa Aktarımı:",
                type=["csv"],
                label_visibility="collapsed"
            )
            xquik_button = st.form_submit_button("Xquik CSV Analiz Et")

            status_placeholder = st.empty()

        if xquik_button:
            st.session_state.pop("analysis_result", None)
            st.session_state["table_page"] = 1

            if xquik_file is None:
                status_placeholder.empty()
                st.error("Bir Xquik CSV dosyası seçin.")
                return

            try:
                self.run_xquik_analysis(xquik_file, status_placeholder)
            except Exception as e:
                status_placeholder.empty()
                st.session_state.pop("analysis_result", None)
                st.error(f"Bir hata oluştu: {e}")

        elif analyze_button:
            st.session_state.pop("analysis_result", None)
            st.session_state["table_page"] = 1
    
            if not self.is_valid_tiktok_url(video_url):
                status_placeholder.empty()
                st.error("Geçerli bir TikTok video bağlantısı girin.")
                return

            try:
                self.run_analysis(video_url, status_placeholder)
            except Exception as e:
                status_placeholder.empty()
                st.session_state.pop("analysis_result", None)
                st.error(f"Bir hata oluştu: {e}")

        self.render_saved_result()


if __name__ == "__main__":
    app = TikTokSentimentAnalyzer()
    app.run()
