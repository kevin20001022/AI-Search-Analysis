import csv
import pandas as pd
from collections import Counter, defaultdict
import jieba
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud
import networkx as nx
import os
import math
import json
from datetime import datetime

# ── 字型設定 ────────────────────────────────────────────────────────────
font_candidates = [
    '/System/Library/Fonts/STHeiti Medium.ttc',
    '/System/Library/Fonts/PingFang.ttc',
    '/Library/Fonts/Arial Unicode MS.ttf',
    '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
]
font_path = None
for fp in font_candidates:
    if os.path.exists(fp):
        font_path = fp
        break

if font_path:
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = '/Users/wupeihua/Desktop/AI_Search_分析'

# ── 讀取資料 ────────────────────────────────────────────────────────────
with open(f'{OUTPUT_DIR}/0204_0309站內搜尋.csv', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    next(reader); next(reader)
    headers = next(reader)
    rows = [r for r in reader if len(r) >= 15]

df = pd.DataFrame(rows, columns=headers[:16] if len(headers) >= 16 else headers)
df.columns = df.columns.str.strip()
df['搜尋文字'] = df['搜尋文字'].str.strip()
df['時間'] = pd.to_datetime(df['時間'], format='%Y/%m/%d %H:%M:%S', errors='coerce')
df['是否為進階搜索'] = df['是否為進階搜索'].str.strip()
df['裝置'] = df['裝置'].str.strip()
df['ip'] = df['ip'].str.strip()

FILTER_COLS = ['類別', '類型', '片長', '風格', '產業', '影視人才職務', '影視人才服務項目', '所在地']

# ════════════════════════════════════════════════════════════════════════
# Task 1：一般 vs 進階搜尋 × 裝置比例
# ════════════════════════════════════════════════════════════════════════
df_text = df[df['搜尋文字'].ne('')].copy()
df_text['搜尋類型'] = df_text['是否為進階搜索'].map({'0': '一般搜尋', '1': '進階搜尋'})

t1_data = {}
for search_type, group in df_text.groupby('搜尋類型'):
    total = len(group)
    t1_data[search_type] = {'total': total, 'devices': {}}
    for device, count in group['裝置'].value_counts().items():
        t1_data[search_type]['devices'][device] = {'count': int(count), 'pct': round(count / total * 100, 1)}

# ════════════════════════════════════════════════════════════════════════
# Task 2：搜尋 vs 篩選 + 詳細篩選分析
# ════════════════════════════════════════════════════════════════════════
def dedup_sessions(sub_df, window_sec=30):
    sub_df = sub_df.sort_values('時間').reset_index(drop=True)
    key_cols = ['ip', '搜尋文字'] + FILTER_COLS
    last_time = {}
    kept_rows = []
    for _, row in sub_df.iterrows():
        key = tuple(str(row.get(c, '')).strip() for c in key_cols)
        t = row['時間']
        if pd.isna(t):
            continue
        prev_t = last_time.get(key)
        if prev_t is None or (t - prev_t).total_seconds() > window_sec:
            kept_rows.append(row)
        last_time[key] = t
    return pd.DataFrame(kept_rows) if kept_rows else pd.DataFrame(columns=sub_df.columns)

df_search_raw = df[df['搜尋文字'].ne('')]
df_filter_raw = df[df['搜尋文字'].eq('') & df[FILTER_COLS].apply(
    lambda row: any(str(v).strip() for v in row), axis=1)]

df_search_dedup = dedup_sessions(df_search_raw)
df_filter_dedup = dedup_sessions(df_filter_raw)

search_count = len(df_search_dedup)
filter_count = len(df_filter_dedup)

# 篩選欄位使用頻率
filter_col_usage = {}
for col in FILTER_COLS:
    used = df_filter_dedup[col].str.strip().ne('').sum() if col in df_filter_dedup.columns else 0
    filter_col_usage[col] = int(used)

# 各欄位的 top 值
filter_col_top_values = {}
for col in FILTER_COLS:
    if col not in df_filter_dedup.columns:
        continue
    # 類型欄位可能是逗號分隔的多值，展開計算
    expanded = []
    for val in df_filter_dedup[col].dropna():
        val = str(val).strip()
        if val:
            for part in val.split(','):
                part = part.strip()
                if part:
                    expanded.append(part)
    top = Counter(expanded).most_common(10)
    filter_col_top_values[col] = [{'value': v, 'count': c} for v, c in top]

# 篩選組合分析（最常見的多欄組合）
combo_counter = Counter()
for _, row in df_filter_dedup.iterrows():
    used_cols = tuple(col for col in FILTER_COLS if str(row.get(col, '')).strip())
    if used_cols:
        combo_counter[used_cols] += 1
top_combos = combo_counter.most_common(10)

# 篩選筆數分布（每次篩選動作用了幾個欄位）
filter_depth = df_filter_dedup[FILTER_COLS].apply(
    lambda row: sum(1 for v in row if str(v).strip()), axis=1).value_counts().sort_index()

# ════════════════════════════════════════════════════════════════════════
# Task 3：熱詞分析圖
# ════════════════════════════════════════════════════════════════════════
STOP_WORDS = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
              '一', '一個', '上', '也', '很', '到', '說', '要', '去', '你',
              '會', '着', '没有', '看', '好', '自己', '這', '那', '啊', '嗎',
              'a', 'an', 'the', 'and', 'or', 'of', 'to', 'in', 'for'}

def tokenize(texts):
    """不切詞，直接以完整搜尋文字為單位計頻。相同文字合併計次。"""
    freq_map = {}
    for text in texts:
        # 清理換行、多餘空白
        text = ' '.join(text.split())
        if text:
            freq_map[text] = freq_map.get(text, 0) + 1
    return [[text] * count for text, count in freq_map.items()]

def build_cooccurrence(tokens_per_doc, top_n=40):
    freq = Counter(w for doc in tokens_per_doc for w in doc)
    top_words = {w for w, _ in freq.most_common(top_n)}
    co = defaultdict(int)
    for doc in tokens_per_doc:
        doc_top = [w for w in doc if w in top_words]
        for i in range(len(doc_top)):
            for j in range(i+1, len(doc_top)):
                pair = tuple(sorted([doc_top[i], doc_top[j]]))
                co[pair] += 1
    return freq, co, top_words

def bubble_pack(words_freqs, iters=400, seed=42):
    """Force-directed circle packing — no overlaps guaranteed."""
    import random
    rng = random.Random(seed)
    max_f = words_freqs[0][1]
    min_r, max_r = 0.35, 2.2
    circles = []
    for word, freq in words_freqs:
        r = min_r + (max_r - min_r) * (freq / max_f) ** 0.52
        angle = rng.uniform(0, 6.28)
        dist  = rng.uniform(0, 0.5)
        circles.append([dist * math.cos(angle), dist * math.sin(angle), r, word, freq])

    for step in range(iters):
        attract = 0.012 + 0.008 * (step / iters)   # gradually pull to centre
        for i, ci in enumerate(circles):
            fx, fy = -ci[0] * attract, -ci[1] * attract
            for j, cj in enumerate(circles):
                if i == j:
                    continue
                dx, dy = ci[0] - cj[0], ci[1] - cj[1]
                dist_ij = max((dx**2 + dy**2) ** 0.5, 1e-6)
                gap = ci[2] + cj[2] + 0.04 - dist_ij
                if gap > 0:
                    strength = gap * 0.52
                    fx += (dx / dist_ij) * strength
                    fy += (dy / dist_ij) * strength
            ci[0] += fx
            ci[1] += fy
    return circles

def draw_hot_word_chart(texts, title, output_path):
    tokens_per_doc = tokenize(texts)
    freq, _, __ = build_cooccurrence(tokens_per_doc, top_n=100)
    if len(freq) == 0:
        print(f"  [{title}] 無足夠文字資料，跳過。")
        return

    BG = '#ffffff'
    fig, ax = plt.subplots(figsize=(24, 11))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis('off')

    wc_kwargs = dict(
        width=2400, height=1100,
        background_color=BG,
        colormap='Dark2',
        max_words=80,
        prefer_horizontal=0.8,
        min_font_size=16,
        max_font_size=280,
        relative_scaling=0.6,
        collocations=False,
    )
    if font_path:
        wc_kwargs['font_path'] = font_path

    wc = WordCloud(**wc_kwargs).generate_from_frequencies(dict(freq))
    ax.imshow(wc, interpolation='bilinear')

    fp_kw = {'fontproperties': prop} if font_path else {}
    ax.set_title(title, color='#2a2018', fontsize=20, fontweight='bold', pad=16, **fp_kw)

    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  已輸出: {output_path}")

texts_general = df[(df['是否為進階搜索'] == '0') & df['搜尋文字'].ne('')]['搜尋文字'].tolist()
texts_advanced = df[(df['是否為進階搜索'] == '1') & df['搜尋文字'].ne('')]['搜尋文字'].tolist()
draw_hot_word_chart(texts_general, '一般搜尋 — 熱詞分析', f'{OUTPUT_DIR}/hotword_一般搜尋.png')
draw_hot_word_chart(texts_advanced, '進階搜尋 — 熱詞分析', f'{OUTPUT_DIR}/hotword_進階搜尋.png')

# ════════════════════════════════════════════════════════════════════════
# 輸出 HTML 報告（Cinematic Editorial 美學）
# ════════════════════════════════════════════════════════════════════════

SHARED_HEAD = '''
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Mono:wght@300;400;500&family=Noto+Sans+TC:wght@300;400;500&display=swap" rel="stylesheet">
'''

SHARED_CSS = '''
:root {
  --bg:       #f5f2ec;
  --bg2:      #ffffff;
  --bg3:      #ede9e1;
  --gold:     #a07828;
  --gold-dim: #c8a84c;
  --cream:    #5c4a1e;
  --muted:    #9b8e78;
  --text:     #3a3128;
  --white:    #1a1410;
  --accent1:  #a07828;
  --accent2:  #6b8b6b;
  --r: 10px;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: "Noto Sans TC", "PingFang TC", "Heiti TC", sans-serif;
  font-weight: 300;
  line-height: 1.7;
  min-height: 100vh;
  overflow-x: hidden;
}

/* grain overlay */
body::before {
  content: "";
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
  opacity: 0.18;
}

.page-wrap { position: relative; z-index: 1; padding: 64px 60px 100px 60px; }

/* ── Header ── */
.page-header { margin-bottom: 72px; }
.eyebrow {
  font-family: "DM Mono", monospace; font-size: 0.7rem; letter-spacing: 0.25em;
  color: var(--gold); text-transform: uppercase; margin-bottom: 16px;
  display: flex; align-items: center; gap: 12px;
}
.eyebrow::after { content: ""; flex: 1; height: 1px; background: var(--gold-dim); max-width: 80px; }
h1 {
  font-family: "Playfair Display", serif; font-size: clamp(2rem, 5vw, 3.2rem);
  font-weight: 900; color: var(--white); line-height: 1.15;
  letter-spacing: -0.02em; margin-bottom: 16px;
}
.subtitle {
  font-family: "DM Mono", monospace; font-size: 0.75rem; color: var(--muted);
  letter-spacing: 0.08em; display: flex; align-items: center; gap: 16px;
}
.subtitle span { color: var(--gold-dim); }
.hairline { width: 100%; height: 1px; background: linear-gradient(90deg, var(--gold-dim) 0%, transparent 70%); margin: 40px 0; }

/* ── Section heading ── */
h2 {
  font-family: "Playfair Display", serif; font-size: 1.3rem; color: var(--white);
  font-weight: 700; margin: 52px 0 24px; display: flex; align-items: baseline; gap: 14px;
}
h2 .h2-num {
  font-family: "DM Mono", monospace; font-size: 0.65rem; color: var(--gold);
  letter-spacing: 0.2em; border: 1px solid var(--gold-dim); border-radius: 2px;
  padding: 2px 6px; position: relative; top: -1px;
}
h3 {
  font-family: "DM Mono", monospace; font-size: 0.72rem; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--gold); margin-bottom: 18px;
  display: flex; align-items: center; gap: 10px;
}
h3::after { content: ""; height: 1px; flex: 1; background: var(--bg3); }

/* ── Cards ── */
.card {
  background: var(--bg2); border: 1px solid rgba(160,120,40,0.18);
  border-radius: var(--r); padding: 32px 36px; margin-bottom: 20px;
  position: relative; overflow: hidden;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.card::before {
  content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--gold-dim), transparent);
}

/* ── Stat grid ── */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 2px; }
.stat {
  background: var(--bg3); padding: 28px 24px; position: relative;
  border-right: 1px solid rgba(160,120,40,0.15);
  animation: fadeUp 0.6s ease both;
}
.stat:last-child { border-right: none; }
.stat .num {
  font-family: "DM Mono", monospace; font-size: 2.4rem; font-weight: 500;
  color: var(--gold); line-height: 1; margin-bottom: 8px; letter-spacing: -0.03em;
}
.stat .lbl { font-size: 0.72rem; color: var(--muted); letter-spacing: 0.05em; line-height: 1.4; font-weight: 400; }
.stat-note { margin-top: 20px; font-family: "DM Mono", monospace; font-size: 0.68rem; color: var(--muted); letter-spacing: 0.05em; }

/* ── Bar chart ── */
.bar-list { display: flex; flex-direction: column; gap: 10px; }
.bar-row { display: grid; grid-template-columns: 140px 1fr 64px 52px; align-items: center; gap: 12px;
           animation: fadeUp 0.5s ease both; }
.bar-label { font-size: 0.82rem; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar-track { background: rgba(0,0,0,0.07); border-radius: 2px; height: 6px; position: relative; overflow: hidden; }
.bar-fill {
  height: 100%; border-radius: 2px; position: relative;
  transition: width 1s cubic-bezier(0.16, 1, 0.3, 1);
}
.bar-fill::after {
  content: ""; position: absolute; right: 0; top: 0; bottom: 0; width: 3px;
  background: rgba(255,255,255,0.6); border-radius: 2px;
}
.bar-count { font-family: "DM Mono", monospace; font-size: 0.75rem; color: var(--cream); text-align: right; }
.bar-pct { font-family: "DM Mono", monospace; font-size: 0.7rem; color: var(--muted); text-align: right; }

/* ── Two col ── */
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 720px) { .two-col { grid-template-columns: 1fr; } }

/* ── Tags ── */
.tag-wrap { display: flex; flex-wrap: wrap; gap: 6px; }
.tag {
  font-family: "DM Mono", monospace; font-size: 0.7rem; letter-spacing: 0.05em;
  border: 1px solid rgba(160,120,40,0.35); border-radius: 2px;
  padding: 3px 10px; color: var(--gold); background: rgba(160,120,40,0.08);
}

/* ── Combo table ── */
.combo-table { width: 100%; border-collapse: collapse; }
.combo-table thead th {
  font-family: "DM Mono", monospace; font-size: 0.65rem; letter-spacing: 0.15em;
  text-transform: uppercase; color: var(--muted); padding: 0 12px 14px;
  text-align: left; border-bottom: 1px solid rgba(160,120,40,0.25);
}
.combo-table thead th:last-child, .combo-table thead th:nth-last-child(2) { text-align: right; }
.combo-table tbody tr { border-bottom: 1px solid rgba(0,0,0,0.06); transition: background 0.15s; }
.combo-table tbody tr:hover { background: rgba(160,120,40,0.05); }
.combo-table td { padding: 10px 12px; font-size: 0.82rem; }
.combo-table td:last-child { text-align: right; }
.combo-table td:nth-last-child(2) { text-align: right; font-family: "DM Mono", monospace; color: var(--gold); font-size: 0.8rem; }
.combo-table td:last-child { font-family: "DM Mono", monospace; font-size: 0.72rem; color: var(--muted); }

/* ── Device donut (CSS only) ── */
.donut-wrap { display: flex; gap: 32px; align-items: center; }
.donut { position: relative; width: 120px; height: 120px; flex-shrink: 0; }
.donut svg { transform: rotate(-90deg); }
.donut-label {
  position: absolute; inset: 0; display: flex; flex-direction: column;
  align-items: center; justify-content: center;
}
.donut-label .d-num { font-family: "DM Mono", monospace; font-size: 1.2rem; color: var(--white); line-height: 1; }
.donut-label .d-lbl { font-size: 0.6rem; color: var(--muted); margin-top: 3px; }
.donut-legend { flex: 1; }
.legend-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; font-size: 0.82rem; }
.legend-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.legend-name { flex: 1; color: var(--text); }
.legend-val { font-family: "DM Mono", monospace; font-size: 0.78rem; color: var(--cream); }
.legend-pct { font-family: "DM Mono", monospace; font-size: 0.7rem; color: var(--muted); width: 44px; text-align: right; }

/* ── Search type cards ── */
.type-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 40px; }
.type-card {
  background: var(--bg2); border: 1px solid rgba(160,120,40,0.18);
  border-radius: var(--r); padding: 32px; position: relative; overflow: hidden;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.type-card::before {
  content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--gold-dim), transparent);
}
.type-card .tc-label { font-family: "DM Mono", monospace; font-size: 0.65rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--gold); margin-bottom: 12px; font-weight: 500; }
.type-card .tc-total { font-family: "DM Mono", monospace; font-size: 1.8rem; color: var(--white); font-weight: 500; margin-bottom: 4px; }
.type-card .tc-sub { font-size: 0.75rem; color: var(--muted); margin-bottom: 24px; font-weight: 400; }

/* ── Animations ── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
.page-header { animation: fadeUp 0.8s ease both; }
.card { animation: fadeUp 0.6s ease both; }
'''

ANIM_JS = '''
<script>
// animate bars on load
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.bar-fill[data-w]').forEach((el, i) => {
    el.style.width = '0%';
    setTimeout(() => {
      el.style.transition = `width 0.9s cubic-bezier(0.16,1,0.3,1) ${i * 40}ms`;
      el.style.width = el.dataset.w + '%';
    }, 100);
  });
  document.querySelectorAll('[data-stroke]').forEach((el, i) => {
    const final = parseFloat(el.dataset.stroke);
    el.style.strokeDashoffset = el.getAttribute('stroke-dasharray');
    setTimeout(() => {
      el.style.transition = `stroke-dashoffset 1s cubic-bezier(0.16,1,0.3,1) ${i * 80}ms`;
      el.style.strokeDashoffset = final;
    }, 200);
  });
  document.querySelectorAll('.stat').forEach((el, i) => {
    el.style.animationDelay = `${i * 80}ms`;
  });
});
</script>
'''

PALETTE = ['#c9a84c','#8b9e6b','#6b8b9e','#9e6b8b','#6b9e8b','#9e8b6b','#7b6b9e','#9e7b6b']

def bar_html_new(data_list, color='#c9a84c', label_width='140px'):
    max_count = max(c for _, c, _ in data_list) if data_list else 1
    rows = ''
    for i, (label, count, pct) in enumerate(data_list):
        w = round(count / max_count * 100, 1)
        rows += f'''
      <div class="bar-row" style="animation-delay:{i*50}ms">
        <div class="bar-label" title="{label}" style="width:{label_width}">{label}</div>
        <div class="bar-track">
          <div class="bar-fill" data-w="{w}" style="background:linear-gradient(90deg,{color}88,{color});width:0%"></div>
        </div>
        <div class="bar-count">{count:,}</div>
        <div class="bar-pct">{pct:.1f}%</div>
      </div>'''
    return f'<div class="bar-list">{rows}</div>'

def donut_svg(pct, color, bg='#1a1a28', r=46, total=100):
    """SVG donut for device split."""
    cx = cy = 60
    circ = 2 * 3.14159 * r
    filled = circ * pct / 100
    gap = circ - filled
    return f'''<svg width="120" height="120" viewBox="0 0 120 120">
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#e0d8cc" stroke-width="10"/>
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="10"
              stroke-dasharray="{circ:.1f}" stroke-linecap="round"
              data-stroke="{gap:.1f}" style="stroke-dashoffset:{circ:.1f};transition:stroke-dashoffset 1s ease"/>
    </svg>'''

def build_combined_html():
    # ── Task 1 片段 ──────────────────────────────────────────────────────
    colors_map   = {'一般搜尋': '#c9a84c', '進階搜尋': '#8b9e6b'}
    device_colors = {'web': '#c9a84c', 'ios': '#6b8b9e'}

    type_cards_html = ''
    for st, info in t1_data.items():
        total   = info['total']
        devices = info['devices']
        legend_rows = ''
        for device, dinfo in sorted(devices.items()):
            dc = device_colors.get(device, '#aaa')
            legend_rows += f'''
            <div class="legend-row">
              <div class="legend-dot" style="background:{dc}"></div>
              <div class="legend-name">{device.upper()}</div>
              <div class="legend-val">{dinfo["count"]:,} 筆</div>
              <div class="legend-pct">{dinfo["pct"]}%</div>
            </div>'''
        primary_device = max(devices.items(), key=lambda x: x[1]['count'])
        primary_pct    = primary_device[1]['pct']
        primary_color  = device_colors.get(primary_device[0], '#c9a84c')
        type_cards_html += f'''
      <div class="type-card">
        <div class="tc-label">{st}</div>
        <div class="tc-total">{total:,}</div>
        <div class="tc-sub">筆有效搜尋紀錄</div>
        <div class="donut-wrap">
          <div class="donut">
            {donut_svg(primary_pct, primary_color)}
            <div class="donut-label">
              <div class="d-num">{primary_pct}%</div>
              <div class="d-lbl">{primary_device[0].upper()}</div>
            </div>
          </div>
          <div class="donut-legend">{legend_rows}</div>
        </div>
      </div>'''

    all_types   = list(t1_data.keys())
    all_devices = sorted({d for info in t1_data.values() for d in info['devices']})
    compare_rows = ''
    for device in all_devices:
        dc = device_colors.get(device, '#aaa')
        compare_rows += f'<h3 style="margin-top:20px">{device.upper()} 裝置各搜尋類型分布</h3>'
        bar_data = [(st, t1_data[st]['devices'].get(device,{'count':0,'pct':0.0})['count'],
                         t1_data[st]['devices'].get(device,{'count':0,'pct':0.0})['pct']) for st in all_types]
        compare_rows += bar_html_new(bar_data, dc)

    # ── Task 2 片段 ──────────────────────────────────────────────────────
    total_act = search_count + filter_count
    s_pct = search_count / total_act * 100 if total_act else 0
    f_pct = filter_count / total_act * 100 if total_act else 0

    stats_html = ''.join([
        f'<div class="stat"><div class="num">{v}</div><div class="lbl">{l}</div></div>'
        for v, l in [
            (f'{search_count:,}', '搜尋次數'),
            (f'{filter_count:,}', '篩選次數'),
            (f'{total_act:,}', '合計次數'),
            (f'{s_pct:.1f}%', '搜尋佔比'),
            (f'{f_pct:.1f}%', '篩選佔比'),
        ]
    ])

    donut_overview = f'''
    <div class="donut-wrap" style="margin-top:28px">
      <div class="donut">
        {donut_svg(f_pct, "#8b9e6b", r=46)}
        <div class="donut-label">
          <div class="d-num">{f_pct:.0f}%</div>
          <div class="d-lbl">篩選</div>
        </div>
      </div>
      <div class="donut-legend">
        <div class="legend-row"><div class="legend-dot" style="background:#8b9e6b"></div><div class="legend-name">純篩選</div><div class="legend-val">{filter_count:,}</div><div class="legend-pct">{f_pct:.1f}%</div></div>
        <div class="legend-row"><div class="legend-dot" style="background:#c9a84c"></div><div class="legend-name">有搜尋文字</div><div class="legend-val">{search_count:,}</div><div class="legend-pct">{s_pct:.1f}%</div></div>
        <div style="margin-top:14px;font-family:'DM Mono',monospace;font-size:0.65rem;color:var(--muted)">
          去重前：搜尋 {len(df_search_raw):,} / 篩選 {len(df_filter_raw):,}
        </div>
      </div>
    </div>'''

    col_rows_sorted = sorted(filter_col_usage.items(), key=lambda x: -x[1])
    col_bar  = bar_html_new([(c, v, v/filter_count*100 if filter_count else 0) for c, v in col_rows_sorted], '#c9a84c', '160px')
    depth_data = [(f'{k} 個條件', int(v), round(v/filter_count*100,1)) for k, v in filter_depth.items()]
    depth_bar = bar_html_new(depth_data, '#8b9e6b', '120px')

    top_val_html = ''
    for i, col in enumerate(FILTER_COLS):
        top_vals = filter_col_top_values.get(col, [])
        if not top_vals:
            continue
        color = PALETTE[i % len(PALETTE)]
        rows_data = [(item['value'], item['count'], item['count']/filter_count*100) for item in top_vals]
        top_val_html += f'<div style="margin-bottom:32px"><h3>{col}</h3>{bar_html_new(rows_data, color, "180px")}</div>'

    combo_rows = ''
    for rank, (combo, cnt) in enumerate(top_combos, 1):
        tags = ''.join(f'<span class="tag">{c}</span>' for c in combo)
        pct  = cnt / filter_count * 100 if filter_count else 0
        combo_rows += f'''
        <tr>
          <td><span style="font-family:'DM Mono',monospace;color:var(--muted);font-size:0.65rem;margin-right:10px">#{rank:02d}</span><div class="tag-wrap" style="display:inline-flex">{tags}</div></td>
          <td>{cnt:,}</td>
          <td>{pct:.1f}%</td>
        </tr>'''

    # ── 側邊導覽 CSS ──────────────────────────────────────────────────────
    insight_css = '''
.insight-box {
  display: flex; gap: 16px; align-items: flex-start;
  background: linear-gradient(135deg, #fffbf0 0%, #fef9ec 100%);
  border: 1px solid rgba(160,120,40,0.25); border-left: 4px solid var(--gold);
  border-radius: var(--r); padding: 24px 28px; margin: 20px 0 48px;
  box-shadow: 0 2px 8px rgba(160,120,40,0.08);
}
.insight-icon { font-size: 1.4rem; line-height: 1; flex-shrink: 0; margin-top: 2px; }
.insight-body { flex: 1; }
.insight-title {
  font-family: "DM Mono", monospace; font-size: 0.65rem; letter-spacing: 0.2em;
  text-transform: uppercase; color: var(--gold); margin-bottom: 10px; font-weight: 500;
}
.insight-body p {
  font-size: 0.88rem; color: var(--text); line-height: 1.75; margin-bottom: 8px;
}
.insight-body p:last-child { margin-bottom: 0; }
.insight-body strong { color: var(--gold); font-weight: 600; }
'''
    nav_css = '''
.sidenav {
  position: fixed; left: 0; top: 0; bottom: 0; width: 200px;
  background: #fff; border-right: 1px solid rgba(160,120,40,0.15);
  padding: 40px 0; display: flex; flex-direction: column; z-index: 100;
  box-shadow: 2px 0 16px rgba(0,0,0,0.04);
}
.sidenav .nav-logo {
  font-family: "DM Mono", monospace; font-size: 0.6rem; letter-spacing: 0.2em;
  text-transform: uppercase; color: var(--gold); padding: 0 24px 28px;
  border-bottom: 1px solid rgba(160,120,40,0.15); margin-bottom: 20px;
}
.sidenav a {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 24px; font-size: 0.8rem; color: var(--muted);
  text-decoration: none; transition: all 0.2s; border-left: 2px solid transparent;
  font-family: "Noto Sans TC", sans-serif;
}
.sidenav a:hover, .sidenav a.active {
  color: var(--gold); border-left-color: var(--gold);
  background: rgba(160,120,40,0.05);
}
.sidenav a .nav-num {
  font-family: "DM Mono", monospace; font-size: 0.6rem; color: var(--gold-dim);
  opacity: 0.7;
}
.sidenav .nav-section-label {
  font-family: "DM Mono", monospace; font-size: 0.6rem; letter-spacing: 0.15em;
  text-transform: uppercase; color: rgba(160,120,40,0.5); padding: 16px 24px 6px;
}
.page-wrap { margin-left: 200px; }
@media (max-width: 860px) {
  .sidenav { display: none; }
  .page-wrap { margin-left: 0; }
}
'''

    nav_js = '''
<script>
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.sidenav a[href^="#"]');
window.addEventListener('scroll', () => {
  let current = '';
  sections.forEach(s => {
    if (window.scrollY >= s.offsetTop - 120) current = s.id;
  });
  navLinks.forEach(a => {
    a.classList.toggle('active', a.getAttribute('href') === '#' + current);
  });
});
</script>
'''

    return f'''<!DOCTYPE html>
<html lang="zh-Hant">
<head>{SHARED_HEAD}
<title>站內搜尋行為分析</title>
<style>{SHARED_CSS}{insight_css}{nav_css}</style>
</head>
<body>

<nav class="sidenav">
  <div class="nav-logo">Search Analytics</div>
  <div class="nav-section-label">Task 01</div>
  <a href="#t1-top"><span class="nav-num">→</span> 裝置分布總覽</a>
  <a href="#t1-compare"><span class="nav-num">→</span> 裝置類型對比</a>
  <div class="nav-section-label" style="margin-top:12px">Task 02</div>
  <a href="#t2-overview"><span class="nav-num">→</span> 行為總覽</a>
  <a href="#t2-freq"><span class="nav-num">→</span> 篩選欄位頻率</a>
  <a href="#t2-topvals"><span class="nav-num">→</span> 熱門篩選選項</a>
  <a href="#t2-combos"><span class="nav-num">→</span> 常見篩選組合</a>
  <div class="nav-section-label" style="margin-top:12px">Task 03</div>
  <a href="#t3-general"><span class="nav-num">→</span> 一般搜尋熱詞</a>
  <a href="#t3-advanced"><span class="nav-num">→</span> 進階搜尋熱詞</a>
</nav>

<div class="page-wrap">

  <!-- ══════════════════ TASK 1 ══════════════════ -->
  <section id="t1-top">
    <header class="page-header">
      <div class="eyebrow">站內搜尋行為分析　·　Task 01</div>
      <h1>搜尋類型<br>× 裝置分布</h1>
      <p class="subtitle">
        <span>2026.02.04</span> — <span>2026.03.09</span>
        &emsp;｜&emsp; 僅計算有輸入搜尋文字之紀錄
      </p>
    </header>
    <div class="hairline"></div>
    <h2><span class="h2-num">01</span> 各搜尋類型裝置分布</h2>
    <div class="type-cards">{type_cards_html}</div>
  </section>

  <section id="t1-compare">
    <div class="card">
      <h3>裝置 × 搜尋類型 對比</h3>
      {compare_rows}
    </div>
    <div class="insight-box">
      <div class="insight-icon">💡</div>
      <div class="insight-body">
        <div class="insight-title">Task 01 結論</div>
        <p>一般搜尋幾乎由 Web 端主導（98.9%），行動裝置使用率極低，顯示此功能主要在桌面情境下被觸發。</p>
        <p>相較之下，進階搜尋的 iOS 佔比明顯偏高（17.4% vs 1.1%），推測 App 端在介面設計上對進階篩選有更顯著的入口或引導，促使行動用戶更容易啟用進階搜尋。</p>
      </div>
    </div>
  </section>

  <!-- ══════════════════ TASK 2 ══════════════════ -->

  <section id="t2-overview" style="margin-top:80px">
    <header class="page-header">
      <div class="eyebrow">站內搜尋行為分析　·　Task 02</div>
      <h1>搜尋與篩選<br>行為深度分析</h1>
      <p class="subtitle">
        <span>2026.02.04</span> — <span>2026.03.09</span>
        &emsp;｜&emsp; 翻頁去重（同 IP + 同參數 30 秒內僅計 1 次）
      </p>
    </header>
    <div class="hairline"></div>
    <h2><span class="h2-num">01</span> 行為總覽</h2>
    <div class="card">
      <div class="stat-grid">{stats_html}</div>
      {donut_overview}
    </div>
  </section>

  <section id="t2-freq">
    <h2><span class="h2-num">02</span> 篩選欄位使用頻率</h2>
    <div class="two-col">
      <div class="card">
        <h3>各欄位使用次數</h3>
        {col_bar}
      </div>
      <div class="card">
        <h3>每次篩選條件數量分布</h3>
        {depth_bar}
      </div>
    </div>
  </section>

  <section id="t2-topvals">
    <h2><span class="h2-num">03</span> 各篩選欄位熱門選項</h2>
    <div class="card">
      <div class="two-col">{top_val_html}</div>
    </div>
  </section>

  <section id="t2-combos">
    <h2><span class="h2-num">04</span> 最常見篩選欄位組合 Top 10</h2>
    <div class="card">
      <table class="combo-table">
        <thead>
          <tr>
            <th>篩選欄位組合</th>
            <th style="text-align:right">次數</th>
            <th style="text-align:right">佔比</th>
          </tr>
        </thead>
        <tbody>{combo_rows}</tbody>
      </table>
    </div>
  </section>

  <div class="insight-box">
      <div class="insight-icon">💡</div>
      <div class="insight-body">
        <div class="insight-title">Task 02 結論</div>
        <p>使用者行為以「純篩選」為絕對主流（{f_pct:.0f}%），直接輸入搜尋文字的比例僅 {s_pct:.0f}%，顯示大多數用戶傾向透過分類條件縮小範圍，而非主動輸入關鍵字。</p>
        <p>在篩選維度上，「類別」使用率最高，其次為「類型」與「影視人才職務」，顯示用戶最在意的是作品與人才的大分類。多數篩選行為僅使用 1 個條件，組合篩選的比例較低，代表進階篩選的潛力尚未被充分發揮。</p>
      </div>
    </div>

  <!-- ══════════════════ TASK 3 ══════════════════ -->
  <section id="t3-general" style="margin-top:80px">
    <header class="page-header">
      <div class="eyebrow">站內搜尋行為分析　·　Task 03</div>
      <h1>熱詞分析</h1>
      <p class="subtitle">
        <span>2026.02.04</span> — <span>2026.03.09</span>
        &emsp;｜&emsp; 僅計算有輸入搜尋文字之紀錄
      </p>
    </header>
    <div class="hairline"></div>
    <h2><span class="h2-num">01</span> 一般搜尋熱詞</h2>
    <div class="card" style="padding:20px">
      <img src="hotword_一般搜尋.png" style="width:100%;border-radius:6px;display:block">
    </div>
  </section>

  <section id="t3-advanced">
    <h2><span class="h2-num">02</span> 進階搜尋熱詞</h2>
    <div class="card" style="padding:20px">
      <img src="hotword_進階搜尋.png" style="width:100%;border-radius:6px;display:block">
    </div>
  </section>

  <div class="insight-box">
    <div class="insight-icon">💡</div>
    <div class="insight-body">
      <div class="insight-title">Task 03 結論</div>
      <p>一般搜尋的關鍵字以<strong>人名、品牌名、產品類別</strong>為主（如人名、家具、汽車、牙刷），屬於目的明確的精準查詢，用戶已知道自己要找什麼。</p>
      <p>進階搜尋則出現較多<strong>完整描述句</strong>（如「醫美診所的形象片」、「有星就是好旅館」），顯示這類用戶傾向用自然語言描述需求，搜尋意圖更偏向「探索發現」而非「精準定位」，未來可考慮強化語意搜尋或 AI 推薦功能來滿足此類需求。</p>
    </div>
  </div>

</div>
{ANIM_JS}
{nav_js}
</body></html>'''

# 寫出單一整合檔案
with open(f'{OUTPUT_DIR}/搜尋行為分析報告.html', 'w', encoding='utf-8') as f:
    f.write(build_combined_html())
print("HTML 已輸出: 搜尋行為分析報告.html")
print("✓ 全部完成")
