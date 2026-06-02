#!/usr/bin/env python3
"""
NAC Framework Figure — spacious, rich palette, no overlaps.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np, os

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')

# Unified palette — cool academic tones
NAC_DARK  = '#1B3A5C'   # deep navy
NAC_MID   = '#3A7CB8'   # steel blue
NAC_LIGHT = '#D0E4F5'   # ice blue
NAC_BG    = '#EDF4FA'   # faint blue

TTC_DARK  = '#8B2E1C'   # deep brick
TTC_MID   = '#C46A4A'   # terracotta
TTC_LIGHT = '#F0D8CE'   # warm beige
TTC_BG    = '#FAF4F0'   # warm white

CLEAN_DARK  = '#2E6B3C' # forest green
CLEAN_MID   = '#5B9E6A' # sage
CLEAN_LIGHT = '#D5E8D8' # pale green

ATTACK_DARK = '#7B2045' # plum
ATTACK_LIGHT = '#F0DCE4' # blush

ACCENT = '#D4A830'       # gold accent
WHITE  = '#FFFFFF'
SLATE  = '#666666'
INK    = '#1C1C1C'
LINE   = '#D5D5D5'

def save(fig, name):
    out = os.path.join(SAVE_DIR)
    os.makedirs(out, exist_ok=True)
    for fmt in ['pdf','png']:
        fig.savefig(os.path.join(out, f'{name}.{fmt}'), dpi=250, bbox_inches='tight',
                    pad_inches=0.15, facecolor=WHITE)
    plt.close(fig)
    print(f'[OK] {name}')

def main():
    fig = plt.figure(figsize=(12, 9), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 18)
    ax.axis('off')

    def T(x, y, t, s=9, c=INK, ha='center', va='center', b=False, i=False, a=1.0):
        kw = {}
        if b: kw['fontweight'] = 'bold'
        if i: kw['fontstyle'] = 'italic'
        if a != 1.0: kw['alpha'] = a
        ax.text(x, y, t, fontsize=s, color=c, ha=ha, va=va, **kw)

    def node(x, y, w, h, title='', sub='', fc=WHITE, ec=SLATE, lw=0.9, ts=9.5, ss=7, tc=INK, sc=SLATE, r=0.12):
        b = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle=f"round,pad={r}",
                           facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
        ax.add_patch(b)
        if title:
            off = 0.15 if sub else 0
            T(x, y+off, title, s=ts, c=tc, b=True)
        if sub:
            T(x, y-0.35, sub, s=ss, c=sc)

    def arrow(x1, y1, x2, y2, c=SLATE, lw=1.1, z=2, ls='-', rad=0):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->', color=c, lw=lw, ls=ls,
                                    connectionstyle=f'arc3,rad={rad}'), zorder=z)

    # ================================================================
    # TITLE
    # ================================================================
    T(12, 17.45, 'NAC: Nesterov Accelerated Counterattack', s=15, c=NAC_DARK, b=True)
    T(12, 16.65, 'Test-Time Adversarial Defense for Vision-Language Models', s=9.5, c=SLATE, i=True)

    # ================================================================
    # SECTION 1 — Attack Pipeline  (y = 13 ~ 16)
    # ================================================================
    by = 14.8  # base y for pipeline

    # Clean Image — with visual icon
    node(3, by, 3.5, 1.5, 'Clean Image', sub='x    224 × 224 × 3', fc=CLEAN_LIGHT, ec=CLEAN_DARK, tc=CLEAN_DARK, sc=CLEAN_MID)
    for px in np.linspace(1.8, 4.2, 8):
        for py in np.linspace(by-0.3, by+0.55, 5):
            ax.plot(px, py, '.', color=CLEAN_MID, markersize=1.6, alpha=0.2)

    arrow(4.75, by, 7.0, by, c=SLATE, lw=1.3)

    # CLIP Encoder
    node(8.0, by, 2.4, 1.1, 'CLIP Encoder', sub='ViT-B/32  (frozen)', fc=WHITE, ec=SLATE, tc=INK, sc=SLATE)

    arrow(9.2, by, 11.5, by, c=SLATE, lw=1.3)

    # Clean Embedding
    node(13.5, by, 3.5, 1.5, 'Clean Embedding', sub='e_clean  =  f (x)', fc=CLEAN_LIGHT, ec=CLEAN_DARK, tc=CLEAN_DARK, sc=CLEAN_MID)

    # PGD Attack arrow (down)
    arrow(13.5, by-0.75, 13.5, 13.4, c=ATTACK_DARK, lw=1.6)

    # Attack node
    node(13.5, 12.75, 4.2, 1.15, 'PGD Attack', sub='10 iterations    ε ∈ {1, 2, 4} / 255', fc=ATTACK_LIGHT, ec=ATTACK_DARK, tc=ATTACK_DARK, sc=ATTACK_DARK)
    np.random.seed(5)
    for _ in range(20):
        ax.plot(13.5+np.random.uniform(-1.4,1.4), 12.75+np.random.uniform(-0.45,0.45),
                '.', color=ATTACK_DARK, markersize=2.2, alpha=0.25)

    # Adversarial image
    arrow(13.5, 12.75-0.6, 13.5, 11.7, c=ATTACK_DARK, lw=1.3)
    node(13.5, 11.1, 3.8, 1.3, 'Adversarial Image', sub='x_adv  =  x  +  δ_att', fc=ATTACK_LIGHT, ec=ATTACK_DARK, tc=ATTACK_DARK, sc=ATTACK_DARK)

    # Counterattack arrow
    arrow(13.5, 11.1-0.65, 13.5, 9.6, c=SLATE, lw=1.2)
    T(14.6, 10.3, 'Counterattack  Optimization', s=9, c=INK, b=True, ha='left')
    T(14.6, 9.85, '2 ~ 4 steps,  training-free', s=7.5, c=SLATE, ha='left')

    # Divider
    ax.axhline(y=9.2, xmin=0.02, xmax=0.98, color=LINE, lw=0.6)

    # ================================================================
    # SECTION 2 — TTC vs NAC  (y = 3 ~ 8.5)
    # ================================================================

    # ---- LEFT: TTC ----
    TX, TY = 5.5, 5.8
    R = 3.5  # circle radius

    ax.add_patch(Circle((TX, TY), R, facecolor=TTC_BG, edgecolor='#E8E0D8', lw=0.5, zorder=0))
    T(TX, TY+R+0.4, 'Embedding  Space', s=7.5, c='#C0B0A0', i=True)

    # Scattered adversarial embeddings
    np.random.seed(3)
    ax.scatter(TX+np.random.randn(55)*1.2, TY+1.5+np.random.randn(55)*1.0,
               s=8, c=TTC_LIGHT, alpha=0.55, edgecolors='none', zorder=1)

    # Clean target star
    ax.plot(TX+0.6, TY-2.3, '*', color=CLEAN_DARK, markersize=26, zorder=7, markeredgewidth=0.4)
    T(TX+0.6, TY-3.0, 'e_clean', s=8, c=CLEAN_DARK, b=True)

    # Current position
    cx, cy = TX-0.5, TY+0.6
    ax.plot(cx, cy, 'o', color=TTC_DARK, markersize=12, zorder=6, markeredgecolor=WHITE, markeredgewidth=2)
    T(cx-0.65, cy, 'δ_t', s=8.5, c=TTC_DARK, b=True, ha='right')

    # Gradient arrow (short, suboptimal)
    arrow(cx, cy, TX+0.1, TY-0.6, c=TTC_MID, lw=2.2, z=5)
    T(TX-0.05, TY-0.95, '∇L(δ_t)', s=7.5, c=TTC_MID, i=True)

    # Next position (far from target)
    ax.plot(TX+0.25, TY-0.5, 'o', color=TTC_DARK, markersize=7, alpha=0.35, zorder=4)
    T(TX+0.95, TY-0.5, 'δ_{t+1}', s=7, c=TTC_DARK, a=0.5)

    # Labels below
    T(TX, TY-R-1.1, 'TTC', s=11, c=TTC_DARK, b=True)
    T(TX, TY-R-1.6, 'gradient at current position', s=8, c=SLATE)
    T(TX, TY-R-2.0, 'O (1/K)   convergence', s=8, c=SLATE, i=True)

    # ---- RIGHT: NAC ----
    NX, NY = 18.5, 5.8

    ax.add_patch(Circle((NX, NY), R, facecolor=NAC_BG, edgecolor='#D5E0EB', lw=0.5, zorder=0))
    T(NX, NY+R+0.4, 'Embedding  Space', s=7.5, c='#A8B8C8', i=True)

    ax.scatter(NX+np.random.randn(55)*1.2, NY+1.5+np.random.randn(55)*1.0,
               s=8, c=NAC_LIGHT, alpha=0.55, edgecolors='none', zorder=1)

    # Clean target
    ax.plot(NX+0.6, NY-2.3, '*', color=CLEAN_DARK, markersize=26, zorder=7, markeredgewidth=0.4)
    T(NX+0.6, NY-3.0, 'e_clean', s=8, c=CLEAN_DARK, b=True)

    # Current position (faint, tiny)
    ax.plot(NX-0.5, NY+0.6, 'o', color=NAC_MID, markersize=7, alpha=0.25, zorder=3)
    T(NX-0.65, NY+0.6, 'δ_t', s=7, c=SLATE, a=0.35, ha='right')

    # LOOK-AHEAD — the key visual element
    lax, lay = NX+1.4, NY+1.1
    ax.plot(lax, lay, 'D', color=ACCENT, markersize=15, zorder=8, markeredgecolor=WHITE, markeredgewidth=2.5)
    T(lax+0.9, lay+0.2, 'δ_t + μ · v_t', s=9, c=NAC_DARK, b=True, ha='left')
    T(lax+0.9, lay-0.2, 'look-ahead  position', s=7.5, c=NAC_MID, ha='left')

    # Dashed momentum arrow
    arrow(NX-0.5, NY+0.6, lax, lay, c=NAC_MID, lw=1.1, z=4, ls='--')
    T(NX+0.5, NY+1.1, 'μ·v_t', s=7, c=NAC_MID, i=True)

    # Gradient from look-ahead (long, accurate)
    arrow(lax, lay, NX+0.7, NY-1.7, c=NAC_DARK, lw=2.8, z=7)
    T(NX+1.1, NY-0.15, '∇L(δ_t + μ·v_t)', s=7.5, c=NAC_DARK, i=True)

    # Next position (close to target)
    ax.plot(NX+0.75, NY-1.6, 'o', color=NAC_DARK, markersize=12, zorder=6, markeredgecolor=WHITE, markeredgewidth=2)
    T(NX+0.75+0.8, NY-1.6, 'δ_{t+1}', s=8.5, c=NAC_DARK, b=True)

    # Labels below
    T(NX, NY-R-1.1, 'NAC  ( Ours )', s=11, c=NAC_DARK, b=True)
    T(NX, NY-R-1.6, 'gradient at look-ahead position', s=8, c=SLATE)
    T(NX, NY-R-2.0, 'O (1/K²)   convergence', s=8, c=SLATE, i=True)

    # ---- CENTER GAP ----
    cx_gap = 12.0
    ax.plot([cx_gap, cx_gap], [3.0, 8.9], color=LINE, lw=0.6, zorder=0)

    # Key insight text (floating, NOT in a box)
    T(cx_gap, 8.55, 'The  only', s=10.5, c=INK, b=True)
    T(cx_gap, 8.1, 'difference:', s=10.5, c=INK, b=True)
    T(cx_gap, 7.4, 'where the', s=9, c=SLATE)
    T(cx_gap, 7.0, 'gradient', s=9, c=SLATE)
    T(cx_gap, 6.6, 'is evaluated', s=9, c=SLATE)

    # ================================================================
    # SECTION 3 — Code comparison (below circles)
    # ================================================================
    # Large, spacious code box
    box_y = 3.5
    code_bg = '#FAF9F6'

    # TTC code line
    b_ttc = FancyBboxPatch((1.0, box_y+0.7), 10.5, 0.9, boxstyle="round,pad=0.15",
                           facecolor=TTC_BG, edgecolor=TTC_LIGHT, linewidth=1.0, zorder=4)
    ax.add_patch(b_ttc)
    T(6.25, box_y+1.15, 'TTC', s=8.5, c=TTC_DARK, b=True, ha='center')
    T(6.25, box_y+0.75, 'g = ∇L (  x_adv  +  δ_t  )               δ_{t+1} = clip(  δ_t + α · sign(g)  )',
      s=8.5, c=INK, ha='center')

    # NAC code line
    b_nac = FancyBboxPatch((12.5, box_y+0.7), 10.5, 0.9, boxstyle="round,pad=0.15",
                           facecolor=NAC_BG, edgecolor=NAC_LIGHT, linewidth=1.0, zorder=4)
    ax.add_patch(b_nac)
    T(17.75, box_y+1.15, 'NAC', s=8.5, c=NAC_DARK, b=True, ha='center')
    T(17.75, box_y+0.75, 'g = ∇L (  x_adv  +  δ_t  +  μ·v_t  )    v = μ·v + α·sign(g),   δ = clip(δ + v)',
      s=8.3, c=INK, ha='center')

    # Highlight the difference
    T(12, box_y+0.25, '←  Only   2   lines   changed,   zero   extra   FLOPs   →', s=7.5, c=SLATE, i=True)

    # ================================================================
    # SECTION 4 — Output
    # ================================================================
    ax.axhline(y=1.6, xmin=0.02, xmax=0.98, color=LINE, lw=0.6)

    # Converging arrows
    arrow(TX, TY-R-2.3, 10.0, 1.15, c=SLATE, lw=0.8, rad=-0.15)
    arrow(NX, NY-R-2.3, 14.0, 1.15, c=SLATE, lw=0.8, rad=0.15)

    node(12, 0.9, 7.5, 1.2, 'Defended  Image', fc=CLEAN_LIGHT, ec=CLEAN_DARK, tc=CLEAN_DARK,
         sub='x̃  =  x_adv  +  Δ     →     CLIP  Zero-Shot  Classification', sc=CLEAN_MID, ts=10, ss=7.5)

    save(fig, 'fig1_framework')

if __name__ == '__main__':
    main()
