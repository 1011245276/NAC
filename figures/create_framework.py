#!/usr/bin/env python3
"""
NAC Framework Figure — Vector Topology philosophy.
Museum-quality academic diagram. Single-column, ~7.5in wide.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (FancyBboxPatch, Circle, Wedge, FancyArrowPatch,
                                 Arc, Ellipse, Polygon, PathPatch)
from matplotlib.path import Path
import numpy as np, os, sys

# --- Font setup ---
FONT_DIR = os.path.expanduser('~/.claude/skills/canvas-design/canvas-fonts')
for f in ['CrimsonPro-Regular.ttf', 'CrimsonPro-Bold.ttf', 'CrimsonPro-Italic.ttf',
          'BricolageGrotesque-Regular.ttf', 'BricolageGrotesque-Bold.ttf',
          'DMMono-Regular.ttf']:
    fp = os.path.join(FONT_DIR, f)
    if os.path.exists(fp):
        matplotlib.font_manager.fontManager.addfont(fp)

plt.rcParams.update({
    'font.family': 'BricolageGrotesque',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'CrimsonPro',
    'mathtext.it': 'CrimsonPro:italic',
    'mathtext.bf': 'CrimsonPro:bold',
    'text.color': '#2D2D2D',
    'axes.edgecolor': 'none',
})

SAVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig1_framework')
if not os.path.exists(os.path.dirname(SAVE)):
    SAVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'fig1_framework')

# --- Palette ---
INDIGO  = '#1A3A5C'   # NAC
STEEL   = '#3A6B96'   # NAC light
SKY     = '#D6E6F5'   # NAC bg
VERMIL  = '#8B2A2A'   # TTC
RUST    = '#C45A4A'   # TTC light
ROSE    = '#F5E6E0'   # TTC bg
EMERALD = '#2D5A3C'   # Clean / target
SAGE    = '#DCE8DD'   # Clean bg
SLATE   = '#6B6B6B'   # Neutral
WARM_GRAY = '#F7F5F2' # Canvas bg
INK     = '#1E1E1E'   # Text
WHITE   = '#FFFFFF'


def save_pdf(fig, name_base):
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
    os.makedirs(out, exist_ok=True)
    for fmt in ['pdf','png']:
        fig.savefig(os.path.join(out, f'{name_base}.{fmt}'), dpi=250, bbox_inches='tight',
                    pad_inches=0.2, facecolor=WHITE, edgecolor='none')
    plt.close(fig)


def main():
    fig = plt.figure(figsize=(8.5, 6.5), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 13)
    ax.axis('off')
    ax.set_facecolor(WHITE)

    # ---- Helper functions ----
    def text(x, y, t, size=8, color=INK, ha='center', va='center', bold=False, italic=False, family=None, alpha=1.0):
        kw = {}
        if bold: kw['fontweight'] = 'bold'
        if italic: kw['fontstyle'] = 'italic'
        if family: kw['fontfamily'] = family
        if alpha != 1.0: kw['alpha'] = alpha
        ax.text(x, y, t, fontsize=size, color=color, ha=ha, va=va, **kw)

    def node(x, y, w, h, label='', fc=WHITE, ec=SLATE, lw=0.8, radius=0.12,
             sublabel='', subcolor=SLATE):
        r = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle=f"round,pad={radius}",
                           facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
        ax.add_patch(r)
        if label:
            text(x, y + (0.08 if sublabel else 0), label, size=7.5, color=INK, bold=True)
        if sublabel:
            text(x, y - 0.22, sublabel, size=6, color=subcolor)

    def arrow(x1, y1, x2, y2, color=SLATE, lw=1.0, z=2, ls='-', style='simple'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw, ls=ls,
                                    connectionstyle='arc3,rad=0'), zorder=z)

    def curved_arrow(x1, y1, x2, y2, rad=0.2, color=SLATE, lw=1.0, z=2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                    connectionstyle=f'arc3,rad={rad}'), zorder=z)

    # ---- Title ----
    text(8.5, 12.65, 'NAC: Nesterov Accelerated Counterattack', size=12, bold=True, color=INDIGO)
    text(8.5, 12.25, 'Test-Time Adversarial Defense for Vision-Language Models', size=8, color=SLATE, italic=True)

    # ================================================================
    # TOP BAND — Attack Pipeline (y ~ 10.2 to 11.6)
    # ================================================================
    band_y = 10.8
    # Section label
    text(0.8, 11.4, '1', size=18, bold=True, color='#DDDDDD', ha='center')

    # Clean image
    node(2.5, band_y, 2.0, 0.9, 'Clean Image', fc=SAGE, ec=EMERALD, lw=1.0, sublabel='x', subcolor=EMERALD)
    # Mini image representation inside
    for px in np.linspace(1.8, 3.2, 6):
        for py in np.linspace(band_y-0.15, band_y+0.25, 4):
            ax.plot(px, py, '.', color=EMERALD, markersize=1.5, alpha=0.3)

    arrow(3.5, band_y, 4.6, band_y, color=SLATE, lw=1.0)

    # CLIP Encoder
    node(5.3, band_y, 1.4, 0.55, 'CLIP', fc=WHITE, ec=SLATE, lw=0.7)
    text(5.3, band_y-0.15, 'Encoder', size=6, color=SLATE)

    arrow(6.0, band_y, 7.1, band_y, color=SLATE, lw=1.0)

    # Clean Embedding
    node(8.0, band_y, 1.8, 0.9, 'Clean Embedding', fc=SAGE, ec=EMERALD, lw=1.0, sublabel='e_clean = f(x)', subcolor=EMERALD)

    # Attack arrow (down and right)
    arrow(8.0, band_y - 0.45, 8.0, 9.6, color=VERMIL, lw=1.2)
    text(7.4, 9.9, 'PGD Attack', size=6.5, color=VERMIL, bold=True)
    text(7.4, 9.55, 'max L(f(x+δ), y)', size=5.8, color=VERMIL, italic=True)

    # Adversarial image
    node(8.0, 9.1, 2.0, 0.85, 'Adversarial Image', fc=ROSE, ec=VERMIL, lw=1.0, sublabel='x_adv = x + δ_att', subcolor=VERMIL)
    # Noise dots
    np.random.seed(1)
    for _ in range(25):
        rx = 8.0 + np.random.uniform(-0.7, 0.7)
        ry = 9.1 + np.random.uniform(-0.3, 0.3)
        ax.plot(rx, ry, '.', color=VERMIL, markersize=1.8, alpha=0.35)

    arrow(8.0, 9.1 - 0.43, 8.0, 8.1, color=SLATE, lw=1.0)

    # "Counterattack" label
    text(7.0, 8.45, 'Counterattack', size=7, color=INK, bold=True, ha='right')
    text(7.0, 8.1, 'Optimization', size=6.5, color=SLATE, ha='right')

    # ================================================================
    # MIDDLE BAND — TTC vs NAC comparison (y ~ 2 to 7.5)
    # ================================================================

    # Divider
    ax.axhline(y=7.7, xmin=0.03, xmax=0.97, color='#E0E0E0', lw=0.5)

    # --- LEFT: TTC ---
    ttc_cx, ttc_cy = 4.2, 4.85

    # Embedding space
    bg_circle = Circle((ttc_cx, ttc_cy), 2.4, facecolor='#FCF9F6', edgecolor='#E0DCD5', lw=0.5, zorder=0)
    ax.add_patch(bg_circle)
    text(ttc_cx, ttc_cy + 2.6, 'Embedding Space', size=6.5, color='#C0B8AD', italic=True)

    # Scattered points (adversarial → scattered)
    np.random.seed(3)
    adv_pts_x = ttc_cx + np.random.randn(40) * 0.9
    adv_pts_y = ttc_cy + 1.0 + np.random.randn(40) * 0.7
    ax.scatter(adv_pts_x, adv_pts_y, s=6, c=ROSE, alpha=0.5, edgecolors='none', zorder=1)

    # Clean target
    ax.plot(ttc_cx + 0.4, ttc_cy - 1.6, '*', color=EMERALD, markersize=18, zorder=5, markeredgewidth=0.5, markeredgecolor=EMERALD)
    text(ttc_cx + 0.4, ttc_cy - 2.05, 'e_clean', size=6.5, color=EMERALD, bold=True)

    # Current position
    cur_x, cur_y = ttc_cx - 0.3, ttc_cy + 0.3
    ax.plot(cur_x, cur_y, 'o', color=VERMIL, markersize=8, zorder=4, markeredgecolor='white', markeredgewidth=1.5)
    text(cur_x - 0.45, cur_y, 'δ_t', size=7, color=VERMIL, bold=True, ha='right')

    # Gradient at current (wrong direction - short arrow)
    arrow(cur_x, cur_y, ttc_cx + 0.1, ttc_cy - 0.4, color=RUST, lw=1.8, z=4)
    text(ttc_cx + 0.0, ttc_cy - 0.65, '∇L(δ_t)', size=6.5, color=RUST, italic=True)

    # Next position (still far from target)
    ax.plot(ttc_cx + 0.2, ttc_cy - 0.3, 'o', color=VERMIL, markersize=5, alpha=0.5, zorder=3)
    text(ttc_cx + 0.65, ttc_cy - 0.3, 'δ_{t+1}', size=6, color=VERMIL, alpha=0.7)

    # Label
    text(ttc_cx, ttc_cy - 2.6, 'TTC', size=10, bold=True, color=VERMIL)
    text(ttc_cx, ttc_cy - 3.0, 'gradient at current position', size=7, color=SLATE)
    text(ttc_cx, ttc_cy - 3.3, 'convergence: O(1/K)', size=7, color=SLATE, italic=True)

    # --- RIGHT: NAC ---
    nac_cx, nac_cy = 12.8, 4.85

    # Embedding space
    bg_circle2 = Circle((nac_cx, nac_cy), 2.4, facecolor='#F7FAFC', edgecolor='#D5DEE8', lw=0.5, zorder=0)
    ax.add_patch(bg_circle2)
    text(nac_cx, nac_cy + 2.6, 'Embedding Space', size=6.5, color='#B0BCC8', italic=True)

    # Scattered points
    np.random.seed(3)
    adv_pts_x2 = nac_cx + np.random.randn(40) * 0.9
    adv_pts_y2 = nac_cy + 1.0 + np.random.randn(40) * 0.7
    ax.scatter(adv_pts_x2, adv_pts_y2, s=6, c=SKY, alpha=0.5, edgecolors='none', zorder=1)

    # Clean target
    ax.plot(nac_cx + 0.4, nac_cy - 1.6, '*', color=EMERALD, markersize=18, zorder=5, markeredgewidth=0.5, markeredgecolor=EMERALD)
    text(nac_cx + 0.4, nac_cy - 2.05, 'e_clean', size=6.5, color=EMERALD, bold=True)

    # Current position
    cur_x2, cur_y2 = nac_cx - 0.3, nac_cy + 0.3
    ax.plot(cur_x2, cur_y2, 'o', color=INDIGO, markersize=6, alpha=0.4, zorder=3)
    text(cur_x2 - 0.45, cur_y2, 'δ_t', size=7, color=SLATE, alpha=0.5, ha='right')

    # Look-ahead position (the key innovation!)
    la_x, la_y = nac_cx + 0.9, nac_cy + 0.7
    ax.plot(la_x, la_y, 'D', color=INDIGO, markersize=10, zorder=5, markeredgecolor='white', markeredgewidth=1.5)
    text(la_x + 0.55, la_y + 0.1, 'δ_t + μ·v_t', size=7.5, color=INDIGO, bold=True, ha='left')
    text(la_x + 0.55, la_y - 0.25, 'look-ahead', size=6.5, color=STEEL, ha='left')

    # Arrow from current to look-ahead (momentum)
    arrow(cur_x2, cur_y2, la_x, la_y, color=STEEL, lw=0.8, z=3, ls='--')
    text(nac_cx + 0.35, nac_cy + 0.65, 'μ·v_t', size=6, color=STEEL, italic=True)

    # Gradient at look-ahead (better direction — longer, more accurate)
    grad_end_x = nac_cx + 0.5
    grad_end_y = nac_cy - 1.2
    arrow(la_x, la_y, grad_end_x, grad_end_y, color=INDIGO, lw=2.2, z=6)
    text(nac_cx + 0.75, nac_cy - 0.15, '∇L(δ_t + μ·v_t)', size=6.5, color=INDIGO, italic=True)

    # Next position (much closer to target!)
    next_x, next_y = nac_cx + 0.5, nac_cy - 1.1
    ax.plot(next_x, next_y, 'o', color=INDIGO, markersize=8, zorder=4, markeredgecolor='white', markeredgewidth=1.5)
    text(next_x + 0.55, next_y, 'δ_{t+1}', size=7, color=INDIGO, bold=True)

    # Label
    text(nac_cx, nac_cy - 2.6, 'NAC (Ours)', size=10, bold=True, color=INDIGO)
    text(nac_cx, nac_cy - 3.0, 'gradient at look-ahead position', size=7, color=SLATE)
    text(nac_cx, nac_cy - 3.3, 'convergence: O(1/K²)', size=7, color=SLATE, italic=True)

    # ---- CENTER DIVIDER ----
    center_x = 8.5
    ax.plot([center_x, center_x], [2.3, 7.4], color='#E8E5E0', lw=0.8, zorder=0)
    text(center_x, 7.2, 'VS', size=9, bold=True, color='#D0CCC5')

    # ---- Key difference annotation ----
    # Arrow from TTC's gradient to NAC's gradient pointing out the difference
    diff_box = FancyBboxPatch((6.2, 6.3), 4.6, 1.0, boxstyle="round,pad=0.12",
                               facecolor='#FFFDF7', edgecolor='#E0D8C0', lw=0.7, zorder=7)
    ax.add_patch(diff_box)
    text(8.5, 6.95, 'The only difference: where gradients are evaluated', size=7.5, bold=True, color=INK)
    text(8.5, 6.55, 'TTC:  ∇L at δ_t               NAC:  ∇L at δ_t + μ·v_t  (look-ahead)', size=7.5, color=INK)
    text(8.5, 6.2, 'Same FLOPs  ·  Zero extra cost  ·  Training-free', size=6.5, color=SLATE, italic=True)

    # ================================================================
    # BOTTOM — Output
    # ================================================================
    ax.axhline(y=2.0, xmin=0.03, xmax=0.97, color='#E0E0E0', lw=0.5)

    # Both converge to defended
    arrow(ttc_cx, ttc_cy - 2.3, center_x - 0.8, 1.3, color=SLATE, lw=0.7, z=1)
    arrow(nac_cx, nac_cy - 2.3, center_x + 0.8, 1.3, color=SLATE, lw=0.7, z=1)

    node(8.5, 1.0, 4.0, 0.75, 'Defended Image', fc=SAGE, ec=EMERALD, lw=1.0,
         sublabel='x̃ = x_adv + Δ   →   CLIP Zero-Shot Classification', subcolor=EMERALD)

    # Bottom tagline
    text(8.5, 0.15, 'NAC gains +3.8 to +7.5 pp over TTC across 6 datasets (PGD ε = 1, 2, 4/255)  ·  Also effective under AutoAttack',
         size=6.5, color=SLATE, italic=True)

    # Section numbers
    text(0.8, 7.4, '2', size=18, bold=True, color='#DDDDDD', ha='center')
    text(0.8, 1.0, '3', size=18, bold=True, color='#DDDDDD', ha='center')

    # ---- Export ----
    save_pdf(fig, 'fig1_framework')
    print('[OK] Framework figure saved')


if __name__ == '__main__':
    main()
