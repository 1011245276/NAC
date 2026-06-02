#!/usr/bin/env python3
"""
NAC Framework Figure — redesigned with generous spacing.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np, os

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')

# Palette
I  = '#1A3A5C'   # NAC indigo
IS = '#5A8AB5'   # NAC light
IB = '#E6F0F8'   # NAC bg
V  = '#8B2A2A'   # TTC vermillion
VS = '#C47A6A'   # TTC light
VB = '#F8F0ED'   # TTC bg
G  = '#2D6B3C'   # Emerald
GS = '#DCE8DD'   # Sage
W  = '#FFFFFF'
SL = '#888888'   # Slate
INK= '#1E1E1E'
LG = '#EEEEEE'

def save(fig, name):
    out = os.path.join(SAVE_DIR)
    os.makedirs(out, exist_ok=True)
    for fmt in ['pdf','png']:
        fig.savefig(os.path.join(out, f'{name}.{fmt}'), dpi=250, bbox_inches='tight',
                    pad_inches=0.15, facecolor=W)
    plt.close(fig)
    print(f'[OK] {name}')

def main():
    fig = plt.figure(figsize=(11, 8.5), facecolor=W)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 17)
    ax.axis('off')

    # ---- Helpers ----
    def T(x, y, t, s=9, c=INK, ha='center', va='center', b=False, i=False, alpha=1.0):
        kw = {}
        if b: kw['fontweight'] = 'bold'
        if i: kw['fontstyle'] = 'italic'
        if alpha != 1.0: kw['alpha'] = alpha
        ax.text(x, y, t, fontsize=s, color=c, ha=ha, va=va, **kw)

    def box(x, y, w, h, label='', sub='', fc=W, ec=SL, lw=0.8, r=0.1, subc=SL):
        rp = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle=f"round,pad={r}",
                            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
        ax.add_patch(rp)
        if label:
            off = 0.12 if sub else 0
            T(x, y+off, label, s=9, c=INK, b=True)
        if sub:
            T(x, y-0.28, sub, s=6.5, c=subc)

    def arr(x1, y1, x2, y2, c=SL, lw=1.0, z=2, ls='-', rad=0):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->', color=c, lw=lw, ls=ls,
                                    connectionstyle=f'arc3,rad={rad}'), zorder=z)

    # ================================================================
    # TITLE
    # ================================================================
    T(11, 16.5, 'NAC: Nesterov Accelerated Counterattack', s=14, c=I, b=True)
    T(11, 15.85, 'Test-Time Adversarial Defense for Vision-Language Models', s=9, c=SL, i=True)

    # ================================================================
    # TOP — Attack Pipeline
    # ================================================================

    # Clean Image
    box(3, 14.5, 3.0, 1.3, 'Clean Image', sub='x  (224×224×3)', fc=GS, ec=G, subc=G)
    # Image icon dots
    for px in np.linspace(2.0, 4.0, 8):
        for py in np.linspace(14.1, 14.8, 5):
            ax.plot(px, py, '.', color=G, markersize=1.8, alpha=0.25)

    arr(4.5, 14.5, 6.5, 14.5, c=SL, lw=1.2)

    # CLIP
    box(7.5, 14.5, 2.2, 1.0, 'CLIP Encoder', sub='ViT-B/32  (frozen)', fc=W, ec=SL, subc=SL)

    arr(8.6, 14.5, 10.6, 14.5, c=SL, lw=1.2)

    # Clean embedding
    box(12.5, 14.5, 3.2, 1.3, 'Clean Embedding', sub='e_clean = f (x)', fc=GS, ec=G, subc=G)

    # PGD Attack (below clean embedding)
    arr(12.5, 14.5-0.65, 12.5, 13.3, c=V, lw=1.5)

    # Attack box
    box(12.5, 12.8, 3.6, 0.9, 'PGD Attack', sub='10 steps,  ε ∈ {1,2,4}/255', fc=VB, ec=V, subc=V)
    # Noise sprinkle
    np.random.seed(5)
    for _ in range(18):
        ax.plot(12.5+np.random.uniform(-1.2,1.2), 12.8+np.random.uniform(-0.35,0.35),
                '.', color=V, markersize=2, alpha=0.3)

    # Adversarial image
    arr(12.5, 12.8-0.45, 12.5, 11.6, c=V, lw=1.2)
    box(12.5, 11.1, 3.2, 1.2, 'Adversarial Image', sub='x_adv = x + δ_att', fc=VB, ec=V, subc=V)

    # Arrow to counterattack
    arr(12.5, 11.1-0.6, 12.5, 10.0, c=SL, lw=1.2)
    T(13.3, 10.5, 'Counterattack', s=8, c=INK, b=True, ha='left')
    T(13.3, 10.1, 'Optimization', s=7, c=SL, ha='left')

    # Divider
    ax.axhline(y=9.4, xmin=0.03, xmax=0.97, color=LG, lw=0.6)

    # ================================================================
    # CENTER — TTC vs NAC
    # ================================================================

    # Left: TTC
    tcx, tcy = 5.5, 5.5

    # Embedding space circle
    ax.add_patch(Circle((tcx, tcy), 3.2, facecolor='#FDFBF9', edgecolor='#E0DBD2', lw=0.5, zorder=0))
    T(tcx, tcy+3.5, 'Embedding Space', s=7, c='#C0B8AD', i=True)

    # Scattered adversarial points
    np.random.seed(3)
    ax.scatter(tcx+np.random.randn(50)*1.1, tcy+1.4+np.random.randn(50)*0.9,
               s=7, c=VB, alpha=0.5, edgecolors='none', zorder=1)

    # Clean target (star)
    ax.plot(tcx+0.5, tcy-2.1, '*', color=G, markersize=22, zorder=6, markeredgewidth=0.5)
    T(tcx+0.5, tcy-2.7, 'e_clean', s=7.5, c=G, b=True)

    # Current position
    cx, cy = tcx-0.4, tcy+0.5
    ax.plot(cx, cy, 'o', color=V, markersize=10, zorder=5, markeredgecolor=W, markeredgewidth=1.8)
    T(cx-0.55, cy, 'δ_t', s=8, c=V, b=True, ha='right')

    # Gradient at current (short, wrong direction)
    arr(cx, cy, tcx+0.1, tcy-0.5, c=VS, lw=2.0, z=4)
    T(tcx+0.05, tcy-0.8, '∇L(δ_t)', s=7, c=VS, i=True)

    # Next position (still far)
    ax.plot(tcx+0.2, tcy-0.4, 'o', color=V, markersize=6, alpha=0.4, zorder=3)
    T(tcx+0.8, tcy-0.4, 'δ_{t+1}', s=6.5, c=V, alpha=0.6)

    T(tcx, tcy-3.4, 'TTC  (CVPR 2025)', s=10, c=V, b=True)
    T(tcx, tcy-3.85, 'gradient at current position', s=7.5, c=SL)
    T(tcx, tcy-4.2, 'O (1/K)  convergence', s=7.5, c=SL, i=True)

    # ---- Center gap ----
    ax.plot([11, 11], [2.8, 8.8], color=LG, lw=0.6, zorder=0)

    # Right: NAC
    ncx, ncy = 16.5, 5.5

    ax.add_patch(Circle((ncx, ncy), 3.2, facecolor='#F7FAFD', edgecolor='#D5DFE8', lw=0.5, zorder=0))
    T(ncx, ncy+3.5, 'Embedding Space', s=7, c='#B0BCC8', i=True)

    ax.scatter(ncx+np.random.randn(50)*1.1, ncy+1.4+np.random.randn(50)*0.9,
               s=7, c=IB, alpha=0.5, edgecolors='none', zorder=1)

    # Clean target
    ax.plot(ncx+0.5, ncy-2.1, '*', color=G, markersize=22, zorder=6, markeredgewidth=0.5)
    T(ncx+0.5, ncy-2.7, 'e_clean', s=7.5, c=G, b=True)

    # Current position (small, faint)
    cx2, cy2 = ncx-0.4, ncy+0.5
    ax.plot(cx2, cy2, 'o', color=I, markersize=6, alpha=0.3, zorder=3)
    T(cx2-0.55, cy2, 'δ_t', s=7, c=SL, alpha=0.4, ha='right')

    # LOOK-AHEAD position (the key visual!)
    lax, lay = ncx+1.2, ncy+0.9
    ax.plot(lax, lay, 'D', color=I, markersize=12, zorder=6, markeredgecolor=W, markeredgewidth=2)
    T(lax+0.75, lay+0.15, 'δ_t + μ·v_t', s=8.5, c=I, b=True, ha='left')
    T(lax+0.75, lay-0.25, '(look-ahead)', s=7, c=IS, ha='left')

    # Dashed line from current to look-ahead
    arr(cx2, cy2, lax, lay, c=IS, lw=1.0, z=3, ls='--')
    T(ncx+0.45, ncy+0.9, 'μ·v_t', s=6.5, c=IS, i=True)

    # Gradient at look-ahead (long, accurate)
    arr(lax, lay, ncx+0.55, ncy-1.5, c=I, lw=2.5, z=6)
    T(ncx+0.9, ncy-0.2, '∇L(δ_t + μ·v_t)', s=7, c=I, i=True)

    # Next position (close to target!)
    ax.plot(ncx+0.6, ncy-1.4, 'o', color=I, markersize=10, zorder=5, markeredgecolor=W, markeredgewidth=1.8)
    T(ncx+0.6+0.65, ncy-1.4, 'δ_{t+1}', s=8, c=I, b=True)

    T(ncx, ncy-3.4, 'NAC  (Ours)', s=10, c=I, b=True)
    T(ncx, ncy-3.85, 'gradient at look-ahead position', s=7.5, c=SL)
    T(ncx, ncy-4.2, 'O (1/K²)  convergence', s=7.5, c=SL, i=True)

    # ---- Key insight text (between circles, not overlapping) ----
    T(11, 8.5, 'The  only  difference :', s=9, c=INK, b=True)
    T(11, 8.05, 'where gradients are evaluated', s=9, c=INK)
    T(11, 7.55, 'Same FLOPs   ·   Zero extra cost   ·   Training-free', s=7.5, c=SL, i=True)

    # "2-line change" annotation
    box(11, 6.5, 6.5, 1.05, '', fc='#FFFDF5', ec='#E0D8B0', lw=0.6)
    T(11, 6.85, 'TTC:  g = ∇L(x_adv + δ_t)        NAC:  g = ∇L(x_adv + δ_t + μ·v_t)', s=8, c=INK)
    T(11, 6.35, 'δ_{t+1} = clip(δ_t + α·sign(g))       v = μ·v + α·sign(g),  δ = clip(δ + v)', s=8, c=INK)

    # ================================================================
    # BOTTOM — Output
    # ================================================================
    ax.axhline(y=2.7, xmin=0.03, xmax=0.97, color=LG, lw=0.6)

    arr(5.5, 5.5-2.4, 9.5, 2.2, c=SL, lw=0.8, rad=-0.1)
    arr(16.5, 5.5-2.4, 12.5, 2.2, c=SL, lw=0.8, rad=0.1)

    box(11, 1.6, 6.0, 1.2, 'Defended Image', fc=GS, ec=G,
        sub='x̃ = x_adv + Δ   →   CLIP Zero-Shot Classification', subc=G)

    T(11, 0.75, 'NAC gains +3.8 ~ +7.5 pp over TTC  ·  6 datasets  ·  PGD + AutoAttack',
      s=7, c=SL, i=True)

    save(fig, 'fig1_framework')

if __name__ == '__main__':
    main()
