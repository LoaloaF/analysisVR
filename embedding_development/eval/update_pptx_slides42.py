"""
Rebuild slides 42 onwards in Neural_Task_Representation_cleanup.pptx with new TempConv images.
"""
import copy, os
from lxml import etree
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor

PPTX_IN  = '/mnt/c/Users/amits/Downloads/Neural_Task_Representation_cleanup.pptx'
PPTX_OUT = '/mnt/c/Users/amits/Downloads/Neural_Task_Representation_cleanup_v3.pptx'
DESK     = '/mnt/c/Users/amits/Desktop/'

prs = Presentation(PPTX_IN)
SW, SH = prs.slide_width, prs.slide_height   # 10" x 5.625"

# ── helpers ────────────────────────────────────────────────────────────────────

def replace_image(slide, img_path):
    """Swap first picture in slide for img_path, keeping exact position/size."""
    for sh in slide.shapes:
        if sh.shape_type == 13:
            l, t, w, h = sh.left, sh.top, sh.width, sh.height
            sh._element.getparent().remove(sh._element)
            slide.shapes.add_picture(img_path, l, t, w, h)
            return
    # no existing picture — add centred
    add_img(slide, img_path, 0.2, 1.15, 9.6, 4.3)

def add_img(slide, img_path, left_in, top_in, w_in, h_in):
    slide.shapes.add_picture(img_path,
        Inches(left_in), Inches(top_in), Inches(w_in), Inches(h_in))

def fix_text(slide):
    """Replace any remaining CEBRA display strings."""
    subs = [
        ('CEBRA-Pred',        'TempConv-Pred'),
        ('CEBRA-Cont',        'TempConv-Cont'),
        ('CEBRA-Contr',       'TempConv-Cont'),
        ('CEBRA',             'TempConv'),
    ]
    for sh in slide.shapes:
        if not sh.has_text_frame:
            continue
        for para in sh.text_frame.paragraphs:
            for run in para.runs:
                for old, new in subs:
                    if old in run.text:
                        run.text = run.text.replace(old, new)

def set_title(slide, text):
    for sh in slide.shapes:
        if sh.has_text_frame:
            tf = sh.text_frame
            if tf.text.strip():
                # first non-empty text box is the title
                for para in tf.paragraphs:
                    for run in para.runs:
                        run.text = ''
                tf.paragraphs[0].runs[0].text = text if tf.paragraphs[0].runs else text
                return

def new_slide(layout_idx=6):
    """Append a blank slide and return it."""
    layout = prs.slide_layouts[layout_idx]
    return prs.slides.add_slide(layout)

def clone_slide_after(ref_idx, title_text, img_path,
                      left=0.2, top=1.15, w=9.6, h=4.3):
    """Clone layout from ref slide, set title, add image."""
    ref = prs.slides[ref_idx]
    layout = ref.slide_layout
    slide = prs.slides.add_slide(layout)
    # set title placeholder if exists
    for ph in slide.placeholders:
        if ph.placeholder_format.idx == 0:
            ph.text = title_text
            break
    else:
        # add a text box as title
        txb = slide.shapes.add_textbox(Inches(0.3), Inches(0.15), Inches(9.4), Inches(0.85))
        tf = txb.text_frame
        tf.text = title_text
        tf.paragraphs[0].runs[0].font.size = Pt(24)
        tf.paragraphs[0].runs[0].font.bold = True
    add_img(slide, img_path, left, top, w, h)
    return slide

# ── S42 (idx=41): Grand-mean R² all 4 models ──────────────────────────────────
s = prs.slides[41]
fix_text(s)
# update bullet text
for sh in s.shapes:
    if sh.has_text_frame and len(sh.text_frame.text) > 30:
        tf = sh.text_frame
        for para in tf.paragraphs:
            for run in para.runs:
                run.text = run.text.replace('CEBRA-Pred', 'TempConv-Pred').replace('CEBRA', 'TempConv')
replace_image(s, DESK + 'r2_comparison_grand_mean.png')
print('Updated S42')

# ── S43 (idx=42): TempConv-Cont per-ensemble bars ─────────────────────────────
s = prs.slides[42]
fix_text(s)
replace_image(s, DESK + 'r2_bar_tempconv_cont.png')
print('Updated S43')

# ── S44 (idx=43): MLP vs TempConv-Cont scatter ────────────────────────────────
s = prs.slides[43]
fix_text(s)
replace_image(s, DESK + 'fig3_mlp_vs_tempconv_scatter.png')
print('Updated S44')

# ── S45 (idx=44): TempConv-Pred overview ──────────────────────────────────────
s = prs.slides[44]
fix_text(s)
replace_image(s, DESK + 'r2_scatter_tempconvpred_vs_mlp.png')
print('Updated S45')

# ── S46 (idx=45): TempConv-Pred per-ensemble ──────────────────────────────────
s = prs.slides[45]
fix_text(s)
replace_image(s, DESK + 'r2_bar_tempconv_pred.png')
print('Updated S46')

# ── S47 (idx=46): TempConv-Cont vs TempConv-Pred scatter (was empty) ──────────
s = prs.slides[46]
fix_text(s)
add_img(s, DESK + 'fig6_tempconv_cont_vs_pred_scatter.png', 0.25, 1.20, 9.50, 4.20)
print('Updated S47')

# ── S48 (idx=47): Two thresholds ──────────────────────────────────────────────
s = prs.slides[47]
fix_text(s)
replace_image(s, DESK + 'two_thresholds_scatter.png')
print('Updated S48')

# ── S49 (idx=48): Embedding consistency ───────────────────────────────────────
s = prs.slides[48]
fix_text(s)
replace_image(s, DESK + 'embedding_consistency_violin.png')
print('Updated S49')

# ── keep S50-52 (example plots) as-is ─────────────────────────────────────────
for idx in [49, 50, 51]:
    fix_text(prs.slides[idx])

# ── NEW SLIDES (appended before example plots, at the end) ───────────────────
# We'll insert after S49. Since python-pptx can only append, we add at the end
# then move the example-plot slides to the very end using XML manipulation.

ref_idx = 48  # copy layout from S49

def append_content_slide(title, img, left=0.15, top=1.10, w=9.70, h=4.40):
    return clone_slide_after(ref_idx, title, img, left, top, w, h)

append_content_slide(
    'Feature Attribution by Group — Model Comparison',
    DESK + 'fig5_grouped_bars.png',
    left=0.15, top=1.05, w=9.70, h=4.45)
print('Added: Feature Attribution grouped bars')

append_content_slide(
    'GPV Attribution Heatmap: Sessions × Feature Groups',
    DESK + 'fig2_gpv_heatmap.png',
    left=0.10, top=1.05, w=9.80, h=4.45)
print('Added: GPV heatmap')

append_content_slide(
    'IG Attribution Heatmap: Sessions × Feature Groups',
    DESK + 'fig3_ig_heatmap.png',
    left=0.10, top=1.05, w=9.80, h=4.45)
print('Added: IG heatmap')

append_content_slide(
    'TempConv Temporal Advantage: Honest Assessment',
    DESK + 'fig_honest_summary.png',
    left=0.15, top=0.95, w=9.70, h=4.55)
print('Added: Honest summary')

append_content_slide(
    'Temporal Structure: Significant Features (IG vs Lagged η² and ρ)',
    DESK + 'fig_E_ig_vs_lagged_rho.png',
    left=0.10, top=0.95, w=9.80, h=4.55)
print('Added: IG vs lagged rho')

append_content_slide(
    'TempConv Advantage Pairs: Stronger Behavioral Signal',
    DESK + 'fig_adv_C_lag_eta2.png',
    left=0.10, top=1.05, w=9.80, h=4.45)
print('Added: Advantage lag eta2')

# ── save ──────────────────────────────────────────────────────────────────────
prs.save(PPTX_OUT)
print(f'\nSaved → {PPTX_OUT}')
print(f'Total slides: {len(prs.slides)}')
