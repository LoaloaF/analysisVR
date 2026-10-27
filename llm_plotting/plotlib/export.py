"""Saving figures and compositing them into multi-session canvases.

Single home for what were three byte-identical copies of the canvas/SVG-merge code
(svm_analysis_2026.py, svm_plotting_lib.py, dim_plotting_lib.py). The docstrings kept
here are the fullest of the three.

`save_figure` is the convention for new and newly-moved figure-level functions: one call
that writes the PNG/SVG pair and hands back both paths, instead of each function growing
its own `save_path`/`save_path_svg`/`save_dir` argument trio.
"""

import os
import re
import xml.etree.ElementTree as ET

from PIL import Image


DPI = 150


def save_figure(fig, save_dir, stem, dpi=DPI, formats=('png', 'svg')):
    """Write `fig` to `save_dir/<stem>.<ext>` for each requested format and return a
    {format: path} dict (empty if `save_dir` is None, so a caller can pass the result
    straight through without branching). Creates `save_dir` if needed.

    The PNG/SVG pair is deliberate rather than either alone: the PNG is what
    `build_canvas` composites for a quick look at a whole session sequence, while the SVG
    is what actually gets hand-assembled into the final poster panel in Inkscape - see
    build_canvas_svg."""
    if save_dir is None:
        return {}
    os.makedirs(save_dir, exist_ok=True)
    paths = {}
    for fmt in formats:
        path = os.path.join(save_dir, f'{stem}.{fmt}')
        fig.savefig(path, dpi=dpi)
        paths[fmt] = path
    return paths


def _paste(canvas, img, xy):
    """`img` onto `canvas` at `xy`, composited over the canvas's white rather than pasted
    raw. PIL's paste without a mask copies the RGB channels and DISCARDS alpha, which for
    the transparent-background panels this module exists to assemble (see the notebooks'
    CCGP_TRANSPARENT_RC) renders every translucent artist - a variance band, a CI band, an
    alpha-masked heatmap cell - at full strength, so a 15%-alpha fill comes out as a solid
    block of colour on the canvas while the panel's own .png is fine. Passing the image as
    its own mask is the alpha composite that was meant."""
    canvas.paste(img, xy, img if img.mode in ('RGBA', 'LA') else None)


def _stack_vertically(image_paths):
    """Stack a session's own plots top to bottom, left-aligned."""
    images = [Image.open(p) for p in image_paths]
    width = max(img.width for img in images)
    height = sum(img.height for img in images)
    stacked = Image.new('RGB', (width, height), color='white')
    y = 0
    for img in images:
        _paste(stacked, img, (0, y))
        y += img.height
    return stacked


def build_canvas(session_image_paths, canvas_path):
    """session_image_paths: one list of PNG paths per session (in the order they should
    stack vertically, e.g. main plot first then any secondary plot - pass a single-element
    list per session, e.g. `[[p] for p in paths]`, for one image per session with no
    vertical stacking). Each session's stack is then placed side by side with every other
    session's, left-aligned, on a white canvas sized to fit all of them (no scaling -
    images are pasted at their native pixel size), then saved to `canvas_path`."""
    session_composites = [_stack_vertically(paths) for paths in session_image_paths]
    total_width = sum(img.width for img in session_composites)
    max_height = max(img.height for img in session_composites)
    canvas = Image.new('RGB', (total_width, max_height), color='white')
    x = 0
    for img in session_composites:
        _paste(canvas, img, (x, 0))
        x += img.width
    canvas.save(canvas_path)
    return canvas


_SVG_NS = 'http://www.w3.org/2000/svg'
_XLINK_NS = 'http://www.w3.org/1999/xlink'
ET.register_namespace('', _SVG_NS)
ET.register_namespace('xlink', _XLINK_NS)


def _length(value):
    """A CSS length attribute as a float, or None - 'pt'/'px' suffixes stripped. Anything
    with real units (mm, %, em) is not something this module can place, so it returns None
    and the caller falls back to the viewBox."""
    if not value:
        return None
    try:
        return float(value.rstrip().rstrip('ptx'))
    except ValueError:
        return None


def _svg_root_and_size(svg_path):
    """(root, width, height, scale_x, scale_y) for one source SVG.

    width/height are the size the file asks to be drawn at, in points; the scales convert
    the root's own user units (its viewBox) into those points. For a matplotlib SVG both are
    1.0 - it writes width/height and viewBox with matching numbers - but a file that was
    edited or exported elsewhere may not, and placing its content by transform (see
    build_canvas_svg) means the scale has to be applied explicitly rather than being handled
    by a nested viewport."""
    root = ET.parse(svg_path).getroot()
    view_box = root.get('viewBox')
    attr_w, attr_h = _length(root.get('width')), _length(root.get('height'))
    if view_box is not None:
        _, _, vb_w, vb_h = (float(v) for v in view_box.split())
    else:
        vb_w, vb_h = attr_w, attr_h
    width, height = (attr_w or vb_w), (attr_h or vb_h)
    scale_x = width / vb_w if vb_w else 1.0
    scale_y = height / vb_h if vb_h else 1.0
    return root, width, height, scale_x, scale_y


def _unique_id(svg_path, used_ids):
    """An id for one panel's group, from its own filename - so Inkscape's objects panel
    lists 'S09_2024-11-28_17-41_condition_panel' instead of an anonymous 'svg1016'.
    Suffixed if the same basename is placed twice in one canvas."""
    stem = re.sub(r'[^A-Za-z0-9_-]', '_', os.path.splitext(os.path.basename(svg_path))[0])
    candidate, n = stem, 1
    while candidate in used_ids:
        n += 1
        candidate = f'{stem}_{n}'
    used_ids.add(candidate)
    return candidate


def build_canvas_svg(session_svg_paths, canvas_path):
    """SVG analog of build_canvas, for the vector (.svg) exports saved alongside the
    PNGs - same one-list-of-paths-per-session layout (stacked vertically per session,
    sessions placed side by side). Nothing is rasterized: each source SVG's own content is
    moved into one master <svg>, so the combined file stays fully vector and editable
    (fonts, paths, everything movable in Inkscape/Illustrator).

    Each panel goes into a plain **<g transform="translate(...)">**, named after its own
    file. It used to be a nested <svg x= y= width= height= viewBox=> per panel, which is
    valid SVG and renders correctly, but Inkscape does not treat a nested viewport as an
    ordinary group: ungrouping the canvas dropped that viewport and the content jumped to a
    different size, a single panel could not be dragged until it had been entered rather
    than selected, and the objects panel listed them as anonymous 'svg1016' entries. A
    translate group has none of that - it is what Inkscape itself produces when you group
    objects - so a panel selects, moves and ungroups like anything else. The file's own
    scale (its width/height against its viewBox) goes into the same transform, since a
    group has no viewport to apply it (see _svg_root_and_size; it is 1.0 for every
    matplotlib SVG).

    One real difference: a nested viewport CLIPPED its panel to its own box, a group does
    not. Anything a figure draws outside its own canvas (`clip_on=False` annotations, a
    label tight_layout did not account for) now spills into the neighbouring panel instead
    of being silently cut off. That is the better failure - it is visible - but it is a
    change in behaviour for a figure that relied on the clip.
    """
    session_layouts = []
    for paths in session_svg_paths:
        parsed = [(path,) + _svg_root_and_size(path) for path in paths]
        col_width = max(entry[2] for entry in parsed)
        col_height = sum(entry[3] for entry in parsed)
        session_layouts.append((parsed, col_width, col_height))

    total_width = sum(col_width for _, col_width, _ in session_layouts)
    max_height = max(col_height for _, _, col_height in session_layouts)

    master = ET.Element(f'{{{_SVG_NS}}}svg', {
        'width': f'{total_width}pt', 'height': f'{max_height}pt',
        'viewBox': f'0 0 {total_width} {max_height}', 'version': '1.1',
    })

    x_cursor = 0.0
    used_ids = set()
    for parsed, col_width, _ in session_layouts:
        y_cursor = 0.0
        for path, root, width, height, scale_x, scale_y in parsed:
            transform = f'translate({x_cursor},{y_cursor})'
            if (scale_x, scale_y) != (1.0, 1.0):
                transform += f' scale({scale_x},{scale_y})'
            group = ET.SubElement(master, f'{{{_SVG_NS}}}g', {
                'id': _unique_id(path, used_ids), 'transform': transform})
            for child in list(root):
                # <metadata> is the source file's own RDF creator/date block - meaningless
                # per panel once several are combined, and the one child that has no
                # business sitting inside a <g>.
                if child.tag != f'{{{_SVG_NS}}}metadata':
                    group.append(child)
            y_cursor += height
        x_cursor += col_width

    ET.ElementTree(master).write(canvas_path, xml_declaration=True, encoding='utf-8')
    return canvas_path
