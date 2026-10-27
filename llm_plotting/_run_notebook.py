"""Execute a `# %%` percent-script notebook headlessly, cell by cell, and report failures.

The end-to-end check that a notebook still runs after a refactor - the smoke harnesses
only exercise library functions, not the notebook's own cells (session selection,
highlight_selections, canvas assembly, the ad-hoc analysis cells between plots).

Two things make this safe to run against the real notebooks:

* **No NAS writes.** Every path under the NAS root is rewritten into `--out`, for
  `os.makedirs`, `Figure.savefig` and `PIL.Image.save` alike. The notebooks otherwise
  write figures straight into Simon's own output directories.
* **No blocking.** Agg backend, `plt.show` neutered, and `get_ipython()` magics stripped.

    python _run_notebook.py svm_analysis_2026.py --out /tmp/nbrun
    python _run_notebook.py svm_analysis_2026.py --out /tmp/nbrun --until 40

Cells are executed in order in one shared namespace, exactly like a kernel. Execution
stops at the first failing cell (a notebook's later cells nearly always depend on earlier
ones, so continuing would just produce a cascade of NameErrors).
"""

import argparse
import os
import re
import sys
import traceback


def _install_path_redirect(nas_root, out_root):
    """Rewrite anything written under `nas_root` to live under `out_root` instead."""
    os.makedirs(out_root, exist_ok=True)
    nas_root = os.path.abspath(nas_root)

    def redirect(path):
        p = os.path.abspath(str(path))
        if p.startswith(nas_root):
            new = os.path.join(out_root, os.path.relpath(p, nas_root))
            os.makedirs(os.path.dirname(new), exist_ok=True)
            return new
        return str(path)

    real_makedirs = os.makedirs

    def patched_makedirs(name, *a, **kw):
        return real_makedirs(redirect(name), *a, **kw)
    os.makedirs = patched_makedirs

    from matplotlib.figure import Figure
    real_savefig = Figure.savefig
    Figure.savefig = lambda self, fname, *a, **kw: real_savefig(self, redirect(fname), *a, **kw)

    from PIL import Image
    real_save = Image.Image.save
    Image.Image.save = lambda self, fp, *a, **kw: real_save(self, redirect(fp), *a, **kw)
    # Reads need redirecting too, not just writes: build_canvas re-opens the paths the
    # notebook recorded when it saved, which are still the original NAS ones.
    real_open = Image.open
    Image.open = lambda fp, *a, **kw: real_open(redirect(fp), *a, **kw)

    import xml.etree.ElementTree as ET
    real_parse = ET.parse
    ET.parse = lambda source, *a, **kw: real_parse(redirect(source), *a, **kw)
    real_write = ET.ElementTree.write
    ET.ElementTree.write = lambda self, fo, *a, **kw: real_write(self, redirect(fo), *a, **kw)
    return redirect


def split_cells(src):
    """Split a percent-script into (start_line, source) cells on top-level `# %%`."""
    lines = src.split('\n')
    starts = [i for i, l in enumerate(lines) if l.startswith('# %%')] or [0]
    # Anything above the first marker is itself a cell - lm4_plots.py keeps its whole
    # import preamble up there (its markers only start at line 225), and dropping it made
    # every later cell fail with NameError on os/plt/session_names.
    if starts[0] != 0:
        starts = [0] + starts
    cells = []
    for j, start in enumerate(starts):
        end = starts[j + 1] if j + 1 < len(starts) else len(lines)
        cells.append((start + 1, '\n'.join(lines[start:end])))
    return cells


def _stub_missing_optional_modules(names=('cv2',)):
    """Register empty stand-ins for optional deps that aren't installed here.

    lm4_plots.py imports dashsrc plot components, which import cv2 at module level; cv2
    isn't installed in this environment, so the notebook's whole import cell dies and
    every later cell then fails on NameError. The cells being tested never call into cv2,
    so a stub is enough to exercise them. Only stubs what is genuinely absent."""
    import importlib.util, types
    for name in names:
        if importlib.util.find_spec(name) is None:
            sys.modules[name] = types.ModuleType(name)
            print(f"  (stubbed missing optional module {name!r} for this run)")


def run(path, out_root, until=None, keep_going=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import analytics_processing.analytics_constants as C
    nas_root = C.device_paths()[0]
    _install_path_redirect(nas_root, out_root)
    _stub_missing_optional_modules()
    plt.show = lambda *a, **kw: None

    src = open(path).read()
    # IPython magics (%matplotlib widget, autoreload) have no meaning headlessly
    src = re.sub(r'^\s*get_ipython\(\).*$', 'pass', src, flags=re.M)

    ns = {'__name__': '__main__', '__file__': os.path.abspath(path)}
    cells = split_cells(src)
    if until:
        cells = cells[:until]
    failures = []
    print(f"{path}: {len(cells)} cells")
    for i, (lineno, cell) in enumerate(cells):
        if not cell.strip() or all(l.strip().startswith('#') or not l.strip() for l in cell.split('\n')):
            continue
        try:
            exec(compile(cell, f'{path}:cell{i}@L{lineno}', 'exec'), ns)
        except Exception:
            print(f"\n=== FAILED at cell {i} (line {lineno}) ===")
            print('\n'.join(f'    {l}' for l in cell.split('\n')[:14]))
            traceback.print_exc()
            failures.append((i, lineno, traceback.format_exc().strip().split(chr(10))[-1]))
            if not keep_going:
                return failures, len(cells)
        finally:
            plt.close('all')
    if failures:
        print(f"\n{len(failures)}/{len(cells)} cells failed:")
        for i, lineno, msg in failures:
            print(f"  cell {i:3d} (line {lineno:5d}): {msg[:150]}")
    else:
        print(f"OK: all {len(cells)} cells ran")
    return failures, len(cells)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('notebook')
    ap.add_argument('--out', required=True)
    ap.add_argument('--until', type=int)
    ap.add_argument('--keep-going', action='store_true')
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.notebook)))
    sys.path.append('/home/vrmaster/Projects/VirtualReality')
    from baseVR.base_functionality import init_import_paths
    init_import_paths()
    failed, total = run(args.notebook, args.out, args.until, args.keep_going)
    sys.exit(1 if failed else 0)
