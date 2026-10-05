"""Shared figure style: Arial / Helvetica, falling back to their metric-compatible clones."""
SANS = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]


def apply(font_size: float = 10.0) -> str:
    """Set the matplotlib style and return the name of the font actually used."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next(name for name in SANS if name in available)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": SANS, "font.size": font_size,
        "mathtext.fontset": "custom", "mathtext.rm": font, "mathtext.it": f"{font}:italic",
        "mathtext.bf": f"{font}:bold", "pdf.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    return font
