"""Visual theme helpers for the Surrogate Model Training Suite GUI.

The GUI code imports these helpers from many places, so this module keeps the
visual constants and styling routines in one spot. That way a rename, palette
update, or plotting tweak does not need to be repeated throughout the window.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyqtgraph as pg
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class Theme:
    """Color tokens shared by widgets, plots, and status badges."""

    background: str = "#f3efe7"
    surface: str = "#fffaf3"
    surface_alt: str = "#f8f3ea"
    border: str = "#d9cfbf"
    text: str = "#1c2733"
    muted_text: str = "#6d7b88"
    accent: str = "#c8653e"
    accent_dark: str = "#9f4829"
    success: str = "#2f7a63"
    warning: str = "#b77a24"
    danger: str = "#a94f4f"
    info: str = "#345f8a"
    topbar_start: str = "#213448"
    topbar_end: str = "#35556d"
    plot_grid: str = "#d8d1c6"
    plot_blue: str = "#2f6d9f"
    plot_orange: str = "#c8653e"
    plot_green: str = "#2f7a63"
    plot_gold: str = "#b77a24"
    plot_purple: str = "#765889"


APP_THEME = Theme()


def apply_application_theme(app: QApplication) -> None:
    """Apply the global Qt palette and stylesheet."""

    # Qt widgets pick up colors from both the palette and the stylesheet. Keeping
    # them aligned avoids "almost right" combinations where native controls ignore
    # part of the custom theme.
    theme = APP_THEME
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(theme.background))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(theme.text))
    palette.setColor(QPalette.ColorRole.Base, QColor(theme.surface))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(theme.surface_alt))
    palette.setColor(QPalette.ColorRole.Text, QColor(theme.text))
    palette.setColor(QPalette.ColorRole.Button, QColor(theme.surface))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(theme.text))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(theme.accent))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    app.setStyleSheet(_stylesheet())


def configure_plot_widget(widget: pg.PlotWidget, *, title: str, x_label: str, y_label: str) -> None:
    """Apply the shared styling to one `pyqtgraph` plot widget."""

    theme = APP_THEME
    widget.setBackground(theme.surface)
    plot_item = widget.getPlotItem()
    # Configure titles, axes, and legend every time a plot is created so all
    # monitoring tabs stay visually consistent even if new plots are added later.
    plot_item.setTitle(title, color=theme.text, size="12pt")
    plot_item.setLabel("bottom", x_label, color=theme.muted_text)
    plot_item.setLabel("left", y_label, color=theme.muted_text)
    plot_item.showGrid(x=True, y=True, alpha=0.25)
    plot_item.getAxis("left").setTextPen(theme.muted_text)
    plot_item.getAxis("bottom").setTextPen(theme.muted_text)
    plot_item.getAxis("left").setPen(theme.border)
    plot_item.getAxis("bottom").setPen(theme.border)
    plot_item.addLegend(
        brush=QColor(theme.surface),
        labelTextColor=theme.text,
        pen=QColor(theme.border),
    )
    pg.setConfigOptions(antialias=True, foreground=theme.text)


def status_colors() -> dict[str, str]:
    """Map human-readable status strings to the palette colors used by badges."""

    return {
        "Connected": APP_THEME.success,
        "Checked Out": APP_THEME.success,
        "High": APP_THEME.success,
        "Ready": APP_THEME.success,
        "Valid": APP_THEME.success,
        "Compatible": APP_THEME.success,
        "Completed": APP_THEME.success,
        "Medium": APP_THEME.warning,
        "Grace": APP_THEME.warning,
        "Released": APP_THEME.warning,
        "Checking": APP_THEME.info,
        "Suggesting": APP_THEME.info,
        "Scanning": APP_THEME.info,
        "Searching": APP_THEME.info,
        "Training": APP_THEME.info,
        "Transfer": APP_THEME.info,
        "Setup Needed": APP_THEME.warning,
        "Missing": APP_THEME.warning,
        "Not Checked": APP_THEME.muted_text,
        "Unconfigured": APP_THEME.muted_text,
        "Stopped": APP_THEME.warning,
        "Low": APP_THEME.danger,
        "Not Available": APP_THEME.muted_text,
        "Not Scanned": APP_THEME.muted_text,
        "Mismatch": APP_THEME.danger,
        "Denied": APP_THEME.danger,
        "Invalid": APP_THEME.danger,
        "License Required": APP_THEME.danger,
        "Error": APP_THEME.danger,
        "Idle": APP_THEME.muted_text,
    }


def plot_color_cycle() -> list[str]:
    """Return the shared series order for plots with multiple overlaid lines."""

    return [
        APP_THEME.plot_blue,
        APP_THEME.plot_orange,
        APP_THEME.plot_green,
        APP_THEME.plot_gold,
        APP_THEME.plot_purple,
        "#4b8b97",
        "#b85450",
    ]


def _stylesheet() -> str:
    """Build the application-wide Qt stylesheet from the theme tokens."""

    theme = APP_THEME
    return f"""
    QWidget {{
        color: {theme.text};
        font-family: "Segoe UI Variable Text", "Segoe UI", sans-serif;
        font-size: 10pt;
    }}
    QMainWindow {{
        background: {theme.background};
    }}
    QFrame#TopBar {{
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 {theme.topbar_start},
            stop: 1 {theme.topbar_end}
        );
        border-radius: 20px;
    }}
    QLabel#TopBarTitle {{
        color: white;
        font-size: 20pt;
        font-weight: 700;
    }}
    QLabel#TopBarSubtitle {{
        color: rgba(255, 255, 255, 0.75);
        font-size: 10pt;
    }}
    QFrame#CardFrame {{
        background: {theme.surface};
        border: 1px solid {theme.border};
        border-radius: 18px;
    }}
    QLabel#SectionTitle {{
        font-size: 12pt;
        font-weight: 700;
    }}
    QLabel#SectionSubtitle {{
        color: {theme.muted_text};
        font-size: 9pt;
    }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
        background: {theme.surface_alt};
        border: 1px solid {theme.border};
        border-radius: 10px;
        padding: 6px 8px;
        selection-background-color: {theme.accent};
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
        border: 1px solid {theme.accent};
    }}
    QPushButton {{
        background: {theme.accent};
        color: white;
        border: none;
        border-radius: 11px;
        padding: 8px 14px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background: {theme.accent_dark};
    }}
    QPushButton:disabled {{
        background: {theme.border};
        color: {theme.muted_text};
    }}
    QPushButton[secondary="true"] {{
        background: {theme.surface_alt};
        color: {theme.text};
        border: 1px solid {theme.border};
    }}
    QPushButton[secondary="true"]:hover {{
        background: {theme.background};
    }}
    QTabWidget::pane {{
        border: 1px solid {theme.border};
        border-radius: 14px;
        top: -1px;
        background: {theme.surface};
    }}
    QTabBar::tab {{
        background: {theme.surface_alt};
        color: {theme.muted_text};
        padding: 9px 16px;
        margin-right: 6px;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
        border: 1px solid {theme.border};
    }}
    QTabBar::tab:selected {{
        background: {theme.surface};
        color: {theme.text};
        font-weight: 700;
    }}
    QHeaderView::section {{
        background: {theme.surface_alt};
        color: {theme.text};
        border: none;
        border-bottom: 1px solid {theme.border};
        padding: 6px;
        font-weight: 700;
    }}
    QTableWidget {{
        background: {theme.surface};
        alternate-background-color: {theme.surface_alt};
        border: 1px solid {theme.border};
        border-radius: 12px;
        gridline-color: {theme.border};
    }}
    QTableWidget::item:selected {{
        background: rgba(200, 101, 62, 0.16);
        color: {theme.text};
    }}
    QProgressBar {{
        background: {theme.surface_alt};
        border: 1px solid {theme.border};
        border-radius: 10px;
        text-align: center;
    }}
    QProgressBar::chunk {{
        background: {theme.accent};
        border-radius: 8px;
    }}
    QScrollArea {{
        border: none;
        background: transparent;
    }}
    QToolButton {{
        color: {theme.muted_text};
        border: none;
        font-weight: 600;
    }}
    QCheckBox {{
        spacing: 8px;
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 5px;
        border: 1px solid {theme.border};
        background: {theme.surface_alt};
    }}
    QCheckBox::indicator:checked {{
        background: {theme.accent};
        border-color: {theme.accent};
    }}
    """
