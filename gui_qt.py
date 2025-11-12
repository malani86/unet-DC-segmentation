"""PySide6 GUI entry point for droplet quantification."""

from __future__ import annotations

import csv
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

from PySide6.QtCore import QObject, Signal, Slot, QThread, QUrl, Qt
from PySide6.QtGui import QDesktopServices, QPixmap

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QSpinBox,
    QDoubleSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtWidgets import QHeaderView


SCRIPT_NAME = "quantify_droplets_batch.py"


def _resolve_batch_script() -> Path:
    """Return the path to the batch quantification script.

    The GUI normally lives alongside ``quantify_droplets_batch.py`` when the
    repository is checked out. Users may, however, launch the interface from a
    different working directory or bundle the GUI file separately. To make the
    subprocess invocation more robust, probe a handful of likely locations
    rather than assuming the script sits right next to ``gui_qt.py``.
    """

    start = Path(__file__).resolve()
    exe_path = Path(sys.argv[0]).resolve()
    frozen_dir = Path(getattr(sys, "_MEIPASS", "")) if getattr(sys, "frozen", False) else None

    search_roots: tuple[Path, ...]
    if frozen_dir and frozen_dir.exists():
        # ``_MEIPASS`` is where PyInstaller extracts bundled files. When the GUI
        # is frozen we prefer those assets first so users do not have to ship
        # the batch script alongside the executable manually.
        search_roots = (frozen_dir, exe_path.parent, Path.cwd(), start.parent, *start.parents)
    else:
        search_roots = (exe_path.parent, Path.cwd(), start.parent, *start.parents)

    candidates: list[Path] = []
    seen: set[Path] = set()
    for directory in search_roots:
        if directory in seen:
            continue
        seen.add(directory)
        candidates.append(directory / SCRIPT_NAME)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    locations = "\n - ".join(str(path.parent) for path in candidates)
    raise FileNotFoundError(
        "Could not locate quantify_droplets_batch.py. Looked in:\n - " + locations
    )


class ProcessWorker(QThread):
    """Execute the batch quantification script in a background thread."""

    succeeded = Signal()
    failed = Signal(str)
    output = Signal(str)

    def __init__(self, args: Sequence[str], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._args = list(args)

    def run(self) -> None:  # type: ignore[override]
        try:
            process = subprocess.Popen(
                self._args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as exc:  # pragma: no cover - GUI flow
            self.failed.emit(str(exc))
            return

        assert process.stdout is not None
        collected: list[str] = []
        try:
            for line in process.stdout:
                cleaned = line.rstrip()
                collected.append(cleaned)
                self.output.emit(cleaned)
        finally:
            process.stdout.close()

        return_code = process.wait()
        if return_code == 0:
            self.succeeded.emit()
            return

        tail = "\n".join(filter(None, collected[-20:]))
        message = tail or f"Process exited with status {return_code}"
        self.failed.emit(message)


class MainWindow(QDialog):
    """Main window for configuring and running droplet quantification."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Droplet Quantification")

        self._worker: ProcessWorker | None = None
        self._last_out_dir: Path | None = None
        self._overlay_paths: list[Path] = []

        self.img_dir_edit = QLineEdit()
        self.ckpt_edit = QLineEdit("best_UNetDC_focal_model.pth")
        self.out_dir_edit = QLineEdit("quant_results")

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 10_000)
        self.batch_spin.setValue(8)

        self.prob_spin = QDoubleSpinBox()
        self.prob_spin.setRange(0.0, 1.0)
        self.prob_spin.setSingleStep(0.01)
        self.prob_spin.setDecimals(3)
        self.prob_spin.setValue(0.3)

        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(0, 10_000_000)
        self.min_area_spin.setValue(1)

        self.px_spin = QDoubleSpinBox()
        self.px_spin.setRange(0.0, 10_000.0)
        self.px_spin.setSingleStep(0.1)
        self.px_spin.setDecimals(3)
        self.px_spin.setValue(0.0)

        self.background_spin = QSpinBox()
        self.background_spin.setRange(0, 10_000)
        self.background_spin.setValue(50)

        self.save_check = QCheckBox("Save overlays")
        self.excel_check = QCheckBox("Generate Excel workbook")
        self.excel_check.setChecked(True)
        self.histogram_check = QCheckBox("Generate histogram plot")
        self.histogram_check.setChecked(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)

        self.run_button = QPushButton("Run")
        self.open_output_button = QPushButton("Open output folder")
        self.open_output_button.setEnabled(False)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)

        self.visual_tabs = QTabWidget()
        self._setup_visualization_tabs()

        self._setup_layout()
        self._connect_signals()

    def _setup_layout(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        layout.addLayout(form)

        form.addRow("Image directory", self._build_path_row(self.img_dir_edit, self._browse_directory))
        form.addRow("Checkpoint path", self._build_path_row(self.ckpt_edit, self._browse_file))
        form.addRow("Output directory", self._build_path_row(self.out_dir_edit, self._browse_directory))

        form.addRow("Batch size", self.batch_spin)
        form.addRow("Probability threshold", self.prob_spin)
        form.addRow("Minimum area", self.min_area_spin)
        form.addRow("Pixels per micron", self.px_spin)
        form.addRow("Background radius", self.background_spin)

        advanced_box = QGroupBox("Outputs")
        advanced_layout = QVBoxLayout()
        advanced_layout.addWidget(self.save_check)
        advanced_layout.addWidget(self.excel_check)
        advanced_layout.addWidget(self.histogram_check)
        advanced_box.setLayout(advanced_layout)
        layout.addWidget(advanced_box)

        layout.addWidget(self.progress_bar)

        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.open_output_button)
        layout.addLayout(buttons)

        layout.addWidget(self.visual_tabs)

        layout.addWidget(QLabel("Run log"))
        layout.addWidget(self.log_output)

    def _setup_visualization_tabs(self) -> None:
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        summary_layout.setContentsMargins(12, 12, 12, 12)
        self.summary_message = QLabel(
            "Summary tables will appear here after a successful run."
        )
        self.summary_message.setAlignment(Qt.AlignCenter)
        self.summary_message.setWordWrap(True)
        summary_layout.addWidget(self.summary_message)

        self.summary_table = QTableWidget()
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        self.summary_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.summary_table.setVisible(False)
        summary_layout.addWidget(self.summary_table)

        self.stats_table = QTableWidget()
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.stats_table.setVisible(False)
        summary_layout.addWidget(self.stats_table)

        self.visual_tabs.addTab(summary_widget, "Summary")

        histogram_container = QScrollArea()
        histogram_container.setWidgetResizable(True)
        histogram_widget = QWidget()
        histogram_layout = QVBoxLayout(histogram_widget)
        histogram_layout.setContentsMargins(12, 12, 12, 12)
        self.histogram_label = QLabel("Histogram preview will appear after a successful run.")
        self.histogram_label.setAlignment(Qt.AlignCenter)
        self.histogram_label.setWordWrap(True)
        histogram_layout.addWidget(self.histogram_label)
        histogram_container.setWidget(histogram_widget)
        self.visual_tabs.addTab(histogram_container, "Histogram")

        overlays_container = QWidget()
        overlays_layout = QHBoxLayout(overlays_container)
        overlays_layout.setContentsMargins(12, 12, 12, 12)
        self.overlay_list = QListWidget()
        self.overlay_list.setMinimumWidth(180)
        self.overlay_list.setEnabled(False)
        self.overlay_list.currentRowChanged.connect(self._on_overlay_selected)
        overlays_layout.addWidget(self.overlay_list)

        overlay_scroll = QScrollArea()
        overlay_scroll.setWidgetResizable(True)
        overlay_widget = QWidget()
        overlay_widget_layout = QVBoxLayout(overlay_widget)
        overlay_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.overlay_image_label = QLabel("Enable ‘Save overlays’ to preview segmentation overlays here.")
        self.overlay_image_label.setAlignment(Qt.AlignCenter)
        self.overlay_image_label.setWordWrap(True)
        overlay_widget_layout.addWidget(self.overlay_image_label)
        overlay_scroll.setWidget(overlay_widget)
        overlays_layout.addWidget(overlay_scroll, 1)

        self._overlay_tab_index = self.visual_tabs.addTab(overlays_container, "Overlays")
        self.visual_tabs.setTabEnabled(self._overlay_tab_index, False)

    def _build_path_row(
        self, line_edit: QLineEdit, browse_handler: Callable[[QLineEdit], None]
    ) -> QWidget:
        container = QWidget()
        row_layout = QHBoxLayout(container)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(line_edit)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(lambda: browse_handler(line_edit))
        row_layout.addWidget(browse_button)
        return container

    def _connect_signals(self) -> None:
        self.run_button.clicked.connect(self._on_run_clicked)
        self.open_output_button.clicked.connect(self._open_output_dir)

    def _browse_directory(self, line_edit: QLineEdit) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select directory")
        if directory:
            line_edit.setText(directory)

    def _browse_file(self, line_edit: QLineEdit) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select file")
        if file_path:
            line_edit.setText(file_path)

    @Slot()
    def _on_run_clicked(self) -> None:
        if self._worker is not None:
            return

        try:
            args = self._build_command()
        except ValueError as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return

        self.log_output.clear()
        self._append_log_line("Running: " + " ".join(shlex.quote(a) for a in args))
        self._clear_visualizations()
        self._toggle_running(True)

        self._worker = ProcessWorker(args, self)
        self._worker.succeeded.connect(self._on_run_succeeded)
        self._worker.failed.connect(self._on_run_failed)
        self._worker.finished.connect(self._cleanup_worker)
        self._worker.output.connect(self._append_log_line)
        self._worker.start()

    def _build_command(self) -> Sequence[str]:
        img_dir = self.img_dir_edit.text().strip()
        ckpt_path = self.ckpt_edit.text().strip()
        out_dir = self.out_dir_edit.text().strip()

        if not img_dir or not ckpt_path or not out_dir:
            raise ValueError("Please fill in all required fields")

        img_dir_path = Path(img_dir)
        if not img_dir_path.is_dir():
            raise ValueError("Image directory does not exist")

        ckpt_path_obj = Path(ckpt_path)
        if not ckpt_path_obj.is_file():
            raise ValueError("Checkpoint file does not exist")

        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        try:
            script_path = _resolve_batch_script()
        except FileNotFoundError as exc:
            raise ValueError(str(exc)) from exc

        self._last_out_dir = out_dir_path

        args = [
            sys.executable,
            str(script_path),
            "--img_dir",
            str(img_dir_path),
            "--ckpt_path",
            str(ckpt_path_obj),
            "--out_dir",
            str(out_dir_path),
            "--batch",
            str(self.batch_spin.value()),
            "--prob_thresh",
            str(self.prob_spin.value()),
            "--min_area",
            str(self.min_area_spin.value()),
            "--background_radius",
            str(self.background_spin.value()),
        ]

        px_value = self.px_spin.value()
        if px_value > 0:
            args.extend(["--px_per_micron", str(px_value)])

        if self.save_check.isChecked():
            args.append("--save_overlays")

        if not self.excel_check.isChecked():
            args.append("--skip_excel")

        if not self.histogram_check.isChecked():
            args.append("--skip_histogram")

        return args

    def _toggle_running(self, running: bool) -> None:
        for widget in (
            self.img_dir_edit,
            self.ckpt_edit,
            self.out_dir_edit,
            self.batch_spin,
            self.prob_spin,
            self.min_area_spin,
            self.px_spin,
            self.background_spin,
            self.save_check,
            self.excel_check,
            self.histogram_check,
        ):
            widget.setEnabled(not running)

        self.run_button.setEnabled(not running)
        self.open_output_button.setEnabled(
            not running
            and self._last_out_dir is not None
            and self._last_out_dir.exists()
        )
        self.progress_bar.setVisible(running)

    @Slot()
    def _cleanup_worker(self) -> None:
        self._toggle_running(False)
        self._worker = None

    @Slot()
    def _on_run_succeeded(self) -> None:
        QMessageBox.information(self, "Done", "Processing complete")
        self._update_visualizations()

    @Slot(str)
    def _on_run_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)

    @Slot()
    def _open_output_dir(self) -> None:
        if self._last_out_dir is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_out_dir)))

    @Slot(str)
    def _append_log_line(self, line: str) -> None:
        self.log_output.appendPlainText(line)

    def _clear_visualizations(self) -> None:
        self._reset_table(self.summary_table)
        self._reset_table(self.stats_table)
        self.summary_message.setText(
            "Summary tables will appear here after a successful run."
        )
        self.summary_message.setVisible(True)
        self.summary_table.setVisible(False)
        self.stats_table.setVisible(False)
        self.histogram_label.setPixmap(QPixmap())
        self.histogram_label.setText("Histogram preview will appear after a successful run.")
        self.overlay_image_label.setPixmap(QPixmap())
        self.overlay_image_label.setText("Enable ‘Save overlays’ to preview segmentation overlays here.")
        self.overlay_list.clear()
        self.overlay_list.setEnabled(False)
        self.visual_tabs.setTabEnabled(self._overlay_tab_index, False)
        self._overlay_paths = []

    def _update_visualizations(self) -> None:
        if self._last_out_dir is None:
            return
        self._load_summary(self._last_out_dir)
        self._load_histogram(self._last_out_dir)
        self._load_overlays(self._last_out_dir)

    def _load_summary(self, out_dir: Path) -> None:
        summary_path = out_dir / "summary_per_image.csv"
        stats_path = out_dir / "droplet_size_stats.csv"

        summary_rows = self._read_csv(summary_path)
        stats_rows = self._read_csv(stats_path)

        if summary_rows:
            headers = list(summary_rows[0].keys())
            self._populate_table(self.summary_table, headers, summary_rows)
            self.summary_table.setVisible(True)
            self.summary_message.setVisible(False)
        else:
            self.summary_table.setVisible(False)

        if stats_rows:
            headers = list(stats_rows[0].keys())
            self._populate_table(self.stats_table, headers, stats_rows)
            self.stats_table.setVisible(True)
        else:
            self.stats_table.setVisible(False)
            if not summary_rows:
                self.summary_message.setText(
                    "Summary files were not generated. Ensure the run completed successfully."
                )

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                return [dict(row) for row in reader]
        except Exception:
            return []

    def _reset_table(self, table: QTableWidget) -> None:
        table.clear()
        table.setRowCount(0)
        table.setColumnCount(0)

    def _populate_table(
        self, table: QTableWidget, headers: list[str], rows: list[dict[str, str]]
    ) -> None:
        self._reset_table(table)
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for col_index, header in enumerate(headers):
                value = row.get(header, "")
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                table.setItem(row_index, col_index, item)
        table.resizeColumnsToContents()

    def _load_histogram(self, out_dir: Path) -> None:
        hist_path = out_dir / "size_histogram.png"
        if hist_path.exists():
            pixmap = QPixmap(str(hist_path))
            if pixmap.isNull():
                self.histogram_label.setPixmap(QPixmap())
                self.histogram_label.setText("Histogram image could not be loaded.")
            else:
                self.histogram_label.setText("")
                self.histogram_label.setPixmap(pixmap)
        else:
            self.histogram_label.setPixmap(QPixmap())
            self.histogram_label.setText("Histogram not generated. Enable histogram output and rerun.")

    def _load_overlays(self, out_dir: Path) -> None:
        overlay_dir = out_dir / "overlays"
        overlay_files = (
            sorted(
                p
                for p in overlay_dir.glob("*")
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            )
            if overlay_dir.exists()
            else []
        )

        self.overlay_list.clear()
        self._overlay_paths = list(overlay_files)

        if not self._overlay_paths:
            self.overlay_list.setEnabled(False)
            self.visual_tabs.setTabEnabled(self._overlay_tab_index, False)
            self.overlay_image_label.setPixmap(QPixmap())
            if overlay_dir.exists():
                self.overlay_image_label.setText("No overlay images were generated.")
            else:
                self.overlay_image_label.setText("Overlays folder not found. Enable ‘Save overlays’ to preview results.")
            return

        self.visual_tabs.setTabEnabled(self._overlay_tab_index, True)
        self.overlay_list.setEnabled(True)
        for path in self._overlay_paths:
            self.overlay_list.addItem(path.name)
        self.overlay_list.setCurrentRow(0)

    @Slot(int)
    def _on_overlay_selected(self, index: int) -> None:
        if index < 0 or index >= len(self._overlay_paths):
            return
        path = self._overlay_paths[index]
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.overlay_image_label.setPixmap(QPixmap())
            self.overlay_image_label.setText(f"Could not load overlay: {path.name}")
            return
        self.overlay_image_label.setText("")
        self.overlay_image_label.setPixmap(pixmap)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
