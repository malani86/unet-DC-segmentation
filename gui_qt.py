"""PySide6 GUI entry point for droplet quantification."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Callable, List

from PySide6.QtCore import QObject, Signal, Slot, QThread
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)


class ProcessWorker(QThread):
    """Execute the batch quantification script in a background thread."""

    succeeded = Signal()
    failed = Signal(str)

    def __init__(self, args: List[str], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._args = args

    def run(self) -> None:  # type: ignore[override]
        try:
            subprocess.run(self._args, check=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - GUI flow
            self.failed.emit(str(exc))
        except Exception as exc:  # pragma: no cover - GUI flow
            self.failed.emit(str(exc))
        else:  # pragma: no cover - GUI flow
            self.succeeded.emit()


class MainWindow(QDialog):
    """Main window for configuring and running droplet quantification."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Droplet Quantification")

        self._worker: ProcessWorker | None = None

        self.img_dir_edit = QLineEdit()
        self.ckpt_edit = QLineEdit("best_UNetDC_focal_model.pth")
        self.out_dir_edit = QLineEdit("quant_results")

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 10_000)
        self.batch_spin.setValue(8)

        self.prob_spin = QDoubleSpinBox()
        self.prob_spin.setRange(0.0, 1.0)
        self.prob_spin.setSingleStep(0.01)
        self.prob_spin.setValue(0.3)

        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(0, 10_000_000)
        self.min_area_spin.setValue(1)

        self.px_spin = QDoubleSpinBox()
        self.px_spin.setRange(0.0, 10_000.0)
        self.px_spin.setSingleStep(0.1)
        self.px_spin.setValue(0.0)

        self.save_check = QCheckBox("Save overlays")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)

        self.run_button = QPushButton("Run")

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

        layout.addWidget(self.save_check)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.run_button)

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

        img_dir = self.img_dir_edit.text().strip()
        ckpt_path = self.ckpt_edit.text().strip()
        out_dir = self.out_dir_edit.text().strip()

        if not img_dir or not ckpt_path or not out_dir:
            QMessageBox.critical(self, "Error", "Please fill in all required fields")
            return

        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantify_droplets_batch.py")
        args = [
            sys.executable,
            script_path,
            "--img_dir",
            img_dir,
            "--ckpt_path",
            ckpt_path,
            "--out_dir",
            out_dir,
            "--batch",
            str(self.batch_spin.value()),
            "--prob_thresh",
            str(self.prob_spin.value()),
            "--min_area",
            str(self.min_area_spin.value()),
        ]

        px_value = self.px_spin.value()
        if px_value > 0:
            args.extend(["--px_per_micron", str(px_value)])

        if self.save_check.isChecked():
            args.append("--save_overlays")

        self.run_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)

        self._worker = ProcessWorker(args, self)
        self._worker.succeeded.connect(self._on_run_succeeded)
        self._worker.failed.connect(self._on_run_failed)
        self._worker.finished.connect(self._cleanup_worker)
        self._worker.start()

    @Slot()
    def _cleanup_worker(self) -> None:
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.run_button.setEnabled(True)
        self._worker = None

    @Slot()
    def _on_run_succeeded(self) -> None:
        QMessageBox.information(self, "Done", "Processing complete")

    @Slot(str)
    def _on_run_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
