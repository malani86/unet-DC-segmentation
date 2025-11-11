"""PySide6 GUI entry point for droplet quantification."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

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

    def __init__(self, args: Sequence[str], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._args = list(args)

    def run(self) -> None:  # type: ignore[override]
        try:
            subprocess.run(
                self._args,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - GUI flow
            message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            self.failed.emit(message)
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

        self.save_check = QCheckBox("Save overlays")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)

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

        try:
            args = self._build_command()
        except ValueError as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return

        self._toggle_running(True)

        self._worker = ProcessWorker(args, self)
        self._worker.succeeded.connect(self._on_run_succeeded)
        self._worker.failed.connect(self._on_run_failed)
        self._worker.finished.connect(self._cleanup_worker)
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

        script_path = Path(__file__).resolve().with_name("quantify_droplets_batch.py")
        if not script_path.exists():
            raise ValueError("quantify_droplets_batch.py was not found next to gui_qt.py")

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
        ]

        px_value = self.px_spin.value()
        if px_value > 0:
            args.extend(["--px_per_micron", str(px_value)])

        if self.save_check.isChecked():
            args.append("--save_overlays")

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
            self.save_check,
        ):
            widget.setEnabled(not running)

        self.run_button.setEnabled(not running)
        self.progress_bar.setVisible(running)

    @Slot()
    def _cleanup_worker(self) -> None:
        self._toggle_running(False)
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
