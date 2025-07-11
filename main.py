#!/usr/bin/env python3
"""
KITTI Dataset Resizer

Script for scaling images and annotations of the KITTI dataset.
All images are scaled to 284x284 pixels without cropping,
and the corresponding bounding box coordinates are recalculated proportionally.
"""

import argparse
import logging
import shutil
import sys
import threading
from pathlib import Path
from typing import List, Tuple
import cv2
from tqdm import tqdm

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    tk = None

LOG_FILE = "resize.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments"""
    p = argparse.ArgumentParser(
        description="Scaling or validation of the KITTI dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  %(prog)s                                    # GUI mode
  %(prog)s --nogui                            # mo GUI mode with progress bar
  %(prog)s --mode validate                    # Verification of processing 
  %(prog)s --width 512 --height 512           # Scaling to  512x512
        """
    )

    p.add_argument("--mode", choices=["resize", "validate"], default="resize",
                   help="Resize or validate mode")

    # Paths for scaling mode
    p.add_argument("--input-images", type=Path, default=Path("images"),
                   help="Orig images folder")
    p.add_argument("--input-annots", type=Path, default=Path("kitti_annotations"),
                   help="Orig annotations folder")
    p.add_argument("--output-images", type=Path, default=Path("images"),
                   help="Folder to save output images")
    p.add_argument("--output-annots", type=Path, default=Path("annotations"),
                   help="Folder to save output annotations")

    # Scaling params
    p.add_argument("--width", type=int, default=284,
                   help="Target image width in pixels")
    p.add_argument("--height", type=int, default=284,
                   help="Target image height in pixels")

    p.add_argument("--nogui", action="store_true",
                   help="No GUI usage")

    # Paths for validation mode
    p.add_argument("--orig-images", type=Path, default=Path("images_orig"),
                   help="Folder with original images for validation")
    p.add_argument("--orig-annots", type=Path, default=Path("kitti_annotations"),
                   help="Folder with original annotations for validation")
    p.add_argument("--resized-images", type=Path, default=Path("images"),
                   help="Folder with scaled images for validation")
    p.add_argument("--resized-annots", type=Path, default=Path("annotations"),
                   help="Folder with scaled annotations for validation")
    p.add_argument("--tolerance", type=float, default=1.0,
                   help="Allowed error margin for coordinate validation in pixels")

    return p.parse_args()


def check_dataset(img_dir: Path, ann_dir: Path) -> Tuple[list[str], list[str]]:
    """Checks the correspondence between image files and annotation files"""
    img_stems = {p.stem for p in img_dir.glob("*.jpg")}
    ann_stems = {p.stem for p in ann_dir.glob("*.txt")}

    miss_ann = sorted(img_stems - ann_stems)  # Images without annotations
    miss_img = sorted(ann_stems - img_stems)  # annotations without images

    return miss_ann, miss_img


def resize_img(src: Path, dst: Path, size: tuple[int, int]) -> Tuple[float, float]:
    """Image scaling"""
    img = cv2.imread(str(src))
    if img is None:
        raise RuntimeError(f"Can not read {src}")

    h, w = img.shape[:2]  # Current dimensions
    tw, th = size  # Target dimensions

    logger.info(f"  Scaling {src.name} {w}x{h} -> {tw}x{th}")

    resized = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)

    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), resized)

    sx = tw / w  # Scaling factors
    sy = th / h

    logger.info(f"  Factor X={sx:.4f} Y={sy:.4f}")

    return sx, sy


def fix_kitti_line(line: str, sx: float, sy: float) -> str:
    """Adjusts bbox coordinates in the annotation line"""
    parts: List[str] = line.strip().split()

    if len(parts) < 8:
        logger.warning(f"Wrong KITTI string {line.strip()}")
        return line

    bbox = list(map(float, parts[4:8]))  # Extracting bbox coordinates
    orig = bbox.copy()

    bbox[0] *= sx  # left
    bbox[2] *= sx  # right
    bbox[1] *= sy  # top
    bbox[3] *= sy  # bottom

    parts[4:8] = [f"{c:.2f}" for c in bbox]

    logger.debug(
        f"    Bbox [{orig[0]:.1f},{orig[1]:.1f},{orig[2]:.1f},{orig[3]:.1f}] -> [{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}]")

    return " ".join(parts) + "\n"


def proc_annot(src: Path, dst: Path, sx: float, sy: float) -> None:
    """Processing annotation file"""
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8") as f_in, \
            dst.open("w", encoding="utf-8") as f_out:
        cnt = 0
        for line in f_in:
            fixed_line = fix_kitti_line(line, sx, sy)
            f_out.write(fixed_line)
            cnt += 1

        logger.info(f"  Has processed {cnt} objects in {src.name}")


def resize_all(args: argparse.Namespace, progress_cb=None) -> None:
    """Scaling main function"""
    logger.info(f"Initiating scaling to size {args.width}x{args.height}")

    miss_ann, miss_img = check_dataset(args.input_images, args.input_annots)

    if miss_ann or miss_img:
        if miss_ann:
            logger.error(f"Annotations missing for images {', '.join(miss_ann)}")
        if miss_img:
            logger.error(f"Images missing for annotations {', '.join(miss_img)}")
        sys.exit(1)

    # Creating a backup if overwriting
    if args.output_images.resolve() == args.input_images.resolve():
        backup = args.input_images.with_name(args.input_images.name + "_orig")
        if not backup.exists():
            logger.info(f"Creating a backup of original files -> {backup}")
            shutil.copytree(args.input_images, backup)

    img_paths = sorted(args.input_images.glob("*.jpg"))
    total = len(img_paths)

    if total == 0:
        logger.error(f"jpg images not found in folder  {args.input_images}")
        sys.exit(1)

    logger.info(f"Found {total} images to process")

    it = enumerate(img_paths, 1)
    if args.nogui and progress_cb is None:
        it = tqdm(it, total=total, unit="img", desc="Processing", ncols=80)

    # Main processing loop
    for i, img_path in it:
        logger.info(f"[{i}/{total}] Processing {img_path.name}")

        out_img = args.output_images / img_path.name

        try:
            sx, sy = resize_img(img_path, out_img, (args.width, args.height))

            in_ann = args.input_annots / f"{img_path.stem}.txt"
            out_ann = args.output_annots / f"{img_path.stem}.txt"

            proc_annot(in_ann, out_ann, sx, sy)

            logger.info(f"  Successfully processed {img_path.name}")

        except Exception as e:
            logger.error(f"  Processing error {img_path.name} {e}")
            continue

        if progress_cb:
            progress_cb(i, total)

    logger.info(f"Done! The total of {total} images have been processed!")


def parse_kitti(line: str) -> Tuple[List[str], List[float]]:
    """KITTI Annotation string parsing """
    parts = line.strip().split()
    if len(parts) < 8:
        raise ValueError("KITTI in not correct")

    meta = parts[:4]  # class, truncated, occluded, alpha
    bbox = list(map(float, parts[4:8]))  # bbox coordinates

    return meta, bbox


def check_bbox(orig: List[float], resized: List[float],
               sx: float, sy: float, tol: float) -> bool:
    """Checks correctness of bbox scaling"""
    for i in (0, 2):  # Checking X coordinates
        exp = orig[i] * sx
        if abs(exp - resized[i]) > tol:
            return False

    for i in (1, 3):  # Checking Y coordinates
        exp = orig[i] * sy
        if abs(exp - resized[i]) > tol:
            return False

    return True


def validate_all(args: argparse.Namespace) -> None:
    """Validates processing correctness"""
    logger.info("Running dataset validation")

    paths = [args.orig_images, args.orig_annots,
             args.resized_images, args.resized_annots]

    for path in paths:
        if not path.exists():
            logger.error(f"Folder {path} is not found")
            sys.exit(1)

    stems = sorted({p.stem for p in args.resized_images.glob("*.jpg")})

    if not stems:
        logger.error(f"No jpg images found in {args.resized_images}")
        sys.exit(1)

    logger.info(f"Validating {len(stems)} images")

    err_size = 0
    err_bbox = 0

    for i, stem in enumerate(stems, 1):
        logger.info(f"[{i}/{len(stems)}] Validating {stem}")

        # File paths
        res_img = args.resized_images / f"{stem}.jpg"
        orig_img = args.orig_images / f"{stem}.jpg"
        res_ann = args.resized_annots / f"{stem}.txt"
        orig_ann = args.orig_annots / f"{stem}.txt"

        if not all(p.exists() for p in [orig_img, orig_ann, res_ann]):
            logger.warning(f"  Skipping {stem}, files are missing")
            continue

        # Checking image size
        img = cv2.imread(str(res_img))
        if (img is None or
                img.shape[1] != args.width or
                img.shape[0] != args.height):
            err_size += 1
            logger.error(f"  Incorrect image size for {stem}, expected {args.width}x{args.height}")
            continue

        logger.info(f"  Image size is correct {args.width}x{args.height}")

        orig_img_data = cv2.imread(str(orig_img))
        if orig_img_data is None:
            continue

        h, w = orig_img_data.shape[:2]
        sx = args.width / w
        sy = args.height / h

        logger.info(f"  Factors X={sx:.4f} Y={sy:.4f}")

        try:
            with open(orig_ann, encoding="utf-8") as f_orig, \
                    open(res_ann, encoding="utf-8") as f_res:

                orig_lines = f_orig.readlines()
                res_lines = f_res.readlines()
        except Exception as e:
            logger.error(f"  Error reading annotations for {stem} {e}")
            err_bbox += 1
            continue

        if len(orig_lines) != len(res_lines):
            err_bbox += 1
            logger.error(f"  Mismatched number of objects in annotations {stem} {len(orig_lines)} vs {len(res_lines)}")
            continue

        logger.info(f"  Checking {len(orig_lines)} objects in the annotation")

        bbox_err = False
        for j, (line_orig, line_res) in enumerate(zip(orig_lines, res_lines), 1):
            try:
                _, bbox_orig = parse_kitti(line_orig)
                _, bbox_res = parse_kitti(line_res)
            except ValueError as e:
                logger.error(f"  Error parsing object {j} in {stem} {e}")
                bbox_err = True
                break

            if not check_bbox(bbox_orig, bbox_res, sx, sy, args.tolerance):
                logger.error(f"  Incorrect object scaling {j} in {stem}")
                logger.error(
                    f"    Original [{bbox_orig[0]:.1f},{bbox_orig[1]:.1f},{bbox_orig[2]:.1f},{bbox_orig[3]:.1f}]")
                logger.error(f"    Result [{bbox_res[0]:.1f},{bbox_res[1]:.1f},{bbox_res[2]:.1f},{bbox_res[3]:.1f}]")
                bbox_err = True
                break

        if bbox_err:
            err_bbox += 1
        else:
            logger.info(f"  All objects in {stem} are scaled correctly")

    # Validation results
    if err_size == 0 and err_bbox == 0:
        logger.info("Validation completed successfully")
        sys.exit(0)
    else:
        logger.error(f"Validation failed: size errors = {err_size}, bbox errors={err_bbox}")
        sys.exit(1)


class LogHandler(logging.Handler):
    """Log handler for GUI simplified version for Windows"""

    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.enabled = True

    def emit(self, record):
        """Sends log to widget"""
        if not self.enabled:
            return

        try:
            msg = self.format(record) + '\n'
            # Simplify for Windows – remove after and make direct call
            if hasattr(self.widget, 'winfo_exists') and self.widget.winfo_exists():
                self._add_log(msg, record.levelname)
        except Exception:
            # Disable on first error to avoid spamming
            self.enabled = False
            pass

    def _add_log(self, msg: str, level: str):
        """Adds message to the widget"""
        try:
            self.widget.config(state='normal')

            colors = {
                'ERROR': 'red',
                'WARNING': 'orange',
                'INFO': 'black',
                'DEBUG': 'gray'
            }
            color = colors.get(level, 'black')

            self.widget.insert(tk.END, msg, level.lower())
            self.widget.tag_config(level.lower(), foreground=color)

            # Limit the number of lines to prevent slowdown
            lines = self.widget.get('1.0', tk.END).count('\n')
            if lines > 1000:
                self.widget.delete('1.0', '100.0')

            self.widget.see(tk.END)
            self.widget.config(state='disabled')

            # Force refresh for Windows
            self.widget.update_idletasks()
        except Exception:
            self.enabled = False
            pass


class GUI:
    """Graphical interface for scaling"""

    def __init__(self, args: argparse.Namespace):
        if tk is None:
            raise RuntimeError("The tkinter library is not available")

        self.args = args
        self.total = 0
        self.running = False

        try:
            self.root = tk.Tk()
            self.root.title("KITTI Dataset Resizer")
            self.root.geometry("800x600")
            self.root.resizable(True, True)

            # Adding window close event handling
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

            self._setup_ui()
            self._setup_logs()
            self._load_defaults()

            # Force refresh gui after creation
            self.root.update_idletasks()

        except Exception as e:
            print(f"Error creating GUI {e}")
            raise RuntimeError(f"Can not create GUI {e}")

    def _on_closing(self):
        """Window close event handling"""
        self.running = False
        if hasattr(self, 'log_handler'):
            try:
                logger.removeHandler(self.log_handler)
            except:
                pass
        self.root.destroy()

    def _setup_ui(self):
        """Interface creation"""
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill='both', expand=True, padx=10, pady=10)

        # Settings tab
        self.settings_frame = ttk.Frame(self.nb)
        self.nb.add(self.settings_frame, text="Settings")
        self._setup_settings()

        # Logs tab
        self.logs_frame = ttk.Frame(self.nb)
        self.nb.add(self.logs_frame, text="Processing logs")
        self._setup_logs_tab()

    def _setup_settings(self):
        """Create settings tab"""
        # Removing canvas for Windows compatibility
        frame = ttk.Frame(self.settings_frame)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        # mode
        mode_group = ttk.LabelFrame(frame, text="Mode", padding=10)
        mode_group.pack(fill='x', padx=10, pady=5)

        self.mode_var = tk.StringVar()
        ttk.Radiobutton(mode_group, text="Scaling (Resize)",
                        variable=self.mode_var, value="resize").pack(anchor='w')
        ttk.Radiobutton(mode_group, text="Validate",
                        variable=self.mode_var, value="validate").pack(anchor='w')

        # Sizes
        size_group = ttk.LabelFrame(frame, text="Image size", padding=10)
        size_group.pack(fill='x', padx=10, pady=5)

        size_frame = ttk.Frame(size_group)
        size_frame.pack(fill='x')

        ttk.Label(size_frame, text="Width").grid(row=0, column=0, sticky='w', padx=(0, 5))
        self.width_var = tk.StringVar()
        ttk.Entry(size_frame, textvariable=self.width_var, width=10).grid(row=0, column=1, padx=(0, 15))

        ttk.Label(size_frame, text="Height").grid(row=0, column=2, sticky='w', padx=(0, 5))
        self.height_var = tk.StringVar()
        ttk.Entry(size_frame, textvariable=self.height_var, width=10).grid(row=0, column=3)

        # resize paths
        resize_group = ttk.LabelFrame(frame, text="Path for scaling mode", padding=10)
        resize_group.pack(fill='x', padx=10, pady=5)

        self._add_path_row(resize_group, 0, "Input images directory", "input_images_var")
        self._add_path_row(resize_group, 1, "Input annotations directory", "input_annots_var")
        self._add_path_row(resize_group, 2, "Output images directory", "output_images_var")
        self._add_path_row(resize_group, 3, "Output annotations directory", "output_annots_var")

        # validate paths
        validate_group = ttk.LabelFrame(frame, text="Paths for validation mode", padding=10)
        validate_group.pack(fill='x', padx=10, pady=5)

        self._add_path_row(validate_group, 0, "Original images folder", "orig_images_var")
        self._add_path_row(validate_group, 1, "Original annotation folder", "orig_annots_var")
        self._add_path_row(validate_group, 2, "Output images folder", "resized_images_var")
        self._add_path_row(validate_group, 3, "Original annotations folder", "resized_annots_var")

        # Error margin
        tol_frame = ttk.Frame(validate_group)
        tol_frame.grid(row=4, column=0, columnspan=3, sticky='ew', pady=(10, 0))

        ttk.Label(tol_frame, text="Error margin (pixels)").pack(side='left')
        self.tolerance_var = tk.StringVar()
        ttk.Entry(tol_frame, textvariable=self.tolerance_var, width=10).pack(side='left', padx=(10, 0))

        # Control
        ctrl_group = ttk.LabelFrame(frame, text="Control", padding=10)
        ctrl_group.pack(fill='x', padx=10, pady=5)

        # Progress
        self.progress = ttk.Progressbar(ctrl_group, length=400, mode="determinate")
        self.progress.pack(pady=5)

        # Status
        self.status = ttk.Label(ctrl_group, text="Ready to run")
        self.status.pack(pady=5)

        # buttons
        btn_frame = ttk.Frame(ctrl_group)
        btn_frame.pack(pady=10)

        self.start_btn = ttk.Button(btn_frame, text="Run",
                                    command=self.start_proc)
        self.start_btn.pack(side='left', padx=(0, 10))

        self.stop_btn = ttk.Button(btn_frame, text="Stop",
                                   command=self.stop_proc, state='disabled')
        self.stop_btn.pack(side='left', padx=(0, 10))

        ttk.Button(btn_frame, text="Reset to default",
                   command=self._load_defaults).pack(side='left')

    def _add_path_row(self, parent, row, label, var_name):
        """Creates a string with a path field"""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=(0, 10), pady=2)

        setattr(self, var_name, tk.StringVar())
        var = getattr(self, var_name)

        ttk.Entry(parent, textvariable=var, width=50).grid(row=row, column=1, sticky='ew', padx=(0, 10), pady=2)

        ttk.Button(parent, text="Overview",
                   command=lambda v=var: self._browse(v)).grid(row=row, column=2, pady=2)

        parent.grid_columnconfigure(1, weight=1)

    def _setup_logs_tab(self):
        """Create log tab"""
        header = ttk.Frame(self.logs_frame)
        header.pack(fill='x', padx=10, pady=5)

        ttk.Label(header, text="Real time log creation",
                  font=('TkDefaultFont', 10, 'bold')).pack(side='left')

        ttk.Button(header, text="Clear logs",
                   command=self._clear_logs).pack(side='right')

        logs_cont = ttk.Frame(self.logs_frame)
        logs_cont.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        self.logs_text = tk.Text(logs_cont, wrap='word', state='disabled',
                                 font=('Consolas', 9), bg='white', fg='black')

        scroll = ttk.Scrollbar(logs_cont, orient='vertical',
                               command=self.logs_text.yview)
        self.logs_text.configure(yscrollcommand=scroll.set)

        self.logs_text.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')

    def _setup_logs(self):
        """Configures logging in the GUI — simplified version for Windows"""
        # Simplifies logging setup for Windows compatibility
        try:
            self.log_handler = LogHandler(self.logs_text)
            self.log_handler.setLevel(logging.INFO)

            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            self.log_handler.setFormatter(fmt)

            logger.addHandler(self.log_handler)
        except Exception as e:
            # If logging in the GUI fails, continue without it
            print(f" Error {e} - GUI is not available")
            pass

    def _load_defaults(self):
        """Loads default values"""
        self.mode_var.set("resize")  # Default mode

        self.width_var.set("284")  # default sizes
        self.height_var.set("284")

        # default paths
        self.input_images_var.set("images")
        self.input_annots_var.set("kitti_annotations")
        self.output_images_var.set("images")
        self.output_annots_var.set("annotations")

        # paths for validate
        self.orig_images_var.set("images_orig")
        self.orig_annots_var.set("kitti_annotations")
        self.resized_images_var.set("images")
        self.resized_annots_var.set("annotations")

        self.tolerance_var.set("1.0")

        logger.info("Default paths have been processed")

    def _browse(self, var):
        """Opens a folder selection window"""
        from tkinter import filedialog

        folder = filedialog.askdirectory(
            title="Choose a folder",
            initialdir=var.get() if var.get() else "."
        )

        if folder:
            var.set(folder)

    def _clear_logs(self):
        """Clear log"""
        self.logs_text.config(state='normal')
        self.logs_text.delete(1.0, tk.END)
        self.logs_text.config(state='disabled')

    def _get_args(self) -> argparse.Namespace:
        """Creates an argument object from GUI settings"""
        args = argparse.Namespace()

        args.mode = self.mode_var.get()

        try:
            args.width = int(self.width_var.get())
            args.height = int(self.height_var.get())
        except ValueError:
            args.width = 284
            args.height = 284
            logger.warning("Invalid dimensions. Using default values 284x284")

        # resize paths
        args.input_images = Path(self.input_images_var.get())
        args.input_annots = Path(self.input_annots_var.get())
        args.output_images = Path(self.output_images_var.get())
        args.output_annots = Path(self.output_annots_var.get())

        # validate paths
        args.orig_images = Path(self.orig_images_var.get())
        args.orig_annots = Path(self.orig_annots_var.get())
        args.resized_images = Path(self.resized_images_var.get())
        args.resized_annots = Path(self.resized_annots_var.get())

        try:
            args.tolerance = float(self.tolerance_var.get())
        except ValueError:
            args.tolerance = 1.0
            logger.warning("Invalid error margin — using default value 1.0")

        args.nogui = False

        return args

    def update_progress(self, cur: int, total: int):
        """Updates progress"""
        if not self.running:
            return

        if self.total != total:
            self.progress["maximum"] = total
            self.total = total

        self.progress["value"] = cur

        pct = (cur / total * 100) if total > 0 else 0
        self.status["text"] = f"Processed {cur}/{total} ({pct:.1f}%)"

        self.root.update_idletasks()

    def start_proc(self):
        """Run processing"""
        if self.running:
            messagebox.showwarning("Caution", "Processing is already in progress")
            return

        args = self._get_args()

        logger.info("Launch processing with settings")
        logger.info(f" {args.mode} mode")
        logger.info(f" {args.width}x{args.height} mode")

        self.running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')

        self.thread = threading.Thread(target=self.run_proc, args=(args,), daemon=True)
        self.thread.start()

    def stop_proc(self):
        """Stops processing"""
        self.running = False
        logger.info("Processing signal has been stopped by user")

        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status["text"] = "`stopped by user`"

    def run_proc(self, args: argparse.Namespace):
        """Starts processing"""
        try:
            if args.mode == "validate":
                validate_all(args)

                self.root.after(0, lambda: messagebox.showinfo(
                    "Validation", "Validation has been completed"
                ))

            else:
                resize_all(args, self.update_progress)

                self.root.after(0, lambda: messagebox.showinfo(
                    "Done", "Scaling has been completed"
                ))

        except Exception as e:
            logger.exception("Error during processing")

            err_msg = f"An error {e} has occured"
            self.root.after(0, lambda: messagebox.showerror("Error", err_msg))

        finally:
            self.root.after(0, self._reset_ui)

    def _reset_ui(self):
        """Resets UI status"""
        self.running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')

        if not hasattr(self, '_stopped'):
            self.status["text"] = "Ready to launch"

    def run(self):
        """Runs GUI"""
        try:
            logger.info("=== KITTI Dataset Resizer ===")
            logger.info("Ready to work. Configure the settings and click Start Processing.")

            self.root.mainloop()

        except Exception as e:
            print(f"Error {e} during processing GUI ")
        finally:
            # Reset
            if hasattr(self, 'log_handler'):
                try:
                    logger.removeHandler(self.log_handler)
                except:
                    pass


def main():
    """Main function"""
    args = parse_args()

    logger.info("Launching KITTI Dataset Resizer")
    logger.info(f"Mode {args.mode}")

    if args.nogui or tk is None:
        logger.info("Launching console mode")

        if args.mode == "validate":
            validate_all(args)
        else:
            resize_all(args)
        return

    # Attempting to launch the GUI
    logger.info("GUI launch")
    try:
        gui = GUI(args)
        gui.run()
    except Exception as e:
        print(f"GUI launch error {e}")
        print("Automatically switching to console mode")
        logger.error(f"Error {e} starting GUI")
        logger.info("Switching to console mode")

        # Console mode
        if args.mode == "validate":
            validate_all(args)
        else:
            resize_all(args)


if __name__ == "__main__":
    main()
