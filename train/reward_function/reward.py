import re
from typing import Any, Dict, List
import os
import subprocess
import tempfile
import numpy as np
import cv2
from PIL import ImageOps
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError
from pdfCropMargins import crop
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import base64
from subprocess import CalledProcessError, DEVNULL, Popen, TimeoutExpired
import pymupdf
from pymupdf import EmptyFileError
from os import getpgid, killpg
from signal import SIGKILL


# ---------------------------------------------------------------------------
# Worker discovery
#
# All three subprocess workers (dreamsim_worker.py, pdf_text_worker.py,
# geometry_sim_batch_worker.py) live in the parent directory of this file
# (i.e. <repo>/train/). They are invoked through `python <worker>.py
# <input.json> <output.json>` so we just need their absolute path.
#
# DreamSim additionally needs its own conda env (because it pins a different
# torch/torchvision than the trainer's env). Set the DREAMSIM_ENV environment
# variable to that env's prefix, e.g.:
#   export DREAMSIM_ENV=/path/to/anaconda3/envs/dreamsim
# ---------------------------------------------------------------------------

_TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_DREAMSIM_WORKER = os.path.join(_TRAIN_DIR, "dreamsim_worker.py")
_PDF_TEXT_WORKER = os.path.join(_TRAIN_DIR, "pdf_text_worker.py")
_GEOMETRY_WORKER = os.path.join(_TRAIN_DIR, "geometry_sim_batch_worker.py")

_DEFAULT_DREAMSIM_ENV = os.environ.get("DREAMSIM_ENV", "")


def dreamsim_batch_reward(
    image_pairs: List[Dict[str, Any]],
    dreamsim_env_path: str = _DEFAULT_DREAMSIM_ENV,
    num_gpus: int = None
) -> List[float]:
    """
    Compute DreamSim similarity for a batch of image pairs using subprocess with temporary files.
    Supports multi-GPU parallel processing for faster computation.
    
    Args:
        image_pairs: List of dictionaries containing rendered_img_bytes and ground_truth_img
        dreamsim_env_path: Path to DreamSim virtual environment
        num_gpus: Number of GPUs to use (default: 4 if None)
    Returns:
        List[float]: DreamSim similarity scores in [0, 1], where higher is more similar
    """
    try:
        # Prepare batch input data
        batch_data = []
        for pair in image_pairs:
            rendered_img_bytes = pair["rendered_img_bytes"]
            ground_truth_img = pair["ground_truth_img"]
            
            # Encode images to base64 for transmission
            rendered_img_base64 = base64.b64encode(rendered_img_bytes).decode('utf-8')
            gt_img_bytes = ground_truth_img.get("bytes")
            ground_truth_img_base64 = base64.b64encode(gt_img_bytes).decode('utf-8')
            
            batch_data.append({
                "rendered_img_base64": rendered_img_base64,
                "ground_truth_img_base64": ground_truth_img_base64
            })
        
        # Prepare input data
        input_data = {
            "image_pairs": batch_data
        }
        
        worker_script_path = _DREAMSIM_WORKER

        # Construct the command to run in the DreamSim environment
        python_executable = os.path.join(dreamsim_env_path, "bin", "python")
        assert os.path.exists(python_executable), (
            f"DreamSim Python not found at {python_executable!r}. "
            "Set DREAMSIM_ENV (or pass dreamsim_env_path=) to the conda env "
            "where DreamSim is installed."
        )

        # Create temporary files for input and output
        input_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        
        input_file_path = input_file.name
        output_file_path = output_file.name
        print("input_file_path", input_file_path)
        print("output_file_path", output_file_path)
        # Write input data to temporary file
        json.dump(input_data, input_file)
        input_file.flush()  # Ensure data is written to disk
        input_file.close()  # Close the file but don't delete it yet
        
        
        try:
            # Build command with additional arguments for multi-GPU support
            cmd = [python_executable, worker_script_path, input_file_path, output_file_path]
            
            # Add optional arguments for GPU configuration
            if num_gpus is not None:
                cmd.extend(['--num_gpus', str(num_gpus)])
                                
            print(f"Running DreamSim subprocess: {' '.join(cmd)}")
            print(f"Number of image pairs: {len(image_pairs)}")
            # Check input data size
            input_size = os.path.getsize(input_file_path)
            print(f"Input file size: {input_size} bytes ({input_size/1024/1024:.2f} MB)")
            
            # Prepare environment variables for subprocess.
            # We *inherit* CUDA_VISIBLE_DEVICES from the trainer so the
            # DreamSim subprocess sees the same GPUs (e.g. when launched
            # through SLURM or a wrapper that pins specific cards). If the
            # caller did not set it, fall back to all 8 GPUs.
            env = os.environ.copy()
            if not env.get("CUDA_VISIBLE_DEVICES"):
                env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
            print(f"DreamSim CUDA_VISIBLE_DEVICES = {env['CUDA_VISIBLE_DEVICES']}")
            print("Starting DreamSim subprocess...")

            result = subprocess.run(
                cmd,
                timeout=600,  # 10 minute timeout for large batch processing
                env=env,
            )

            if result.returncode != 0:
                print(f"DreamSim subprocess failed with return code {result.returncode}")
                print(f"stderr: {result.stderr.decode('utf-8', errors='ignore')}")
                return [0.0] * len(image_pairs)
            
            # Read output from temporary file
            if os.path.exists(output_file_path):
                with open(output_file_path, 'r') as f:
                    output_data = json.load(f)
            else:
                print("Output file not found")
                return [0.0] * len(image_pairs)
            
            if output_data.get("success", False):
                results = output_data["results"]
                # Sort by index to maintain order
                results.sort(key=lambda x: x["index"])
                return [result["similarity"] for result in results]
            else:
                print(f"DreamSim computation failed: {output_data.get('error', 'Unknown error')}")
                return [0.0] * len(image_pairs)
                
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(input_file_path):
                    os.unlink(input_file_path)
                if os.path.exists(output_file_path):
                    os.unlink(output_file_path)
            except OSError:
                pass  # Ignore cleanup errors
        
    except subprocess.TimeoutExpired:
        print("DreamSim computation timed out")
        return [0.0] * len(image_pairs)
    except Exception as e:
        print(f"Unexpected error computing DreamSim reward: {e}")
        return [0.0] * len(image_pairs)

def pdf_text_batch_reward(
    pdf_pairs: List[Dict[str, Any]],
    num_processes: int = None
) -> List[float]:
    """
    Compute PDF text layout similarity for a batch of PDF pairs using subprocess.
    Uses PyMuPDF for direct text extraction from PDF and Hungarian algorithm for optimal text box matching.
    Supports multi-process parallel processing for faster computation.
    
    Args:
        pdf_pairs: List of dictionaries containing rendered_pdf_bytes and ground_truth_img
        num_processes: Number of processes to use (default: 48 if None)
    Returns:
        List[float]: Layout similarity scores in [0, 1], where higher is more similar
    """
    try:
        # Prepare batch input data
        batch_data = []
        for pair in pdf_pairs:
            rendered_pdf_bytes = pair["rendered_pdf_bytes"]
            ground_truth_img = pair["ground_truth_img"]
            
            # Encode PDF bytes to base64 for transmission
            rendered_pdf_base64 = base64.b64encode(rendered_pdf_bytes).decode('utf-8')
            
            # Encode ground truth image to base64
            gt_img_bytes = ground_truth_img.get("bytes")
            ground_truth_img_base64 = base64.b64encode(gt_img_bytes).decode('utf-8')
            
            batch_data.append({
                "rendered_pdf_base64": rendered_pdf_base64,
                "ground_truth_img_base64": ground_truth_img_base64
            })
        
        # Prepare input data
        input_data = {
            "image_pairs": batch_data
        }
        
        worker_script_path = _PDF_TEXT_WORKER
        
        # Use the system Python (no need for special environment for PDF processing)
        python_executable = "python"

        # Create temporary files for input and output
        input_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        
        input_file_path = input_file.name
        output_file_path = output_file.name
        print("PDF text extraction input_file_path", input_file_path)
        print("PDF text extraction output_file_path", output_file_path)
        
        # Write input data to temporary file
        json.dump(input_data, input_file)
        input_file.flush()  # Ensure data is written to disk
        input_file.close()  # Close the file but don't delete it yet
        
        

        try:
            # Build command with additional arguments for multi-process support
            cmd = [python_executable, worker_script_path, input_file_path, output_file_path]
            
            # Add optional arguments for process configuration
            if num_processes is not None:
                cmd.extend(['--num_processes', str(num_processes)])
                cmd.extend(['--skip_stage2', 'True'])
                                
            # Prepare environment variables for subprocess
            env = os.environ.copy()
            # PDF text extraction doesn't need CUDA
                
            
            # Run subprocess with timeout (increased for batch processing)
            print("Starting PDF text extraction subprocess...")
            print(f"Number of PDF pairs: {len(pdf_pairs)}")
            
            # Check input data size
            input_size = os.path.getsize(input_file_path)
            print(f"PDF text extraction input file size: {input_size} bytes ({input_size/1024/1024:.2f} MB)")
            
            result = subprocess.run(
                cmd,
                timeout=600,  # 10 minute timeout for PDF text processing
                env=env,  # Pass the modified environment
            )
            
            if result.returncode != 0:
                print(f"PDF text extraction subprocess failed with return code {result.returncode}")
                print(f"stderr: {result.stderr}")
                return [0.0] * len(pdf_pairs)
            
            # Read output from temporary file
            if os.path.exists(output_file_path):
                with open(output_file_path, 'r') as f:
                    output_data = json.load(f)
            else:
                print("PDF text extraction output file not found")
                return [0.0] * len(pdf_pairs)
            
            if output_data.get("success", False):
                results = output_data["results"]
                # Sort by index to maintain order
                results.sort(key=lambda x: x["index"])
                return [result["layout_score"] for result in results]
            else:
                print(f"PDF text extraction computation failed: {output_data.get('error', 'Unknown error')}")
                return [0.0] * len(pdf_pairs)
                
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(input_file_path):
                    os.unlink(input_file_path)
                if os.path.exists(output_file_path):
                    os.unlink(output_file_path)
            except OSError:
                pass  # Ignore cleanup errors
        
    except subprocess.TimeoutExpired:
        print("PDF text extraction computation timed out")
        return [0.0] * len(pdf_pairs)
    except Exception as e:
        print(f"Unexpected error computing PDF text extraction reward: {e}")
        return [0.0] * len(pdf_pairs)

def geometry_sim_batch_reward(
    pdf_pairs: List[Dict[str, Any]],
    num_processes: int = None
) -> List[float]:
    """
    Compute PDF geometry similarity for a batch of PDF pairs via subprocess.
    Uses a simplified geometric-element extraction + matching algorithm for
    fast similarity scoring, with multi-process parallelism.
    
    Args:
        pdf_pairs: List of dictionaries containing rendered_pdf_bytes and ground_truth_pdf_bytes
        num_processes: Number of processes to use (default: 48 if None)
    Returns:
        List[float]: Geometry similarity scores in [0, 1], where higher is more similar
    """
    try:
        # Prepare batch input data
        batch_data = []
        for pair in pdf_pairs:
            rendered_pdf_bytes = pair["rendered_pdf_bytes"]
            ground_truth_pdf_bytes = pair["ground_truth_pdf_bytes"]
            
            # Encode PDF bytes to base64 for transmission
            rendered_pdf_base64 = base64.b64encode(rendered_pdf_bytes).decode('utf-8')
            ground_truth_pdf_base64 = base64.b64encode(ground_truth_pdf_bytes).decode('utf-8')
            
            batch_data.append({
                "rendered_pdf_base64": rendered_pdf_base64,
                "ground_truth_pdf_base64": ground_truth_pdf_base64
            })
        
        # Prepare input data
        input_data = {
            "image_pairs": batch_data  # keep interface name consistent across workers
        }
        
        worker_script_path = _GEOMETRY_WORKER
        
        # Use the system Python (no need for special environment for geometry processing)
        python_executable = "python"

        # Create temporary files for input and output
        input_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        
        input_file_path = input_file.name
        output_file_path = output_file.name
        print("Geometry similarity input_file_path", input_file_path)
        print("Geometry similarity output_file_path", output_file_path)
        
        # Write input data to temporary file
        json.dump(input_data, input_file)
        input_file.flush()  # Ensure data is written to disk
        input_file.close()  # Close the file but don't delete it yet
        
        try:
            # Build command with additional arguments for multi-process support
            cmd = [python_executable, worker_script_path, input_file_path, output_file_path]
            
            # Add optional arguments for process configuration
            if num_processes is not None:
                cmd.extend(['--num_processes', str(num_processes)])
                                
            # Prepare environment variables for subprocess
            env = os.environ.copy()
            # Geometry similarity doesn't need CUDA
                
            # Run subprocess with timeout (increased for batch processing)
            print("Starting geometry similarity subprocess...")
            print(f"Number of PDF pairs: {len(pdf_pairs)}")
            
            # Check input data size
            input_size = os.path.getsize(input_file_path)
            print(f"Geometry similarity input file size: {input_size} bytes ({input_size/1024/1024:.2f} MB)")
            
            result = subprocess.run(
                cmd,
                timeout=600,  # 10 minute timeout for geometry processing
                env=env,  # Pass the modified environment
            )
            
            if result.returncode != 0:
                print(f"Geometry similarity subprocess failed with return code {result.returncode}")
                print(f"stderr: {result.stderr}")
                return [0.0] * len(pdf_pairs)
            
            # Read output from temporary file
            if os.path.exists(output_file_path):
                with open(output_file_path, 'r') as f:
                    output_data = json.load(f)
            else:
                print("Geometry similarity output file not found")
                return [0.0] * len(pdf_pairs)
            
            if output_data.get("success", False):
                results = output_data["results"]
                # Sort by index to maintain order
                results.sort(key=lambda x: x["index"])
                return [result["geometry_similarity"] for result in results]
            else:
                print(f"Geometry similarity computation failed: {output_data.get('error', 'Unknown error')}")
                return [0.0] * len(pdf_pairs)
                
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(input_file_path):
                    os.unlink(input_file_path)
                if os.path.exists(output_file_path):
                    os.unlink(output_file_path)
            except OSError:
                pass  # Ignore cleanup errors
        
    except subprocess.TimeoutExpired:
        print("Geometry similarity computation timed out")
        return [0.0] * len(pdf_pairs)
    except Exception as e:
        print(f"Unexpected error computing geometry similarity reward: {e}")
        return [0.0] * len(pdf_pairs)

def normalize_img(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mean = img.mean()
    std = img.std() if img.std() > 1e-6 else 1.0
    return (img - mean) / std

def canny_edge(img: np.ndarray) -> np.ndarray:
    edges = cv2.Canny((img * 255).astype(np.uint8), 100, 200)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.GaussianBlur(edges, (13, 13), 0)
    return edges

def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def run(*popenargs, timeout=None, **kwargs):
    with Popen(*popenargs, start_new_session=True, **kwargs) as p:
        try:
            stdout, stderr = p.communicate(timeout=timeout)
        except TimeoutExpired:
            killpg(getpgid(p.pid), SIGKILL)
            p.wait()
            raise
        except Exception:
            killpg(getpgid(p.pid), SIGKILL)
            raise
        if retcode := p.poll():
            raise CalledProcessError(retcode, p.args, output=stdout, stderr=stderr)


def _run_tikz_compilation(
    code: str, timeout: int = 20, output_image_path: str = None, output_pdf_path: str = None, size: int=384
) -> bool:
    try:
        if not code or not code.strip():
            print("Error: Empty or invalid LaTeX code provided.")
            return False
            
        codelines = code.split("\n")
        # make sure we don't have page numbers in compiled pdf (for cropping)
        codelines.insert(1, r"{cmd}\AtBeginDocument{{{cmd}}}".format(cmd=r"\thispagestyle{empty}\pagestyle{empty}"))

        def try_compile(file):
            try:
                open(f"{file}.bbl", 'a').close() # some classes expect a bibfile
            except Exception as e:
                print(f"Warning: Could not create .bbl file: {e}")
                
            for engine in ["pdflatex"]: # could also try: https://tex.stackexchange.com/a/495999
                try:
                    run(
                        args=["latexmk", "-nobibtex", "-norc", "-interaction=nonstopmode", f"-{engine}", file],
                        cwd=tmpdir,
                        stdout=DEVNULL,
                        stderr=DEVNULL,
                        timeout=timeout
                    )
                    return f"{file}.pdf"
                except CalledProcessError as e:
                    # ignore common error print for short logs
                    # print(f"Warning: {engine} compilation failed: {e}")
                    continue
                except TimeoutExpired:
                    print(f"Warning: {engine} compilation timed out after {timeout} seconds")
                    continue
                except Exception as e:
                    print(f"Warning: Unexpected error during {engine} compilation: {e}")
                    continue
            # print("Error: Couldn't compile latex source with any engine.")
            return False

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            tex_path = os.path.join(tmpdir, "temp.tex")

            try:
                with open(tex_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(codelines))
            except Exception as e:
                print(f"Error: Could not write LaTeX file: {e}")
                return False

            pdfname = try_compile(tex_path.replace(".tex", ""))
            if not pdfname:
                return False

            try:
                doc = pymupdf.open(pdfname)
                doc.select([len(doc)-1])
                doc.saveIncr()
            except EmptyFileError:
                print("Error: Generated PDF is empty.")
                return False
            except Exception as e:
                print(f"Error: Could not process PDF with PyMuPDF: {e}")
                return False

            # Step 4: Crop PDF
            cropname = tex_path.replace(".tex", "-cropped.pdf")
            try:
                crop(["-c", "gb", "-p", "0", "-a", "-1", "-o", cropname, pdfname], quiet=True)
            except Exception as e:
                print(f"Error: Could not crop PDF: {e}")
                return False
            
            # Step 5: Convert PDF to PNG
            try:
                image = convert_from_path(cropname, size=size, single_file=True)[0]
                image = ImageOps.pad(image, (size, size), color='white')
            except PDFPageCountError:
                print("Error: PDF has no pages or invalid page count.")
                return False
            except Exception as e:
                print(f"Error: Could not convert PDF to image: {e}")
                return False


            if image.getcolors(1) is not None:
                print("Warning: Provided code compiled to an empty image.")
                return False
            
            # Save the image
            if output_image_path:
                image.save(output_image_path, "PNG")
            
            # Also save the cropped PDF if path is provided
            if output_pdf_path:
                try:
                    import shutil
                    shutil.copy2(cropname, output_pdf_path)
                except Exception as e:
                    print(f"Warning: Could not save PDF file: {e}")

        return True
        
    except Exception as e:
        print(f"Unexpected error in _run_tikz_compilation: {e}")
        return False



def load_ground_truth_img(ground_truth_img: Dict[str, Any]) -> np.ndarray:
    """
    Load ground truth image from byte data as grayscale float32 numpy array.
    """
    img_bytes = ground_truth_img.get("bytes")
    if img_bytes is None:
        raise ValueError("Missing image bytes in ground_truth_img")

    # Convert bytes to numpy array and decode
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Failed to decode image from bytes")

    return img.astype(np.float32) / 255.0

def reconstruction_reward(
    rendered_img: np.ndarray,
    ground_truth_img: dict,
) -> dict:
    """
    Calculate both edge-aware and non-edge-aware MSE reconstruction scores
    
    Args:
        rendered_img: Rendered image (numpy array, float32, range [0,1])
        ground_truth_img: Ground truth image dictionary containing bytes data
    
    Returns:
        dict: Dictionary containing both 'edge_aware' and 'non_edge_aware' scores
    """
    if rendered_img is not None:
        try:
            gt_img = load_ground_truth_img(ground_truth_img)
        except Exception as e:
            print(f"Error loading ground truth image: {e}")
            return {"edge_aware": 0.0, "non_edge_aware": 0.0}
        
        pred_img = rendered_img
        if pred_img.shape != gt_img.shape:
            pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))
        
        # Calculate non-edge-aware MSE score
        pred_img_normal = normalize_img(pred_img.copy())
        gt_img_normal = normalize_img(gt_img.copy())
        l2_normal = np.mean((pred_img_normal - gt_img_normal) ** 2)
        non_edge_aware_score = 1 - l2_normal
        
        # Calculate edge-aware MSE score
        pred_img_edge = canny_edge(pred_img)
        gt_img_edge = canny_edge(gt_img)
        pred_img_edge = normalize_img(pred_img_edge)
        gt_img_edge = normalize_img(gt_img_edge)
        l2_edge = np.mean((pred_img_edge - gt_img_edge) ** 2)
        edge_aware_score = 1 - l2_edge
        
        return {
            "edge_aware": float(np.clip(edge_aware_score, -1, 1)),
            "non_edge_aware": float(np.clip(non_edge_aware_score, -1, 1))
        }
    
    return {"edge_aware": 0.0, "non_edge_aware": 0.0}



def _process_compilation(args):
    """First stage: LaTeX compilation (CPU-intensive, use multiprocessing)"""
    reward_input, i = args
    response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
    
    # Format score
    format_score = format_reward(response)
    
    # Extract LaTeX code
    latex_code = None
    patterns = [
        r'```latex\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'<answer>(.*?)</answer>',
    ]
    for pattern in patterns:
        latex_match = re.search(pattern, response, re.DOTALL)
        if latex_match:
            latex_code = latex_match.group(1).strip()
            break
    if latex_code is None:
        latex_code = response

    # Compilation score
    compiled = False
    compilation_score = 0
    rendered_img_bytes = None
    rendered_pdf_bytes = None
    
    if latex_code is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, f"pred_{i}.png")
            pdf_path = os.path.join(tmpdir, f"pred_{i}.pdf")
            compiled = _run_tikz_compilation(code=latex_code, output_image_path=img_path, output_pdf_path=pdf_path)
            compilation_score = 1 if compiled else 0
            
            # If compiled successfully, read both image and PDF as bytes
            if compiled:
                if img_path and os.path.exists(img_path):
                    with open(img_path, 'rb') as f:
                        rendered_img_bytes = f.read()
                
                # Read PDF file as bytes directly
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, 'rb') as f:
                        rendered_pdf_bytes = f.read()

    return {
        "index": i,
        "format_score": format_score,
        "compilation_score": compilation_score,
        "compiled": compiled,
        "rendered_img_bytes": rendered_img_bytes,
        "rendered_pdf_bytes": rendered_pdf_bytes,
        "ground_truth_img": reward_input["multi_modal_data"]["images"][0]
    }


def _process_reconstruction(args):
    """Second stage: Reconstruction score calculation (CPU-intensive)"""
    compilation_result = args
    
    # Initialize reconstruction scores
    reconstruction_scores = {"edge_aware": -1, "non_edge_aware": -1}
    
    if compilation_result["compiled"]:
        assert compilation_result["rendered_img_bytes"] is not None
        # Convert bytes to numpy array
        img_array = np.frombuffer(compilation_result["rendered_img_bytes"], dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            rendered_img = img.astype(np.float32) / 255.0
            reconstruction_scores = reconstruction_reward(
                rendered_img, compilation_result["ground_truth_img"]
            )

    return {
        "index": compilation_result["index"],
        "reconstruction_edge_aware": reconstruction_scores["edge_aware"],
        "reconstruction_non_edge_aware": reconstruction_scores["non_edge_aware"],
    }





def _process_dreamsim_batch(compilation_results, dreamsim_env_path, num_gpus=None):
    """Third stage: DreamSim inference (GPU-intensive, use batch processing)"""
    # Initialize all scores to 0.0 (default for compilation failures)
    result_scores = [0.0] * len(compilation_results)
    
    # Collect all valid image pairs for batch processing
    image_pairs = []
    valid_indices = []
    
    for i, compilation_result in enumerate(compilation_results):
        # Only process images that compiled successfully
        if compilation_result["compiled"]:
            image_pairs.append({
                "rendered_img_bytes": compilation_result["rendered_img_bytes"],
                "ground_truth_img": compilation_result["ground_truth_img"]
            })
            valid_indices.append(i)
        # For compilation failures, result_scores[i] remains 0.0
    
    # If no valid images, return all zeros (all compilation failures)
    if not image_pairs:
        print("No successfully compiled images found for DreamSim processing")
        return result_scores
    
    # Process batch for successfully compiled images only with GPU configuration
    dreamsim_scores = dreamsim_batch_reward(
        image_pairs, 
        dreamsim_env_path, 
        num_gpus=num_gpus
    )
    
    # Map results back to original indices
    for idx, score in zip(valid_indices, dreamsim_scores):
        result_scores[idx] = score
    
    return result_scores

def _process_pdf_text_batch(compilation_results, num_processes=None):
    """Fifth stage: PDF text layout analysis (text detection using DIoU)"""
    # Initialize all scores to 0.0 (default for compilation failures)
    result_scores = [0.0] * len(compilation_results)
    
    # Collect all valid PDF files for batch processing
    pdf_pairs = []
    valid_indices = []
    
    for i, compilation_result in enumerate(compilation_results):
        # Only process PDFs that compiled successfully and have PDF bytes
        if compilation_result["compiled"] and compilation_result.get("rendered_pdf_bytes"):
            pdf_pairs.append({
                "rendered_pdf_bytes": compilation_result["rendered_pdf_bytes"],
                "ground_truth_img": compilation_result["ground_truth_img"]
            })
            valid_indices.append(i)
        # For compilation failures, result_scores[i] remains 0.0
    
    # If no valid PDFs, return all zeros (all compilation failures)
    if not pdf_pairs:
        print("No successfully compiled PDFs found for PDF text processing")
        return result_scores
    
    # Process batch for successfully compiled PDFs only with multi-process configuration
    pdf_text_scores = pdf_text_batch_reward(
        pdf_pairs, 
        num_processes=num_processes
    )
    
    # Map results back to original indices
    for idx, score in zip(valid_indices, pdf_text_scores):
        result_scores[idx] = score
    
    return result_scores

def _process_geometry_similarity_batch(compilation_results, num_processes=None):
    """Stage 5: PDF geometry similarity analysis (element extraction + matching)."""
    # Initialize all scores to 0.0 (default for compilation failures)
    result_scores = [0.0] * len(compilation_results)
    
    # Collect all valid PDF files for batch processing
    pdf_pairs = []
    valid_indices = []
    
    for i, compilation_result in enumerate(compilation_results):
        # Only process PDFs that compiled successfully and have PDF bytes for both predicted and ground truth
        if (compilation_result["compiled"] and 
            compilation_result.get("rendered_pdf_bytes") and 
            compilation_result.get("ground_truth_pdf_bytes")):
            pdf_pairs.append({
                "rendered_pdf_bytes": compilation_result["rendered_pdf_bytes"],
                "ground_truth_pdf_bytes": compilation_result["ground_truth_pdf_bytes"]
            })
            valid_indices.append(i)
        # For compilation failures, result_scores[i] remains 0.0
    
    # If no valid PDFs, return all zeros (all compilation failures)
    if not pdf_pairs:
        print("No successfully compiled PDFs found for geometry similarity processing")
        return result_scores
    
    # Process batch for successfully compiled PDFs only with multi-process configuration
    geometry_sim_scores = geometry_sim_batch_reward(
        pdf_pairs, 
        num_processes=num_processes
    )
    
    # Map results back to original indices
    for idx, score in zip(valid_indices, geometry_sim_scores):
        result_scores[idx] = score
    
    return result_scores

def compute_score(
    reward_inputs: List[Dict[str, Any]],
    max_workers: int = 48,
    dreamsim_env_path: str = _DEFAULT_DREAMSIM_ENV,
    dreamsim_num_gpus: int = 8,
    pdf_text_num_processes: int = 48,
    geometry_sim_num_processes: int = 48,
) -> List[Dict[str, float]]:
    assert isinstance(reward_inputs, list), "Please use `reward_type=batch` for tikz reward function."
    
    assert os.path.exists(dreamsim_env_path), "DreamSim environment doesn't exist."

    # Stage 1: LaTeX compilation using multiprocessing (CPU-intensive)
    print("Stage 1: LaTeX compilation with multiprocessing...")
    compilation_args_list = [(copy.deepcopy(r), i) for i, r in enumerate(reward_inputs)]
    compilation_results = [None] * len(compilation_args_list)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_process_compilation, args): i
            for i, args in enumerate(compilation_args_list)
        }

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
                compilation_results[idx] = result
            except Exception as e:
                print(f"Error in compilation stage for index {idx}: {e}")
                compilation_results[idx] = {
                    "index": idx,
                    "format_score": 0.0,
                    "compilation_score": 0.0,
                    "compiled": False,
                    "rendered_img_bytes": None,
                    "rendered_pdf_bytes": None,
                    "ground_truth_img": reward_inputs[idx]["multi_modal_data"]["images"][0],
                    "ground_truth_pdf_bytes": None,
                }

    # Stage 2: Reconstruction score calculation using multiprocessing (CPU-intensive)
    print("Stage 2: Reconstruction score calculation with multiprocessing...")
    reconstruction_args_list = [compilation_results[i] for i in range(len(compilation_results))]
    reconstruction_results = [None] * len(reconstruction_args_list)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_process_reconstruction, args): i
            for i, args in enumerate(reconstruction_args_list)
        }

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
                reconstruction_results[idx] = result
            except Exception as e:
                print(f"Error in reconstruction stage for index {idx}: {e}")
                reconstruction_results[idx] = {
                    "index": idx,
                    "reconstruction_edge_aware": -1.0,
                    "reconstruction_non_edge_aware": -1.0,
                }

    # Stage 3: DreamSim inference using batch processing (GPU-intensive)
    print("Stage 3: DreamSim inference with batch processing...")

    try:
        dreamsim_scores = _process_dreamsim_batch(
            compilation_results, 
            dreamsim_env_path, 
            num_gpus=dreamsim_num_gpus
        )
    except Exception as e:
        print(f"Error in DreamSim batch processing: {e}")
        dreamsim_scores = [0.0] * len(compilation_results)

    # Stage 4: PDF text layout analysis (text detection)
    print("Stage 4: PDF text layout analysis...")

    try:
        pdf_text_scores = _process_pdf_text_batch(
            compilation_results, 
            num_processes=pdf_text_num_processes
        )
    except Exception as e:
        print(f"Error in PDF text batch processing: {e}")
        pdf_text_scores = [0.0] * len(compilation_results)

    # Stage 5: Geometry similarity analysis (geometry elements extraction and matching)
    print("Stage 5: Geometry similarity analysis...")

    try:
        geometry_sim_scores = _process_geometry_similarity_batch(
            compilation_results, 
            num_processes=geometry_sim_num_processes
        )
    except Exception as e:
        print(f"Error in geometry similarity batch processing: {e}")
        geometry_sim_scores = [0.0] * len(compilation_results)
    

    # Weights configuration: MSE (reconstruction), DSIM (dreamsim),
    # Layout IoU (PDF text), Geometry similarity.
    format_weight = 0
    compilation_weight = 0
    
    # MSE weights - set one to 0.5 and the other to 0 based on preference
    reconstruction_edge_aware_weight = 0    # Edge-aware MSE weight
    reconstruction_non_edge_aware_weight = 0.5  # Non-edge-aware MSE weight
    
    dreamsim_weight = 0.5      # DSIM weight
    pdf_text_weight = 0.5      # Layout IoU weight
    geometry_sim_weight = 0.5   # Geometry similarity weight

    # Combine all results
    scores = []
    for i in range(len(compilation_results)):
        format_score = compilation_results[i]["format_score"]
        compilation_score = compilation_results[i]["compilation_score"]
        reconstruction_edge_aware_score = reconstruction_results[i]["reconstruction_edge_aware"]
        reconstruction_non_edge_aware_score = reconstruction_results[i]["reconstruction_non_edge_aware"]
        dreamsim_score = dreamsim_scores[i]
        pdf_text_score = pdf_text_scores[i]
        geometry_sim_score = geometry_sim_scores[i]
        
        overall_score = (format_weight * format_score + 
                        compilation_weight * compilation_score + 
                        reconstruction_edge_aware_weight * reconstruction_edge_aware_score + 
                        reconstruction_non_edge_aware_weight * reconstruction_non_edge_aware_score +
                        dreamsim_weight * dreamsim_score +
                        pdf_text_weight * pdf_text_score +
                        geometry_sim_weight * geometry_sim_score)
        
        scores.append({
            "overall": overall_score,
            "format": format_score,
            "compilation": compilation_score,
            "reconstruction": reconstruction_edge_aware_score,
            "reconstruction_non_edge_aware": reconstruction_non_edge_aware_score,
            "dreamsim": dreamsim_score,
            "pdf_text": pdf_text_score,
            "geometry_sim": geometry_sim_score,
        })

    return scores