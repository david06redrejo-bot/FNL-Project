import os
import subprocess
import sys
import urllib.request
import zipfile
import shutil

def download_tectonic(target_dir):
    """Downloads the Tectonic portable LaTeX engine for Windows."""
    print("Downloading portable LaTeX compiler (Tectonic)... This might take a minute.")
    url = "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-pc-windows-msvc.zip"
    zip_path = os.path.join(target_dir, "tectonic.zip")
    
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting compiler...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
        
    os.remove(zip_path)
    return os.path.join(target_dir, "tectonic.exe")

def compile_latex_report():
    """
    Compiles the Report.tex file into Report.pdf.
    It first tries 'pdflatex'. If not found, it downloads and uses 'tectonic'.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    tex_file = "Report.tex"
    base_name = "Report"
    
    if not os.path.exists(tex_file):
        print(f"Error: Could not find {tex_file} in {script_dir}")
        sys.exit(1)
        
    print(f"Starting compilation of {tex_file}...")
    
    # Try pdflatex first
    try:
        subprocess.run(["pdflatex", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        has_pdflatex = True
    except (FileNotFoundError, subprocess.CalledProcessError):
        has_pdflatex = False

    if has_pdflatex:
        print("Using local 'pdflatex'...")
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_file], check=False)
        subprocess.run(["bibtex", base_name], check=False)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_file], check=False)
        result = subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_file], check=False)
        
        if result.returncode == 0:
            print(f"\n[SUCCESS] Report.pdf has been generated/overwritten.")
        else:
            print(f"\n[WARNING] Compilation finished with exit code {result.returncode}.")
    else:
        print("Local 'pdflatex' not found.")
        tectonic_exe = os.path.join(script_dir, "tectonic.exe")
        
        if not os.path.exists(tectonic_exe):
            tectonic_exe = download_tectonic(script_dir)
            
        print("\n--- Running Tectonic (Automated LaTeX Engine) ---")
        # Tectonic automatically handles bibtex and multiple passes
        result = subprocess.run([tectonic_exe, tex_file], check=False)
        
        if result.returncode == 0:
            print(f"\n[SUCCESS] Report.pdf has been generated/overwritten using Tectonic.")
        else:
            print(f"\n[ERROR] Tectonic compilation failed with exit code {result.returncode}.")

if __name__ == "__main__":
    compile_latex_report()
