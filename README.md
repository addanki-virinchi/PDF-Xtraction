# 📄 PDF Data Extractor

An AI-powered application that extracts structured data from PDF documents using Qwen Vision-Language models. Upload a PDF and an Excel template defining the fields you want to extract, and the application will populate the Excel file with the extracted data.

## 🌟 Features

- **Vision-based extraction**: Process PDFs as images for accurate extraction from complex layouts
- **Excel template support**: Define extraction fields using Excel column headers
- **Multiple model support**: Choose from Qwen2-VL (2B, 7B) or Qwen2.5-VL models
- **CPU and GPU support**: Optimized for both CPU-only systems and GPU acceleration
- **Web interface**: Simple browser-based UI for uploading files and downloading results
- **Configurable performance**: Tune image quality, token limits, and threading for your hardware

## 💻 System Requirements

### Minimum Requirements (Qwen2-VL-2B)

| Component | Requirement |
|-----------|-------------|
| RAM | 16GB |
| Storage | 10GB free space |
| Python | 3.10+ |
| OS | Windows 10/11, Linux, macOS |

### Recommended for Larger Models

| Model | RAM Required | GPU VRAM | Download Size |
|-------|--------------|----------|---------------|
| Qwen2-VL-2B-Instruct | 5-6 GB | Optional | ~4.5 GB |
| Qwen2-VL-7B-Instruct | 14-16 GB | 16GB+ recommended | ~15 GB |
| Qwen2.5-VL-3B-Instruct | 8-10 GB | Optional | ~7.5 GB |
| Qwen2.5-VL-7B-Instruct | 16-17 GB | 16GB+ recommended | ~16 GB |

### Additional Requirements

- **Poppler**: Required for PDF to image conversion
  - Windows: Download from [poppler releases](https://github.com/osber98/poppler-windows/releases) and add to PATH
  - Linux: `sudo apt-get install poppler-utils`
  - macOS: `brew install poppler`

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/PDF_Extraction.git
cd PDF_Extraction
```

### 2. Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install Faster Downloads

```bash
pip install hf_transfer
```

This provides 3-5x faster model downloads from Hugging Face.

## ⚙️ Configuration

### Environment Variables

All settings can be configured via environment variables. Here are the available options:

#### Model Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen3-VL-4B-Instruct` | Hugging Face model ID |
| `MAX_NEW_TOKENS` | `2048` | Maximum tokens to generate (lower = faster) |
| `USE_FLASH_ATTENTION` | `false` | Enable Flash Attention 2 (GPU only) |
| `QUANTIZATION_MODE` | `none` | Quantization: `none`, `4bit`, `8bit` (GPU only) |

#### Cache and Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HOME` | `D:\huggingface_cache` | Hugging Face cache directory |
| `TEMP_DIR` | `D:\temp\pdf_extraction` | Temporary files directory |

#### CPU Optimization

| Variable | Default | Description |
|----------|---------|-------------|
| `CPU_THREADS` | `4` | Number of CPU threads (0 = auto-detect half cores) |
| `PDF_DPI` | `100` | PDF rendering DPI (72-150, lower = faster) |
| `PDF_MAX_DIMENSION` | `800` | Max image dimension in pixels (640-2048) |
| `PDF_MAX_PAGES` | `0` | Max pages to process (0 = all pages) |

### Setting Environment Variables

**Windows (PowerShell) - Temporary:**
```powershell
$env:MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
$env:CPU_THREADS = "4"
$env:PDF_DPI = "100"
$env:MAX_NEW_TOKENS = "2048"
```

**Windows (PowerShell) - Permanent:**
```powershell
[Environment]::SetEnvironmentVariable("MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct", "User")
```

**Linux/macOS - Temporary:**
```bash
export MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
export CPU_THREADS="4"
export PDF_DPI="100"
```

**Linux/macOS - Permanent (add to ~/.bashrc or ~/.zshrc):**
```bash
echo 'export MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"' >> ~/.bashrc
source ~/.bashrc
```

### Recommended Settings by Hardware

#### 16GB RAM, CPU-only (Laptop/Desktop)
```powershell
$env:MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
$env:CPU_THREADS = "4"
$env:PDF_DPI = "100"
$env:PDF_MAX_DIMENSION = "800"
$env:MAX_NEW_TOKENS = "2048"
```

#### 32GB RAM, CPU-only
```powershell
$env:MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
$env:CPU_THREADS = "8"
$env:PDF_DPI = "120"
$env:PDF_MAX_DIMENSION = "1024"
$env:MAX_NEW_TOKENS = "4096"
```

#### GPU with 16GB+ VRAM
```powershell
$env:MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
$env:USE_FLASH_ATTENTION = "true"
$env:PDF_DPI = "150"
$env:PDF_MAX_DIMENSION = "1280"
$env:MAX_NEW_TOKENS = "8192"
```

## 📥 Pre-downloading the Model

To avoid download delays on first run, pre-download the model using the provided script:

```powershell
# Activate virtual environment
.\venv\Scripts\Activate  # Windows
source venv/bin/activate  # Linux/macOS

# Download default model (Qwen3-VL-4B-Instruct)
python download_model.py

# Download a specific model
python download_model.py --model Qwen/Qwen2.5-VL-3B-Instruct

# Verify existing download only (no new downloads)
python download_model.py --verify-only
```

### Expected Download Times

| Model | Size | Time (100 Mbps) | Time with hf_transfer |
|-------|------|-----------------|----------------------|
| Qwen2-VL-2B | ~4.5 GB | ~6 min | ~2 min |
| Qwen2.5-VL-3B | ~7.5 GB | ~10 min | ~3 min |
| Qwen2-VL-7B | ~15 GB | ~20 min | ~6 min |

## ▶️ Running the Application

### Start the Flask Server

```powershell
# Activate virtual environment
.\venv\Scripts\Activate  # Windows
source venv/bin/activate  # Linux/macOS

# Run the application
python app.py
```

The server will start on `http://localhost:5000`.

### Access the Web Interface

1. Open your browser and go to `http://localhost:5000`
2. Wait for the "AI Model Ready" status indicator
3. Upload your PDF document
4. Upload an Excel template with column headers defining the fields to extract
5. (Optional) Add custom instructions for the AI
6. Click "Extract Data"
7. The extracted data will be downloaded as an Excel file

### Using the API Directly

```bash
# Extract data and download Excel
curl -X POST http://localhost:5000/api/extract \
  -F "pdf_file=@document.pdf" \
  -F "excel_template=@template.xlsx" \
  -F "use_vision=true" \
  -o extracted_data.xlsx

# Get JSON response instead
curl -X POST http://localhost:5000/api/extract \
  -F "pdf_file=@document.pdf" \
  -F "excel_template=@template.xlsx" \
  -F "use_vision=true" \
  -F "return_json=true"

# Check model status
curl http://localhost:5000/api/model-status
```

## ⚡ Performance Optimization

### Processing Time Expectations

| Hardware | Model | 1-page PDF | 5-page PDF |
|----------|-------|------------|------------|
| 16GB RAM, CPU | Qwen2-VL-2B | 2-4 min | 8-15 min |
| 32GB RAM, CPU | Qwen2.5-VL-3B | 3-5 min | 10-20 min |
| GPU 16GB VRAM | Qwen2.5-VL-7B | 15-30 sec | 1-2 min |

### Speed vs Quality Trade-offs

| Setting | Faster Processing | Higher Quality |
|---------|-------------------|----------------|
| `PDF_DPI` | 72 | 150 |
| `PDF_MAX_DIMENSION` | 640 | 1280 |
| `MAX_NEW_TOKENS` | 1024 | 8192 |
| `CPU_THREADS` | 4 | 8 (if RAM allows) |

### Tips for Faster Processing

1. **Reduce image size**: Lower `PDF_DPI` and `PDF_MAX_DIMENSION`
2. **Limit pages**: Set `PDF_MAX_PAGES` to process only first N pages
3. **Reduce tokens**: Lower `MAX_NEW_TOKENS` if extractions are concise
4. **Use SSD**: Ensure cache directories are on SSD, not HDD

## 🔧 Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| "Model loading..." stuck | First load takes 5-15 min. Check RAM usage in Task Manager |
| Out of memory error | Use smaller model or reduce `PDF_MAX_DIMENSION` |
| System hangs during processing | Reduce `CPU_THREADS` to 4 or lower |
| Slow processing | Reduce `PDF_DPI`, `PDF_MAX_DIMENSION`, and `MAX_NEW_TOKENS` |
| "Poppler not found" error | Install Poppler and add to system PATH |
| JSON parsing error in browser | Clear browser cache and retry |
| Excel has data in single row | Model response format issue - check model output in JSON mode |

### Checking Model Status

```powershell
# Via API
curl http://localhost:5000/api/model-status

# Expected response when ready:
# {"is_ready": true, "state": "ready", "model_name": "Qwen/Qwen2-VL-2B-Instruct"}
```

### Clearing Cache

If you encounter issues, try clearing the Hugging Face cache:

```powershell
# Windows
Remove-Item -Recurse -Force D:\huggingface_cache\*

# Linux/macOS
rm -rf ~/.cache/huggingface/*
```

## 📁 Project Structure

```
PDF_Extraction/
├── app.py              # Flask application entry point
├── config.py           # Configuration settings and environment variables
├── download_model.py   # Script to pre-download models
├── excel_handler.py    # Excel reading/writing and data parsing
├── pdf_extractor.py    # PDF to image conversion
├── prompt_builder.py   # AI prompt construction
├── qwen_model.py       # Qwen model loading and inference
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Web interface
├── uploads/            # Temporary uploaded files
└── outputs/            # Generated Excel files
```

### Key Files

| File | Purpose |
|------|---------|
| `app.py` | Flask routes, file handling, orchestrates extraction |
| `config.py` | All configuration variables, environment setup |
| `qwen_model.py` | Model loading, inference, GPU/CPU handling |
| `prompt_builder.py` | Constructs prompts for the vision model |
| `excel_handler.py` | Parses model responses, writes Excel files |
| `pdf_extractor.py` | Converts PDF pages to images |

## 🚀 Deployment to RunPod

See the [RunPod SSH Setup Guide](#runpod-ssh-setup-guide) section below for deploying to RunPod with GPU support.

---

## RunPod SSH Setup Guide

### Prerequisites

- RunPod account with credits
- VS Code with "Remote - SSH" extension installed
- SSH client on your local machine

### Step 1: Create a RunPod Pod

1. Log in to [RunPod.io](https://runpod.io)
2. Click **"+ Deploy"** or **"Pods"** → **"+ New Pod"**
3. Select a GPU template:
   - For Qwen2.5-VL-7B: Choose 24GB+ VRAM (e.g., RTX 4090, A5000)
   - For Qwen2-VL-2B: 8GB+ VRAM is sufficient
4. Select a template:
   - Recommended: **"RunPod Pytorch 2.x"** or **"RunPod Ubuntu"**
5. Configure storage:
   - **Container Disk**: 20GB minimum
   - **Volume Disk**: 50GB+ (for model cache, persistent)
   - Mount path: `/workspace` (default)
6. Click **"Deploy On-Demand"** or **"Deploy Spot"**

### Step 2: Enable SSH Access

1. Once the pod is running, click on the pod name to open details
2. Look for **"Connect"** button or **"SSH"** section
3. Click **"Connect via SSH"** or find the SSH connection details:
   - You'll see something like:
     ```
     ssh root@<pod-ip> -p <port> -i ~/.ssh/id_rsa
     ```
   - Or a connection string like:
     ```
     ssh -p 22022 root@205.196.164.25
     ```
4. Copy the SSH connection command

### Step 3: Configure VS Code Remote SSH

1. Open VS Code
2. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS)
3. Type **"Remote-SSH: Open SSH Configuration File"**
4. Select your SSH config file (usually `~/.ssh/config`)
5. Add your RunPod connection:

```
Host runpod-gpu
    HostName 205.196.164.25
    User root
    Port 22022
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking no
```

Replace the `HostName` and `Port` with your actual RunPod details.

### Step 4: Connect to RunPod

1. Press `Ctrl+Shift+P` → **"Remote-SSH: Connect to Host"**
2. Select **"runpod-gpu"** from the list
3. Wait for VS Code to install the server on the remote machine
4. Once connected, you'll see "SSH: runpod-gpu" in the bottom-left corner

### Step 5: Set Up Development Environment

After connecting via SSH:

```bash
# Navigate to persistent storage
cd /workspace

# Clone your repository
git clone https://github.com/yourusername/PDF_Extraction.git
cd PDF_Extraction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up GPU-optimized environment
export MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
export HF_HOME="/workspace/huggingface_cache"
export USE_FLASH_ATTENTION="true"
export MAX_NEW_TOKENS="8192"

# Pre-download the model
python download_model.py
```

### Step 6: Port Forwarding for Flask

To access the Flask app from your local browser:

**Option A: VS Code Port Forwarding (Recommended)**

1. In VS Code, go to the **"Ports"** tab (bottom panel)
2. Click **"Forward a Port"**
3. Enter `5000` and press Enter
4. Access the app at `http://localhost:5000` in your local browser

**Option B: RunPod Expose HTTP Port**

1. In RunPod dashboard, click on your pod
2. Find **"HTTP Service Ports"** or **"Expose HTTP"**
3. Add port `5000`
4. RunPod will provide a public URL like `https://xxxxx-5000.proxy.runpod.net`

### Step 7: Run the Application

```bash
# In the VS Code terminal (connected to RunPod)
cd /workspace/PDF_Extraction
source venv/bin/activate
python app.py
```

### RunPod Tips

| Tip | Details |
|-----|---------|
| **Persistent Storage** | Store models and code in `/workspace` - it persists across restarts |
| **Spot Instances** | Cheaper but can be interrupted - save work frequently |
| **Templates** | Create a custom template with dependencies pre-installed |
| **Auto-shutdown** | Set idle timeout to avoid unexpected charges |
| **GPU Monitoring** | Use `nvidia-smi` to check GPU memory usage |

### Example RunPod Startup Script

Create `/workspace/start.sh` to quickly restart your environment:

```bash
#!/bin/bash
cd /workspace/PDF_Extraction
source venv/bin/activate
export MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
export HF_HOME="/workspace/huggingface_cache"
export USE_FLASH_ATTENTION="true"
python app.py
```

Make it executable: `chmod +x /workspace/start.sh`

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.
