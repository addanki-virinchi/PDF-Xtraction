# Deploy Document

This document describes how to deploy the PDF Extraction app on a Linux VPS
(DigitalOcean, Hostinger, or similar). The steps use Gunicorn + Nginx.

## 1) Server Requirements

- Ubuntu 22.04+ (or similar)
- Python 3.10+ and pip
- 4+ CPU cores and 16 GB RAM recommended for larger models
- 50+ GB disk if you plan to download large models

## 2) Copy the Project to the Server

Upload the project folder to your server (SCP/SFTP/Zip).

Example with SCP:

```bash
scp -r PDF_Extraction user@your-server-ip:/opt/pdf_extraction
```

## 3) Create a Virtual Environment

```bash
cd /opt/pdf_extraction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 4) Configure Environment Variables

Create an env file (example name: `.env`) or export variables directly.

Minimal example:

```bash
export MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
export MAX_NEW_TOKENS="4096"
export PDF_DPI="100"
export PDF_MAX_DIMENSION="1024"
export CPU_THREADS="4"
export SECRET_KEY="change-me"
```

Optional cache/temp paths (recommended for large models):

```bash
export TEMP_DIR="/opt/pdf_extraction/tmp"
export HF_HOME="/opt/pdf_extraction/hf_cache"
```

## 5) Test the App Locally on the Server

```bash
source venv/bin/activate
python app.py
```

Visit:
`http://your-server-ip:5000`

## 6) Run with Gunicorn

Install Gunicorn (if not already in requirements):

```bash
pip install gunicorn
```

Start:

```bash
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

Adjust workers based on CPU/RAM. 2 workers is a safe default.

## 7) Nginx Reverse Proxy

Install Nginx:

```bash
sudo apt update
sudo apt install nginx
```

Create `/etc/nginx/sites-available/pdf_extraction`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/pdf_extraction /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 8) Systemd Service

Create `/etc/systemd/system/pdf_extraction.service`:

```ini
[Unit]
Description=PDF Extraction App
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/pdf_extraction
Environment="MODEL_NAME=Qwen/Qwen2-VL-2B-Instruct"
Environment="MAX_NEW_TOKENS=4096"
Environment="PDF_DPI=100"
Environment="PDF_MAX_DIMENSION=1024"
Environment="CPU_THREADS=4"
Environment="SECRET_KEY=change-me"
Environment="TEMP_DIR=/opt/pdf_extraction/tmp"
Environment="HF_HOME=/opt/pdf_extraction/hf_cache"
ExecStart=/opt/pdf_extraction/venv/bin/gunicorn -w 2 -b 127.0.0.1:8000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable pdf_extraction
sudo systemctl start pdf_extraction
sudo systemctl status pdf_extraction
```

## 9) TLS (Optional)

If you have a domain, use Certbot:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## 10) Common Issues

- Out of memory: use a smaller model (e.g. `Qwen/Qwen2-VL-2B-Instruct`)
- Slow CPU: reduce `MAX_NEW_TOKENS` and `PDF_MAX_DIMENSION`
- 413 Request Entity Too Large: increase `client_max_body_size` in Nginx

## 11) Hostinger Notes

If you are using a shared hosting plan, you likely cannot run Gunicorn.
Use a VPS plan instead and follow the same steps above.
