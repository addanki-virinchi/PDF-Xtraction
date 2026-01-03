import subprocess, time, re, threading, sys

def stream_output(proc, prefix=""):
    for line in proc.stdout:
        print(prefix + line, end="")  # keep original formatting
        sys.stdout.flush()

# Start Flask app (unbuffered so prints appear immediately)
flask = subprocess.Popen(
    ["python", "-u", "app.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# Start a thread to stream Flask logs
threading.Thread(target=stream_output, args=(flask, "[FLASK] "), daemon=True).start()

time.sleep(3)

# Start Cloudflare tunnel
cf = subprocess.Popen(
    ["./cloudflared", "tunnel", "--url", "http://localhost:5000", "--no-autoupdate"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# Stream cloudflared logs + capture public URL
public_url = None
for line in cf.stdout:
    print("[CLOUDFLARE] " + line, end="")
    m = re.search(r"(https://[-\w]+\.trycloudflare\.com)", line)
    if m and not public_url:
        public_url = m.group(1)
        print("\n✅ Your public link:", public_url, "\n")
