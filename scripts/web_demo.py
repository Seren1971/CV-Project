"""Browser demo for the final model: upload any photo, get the detections.

    python scripts/web_demo.py --ckpt path/to/best.pt [--threshold 0.9] [--port 8765]

Open http://127.0.0.1:8765 (the script tries to open it automatically).
Drag a JPG/PNG onto the page or click to choose a file; inference runs locally.
Ctrl+C stops the server.  --once image.jpg runs one file and exits (no server).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root

import torch
from PIL import Image, ImageDraw, ImageFont

from src.predict import load_model
from src.models.detector import set_eval_size
from scripts.demo import COLORS

MAX_UPLOAD = 25 * 1024 * 1024

HTML = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>P16 Wildlife Detection</title><style>
body{margin:0;background:#101418;color:#e8edf2;font:16px system-ui,sans-serif}main{max-width:1100px;margin:26px auto;padding:0 22px}
h1{font-size:26px;margin:0 0 4px}#model{color:#3cb44b;font-weight:600;margin-bottom:18px}
#drop{border:2px dashed #3a4653;border-radius:10px;padding:34px;text-align:center;color:#9aa5b1;cursor:pointer;background:#151b21}
#drop.over{border-color:#3cb44b;color:#e8edf2}
.controls{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:18px 0}
button{font:inherit;padding:9px 18px;border:0;border-radius:6px;background:#3cb44b;color:#08110b;font-weight:600;cursor:pointer}
button:disabled{opacity:.45;cursor:wait}
input[type=range]{accent-color:#3cb44b}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0}img{width:100%;border-radius:8px;background:#000;display:block}
figcaption{font-weight:600;margin:0 0 8px;color:#9aa5b1}#status{min-height:22px;color:#ffd966}
table{border-collapse:collapse;margin-top:16px;font-size:14px}td,th{border:1px solid #2a333d;padding:5px 12px;text-align:left}
.small{font-size:13px;color:#7c8894;margin-top:16px}@media(max-width:800px){.pair{grid-template-columns:1fr}}
</style><main><h1>P16 · Wildlife detection in camera-trap images</h1><div id="model">Loading model…</div>
<div id="drop">Drop a camera-trap photo here or click to choose a file<input type="file" id="file" accept="image/*" hidden></div>
<div class="controls"><label>Threshold <input type="range" id="thr" min="0.1" max="0.99" step="0.01" value="0.9"><span id="thrv">0.90</span></label>
<button id="run" disabled>Run detection</button><span id="status"></span></div>
<div class="pair"><figure><figcaption>Input</figcaption><img id="input" alt=""></figure>
<figure><figcaption>Model output</figcaption><img id="result" alt=""></figure></div>
<div id="dets"></div><p class="small">Inference runs locally on this machine; the photo never leaves it.
Green-box classes come from the CCT20 label set; confidence is the model score above the chosen threshold.</p>
</main><script>
const el=id=>document.getElementById(id);let file=null;
el('thr').oninput=()=>el('thrv').textContent=(+el('thr').value).toFixed(2);
fetch('/api/info').then(r=>r.json()).then(d=>{el('model').textContent=d.model+' · '+d.device+' · default threshold '+d.threshold.toFixed(2);el('thr').value=d.threshold;el('thrv').textContent=d.threshold.toFixed(2)});
function setFile(f){if(!f)return;file=f;el('input').src=URL.createObjectURL(f);el('run').disabled=false;el('status').textContent=f.name+' ready ('+(f.size/1048576).toFixed(1)+' MB)'}
el('drop').onclick=()=>el('file').click();el('file').onchange=e=>setFile(e.target.files[0]);
el('drop').ondragover=e=>{e.preventDefault();el('drop').classList.add('over')};
el('drop').ondragleave=()=>el('drop').classList.remove('over');
el('drop').ondrop=e=>{e.preventDefault();el('drop').classList.remove('over');setFile(e.dataTransfer.files[0])};
el('run').onclick=async()=>{if(!file)return;el('run').disabled=true;el('status').textContent='Running the model…';
try{const r=await fetch('/api/predict?thr='+el('thr').value,{method:'POST',body:file});const o=await r.json();if(!r.ok)throw Error(o.error);
el('result').src=o.image;el('status').textContent=o.detections.length+' detections · inference '+o.inference_seconds.toFixed(2)+' s';
el('dets').innerHTML=o.detections.length?'<table><tr><th>#</th><th>Class</th><th>Confidence</th><th>Box (x1,y1,x2,y2)</th></tr>'+o.detections.map((d,i)=>'<tr><td>'+(i+1)+'</td><td>'+d.label+'</td><td>'+d.confidence.toFixed(2)+'</td><td>'+d.bbox_xyxy.map(v=>v.toFixed(0)).join(', ')+'</td></tr>').join('')+'</table>':'<p>Nothing above the threshold — try lowering it.</p>'}
catch(e){el('status').textContent='Error: '+e.message}finally{el('run').disabled=false}};
</script></html>'''


class Demo:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.classes = load_model(args.ckpt, coco_reference=False, device=self.device)
        self.model.eval()
        set_eval_size(self.model)
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        self.name = f"variant {ck.get('variant', '?')} · val mAP@0.5 {ck.get('val_map50', 0):.3f}"
        self.threshold = args.threshold if args.threshold is not None else ck.get("threshold", 0.9)
        self.lock = threading.Lock()
        print(f"Loaded {args.ckpt} ({self.name}); device={self.device}", flush=True)

    @torch.no_grad()
    def infer(self, raw: bytes, thr: float):
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        from torchvision.transforms.functional import to_tensor
        t = to_tensor(im).to(self.device)
        with self.lock:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
                out = self.model([t])[0]
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
        keep = out["scores"] >= thr
        boxes, scores, labels = out["boxes"][keep].cpu(), out["scores"][keep].cpu(), out["labels"][keep].cpu()
        d = ImageDraw.Draw(im)
        font = ImageFont.load_default(size=max(14, im.width // 70))
        for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            col = COLORS[(l - 1) % len(COLORS)]
            d.rectangle(b, outline=col, width=3)
            d.text((b[0] + 3, b[1] + 3), f"{self.classes[l - 1]} {s:.2f}", fill=col, font=font)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        dets = [{"label": self.classes[l - 1], "confidence": round(s, 4),
                 "bbox_xyxy": [round(v, 1) for v in b]}
                for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist())]
        return {"image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                "detections": dets, "inference_seconds": elapsed, "threshold": thr}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--once", type=Path, help="run one image and exit (no server)")
    args = p.parse_args()
    demo = Demo(args)
    if args.once:
        r = demo.infer(args.once.read_bytes(), demo.threshold)
        r.pop("image")
        print(json.dumps(r, indent=1))
        return

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def respond(self, body, status=200, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                return self.respond(HTML, ctype="text/html; charset=utf-8")
            if self.path == "/api/info":
                return self.respond({"model": demo.name, "device": str(demo.device),
                                     "threshold": demo.threshold})
            self.respond({"error": "not found"}, 404)

        def do_POST(self):
            if not self.path.startswith("/api/predict"):
                return self.respond({"error": "not found"}, 404)
            try:
                from urllib.parse import urlparse, parse_qs
                thr = float(parse_qs(urlparse(self.path).query).get("thr", [demo.threshold])[0])
                thr = min(max(thr, 0.01), 0.99)
                n = int(self.headers.get("Content-Length", 0))
                if not 0 < n <= MAX_UPLOAD:
                    raise ValueError("empty or too large upload (25 MB max)")
                self.respond(demo.infer(self.rfile.read(n), thr))
            except Exception as e:                       # noqa: BLE001 - report to the page
                self.respond({"error": str(e)}, 400)

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as e:
        p.exit(2, f"Cannot bind port {args.port}: {e}. Choose another --port.\n")
    url = f"http://127.0.0.1:{args.port}"
    print(f"Open {url} — Ctrl+C stops the demo.", flush=True)
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
