"""Luna native HTTP server — Ollama + OpenAI-compat API.

完整异构栈（VLM/VLA/WA）走此服务；Ollama 蒸馏版见 export/。
所有文本/动作请求过 safety 三层门。
"""

from __future__ import annotations

import base64
import io
import json
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import torch

ROOT = Path(__file__).resolve().parents[1]


class LunaEngine:
    """懒加载 tiny/指定 preset 模型。"""

    def __init__(self, preset: str = "tiny", device_pref: str = "auto"):
        from serve.device_router import pick_device
        from config import LunaConfig
        from modeling_luna_ultimate import LunaUltimateFused

        self.choice = pick_device(device_pref)
        self.device = self.choice.torch_device
        self.cfg = LunaConfig.from_preset(preset)
        # serve 默认开 V-JEPA（VLM）；tiny 上用轻量配置
        self.cfg.vjepa_enabled = True
        self.model = LunaUltimateFused(self.cfg, decode_mode="meaning_first").to(self.device)
        self.model.eval()
        self.preset = preset
        self._init_safety()

    def _init_safety(self) -> None:
        from safety.audit import AuditLog
        from safety.locks import SafetyLock
        base = ROOT / ".luna"
        base.mkdir(exist_ok=True)
        self.audit = AuditLog(str(base / "serve_audit.log"))
        try:
            from safety.cognition import default_safety_ctm
            ctm = default_safety_ctm()
        except Exception:
            ctm = None
        self.lock = SafetyLock(self.audit, safety_ctm=ctm, base_dir=str(base))

    def gate(self, text: str) -> Tuple[bool, str]:
        dec = self.lock.gate(text)
        return bool(dec.allowed), dec.reason

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 32,
        images: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        allowed, reason = self.gate(prompt)
        if not allowed:
            return {
                "response": f"[REFUSED] {reason}",
                "done": True,
                "allowed": False,
                "reason": reason,
            }
        ids = self._tokenize(prompt)
        visual = self._decode_images(images) if images else None
        out = self.model(
            ids,
            visual_input=visual,
            route_text=prompt,
            return_all_losses=False,
        )
        gen = self.model.generate_meaning_first(
            ids, max_new_tokens=max_tokens, route_text=prompt
        )
        collapse = gen["collapse"]
        text = self._detokenize(gen["generated_ids"][0])
        return {
            "response": text,
            "done": True,
            "allowed": True,
            "decode_mode": "meaning_first",
            "collapse": bool(collapse.collapsed),
            "collapse_reason": collapse.reason,
            "logits_shape": list(out["logits"].shape),
            "device": self.choice.name,
            "backend": self.choice.backend,
        }

    @torch.no_grad()
    def predict_action(
        self, prompt: str, images: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        allowed, reason = self.gate(f"vla action: {prompt}")
        if not allowed:
            return {"allowed": False, "reason": reason, "actions": []}
        ids = self._tokenize(prompt)
        visual = self._decode_images(images) if images else None
        out = self.model(
            ids,
            visual_input=visual,
            route_text=prompt,
            compute_vla=True,
            compute_wa=True,
        )
        names = out.get("action_names") or ["noop"]
        # 二次门：动作名拼进文本再检
        act = names[0] if names else "noop"
        ok2, reason2 = self.gate(f"execute robot action {act} for: {prompt}")
        if not ok2:
            return {"allowed": False, "reason": reason2, "actions": [], "proposed": act}
        return {
            "allowed": True,
            "actions": names,
            "action_ids": out["action_ids"].tolist(),
            "action_mu": out["action_mu"].tolist() if out.get("action_mu") is not None else None,
            "world_next": out["world_next"].mean().item() if out.get("world_next") is not None else None,
            "device": self.choice.name,
        }

    def _tokenize(self, text: str) -> torch.Tensor:
        # 无外部 tokenizer：字符/字节哈希映射到 vocab
        vs = self.cfg.vocab_size
        ids = [(ord(c) * 131 + 7) % vs for c in (text or " ")[:64]]
        if not ids:
            ids = [0]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _detokenize(self, ids: torch.Tensor) -> str:
        # 冒烟可读标记（非真实词表）
        vals = ids.detach().cpu().tolist()
        return " ".join(f"<{v}>" for v in vals[:48])

    def _decode_images(self, images: List[Any]) -> Optional[torch.Tensor]:
        """支持 base64 PNG/JPEG 或跳过；失败则 None。"""
        if not images:
            return None
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            # 无 PIL：合成随机图匹配 tiny 配置
            h, w = self.cfg.vjepa_config.get("img_size", (32, 32))
            return torch.randn(1, 3, h, w, device=self.device)

        tensors = []
        h, w = self.cfg.vjepa_config.get("img_size", (32, 32))
        for img in images[:1]:
            if isinstance(img, str):
                raw = base64.b64decode(img.split(",")[-1])
                im = Image.open(io.BytesIO(raw)).convert("RGB").resize((w, h))
                arr = np.asarray(im).astype("float32") / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1)  # C,H,W
                tensors.append(t)
        if not tensors:
            return None
        return torch.stack(tensors, dim=0).to(self.device)


_ENGINE: Optional[LunaEngine] = None


def get_engine(preset: str = "tiny", device: str = "auto") -> LunaEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = LunaEngine(preset=preset, device_pref=device)
    return _ENGINE


def make_handler(preset: str, device: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"[luna_serve] {self.address_string()} {fmt % args}")

        def _json(self, code: int, obj: Dict[str, Any]):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8"))

        def do_GET(self):
            path = urlparse(self.path).path
            eng = get_engine(preset, device)
            if path in ("/", "/api/tags", "/api/version"):
                self._json(200, {
                    "models": [{
                        "name": f"luna-{eng.preset}",
                        "model": f"luna-{eng.preset}",
                        "details": {
                            "family": "luna-ultimate",
                            "track": "native",
                            "device": eng.choice.name,
                            "backend": eng.choice.backend,
                            "capabilities": ["llm", "vlm", "vla", "wa"],
                        },
                    }],
                    "version": "luna-serve-0.1",
                })
                return
            self._json(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                eng = get_engine(preset, device)
                data = self._read_json()
                if path == "/api/generate":
                    prompt = data.get("prompt") or data.get("input") or ""
                    max_tokens = int(data.get("options", {}).get("num_predict", data.get("max_tokens", 32)))
                    images = data.get("images")
                    result = eng.generate_text(prompt, max_tokens=max_tokens, images=images)
                    self._json(200, {
                        "model": f"luna-{eng.preset}",
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "response": result.get("response", ""),
                        "done": True,
                        "luna": result,
                    })
                    return
                if path in ("/api/chat", "/v1/chat/completions"):
                    messages = data.get("messages") or []
                    prompt = ""
                    images = data.get("images")
                    for m in messages:
                        if m.get("role") == "user":
                            c = m.get("content")
                            if isinstance(c, str):
                                prompt = c
                            elif isinstance(c, list):
                                parts = []
                                for p in c:
                                    if isinstance(p, dict) and p.get("type") == "text":
                                        parts.append(p.get("text", ""))
                                    if isinstance(p, dict) and p.get("type") == "image_url":
                                        url = (p.get("image_url") or {}).get("url", "")
                                        if url.startswith("data:"):
                                            images = (images or []) + [url]
                                prompt = " ".join(parts)
                    max_tokens = int(data.get("max_tokens", 32))
                    result = eng.generate_text(prompt, max_tokens=max_tokens, images=images)
                    if path.startswith("/v1/"):
                        self._json(200, {
                            "id": f"luna-{int(time.time())}",
                            "object": "chat.completion",
                            "model": f"luna-{eng.preset}",
                            "choices": [{
                                "index": 0,
                                "message": {"role": "assistant", "content": result.get("response", "")},
                                "finish_reason": "stop",
                            }],
                            "luna": result,
                        })
                    else:
                        self._json(200, {
                            "model": f"luna-{eng.preset}",
                            "message": {"role": "assistant", "content": result.get("response", "")},
                            "done": True,
                            "luna": result,
                        })
                    return
                if path == "/api/vla":
                    prompt = data.get("prompt") or data.get("instruction") or ""
                    result = eng.predict_action(prompt, images=data.get("images"))
                    self._json(200, result)
                    return
                self._json(404, {"error": f"unknown path {path}"})
            except Exception as e:
                self._json(500, {"error": str(e), "trace": traceback.format_exc()[-2000:]})

    return Handler


def serve(host: str = "127.0.0.1", port: int = 11435, preset: str = "tiny", device: str = "auto"):
    handler = make_handler(preset, device)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"[luna_serve] http://{host}:{port} preset={preset} device={device}")
    print("[luna_serve] endpoints: GET /api/tags | POST /api/generate | /api/chat | /v1/chat/completions | /api/vla")
    httpd.serve_forever()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Luna native serve（完整异构轨）")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=11435)
    p.add_argument("--preset", default="tiny")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mlx", "cann", "dml", "cpu"])
    args = p.parse_args()
    serve(args.host, args.port, args.preset, args.device)
