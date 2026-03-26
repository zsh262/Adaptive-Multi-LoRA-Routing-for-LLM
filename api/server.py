from __future__ import annotations

import json
import os
import sys
import sqlite3
from datetime import datetime
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from router.router_embedding import EmbeddingRouter


BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "models/TinyLlama-1.1B-Chat-v1.0")
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "1") == "1"
HOST = os.getenv("API_HOST", "127.0.0.1")
PORT = int(os.getenv("API_PORT", "8000"))
ROUTE_AVAILABLE_ONLY = os.getenv("ROUTE_AVAILABLE_ONLY", "0") == "1"
DB_PATH = Path(os.getenv("CHAT_DB_PATH", str(ROOT_DIR / "data" / "chat.db")))

_tokenizer = None
_base_model = None
_current_base_path = None
_router: Optional[EmbeddingRouter] = None


def _cors_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }

def _now():
    return datetime.utcnow().isoformat()

def _ensure_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, title TEXT, created_at TEXT, updated_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT, role TEXT, content TEXT, created_at TEXT)"
    )
    conn.commit()
    conn.close()

def _db():
    return sqlite3.connect(DB_PATH)

def load_model_and_tokenizer(base_model_path: str, load_in_4bit: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def ensure_base_model(base_model_path: str) -> Tuple[Any, Any]:
    global _base_model, _tokenizer, _current_base_path
    if _base_model is None or _current_base_path != base_model_path:
        _base_model, _tokenizer = load_model_and_tokenizer(base_model_path, LOAD_IN_4BIT)
        _current_base_path = base_model_path
    return _base_model, _tokenizer


def ensure_router() -> EmbeddingRouter:
    global _router
    if _router is None:
        _router = EmbeddingRouter()
    return _router


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    *,
    do_sample: bool = True,
    temperature: float = 0.1,
    top_p: float = 0.9,
):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if full_text.startswith(prompt):
        return full_text[len(prompt):].strip()
    return full_text


def _available_domains() -> list[str]:
    result = []
    for d in ["coding", "paper", "speech"]:
        p = Path(f"models/lora/{d}_lora")
        if p.exists():
            result.append(d)
    return result


def choose_domain(query: str, domain: Optional[str], route_available_only: bool) -> Tuple[str, Dict[str, float], list[str]]:
    r = ensure_router()
    if domain and domain in {"coding", "paper", "speech"}:
        _, scores = r.route(query)
        return domain, scores, _available_domains()
    best_domain, scores = r.route(query)
    if route_available_only:
        avail = _available_domains()
        if avail:
            best = None
            best_score = None
            for d in avail:
                s = scores.get(d, float("-inf"))
                if best_score is None or s > best_score:
                    best = d
                    best_score = s
            if best is not None:
                return best, scores, avail
    return best_domain, scores, _available_domains()

DOMAIN_PROMPTS = {
    "paper": (
        "You are a research assistant specializing in academic papers. "
        "Please answer in a formal academic style with structured points. "
        "Use numbered lists for contributions when appropriate. "
        "Begin directly with the answer.\n\n"
        "Instruction: {instruction}\n"
        "Answer:"
    ),
    "coding": (
        "You are an expert programmer. "
        "Provide clear, efficient, and well-documented code solutions. "
        "Include comments in the code and example usage when helpful.\n\n"
        "Instruction: {instruction}\n"
        "Answer:"
    ),
    "speech": (
        "You are a speech technology expert. "
        "Explain speech processing concepts clearly and accurately.\n\n"
        "Instruction: {instruction}\n"
        "Answer:"
    ),
}

def _get_prompt_template(domain: str) -> str:
    return DOMAIN_PROMPTS.get(domain, "Instruction: {instruction}\nInput: \nAnswer:")


def _build_code_suffix(query: str, domain: str) -> str:
    return ""


def _should_disable_lora(query: str, domain: str) -> bool:
    """
    只有 coding 领域的理论分析类问题才跳过 LoRA。
    因为 coding_lora 数据分布偏向"写代码"，不适合复杂度分析等理论问题。
    speech_lora 和 paper_lora 应该更适合各自领域的原理问题。
    """
    if domain != "coding":
        return False

    analysis_keywords = [
        "复杂度", "时间复杂度", "空间复杂度", "原理", "为什么",
        "区别", "分析", "是什么", "概念", "定义", "思路",
        "优缺点", "特点", "步骤", "过程", "如何理解"
    ]
    q_lower = query.lower()
    return any(k in q_lower for k in analysis_keywords)


def _get_conversation_messages(conv_id: str) -> list[dict]:
    """从数据库获取对话历史消息"""
    conn = _db()
    cur = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conv_id,),
    )
    rows = [{"role": r[0], "content": r[1]} for r in cur.fetchall()]
    conn.close()
    return rows


def _clean_message_content(content: str) -> str:
    """清理消息内容，移除 prompt 格式标记"""
    import re
    content = re.sub(r"Instruction:\s*", "", content)
    content = re.sub(r"Input:\s*", "", content)
    content = re.sub(r"Answer:\s*", "", content)
    content = re.sub(r"User:\s*", "", content)
    content = re.sub(r"Assistant:\s*", "", content)
    return content.strip()


def _build_prompt_with_history(messages: list[dict], query: str, suffix: str, disable_lora: bool = False, domain: str = None, max_history_chars: int = 2000) -> str:
    """
    构建带对话历史的 prompt。
    如果历史超过 max_history_chars，进行截断（保留最近的消息）。
    disable_lora=True 时，使用中文 prompt 格式并要求模型用中文回答。
    domain 用于选择领域特定的 prompt 模板。
    """
    if not messages:
        if disable_lora:
            return f"请用中文回答。\n指令：{query}{suffix}\n输入：\n回答："
        template = _get_prompt_template(domain) if domain else "Instruction: {instruction}\nInput: \nAnswer:"
        return template.format(instruction=query) + suffix + "\n"

    history_lines = []
    total_chars = 0

    for msg in reversed(messages):
        cleaned_content = _clean_message_content(msg["content"])
        line = f"{msg['role'].capitalize()}: {cleaned_content}"
        line_chars = len(line)
        if total_chars + line_chars > max_history_chars:
            break
        history_lines.insert(0, line)
        total_chars += line_chars

    history_text = "\n".join(history_lines)

    if disable_lora:
        return (
            f"请用中文回答。\n{history_text}\n\n"
            f"指令：{query}{suffix}\n"
            f"输入：\n"
            f"回答："
        )

    template = _get_prompt_template(domain) if domain else "Instruction: {instruction}\nInput: \nAnswer:"
    formatted = template.format(instruction=query)
    return (
        f"{history_text}\n\n"
        f"{formatted}{suffix}\n"
    )

class Handler(BaseHTTPRequestHandler):
    def _send_text(self, status: int, body: str, content_type: str = "text/plain; charset=utf-8"):
        data = body.encode("utf-8")
        self.send_response(status)
        for k, v in _cors_headers().items():
            self.send_header(k, v)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status: int, payload: Dict[str, Any]):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        for k, v in _cors_headers().items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, payload: Dict[str, Any]):
        body = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
        self.send_response(200)
        for k, v in _cors_headers().items():
            self.send_header(k, v)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in _cors_headers().items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if self.path == "/" or self.path == "/index.html":
            index_path = ROOT_DIR / "web" / "index.html"
            if index_path.exists():
                html = index_path.read_text(encoding="utf-8")
                self._send_text(200, html, "text/html; charset=utf-8")
                return
        if self.path.startswith("/api/conversations"):
            parsed = urlparse(self.path)
            parts = parsed.path.strip("/").split("/")
            if len(parts) == 2:
                conn = _db()
                cur = conn.execute("SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC")
                rows = [{"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]} for r in cur.fetchall()]
                conn.close()
                self._send_json(200, {"items": rows})
                return
            if len(parts) == 3:
                conv_id = parts[2]
                conn = _db()
                cur = conn.execute("SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?", (conv_id,))
                conv = cur.fetchone()
                if not conv:
                    conn.close()
                    self._send_json(404, {"error": "conversation not found"})
                    return
                cur = conn.execute(
                    "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                    (conv_id,),
                )
                messages = [{"role": r[0], "content": r[1], "created_at": r[2]} for r in cur.fetchall()]
                conn.close()
                self._send_json(200, {
                    "id": conv[0],
                    "title": conv[1],
                    "created_at": conv[2],
                    "updated_at": conv[3],
                    "messages": messages,
                })
                return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/api/chat/stream":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(raw.decode("utf-8"))
            except Exception as e:
                self._send_sse({"error": f"invalid json: {e}"})
                return
            query = (
                str(data.get("query") or data.get("prompt") or data.get("message") or "").strip()
            )
            if not query:
                self._send_sse({"error": "query is required"})
                return
            base_model_path = data.get("base_model_path") or BASE_MODEL_PATH
            max_new_tokens = int(data.get("max_new_tokens", 256))
            fast_mode = bool(data.get("fast_mode", False))
            code_language = str(data.get("code_language", "")).strip()
            conv_id = str(data.get("conversation_id", "")).strip()
            try:
                model, tokenizer = ensure_base_model(base_model_path)
                best_domain, scores, avail = choose_domain(
                    query,
                    None,
                    bool(data.get("route_available_only", ROUTE_AVAILABLE_ONLY)),
                )
                adapter_path = f"models/lora/{best_domain}_lora"
                used_adapter = False
                model_to_use = model
                route_avail_only = bool(data.get("route_available_only", ROUTE_AVAILABLE_ONLY))

                if route_avail_only and not Path(adapter_path).exists():
                    avail_loras = _available_domains()
                    if avail_loras:
                        best_score = float("-inf")
                        best_domain = None
                        for d in avail_loras:
                            s = scores.get(d, float("-inf"))
                            if s > best_score:
                                best_score = s
                                best_domain = d
                        adapter_path = f"models/lora/{best_domain}_lora" if best_domain else None

                if adapter_path and Path(adapter_path).exists() and not _should_disable_lora(query, best_domain):
                    model_to_use = PeftModel.from_pretrained(model, adapter_path)
                    used_adapter = True

                suffix = _build_code_suffix(query, best_domain)
                messages = _get_conversation_messages(conv_id) if conv_id else []
                disable_lora = _should_disable_lora(query, best_domain)
                prompt = _build_prompt_with_history(messages, query, suffix, disable_lora, best_domain)

                if fast_mode:
                    answer = generate_response(
                        model_to_use,
                        tokenizer,
                        prompt,
                        min(max_new_tokens, 256),
                        do_sample=False,
                        temperature=0.0,
                        top_p=1.0,
                    )
                else:
                    answer = generate_response(model_to_use, tokenizer, prompt, max_new_tokens)
                self._send_sse({
                    "answer": answer,
                    "selected_domain": best_domain,
                    "scores": scores,
                    "used_adapter": used_adapter,
                    "available_domains": avail,
                    "base_model_path": base_model_path,
                })
            except Exception as e:
                self._send_sse({"error": f"{e.__class__.__name__}: {e}"})
            return
        if self.path == "/api/llm/ask":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception as e:
                self._send_json(400, {"error": f"invalid json: {e}"})
                return

            query = str(data.get("query") or "").strip()
            domain = str(data.get("domain") or "general").strip().lower()

            if not query:
                self._send_json(400, {"error": "query is required"})
                return

            if domain not in {"coding", "paper", "speech", "general"}:
                self._send_json(400, {"error": f"invalid domain: {domain}. Must be one of: coding, paper, speech, general"})
                return

            try:
                model, tokenizer = ensure_base_model(BASE_MODEL_PATH)
                best_domain = domain if domain != "general" else None

                if best_domain is None:
                    best_domain, scores, _ = choose_domain(query, None, True)
                else:
                    scores = {}

                adapter_path = f"models/lora/{best_domain}_lora" if best_domain else None
                used_adapter = False
                model_to_use = model

                if adapter_path and Path(adapter_path).exists():
                    if best_domain != "coding" or not _should_disable_lora(query, best_domain):
                        model_to_use = PeftModel.from_pretrained(model, adapter_path)
                        used_adapter = True

                suffix = _build_code_suffix(query, best_domain)
                disable_lora = _should_disable_lora(query, best_domain) if best_domain else False
                prompt = _build_prompt_with_history([], query, suffix, disable_lora, best_domain)

                answer = generate_response(model_to_use, tokenizer, prompt, 256)

                self._send_json(200, {
                    "result": answer,
                    "domain": best_domain,
                    "used_adapter": used_adapter,
                })
            except Exception as e:
                self._send_json(500, {"error": f"{e.__class__.__name__}: {e}"})
            return
        if self.path == "/api/conversations":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(raw.decode("utf-8")) if raw else {}
            title = str(data.get("title", "新对话")).strip() or "新对话"
            conv_id = str(int(datetime.utcnow().timestamp() * 1000))
            now = _now()
            conn = _db()
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conv_id, title, now, now),
            )
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conv_id, "assistant", "你好，我是 Adaptive Multi-LoRA 助手。你可以问我代码、论文或语音相关问题。", now),
            )
            conn.commit()
            conn.close()
            self._send_json(200, {"id": conv_id, "title": title})
            return
        if self.path.startswith("/api/conversations/") and self.path.endswith("/messages"):
            parts = self.path.strip("/").split("/")
            if len(parts) != 4:
                self._send_json(404, {"error": "not found"})
                return
            conv_id = parts[2]
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(raw.decode("utf-8")) if raw else {}
            role = str(data.get("role", "")).strip()
            content = str(data.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                self._send_json(400, {"error": "role and content are required"})
                return
            now = _now()
            conn = _db()
            cur = conn.execute("SELECT id FROM conversations WHERE id = ?", (conv_id,))
            if not cur.fetchone():
                conn.close()
                self._send_json(404, {"error": "conversation not found"})
                return
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conv_id, role, content, now),
            )
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))
            conn.commit()
            conn.close()
            self._send_json(200, {"status": "ok"})
            return
        if self.path != "/generate":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(raw.decode("utf-8"))
        except Exception as e:
            self._send_json(400, {"error": f"invalid json: {e}"})
            return

        query = str(data.get("query", "")).strip()
        if not query:
            self._send_json(400, {"error": "query is required"})
            return
        domain = data.get("domain")
        base_model_path = data.get("base_model_path") or BASE_MODEL_PATH
        max_new_tokens = int(data.get("max_new_tokens", 256))
        fast_mode = bool(data.get("fast_mode", False))
        code_language = str(data.get("code_language", "")).strip()
        conv_id = str(data.get("conversation_id", "")).strip()

        try:
            model, tokenizer = ensure_base_model(base_model_path)
            route_available_only = bool(data.get("route_available_only", ROUTE_AVAILABLE_ONLY))
            best_domain, scores, avail = choose_domain(
                query,
                None if domain in (None, "", "auto") else domain,
                route_available_only,
            )
            adapter_path = f"models/lora/{best_domain}_lora"
            used_adapter = False
            model_to_use = model

            if route_available_only and not Path(adapter_path).exists():
                avail_loras = _available_domains()
                if avail_loras:
                    best_score = float("-inf")
                    best_domain = None
                    for d in avail_loras:
                        s = scores.get(d, float("-inf"))
                        if s > best_score:
                            best_score = s
                            best_domain = d
                    adapter_path = f"models/lora/{best_domain}_lora" if best_domain else None

            if adapter_path and Path(adapter_path).exists() and not _should_disable_lora(query, best_domain):
                model_to_use = PeftModel.from_pretrained(model, adapter_path)
                used_adapter = True

            suffix = _build_code_suffix(query, best_domain)
            messages = _get_conversation_messages(conv_id) if conv_id else []
            disable_lora = _should_disable_lora(query, best_domain)
            prompt = _build_prompt_with_history(messages, query, suffix, disable_lora, best_domain)

            if fast_mode:
                answer = generate_response(
                    model_to_use,
                    tokenizer,
                    prompt,
                    min(max_new_tokens, 256),
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                )
            else:
                answer = generate_response(model_to_use, tokenizer, prompt, max_new_tokens)

            self._send_json(200, {
                "selected_domain": best_domain,
                "scores": scores,
                "used_adapter": used_adapter,
                "available_domains": avail,
                "route_available_only": route_available_only,
                "base_model_path": base_model_path,
                "answer": answer,
            })
        except Exception as e:
            self._send_json(500, {"error": f"{e.__class__.__name__}: {e}"})

    def do_PUT(self):
        if not self.path.startswith("/api/conversations/"):
            self._send_json(404, {"error": "not found"})
            return
        parts = self.path.strip("/").split("/")
        if len(parts) != 3:
            self._send_json(404, {"error": "not found"})
            return
        conv_id = parts[2]
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        data = json.loads(raw.decode("utf-8")) if raw else {}
        title = str(data.get("title", "")).strip()
        if not title:
            self._send_json(400, {"error": "title is required"})
            return
        now = _now()
        conn = _db()
        cur = conn.execute("SELECT id FROM conversations WHERE id = ?", (conv_id,))
        if not cur.fetchone():
            conn.close()
            self._send_json(404, {"error": "conversation not found"})
            return
        conn.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?", (title, now, conv_id))
        conn.commit()
        conn.close()
        self._send_json(200, {"status": "ok"})


def run_server():
    _ensure_db()
    server = HTTPServer((HOST, PORT), Handler)
    print(f"Server listening on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
