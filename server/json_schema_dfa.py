"""Zero-dependency incremental JSON & JSON-Schema DFA and vocabulary masker for DankAttn.

Supports OpenAI `response_format`:
  - {"type": "json_object"}
  - {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}, "strict": true}}

Provides:
  1. `JsonSchemaDFA`: character-level incremental parser/validator that accepts valid JSON
     prefixes matching the schema and can test any candidate token string in O(len(token)).
  2. `VocabMasker`: caches decoded token strings and computes allowed token ID sets / bitmasks
     for any DFA state.
  3. `validate_json_schema(obj, schema)`: full post-parse schema validation.
  4. `schema_system_hint(response_format)`: concise schema instruction for prompt guidance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def _resolve_ref(schema: dict, root: dict) -> dict:
    if "$ref" not in schema:
        return schema
    ref = schema["$ref"]
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema
    cur: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return schema
    return cur if isinstance(cur, dict) else schema


def validate_json_schema(val: Any, schema: dict | None, root: dict | None = None) -> tuple[bool, str]:
    """Validate a parsed JSON value against a JSON Schema subset (object/array/string/number/integer/boolean/null/enum/anyOf/oneOf)."""
    if not schema:
        return True, ""
    if root is None:
        root = schema
    schema = _resolve_ref(schema, root)

    if "const" in schema:
        if val != schema["const"] or type(val) is not type(schema["const"]):
            return False, f"expected const {schema['const']!r}"
        return True, ""

    if "enum" in schema:
        if not any(val == e and type(val) is type(e) for e in schema["enum"]):
            return False, f"value {val!r} not in enum {schema['enum']!r}"
        return True, ""

    for comb in ("anyOf", "oneOf"):
        if comb in schema:
            hits = [validate_json_schema(val, sub, root)[0] for sub in schema[comb]]
            ok = sum(hits) >= 1 if comb == "anyOf" else sum(hits) == 1
            return (True, "") if ok else (False, f"value did not satisfy {comb}")

    t = schema.get("type")
    if isinstance(t, list):
        for st in t:
            ok, _ = validate_json_schema(val, dict(schema, type=st), root)
            if ok:
                return True, ""
        return False, f"value did not match any type in {t}"

    if t == "null":
        return (val is None, "expected null")
    if t == "boolean":
        return (isinstance(val, bool), "expected boolean")
    if t == "integer":
        return (isinstance(val, int) and not isinstance(val, bool), "expected integer")
    if t == "number":
        return (isinstance(val, (int, float)) and not isinstance(val, bool), "expected number")
    if t == "string":
        if not isinstance(val, str):
            return False, "expected string"
        if "minLength" in schema and len(val) < schema["minLength"]:
            return False, "string shorter than minLength"
        if "maxLength" in schema and len(val) > schema["maxLength"]:
            return False, "string longer than maxLength"
        return True, ""
    if t == "array":
        if not isinstance(val, list):
            return False, "expected array"
        if "minItems" in schema and len(val) < schema["minItems"]:
            return False, "array shorter than minItems"
        if "maxItems" in schema and len(val) > schema["maxItems"]:
            return False, "array longer than maxItems"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, el in enumerate(val):
                ok, err = validate_json_schema(el, item_schema, root)
                if not ok:
                    return False, f"[{idx}]: {err}"
        return True, ""
    if t == "object" or "properties" in schema:
        if not isinstance(val, dict):
            return False, "expected object"
        props = schema.get("properties") or {}
        req = schema.get("required") or []
        for k in req:
            if k not in val:
                return False, f"missing required key {k!r}"
        addl = schema.get("additionalProperties", True)
        for k, v in val.items():
            if k in props:
                ok, err = validate_json_schema(v, props[k], root)
                if not ok:
                    return False, f".{k}: {err}"
            elif addl is False:
                return False, f"unexpected additional property {k!r}"
            elif isinstance(addl, dict):
                ok, err = validate_json_schema(v, addl, root)
                if not ok:
                    return False, f".{k}: {err}"
        return True, ""
    return True, ""


_WS = frozenset(" \t\r\n")
_DIGITS = frozenset("0123456789")
_HEX = frozenset("0123456789abcdefABCDEF")
_ESCAPES = frozenset('"\\/bfnrtu')


@dataclass
class _Frame:
    kind: str  # "obj" | "arr"
    schema: dict
    expect: str  # obj: "key_or_end"|"key"|"colon"|"val"|"comma_or_end"; arr: "val_or_end"|"val"|"comma_or_end"
    cur_key: str = ""
    seen_keys: set[str] = field(default_factory=set)

    def clone(self) -> "_Frame":
        return _Frame(self.kind, self.schema, self.expect, self.cur_key, set(self.seen_keys))


class JsonSchemaDFA:
    """Character-by-character incremental JSON + JSON-Schema state machine.

    Tracks exact JSON syntax plus object property names / required keys / value types so any
    candidate token string can be checked for prefix validity in microseconds without C deps.
    """

    def __init__(self, schema: dict | None = None, root: dict | None = None):
        self.root = root or schema or {}
        self.schema = _resolve_ref(schema or {}, self.root)
        self.stack: list[_Frame] = []
        self.mode: str = "value"  # "value"|"string"|"str_esc"|"str_u1".."str_u4"|"num_sign"|"num_zero"|"num_int"|"num_dot"|"num_frac"|"num_e"|"num_esign"|"num_exp"|"lit"|"done"
        self.cur_schema: dict = self.schema
        self.str_buf: str = ""
        self.str_is_key: bool = False
        self.lit_target: str = ""
        self.lit_pos: int = 0
        self.num_is_int: bool = False
        self.dead: bool = False

    def clone(self) -> "JsonSchemaDFA":
        c = JsonSchemaDFA.__new__(JsonSchemaDFA)
        c.root = self.root
        c.schema = self.schema
        c.stack = [f.clone() for f in self.stack]
        c.mode = self.mode
        c.cur_schema = self.cur_schema
        c.str_buf = self.str_buf
        c.str_is_key = self.str_is_key
        c.lit_target = self.lit_target
        c.lit_pos = self.lit_pos
        c.num_is_int = self.num_is_int
        c.dead = self.dead
        return c

    def _allowed_types(self, sch: dict) -> set[str]:
        sch = _resolve_ref(sch, self.root)
        if "enum" in sch:
            out: set[str] = set()
            for v in sch["enum"]:
                if v is None: out.add("null")
                elif isinstance(v, bool): out.add("boolean")
                elif isinstance(v, int): out.add("integer")
                elif isinstance(v, float): out.add("number")
                elif isinstance(v, str): out.add("string")
                elif isinstance(v, list): out.add("array")
                elif isinstance(v, dict): out.add("object")
            return out
        if "anyOf" in sch or "oneOf" in sch:
            out = set()
            for sub in sch.get("anyOf") or sch.get("oneOf") or ():
                out |= self._allowed_types(sub)
            return out
        t = sch.get("type")
        if isinstance(t, str):
            return {t}
        if isinstance(t, list):
            return set(t)
        if "properties" in sch:
            return {"object"}
        return {"object", "array", "string", "number", "integer", "boolean", "null"}

    def _finish_value(self):
        if not self.stack:
            self.mode = "done"
            return
        top = self.stack[-1]
        top.expect = "comma_or_end"
        self.mode = "container"

    def _allowed_key_prefix(self, sch: dict, seen: set[str], prefix: str, closed: bool) -> bool:
        sch = _resolve_ref(sch, self.root)
        addl = sch.get("additionalProperties", True)
        if addl is not False:
            return True
        props = sch.get("properties") or {}
        if closed:
            return prefix in props and prefix not in seen
        return any(k not in seen and k.startswith(prefix) for k in props)

    def feed_char(self, ch: str) -> bool:
        if self.dead:
            return False

        if self.mode == "done":
            if ch in _WS:
                return True
            self.dead = True
            return False

        if self.mode == "string":
            if ch == '"':
                if self.str_is_key:
                    top = self.stack[-1]
                    if not self._allowed_key_prefix(top.schema, top.seen_keys, self.str_buf, closed=True):
                        self.dead = True
                        return False
                    top.cur_key = self.str_buf
                    top.seen_keys.add(self.str_buf)
                    top.expect = "colon"
                    self.mode = "container"
                else:
                    self._finish_value()
                self.str_buf = ""
                return True
            if ch == "\\":
                self.mode = "str_esc"
                return True
            if ord(ch) < 0x20:
                self.dead = True
                return False
            self.str_buf += ch
            if self.str_is_key:
                top = self.stack[-1]
                if not self._allowed_key_prefix(top.schema, top.seen_keys, self.str_buf, closed=False):
                    self.dead = True
                    return False
            return True

        if self.mode == "str_esc":
            if ch not in _ESCAPES:
                self.dead = True
                return False
            self.mode = "str_u1" if ch == "u" else "string"
            return True

        if self.mode.startswith("str_u"):
            if ch not in _HEX:
                self.dead = True
                return False
            step = int(self.mode[-1])
            self.mode = "string" if step == 4 else f"str_u{step + 1}"
            return True

        if self.mode == "lit":
            if self.lit_pos < len(self.lit_target) and ch == self.lit_target[self.lit_pos]:
                self.lit_pos += 1
                if self.lit_pos == len(self.lit_target):
                    self._finish_value()
                return True
            self.dead = True
            return False

        if self.mode.startswith("num_"):
            if self.mode == "num_sign":
                if ch == "0":
                    self.mode = "num_zero"
                    return True
                if ch in "123456789":
                    self.mode = "num_int"
                    return True
                self.dead = True
                return False
            if self.mode == "num_zero":
                if ch == "." and not self.num_is_int:
                    self.mode = "num_dot"
                    return True
                if ch in "eE" and not self.num_is_int:
                    self.mode = "num_e"
                    return True
                self._finish_value()
                return self.feed_char(ch)
            if self.mode == "num_int":
                if ch in _DIGITS:
                    return True
                if ch == "." and not self.num_is_int:
                    self.mode = "num_dot"
                    return True
                if ch in "eE" and not self.num_is_int:
                    self.mode = "num_e"
                    return True
                self._finish_value()
                return self.feed_char(ch)
            if self.mode == "num_dot":
                if ch in _DIGITS:
                    self.mode = "num_frac"
                    return True
                self.dead = True
                return False
            if self.mode == "num_frac":
                if ch in _DIGITS:
                    return True
                if ch in "eE":
                    self.mode = "num_e"
                    return True
                self._finish_value()
                return self.feed_char(ch)
            if self.mode == "num_e":
                if ch in "+-":
                    self.mode = "num_esign"
                    return True
                if ch in _DIGITS:
                    self.mode = "num_exp"
                    return True
                self.dead = True
                return False
            if self.mode == "num_esign":
                if ch in _DIGITS:
                    self.mode = "num_exp"
                    return True
                self.dead = True
                return False
            if self.mode == "num_exp":
                if ch in _DIGITS:
                    return True
                self._finish_value()
                return self.feed_char(ch)

        if ch in _WS:
            return True

        if self.mode == "container":
            top = self.stack[-1]
            if top.kind == "obj":
                if top.expect in ("key_or_end", "comma_or_end") and ch == "}":
                    req = top.schema.get("required") or []
                    if any(k not in top.seen_keys for k in req):
                        self.dead = True
                        return False
                    self.stack.pop()
                    self._finish_value()
                    return True
                if top.expect in ("key_or_end", "key") and ch == '"':
                    self.mode = "string"
                    self.str_is_key = True
                    self.str_buf = ""
                    return True
                if top.expect == "colon" and ch == ":":
                    props = top.schema.get("properties") or {}
                    addl = top.schema.get("additionalProperties", True)
                    sub = props.get(top.cur_key) if top.cur_key in props else (addl if isinstance(addl, dict) else {})
                    self.cur_schema = _resolve_ref(sub or {}, self.root)
                    top.expect = "val"
                    self.mode = "value"
                    return True
                if top.expect == "comma_or_end" and ch == ",":
                    top.expect = "key"
                    return True
                self.dead = True
                return False
            else:
                if top.expect in ("val_or_end", "comma_or_end") and ch == "]":
                    self.stack.pop()
                    self._finish_value()
                    return True
                if top.expect == "comma_or_end" and ch == ",":
                    item_s = top.schema.get("items") if isinstance(top.schema.get("items"), dict) else {}
                    self.cur_schema = _resolve_ref(item_s, self.root)
                    top.expect = "val"
                    self.mode = "value"
                    return True
                if top.expect == "val_or_end":
                    item_s = top.schema.get("items") if isinstance(top.schema.get("items"), dict) else {}
                    self.cur_schema = _resolve_ref(item_s, self.root)
                    self.mode = "value"
                    return self.feed_char(ch)
                self.dead = True
                return False

        # self.mode == "value"
        allowed = self._allowed_types(self.cur_schema)
        if ch == "{" and "object" in allowed:
            self.stack.append(_Frame("obj", _resolve_ref(self.cur_schema, self.root), "key_or_end"))
            self.mode = "container"
            return True
        if ch == "[" and "array" in allowed:
            self.stack.append(_Frame("arr", _resolve_ref(self.cur_schema, self.root), "val_or_end"))
            self.mode = "container"
            return True
        if ch == '"' and "string" in allowed:
            self.mode = "string"
            self.str_is_key = False
            self.str_buf = ""
            return True
        if ch == "t" and "boolean" in allowed:
            self.mode, self.lit_target, self.lit_pos = "lit", "true", 1
            return True
        if ch == "f" and "boolean" in allowed:
            self.mode, self.lit_target, self.lit_pos = "lit", "false", 1
            return True
        if ch == "n" and "null" in allowed:
            self.mode, self.lit_target, self.lit_pos = "lit", "null", 1
            return True
        if (ch == "-" or ch in _DIGITS) and ("number" in allowed or "integer" in allowed):
            self.num_is_int = "number" not in allowed
            if ch == "-":
                self.mode = "num_sign"
            elif ch == "0":
                self.mode = "num_zero"
            else:
                self.mode = "num_int"
            return True
        self.dead = True
        return False

    def feed(self, text: str) -> bool:
        for ch in text:
            if not self.feed_char(ch):
                return False
        return True

    def is_complete(self) -> bool:
        return not self.dead and self.mode == "done"

    def accepts_token(self, tok_str: str) -> bool:
        c = self.clone()
        return c.feed(tok_str)


class VocabMasker:
    """Computes allowed token IDs for any JsonSchemaDFA state over a HuggingFace Tokenizer."""

    def __init__(self, tok):
        vocab = tok.get_vocab()
        self.size = max(vocab.values()) + 1 if vocab else 0
        self.id_to_str: list[str] = [""] * self.size
        for s, tid in vocab.items():
            self.id_to_str[tid] = tok.decode([tid], skip_special_tokens=False)

    def allowed_ids(self, dfa: JsonSchemaDFA, eos_ids: tuple[int, ...] = ()) -> list[int]:
        if dfa.is_complete():
            return list(eos_ids)
        out: list[int] = []
        for tid, s in enumerate(self.id_to_str):
            if s and dfa.accepts_token(s):
                out.append(tid)
        return out


def parse_response_format(body: dict) -> dict | None:
    """Extract normalized response_format spec from an OpenAI chat request body."""
    rf = body.get("response_format")
    if not isinstance(rf, dict):
        return None
    rtype = rf.get("type")
    if rtype == "json_object":
        return {"type": "json_object", "schema": {"type": "object"}}
    if rtype == "json_schema":
        js = rf.get("json_schema")
        if isinstance(js, dict) and isinstance(js.get("schema"), dict):
            return {"type": "json_schema", "name": js.get("name", "response"), "schema": js["schema"]}
    return None


def schema_system_hint(rf: dict) -> str:
    """Build a concise system instruction enforcing the requested JSON / JSON-Schema output."""
    if rf["type"] == "json_object":
        return "Respond ONLY with a valid JSON object. Do not include markdown fences or extra text."
    schema_txt = json.dumps(rf["schema"], ensure_ascii=False, separators=(",", ":"))
    return (
        f"Respond ONLY with valid JSON conforming strictly to this JSON Schema ({rf.get('name', 'response')}): "
        f"{schema_txt}\nDo not include markdown fences or any text outside the JSON."
    )
