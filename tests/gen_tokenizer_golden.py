#!/usr/bin/env python3
"""Generate golden tokenizer / chat-template test cases from the HF reference.

Requires: pip install tokenizers jinja2

Usage:
    python tests/gen_tokenizer_golden.py [model_dir]

Writes tests/tokenizer_golden.txt and tests/chat_template_golden.txt, which
tests/test_tokenizer.c checks against gemma3_tokenizer.c.

tokenizer_golden.txt format (one case = three lines, escaped with \\\\, \\n,
\\t, \\r, \\xNN):
    T <text>
    I <token ids, space separated>          (encode, add_special_tokens=False)
    D <decoded text>                        (decode, skip_special_tokens=True)
"""

import json
import os
import random
import sys

from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


def esc(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif o < 0x20 or o == 0x7F:
            out.append("\\x%02x" % o)
        else:
            out.append(ch)
    # Make trailing spaces explicit so editors cannot silently strip them
    n = 0
    while n < len(out) and out[len(out) - 1 - n] == " ":
        n += 1
    if n:
        out[len(out) - n:] = ["\\x20"] * n
    return "".join(out)


PROSE = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello, world!",
    "Language models are trained on large amounts of text. They learn statistical "
    "patterns that let them predict the next token, and with enough data and "
    "parameters those patterns start to look a lot like understanding.",
    "It's a beautiful day, isn't it? \"Yes,\" she said -- 'absolutely'.",
    "Dr. Smith (born 1970) earned $1,234.56 on 2024-03-15 at 10:30am; that's 12.5% more.",
    "e.g. i.e. etc. vs. U.S.A. Mr. Mrs. Ph.D.",
    "In 1492, Columbus sailed the ocean blue.",
    "What is the capital of France? Answer in one sentence.",
]

CODE = [
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
    "#include <stdio.h>\n\nint main(void) {\n\tprintf(\"Hello, %s!\\n\", \"world\");\n\treturn 0;\n}\n",
    "for (int i = 0; i < n; i++) {\n\t\tsum += a[i] * b[i];\n\t}",
    "const x = {a: 1, b: [2, 3], c: \"four\"};\nconsole.log(JSON.stringify(x, null, 2));",
    "SELECT name, COUNT(*) FROM users WHERE age >= 18 GROUP BY name ORDER BY 2 DESC;",
    "<html>\n  <body>\n    <div class=\"x\"><b>bold</b> and <i>italic</i></div>\n  </body>\n</html>",
    "<table><tr><td>1</td><td>2</td></tr></table>",
    "x = a<b and b>c; y = a << 2 >> 1; z = <c> </c> <unused> <unused5 <unused5> <unused99999>",
    "if (a < b && c > d) { return a <= b ? c : d; }",
    "    four spaces\n\ttab\n  \t mixed\n",
    "$ make -j8 && ./gemma3 -p \"Hi\" --greedy -n 32",
    "https://example.com/path/to/page?query=1&other=two#fragment",
    "mailto:someone@example.org, ftp://files.example.net/pub/README.txt",
    "{\"key\": \"value\", \"list\": [1, 2.5, -3e10, true, false, null]}",
    "| col1 | col2 |\n|------|------|\n| a    | b    |",
    "# Title\n\n## Subtitle\n\n- item one\n- item two\n\n> quote\n\n```python\nprint('x')\n```",
]

MULTI = [
    "naïve café résumé façade jalapeño über straße",
    "Le cœur a ses raisons que la raison ne connaît point.",
    "Ich möchte ein Glas Wasser, bitte.",
    "¿Dónde está la biblioteca? ¡Hola!",
    "Привет, как дела? Всё хорошо.",
    "Καλημέρα κόσμε",
    "مرحبا بالعالم",
    "שלום עולם",
    "नमस्ते दुनिया",
    "日本語のテキストです。東京は大きい都市です。",
    "中文文本测试，今天天气很好。",
    "한국어 텍스트입니다.",
    "สวัสดีชาวโลก",
    "Tiếng Việt có dấu",
    "emoji: 🙂 😀 🎉 👍🏽 👨‍👩‍👧‍👦 🏳️‍🌈 ❤️",
    "math: ∑_{i=1}^{n} x_i² ≤ ∫ f(x) dx ≈ π ≠ ∞ → ∀ε>0 ∃δ",
    "rare: \U00020000 \U0002A6D6  \U000F0000 \U0010FFFD ☃ ༀ ᚠ",
    "combining: é ä ñ क्ष",
    "zero width: a​b‌c‍d﻿e",
    "nbsp: x y z　w",
    "box: ┌─┬─┐ │ │ │ └─┴─┘ ░▒▓█",
    "currency: € £ ¥ ₹ ₽ ₿ ¢",
    "fullwidth: ＡＢＣ １２３ ！？",
]

EDGE = [
    "",
    " ",
    "  ",
    "   ",
    " " * 31,
    " " * 32,
    " " * 70,
    "a",
    " a",
    "a ",
    " a ",
    "  two  spaces  ",
    "\n",
    "\n\n",
    "\n" * 31,
    "\n" * 40,
    "\t",
    "\t" * 35,
    "\r\n",
    "line1\r\nline2\r\n",
    "trailing newline\n",
    "\nleading newline",
    " \n \n ",
    "a\n b\n  c\n   d",
    "▁",
    "▁▁",
    "▁▁▁ x",
    "word▁word",
    "<",
    ">",
    "<>",
    "< >",
    "a < b <c>",
    "<bos>",
    "<eos>",
    "<pad>",
    "<unk>",
    "<mask>",
    "[multimodal]",
    "<start_of_turn>",
    "<end_of_turn>",
    "<start_of_image>",
    "<end_of_image>",
    "<image_soft_token>",
    "<unused0>",
    "<unused5>",
    "<unused6241>",
    "<unused6242>",
    "<start_of_turn><end_of_turn>",
    "<bos><bos>",
    "x<bos>y",
    "<start_of_turnn>",
    "<start_of_tur>",
    "<<start_of_turn>>",
    "<0x41>",
    "<0x0A>",
    "<0xZZ>",
    "\x01\x02\x7f",
    "\x1b[31mred\x1b[0m",
    "1234567890",
    "3.14159265358979323846",
    "1,000,000 and 1.000.000",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "==========================================================",
    "----------------------------------------------------------",
    "..........................................................",
    "!!!???...,,,;;;:::",
    "CamelCaseIdentifier snake_case_identifier kebab-case-identifier",
    "ALL CAPS SENTENCE WITH WORDS",
    "<b>bold</b><i>it</i><code>c</code></code>",
    "<h1>Title</h1>\n<p>para</p>",
]

CHAT = (
    "<bos><start_of_turn>user\nYou are a helpful assistant.\n\nHello there!<end_of_turn>\n"
    "<start_of_turn>model\nHi! How can I help you today?<end_of_turn>\n"
    "<start_of_turn>user\nWrite a haiku about the sea.<end_of_turn>\n"
    "<start_of_turn>model\n"
)


def random_strings(rng: random.Random, count: int):
    pools = [
        " ", " ", " ", "\n", "\t", "<", ">", "/", "_", "-", ".", ",", "'", "\"",
        "a", "b", "e", "t", "the", "ing", "tion", "▁", "unused", "start_of_turn",
        "<b>", "</b>", "<unused7>", "<bos>", "<end_of_turn>", "0", "1", "42",
        "é", "ß", "ø", "日", "本", "語", "🙂", "\U0001F600", "‍", " ",
        "Hello", "world", "Gemma", "token", "\r\n", "  ", "\n\n", "\t\t",
    ]
    out = []
    for _ in range(count):
        n = rng.randint(1, 60)
        s = "".join(rng.choice(pools) for _ in range(n))
        out.append(s)
    # random code points
    for _ in range(count // 3):
        n = rng.randint(1, 40)
        cps = []
        while len(cps) < n:
            r = rng.random()
            if r < 0.5:
                cp = rng.randint(0x20, 0x7E)
            elif r < 0.8:
                cp = rng.randint(0xA0, 0x2FFF)
            elif r < 0.95:
                cp = rng.randint(0x3000, 0xFFFD)
            else:
                cp = rng.randint(0x10000, 0x10FFFD)
            if 0xD800 <= cp <= 0xDFFF:
                continue
            cps.append(chr(cp))
        out.append("".join(cps))
    return out


def long_texts(rng: random.Random):
    texts = []
    blocks = PROSE + CODE + MULTI
    for k in range(6):
        parts = []
        size = 0
        target = 2000 + 1500 * k
        while size < target:
            b = rng.choice(blocks)
            sep = rng.choice([" ", "\n", "\n\n", "  ", "\t", " <b>", "<end_of_turn>\n"])
            parts.append(b + sep)
            size += len(b) + len(sep)
        texts.append("".join(parts))
    return texts


def chat_cases():
    S, U, M = "system", "user", "model"
    return [
        [(U, "Hello!")],
        [(S, "You are a helpful assistant."), (U, "Hello!")],
        [(S, "You are a pirate."), (U, "  What's up?  \n"), (M, " Arr, matey! \n"), (U, "Tell me more.")],
        [(U, "One"), (M, "Two"), (U, "Three"), (M, "Four"), (U, "Five")],
        [(S, "  spaced system  "), (U, "\n\ttrim me 　")],
        [(U, "multi\nline\n\ncontent\n")],
        [(U, "")],
        [(S, "Only system")],
        [(U, "日本語で答えてください。 "), (M, "はい、わかりました。")],
        [(U, "Code:\n```c\nint x = 1;\n```")],
    ]


def render_chat(template_src, msgs):
    import jinja2

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(msg):
        raise RuntimeError(msg)

    env.globals["raise_exception"] = raise_exception
    tmpl = env.from_string(template_src)
    role_map = {"system": "system", "user": "user", "model": "assistant"}
    messages = [{"role": role_map[r], "content": c} for r, c in msgs]
    return tmpl.render(messages=messages, bos_token="<bos>", add_generation_prompt=True)


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "gemma-3-4b-it")
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    rng = random.Random(1234)

    cases = []
    cases += EDGE + PROSE + CODE + MULTI + [CHAT]
    cases += [p + " " + q for p, q in zip(PROSE, MULTI)]
    cases += random_strings(rng, 240)
    cases += long_texts(rng)

    path = os.path.join(HERE, "tokenizer_golden.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Generated by tests/gen_tokenizer_golden.py from tokenizer.json - do not edit\n")
        for text in cases:
            enc = tok.encode(text, add_special_tokens=False)
            dec = tok.decode(enc.ids, skip_special_tokens=True)
            f.write("T " + esc(text) + "\n")
            f.write("I " + " ".join(str(i) for i in enc.ids) + "\n")
            f.write("D " + esc(dec) + "\n")
    print(f"wrote {len(cases)} tokenizer cases to {path}")

    cfg = json.load(open(os.path.join(model_dir, "tokenizer_config.json")))
    template_src = cfg["chat_template"]
    path = os.path.join(HERE, "chat_template_golden.txt")
    chats = chat_cases()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Generated by tests/gen_tokenizer_golden.py from the chat_template - do not edit\n")
        for msgs in chats:
            f.write("C %d\n" % len(msgs))
            for role, content in msgs:
                f.write("M %s %s\n" % (role, esc(content)))
            f.write("E " + esc(render_chat(template_src, msgs)) + "\n")
    print(f"wrote {len(chats)} chat cases to {path}")


if __name__ == "__main__":
    main()
