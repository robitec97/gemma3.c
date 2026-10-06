#!/usr/bin/env python3
"""Run a gemma3 command in a pseudo-terminal and save the colored output as an SVG.

The SVG is a faithful rendering of what the terminal showed (ANSI colors
included), used for the screenshots in the README.

Examples:
  python scripts/terminal_capture.py --out docs/images/single-prompt.svg -- \
      ./gemma3 -p "Write a haiku about C" --stats
  python scripts/terminal_capture.py --out docs/images/chat.svg \
      --input "My name is Ada." --input "What is my name?" --input "/exit" -- \
      ./gemma3 -i --stats

Requires: pip install rich
"""

import argparse
import os
import pty
import re
import select
import shlex
import subprocess
import sys
import time

from rich.console import Console
from rich.text import Text


def run_in_pty(cmd, inputs, prompt_re, timeout, cols):
    """Run cmd attached to a pty; feed each input after a prompt appears."""
    master, slave = pty.openpty()
    env = dict(os.environ, COLUMNS=str(cols), TERM="xterm-256color")
    env.pop("NO_COLOR", None)
    proc = subprocess.Popen(cmd, stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True)
    os.close(slave)

    out = b""
    pending = list(inputs)
    seen_prompts = 0
    deadline = time.time() + timeout
    while time.time() < deadline:
        r, _, _ = select.select([master], [], [], 0.2)
        if r:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
        prompts = len(re.findall(prompt_re, out))
        if pending and prompts > seen_prompts:
            seen_prompts = prompts
            line = pending.pop(0)
            time.sleep(0.3)
            os.write(master, (line + "\n").encode())
        if proc.poll() is not None and not r:
            break
    proc.wait(timeout=10)
    os.close(master)
    return out.decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output .svg path")
    ap.add_argument("--title", default="gemma3.c", help="window title")
    ap.add_argument("--input", action="append", default=[], help="line to type at each prompt")
    ap.add_argument("--prompt-re", default=r">\x1b\[0m ", help="regex marking an input prompt")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--cols", type=int, default=100)
    ap.add_argument("--show-command", action="store_true", help="prefix the output with '$ <cmd>'")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("missing command")

    raw = run_in_pty(cmd, args.input, args.prompt_re.encode(), args.timeout, args.cols)
    raw = raw.replace("\r\n", "\n").replace("\r", "")
    if args.show_command:
        shown = " ".join(shlex.quote(c) if " " in c else c for c in cmd)
        raw = f"\x1b[1;32m$\x1b[0m {shown}\n" + raw

    console = Console(record=True, width=args.cols, force_terminal=True, color_system="truecolor",
                      file=open(os.devnull, "w"))
    console.print(Text.from_ansi(raw.rstrip("\n")))
    console.save_svg(args.out, title=args.title)
    sys.stdout.write(raw)
    print(f"\nsaved {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
