"""User-supplied local source dumper. Do not publish internal source dumps."""

import os
import sys

CODE_EXTENSIONS = {
    ".h",
    ".hpp",
    ".hxx",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".cce",
    ".cuh",
    ".cu",
    ".py",
    ".sh",
    ".bat",
    ".cmake",
    ".mk",
    ".proto",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
}
SKIP_DIRS = {
    ".git",
    ".codegraph",
    ".opencode",
    ".vscode",
    "__pycache__",
    "node_modules",
    "build",
    "out",
    "dist",
    ".idea",
    ".vs",
}
SKIP_FILES = {"code_dump.txt"}
SKIP_BINARY_EXTENSIONS = {
    ".db",
    ".db-shm",
    ".db-wal",
    ".lock",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".exe",
    ".dll",
    ".so",
    ".o",
    ".a",
    ".obj",
    ".bin",
    ".dat",
    ".woff",
    ".ttf",
}


def dump_project(root):
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name in SKIP_FILES:
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in SKIP_BINARY_EXTENSIONS or ext not in CODE_EXTENSIONS:
                continue
            entries.append(os.path.join(dirpath, name))
    entries.sort()
    output_path = os.path.join(root, "code_dump.txt")
    with open(output_path, "w", encoding="utf-8") as out:
        for idx, full_path in enumerate(entries, 1):
            rel = os.path.relpath(full_path, root)
            out.write(f"########## [{idx}/{len(entries)}] {rel} ##########\n")
            try:
                with open(full_path, encoding="utf-8") as f:
                    content = f.read()
            except (UnicodeDecodeError, OSError):
                content = ""
            out.write(content)
            if not content.endswith("\n"):
                out.write("\n")
            out.write("\n")
    return output_path, len(entries)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) > 1:
        root = os.path.abspath(sys.argv[1])
    out_file, count = dump_project(root)
    print(f"Done: {count} files -> {out_file}")
