import argparse
import os
import runpy
import ssl
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional

_DEFAULT_MODEL_URL = (
    "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/"
    "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
)
_DEFAULT_MODEL_FILENAME = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"


def _get_cache_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
        return Path(base) / "llama-cpp-py-sync" / "models"

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "llama-cpp-py-sync" / "models"

    return Path.home() / ".cache" / "llama-cpp-py-sync" / "models"


def _ensure_llama_library() -> None:
    if os.environ.get("LLAMA_CPP_LIB"):
        return

    try:
        import llama_cpp_py_sync

        package_dir = Path(llama_cpp_py_sync.__file__).resolve().parent
    except Exception:
        package_dir = None

    dll_candidates: list[Path] = []
    if package_dir:
        dll_candidates.extend(
            [
                package_dir / "llama.dll",
                package_dir / "libllama.dll",
                package_dir / "libllama.so",
                package_dir / "libllama.dylib",
            ]
        )

    existing = next((p for p in dll_candidates if p.exists()), None)
    if existing is not None:
        os.environ["LLAMA_CPP_LIB"] = str(existing)
        return

    raise RuntimeError(
        "No llama shared library was found inside the installed package. "
        "Install a prebuilt wheel for your platform or set LLAMA_CPP_LIB to a valid library path."
    )


def _download_with_progress(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100.0, downloaded * 100.0 / total_size)
        sys.stdout.write(f"\rDownloading model: {pct:5.1f}%")
        sys.stdout.flush()

    sys.stdout.write(f"Downloading model to: {dest}\n")
    sys.stdout.flush()

    tmp = dest.with_suffix(dest.suffix + ".partial")
    if tmp.exists():
        tmp.unlink(missing_ok=True)

    ctx: ssl.SSLContext | None = None
    try:
        import certifi  # type: ignore

        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        # Fall back to system defaults. On some macOS Python installs this may fail
        # if system certificates are not configured; installing certifi fixes it.
        ctx = None

    req = urllib.request.Request(url, headers={"User-Agent": "llama-cpp-py-sync"})
    with urllib.request.urlopen(req, context=ctx) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        downloaded = 0
        chunk_size = 1024 * 256
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            if total > 0:
                downloaded += len(chunk)
                pct = min(100.0, downloaded * 100.0 / total)
                sys.stdout.write(f"\rDownloading model: {pct:5.1f}%")
                sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()

    tmp.replace(dest)


def _resolve_model_path(args_model: Optional[Path]) -> Path:
    if args_model:
        return args_model

    env_model = os.environ.get("LLAMA_MODEL")
    if env_model:
        return Path(env_model)

    cache_dir = _get_cache_dir()
    return cache_dir / _DEFAULT_MODEL_FILENAME


def _confirm_download(url: str, dest: Path) -> bool:
    if not sys.stdin.isatty():
        return False

    print(f"Model file not found: {dest}")
    print(f"Download default model from: {url}")
    resp = input("Proceed? [y/N]: ").strip().lower()
    return resp in {"y", "yes"}


def _chat(args: argparse.Namespace) -> int:
    if args.prompt is None and not sys.stdin.isatty():
        args.prompt = "Say 'ok'."

    model_path = _resolve_model_path(args.model)
    if not model_path.exists():
        is_default = args.model is None and not os.environ.get("LLAMA_MODEL")
        if not is_default:
            print(f"Model file not found: {model_path}")
            return 2

        if not getattr(args, "yes", False) and not _confirm_download(_DEFAULT_MODEL_URL, model_path):
            print("Download cancelled.")
            print("To auto-download without prompting, pass --yes.")
            return 2

        _download_with_progress(_DEFAULT_MODEL_URL, model_path)

    _ensure_llama_library()

    os.environ["LLAMA_LOG_LEVEL"] = args.log_level

    from llama_cpp_py_sync.llama import Llama

    print(f"Loading model (this may take a few seconds): {model_path}")
    sys.stdout.flush()

    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1

    if args.prompt:
        try:
            out = llm.generate(
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(out)
        except Exception as e:
            print(f"Generation failed: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()
            return 1
        return 0

    print("Interactive chat. Ctrl+C or blank line to exit.")
    sys.stdout.flush()

    history: list[str] = []
    try:
        while True:
            user_msg = input("You: ").strip()
            if not user_msg:
                break

            prompt = "".join(history) + f"User: {user_msg}\nAssistant:"
            try:
                out = llm.generate(
                    prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    stop_sequences=["\nUser:", "\nYou:", "\nAssistant:"],
                )
                out_text = str(out).strip()
                print(f"Assistant: {out_text}")
                history.append(f"User: {user_msg}\nAssistant: {out_text}\n")
            except ValueError as e:
                if "Prompt too long" in str(e):
                    history = []
                    out = llm.generate(
                        f"User: {user_msg}\nAssistant:",
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        stop_sequences=["\nUser:", "\nYou:", "\nAssistant:"],
                    )
                    out_text = str(out).strip()
                    print(f"Assistant: {out_text}")
                    history.append(f"User: {user_msg}\nAssistant: {out_text}\n")
                    continue
                raise
    except KeyboardInterrupt:
        print("\nExiting.")

    return 0


def _run_module_main(module_name: str, argv: List[str]) -> int:
    forwarded_argv = list(argv)
    if forwarded_argv and forwarded_argv[0] == "--":
        forwarded_argv = forwarded_argv[1:]

    old_argv = sys.argv
    sys.argv = [module_name, *forwarded_argv]
    try:
        runpy.run_module(module_name, run_name="__main__")
        return 0
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 1
    finally:
        sys.argv = old_argv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m llama_cpp_py_sync")
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="Start a simple interactive chat")
    chat.add_argument("--model", type=Path, default=None, help="Path to GGUF model")
    chat.add_argument(
        "--yes",
        action="store_true",
        help="Automatically confirm prompts (e.g., default model download).",
    )
    chat.add_argument("--prompt", type=str, default=None, help="One-shot prompt")
    chat.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("LLAMA_MAX_TOKENS", "128")),
        help="Max tokens per response.",
    )
    chat.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("LLAMA_TEMPERATURE", "0.7")),
        help="Sampling temperature.",
    )
    chat.add_argument(
        "--n-ctx",
        type=int,
        default=int(os.environ.get("LLAMA_N_CTX", "2048")),
        help="Context length.",
    )
    chat.add_argument(
        "--n-gpu-layers",
        type=int,
        default=int(os.environ.get("LLAMA_N_GPU_LAYERS", "0")),
        help="GPU layers (0 = CPU).",
    )
    chat.add_argument("--verbose", action="store_true", help="Verbose model load/logs.")
    chat.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set llama_cpp_py_sync log level (default: INFO)",
    )
    chat.add_argument("--debug", action="store_true", help="Verbose troubleshooting.")

    converters = {
        "convert-hf-to-gguf": "Delegate to the bundled Hugging Face to GGUF converter.",
        "convert-lora-to-gguf": "Delegate to the bundled LoRA to GGUF converter.",
        "convert-llama-ggml-to-gguf": "Delegate to the bundled GGML to GGUF converter.",
    }
    for command, help_text in converters.items():
        converter = sub.add_parser(command, help=help_text, add_help=False)
        converter.add_argument(
            "args",
            nargs=argparse.REMAINDER,
            help="Arguments passed through to the underlying upstream converter.",
        )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    converter_modules = {
        "convert-hf-to-gguf": "llama_cpp_py_sync.convert_hf_to_gguf",
        "convert-lora-to-gguf": "llama_cpp_py_sync.convert_lora_to_gguf",
        "convert-llama-ggml-to-gguf": "llama_cpp_py_sync.convert_llama_ggml_to_gguf",
    }

    argv = list(argv) if argv is not None else sys.argv[1:]
    if argv and argv[0] in converter_modules:
        return _run_module_main(converter_modules[argv[0]], argv[1:])

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "chat":
        return _chat(args)

    parser.print_help()
    return 2
