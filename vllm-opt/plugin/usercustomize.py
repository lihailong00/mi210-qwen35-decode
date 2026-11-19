# Loaded automatically by every Python process in the container when this directory is on
# PYTHONPATH (site.ENABLE_USER_SITE is True in the vLLM image). Installs post-import hooks that
# patch vLLM modules right after they are first imported. Active only when VOPT_PATCHES is set.
#   VOPT_PATCHES: comma list of patch names (fp8gemv | lmhead,greedy,topk,bf16gemv | norms,act,gdn,attn)
#   VOPT_PKGS:    comma list of patch packages (default: all four; vopt_spec = MTP draft head + greedy verify, own env switches); each ignores names it does not own.
import os

if os.environ.get("VOPT_PATCHES"):
    import importlib
    for _pkg in os.environ.get("VOPT_PKGS", "vopt_fp8gemv,vopt_lmhead,vopt_rest,vopt_spec").split(","):
        if not _pkg.strip():
            continue
        try:
            importlib.import_module(_pkg.strip()).install_hooks(os.environ["VOPT_PATCHES"])
        except Exception as e:  # never break interpreter start-up
            import sys
            print(f"[vopt] hook install failed for {_pkg}: {e!r}", file=sys.stderr)
