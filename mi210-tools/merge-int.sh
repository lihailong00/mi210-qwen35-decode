#!/usr/bin/env bash
# usage (host): /data/tools/mi210-opt/merge-int.sh branch1 [branch2 ...]  -> resets integration to master and merges given branches
# Ignores CR/LF-only differences and normalizes merged files to LF.
set -o pipefail
cd /data/tools/mi210-opt/wt-int
git merge --abort >/dev/null 2>&1
git reset -q --hard master || exit 1
for b in "$@"; do
  if ! git merge -q --no-edit -X ignore-space-at-eol $b; then echo "MERGE CONFLICT on $b"; git diff --name-only --diff-filter=U; exit 1; fi
done
for pf in /data/tools/mi210-opt/int-patches/*.diff; do [ -f "$pf" ] || continue; if git apply --check "$pf" 2>/dev/null; then git apply "$pf" && git commit -qam "integration patch: $(basename $pf)" && echo "applied $(basename $pf)"; else echo "SKIP (does not apply): $(basename $pf)"; fi; done
crlf=$(git diff --name-only master HEAD | while read f; do [ -f "$f" ] && grep -lq $(printf "\r") "$f" && echo "$f"; done)
if [ -n "$crlf" ]; then for f in $crlf; do sed -i "s/\r$//" "$f"; done; git commit -qam "normalize line endings (LF)"; echo "normalized: $crlf"; fi
git log --oneline -1
