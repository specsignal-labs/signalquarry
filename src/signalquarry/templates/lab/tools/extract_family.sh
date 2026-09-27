#!/bin/sh
# Transfer drill / sale: extract one family into a standalone repository and verify it.
#
#   tools/extract_family.sh FAMILY DEST
#
# Needs git-filter-repo (pip install git-filter-repo). The family's directory becomes the
# new repository's root, so its evidence ledgers keep their hashes; the tree hash of
# families/FAMILY equals the new root tree hash (this is what c_code commitments bind).
set -eu
family="$1"; dest="$2"
command -v git-filter-repo >/dev/null || { echo "install git-filter-repo first"; exit 2; }
before=$(git rev-parse "HEAD:families/$family")
git clone --quiet --no-local . "$dest"
cd "$dest"
git filter-repo --quiet --path "families/$family/" --path-rename "families/$family/:"
after=$(git rev-parse "HEAD^{tree}")
echo "family tree before: $before"
echo "root tree after:    $after"
[ "$before" = "$after" ] || { echo "TREE_HASH_CHANGED"; exit 1; }
cat > signalquarry.toml <<TOML
[project]
name = "$family"

[strategies]
src_dirs = ["src"]
modules = []   # re-add this family's strategy modules
TOML
echo "extracted $family to $dest; now run: sqy evidence verify --project $dest"
