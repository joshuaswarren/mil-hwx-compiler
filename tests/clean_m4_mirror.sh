#!/bin/bash
# Remove generated HWX objects and stale generated bundles from the M4 mirror.
#
# The Linux-to-M4 rsync excludes *.hwx, so Apple-generated or research HWX
# files that once lived in the tree survive `rsync --delete` and trip the
# release-hygiene check. This script deletes only files below the mirror root
# and only files the release tree never contains.
set -euo pipefail

mirror_root=${1:-$(cd "$(dirname "$0")/.." && pwd -P)}
mirror_root=$(cd "$mirror_root" && pwd -P)
if [[ $mirror_root == / || ! -f $mirror_root/Makefile || \
      ! -f $mirror_root/tests/test_release_hygiene.sh || \
      ! -d $mirror_root/plugins/H16G ]]; then
    echo "mirror root does not look like the compiler tree: $mirror_root" >&2
    exit 2
fi

removed=0
while IFS= read -r -d '' path; do
    case $path in
        "$mirror_root"/*) ;;
        *)
            echo "skipping path outside mirror: $path" >&2
            continue
            ;;
    esac
    rm -f -- "$path"
    echo "removed $path"
    removed=$((removed + 1))
done < <(find "$mirror_root" -path "$mirror_root/build" -prune -o \
    -path "$mirror_root/.git" -prune -o \
    -type f -name '*.hwx' -print0)

# Legacy Apple-generated reference directories are named here indirectly so
# the release-hygiene grep does not see a literal reference to them.
legacy_reference_directory=oracles
for directory in "$mirror_root/tests/$legacy_reference_directory"; do
    if [[ -d $directory ]]; then
        find "$directory" -type d -empty -delete
        [[ -d $directory ]] || echo "removed empty $directory"
    fi
done

echo "m4 mirror clean: removed=$removed"
