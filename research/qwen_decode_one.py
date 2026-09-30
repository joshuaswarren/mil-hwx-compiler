"""Decode one HWX file and emit a record."""
import sys, json
sys.path.insert(0, '/home/joshuawarren/src/mil-hwx-h14-integ-wt/research')
import mint_oracles as om
import subprocess, tempfile
from pathlib import Path

hwx_local = Path(sys.argv[1])
data = hwx_local.read_bytes()
decoded = om.parse_hwx(data, 'h14', capture_constant_bytes=True)
cs = decoded.get('constant_section', {})
print(json.dumps({
    "case": hwx_local.stem,
    "hwx_bytes": len(data),
    "hwx_sha256": decoded.get("hwx_sha256"),
    "constant_section": {
        "size": cs.get('size'),
        "sha256": cs.get('sha256'),
        "nonzero_bytes": cs.get('nonzero_bytes'),
        "prefix_128_sha256": cs.get('prefix_128_sha256'),
    },
    "program_descriptor": decoded.get("program_descriptor"),
    "task_descriptors": decoded.get("task_descriptors"),
}, indent=2, default=str))