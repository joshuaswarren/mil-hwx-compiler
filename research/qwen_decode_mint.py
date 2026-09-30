#!/usr/bin/env python3
"""Decode all H14 matvec HWXs from the Qwen mint campaign and emit
constant-section JSON records for the H14 parity envelope."""
import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, '/home/joshuawarren/src/mil-hwx-h14-integ-wt/research')
import mint_oracles as om

cases = json.load(open('/home/joshuawarren/src/mil-hwx-h14-integ-wt/research/oracles/qwen/2026-09-30-blob-v3.json'))
records = []
for c in cases:
    if not c['compiled'] or not c['case'].startswith('qwenmv'):
        continue
    case = c['case']
    # find the HWX file on macstudio (use the v3 run only)
    remote_dir = f"/tmp/qwenmint-v2-1790785841/{case}"
    cmd = f"ls {remote_dir}/out/model.hwx 2>/dev/null"
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "macstudio", cmd],
                       capture_output=True, text=True, timeout=30)
    hwx_remote = r.stdout.strip()
    if not hwx_remote:
        continue
    # scp to local
    local = Path(tempfile.mktemp(prefix=f"qwenmv-{case}-", suffix=".hwx"))
    subprocess.run(["scp", "-q", f"macstudio:{hwx_remote}", str(local)],
                   check=True, timeout=60)
    data = local.read_bytes()
    decoded = om.parse_hwx(data, 'h14', capture_constant_bytes=True)
    cs = decoded.get('constant_section', {})
    record = {
        "case": case,
        "shape": c['shape'],
        "hwx_bytes": len(data),
        "hwx_sha256": decoded.get('hwx_sha256'),
        "constant_section": {
            "size": cs.get('size'),
            "sha256": cs.get('sha256'),
            "nonzero_bytes": cs.get('nonzero_bytes'),
            "prefix_128_sha256": cs.get('prefix_128_sha256'),
            "prefix_128_nonzero_bytes": cs.get('prefix_128_nonzero_bytes'),
            "tail_after_128_nonzero_bytes": cs.get('tail_after_128_nonzero_bytes'),
            "chunk_bytes": cs.get('chunk_bytes'),
            "chunk_count": cs.get('chunk_count'),
            "first_chunk": (cs.get('chunks') or [{}])[0],
        },
        "program_descriptor": decoded.get('program_descriptor'),
        "task_descriptors": decoded.get('task_descriptors'),
    }
    records.append(record)
    print(f"{case}: size={cs.get('size')} sha256={cs.get('sha256','')[:16]}", flush=True)
    local.unlink()
out = Path('/home/joshuawarren/src/mil-hwx-h14-integ-wt/research/oracles/qwen/2026-09-30-decoded.json')
json.dump(records, open(out, 'w'), indent=1, default=str)
print(f"\nWROTE {len(records)} decoded records to {out}")