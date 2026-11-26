"""Failing-first probes for the two reductions in the Parakeet encoder MIL.

Run after `make build/mil-hwxc`:
  python3 tests/test_parakeet_reductions.py build/mil-hwxc
  python3 tests/test_parakeet_reductions.py build/mil-hwxc --target H18G
"""

import argparse
import json
import subprocess
import tempfile
import unittest
from math import prod
from pathlib import Path


def reduction_mil(op, shape, axis, result, keep_dims=False):
    keep = str(keep_dims).lower()
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, {list(shape)}> x) {{
    tensor<int32, [1]> axes = const()[name = string("axes"), val = tensor<int32, [1]>([{axis}])];
    tensor<fp16, {list(result)}> y = {op}(x = x, axes = axes, keep_dims = bool({keep}))[name = string("y")];
  }} -> (y);
}}
'''


class ParakeetReductions(unittest.TestCase):
    compiler = Path('build/mil-hwxc').resolve()
    target = 'H16G'

    def assert_lowers(self, op, shape, axis, result):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mil = root / 'op.mil'
            mil.write_text(reduction_mil(op, shape, axis, result))
            out = root / 'out'
            command = [self.compiler, '--target', self.target, '--mil', mil,
                       '--model-root', root, '--output', out]
            if self.target == 'H18G':
                sample = root / 'input.fp16'
                sample.write_bytes(bytes(2 * prod(shape)))
                command += ['--input', sample]
            proc = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertEqual(manifest['target'], self.target)
            suffix = 'anep' if self.target == 'H18G' else 'hwx'
            self.assertGreater((out / f'program-0.{suffix}').stat().st_size, 0)
            if self.target == 'H16G':
                outputs = [binding['logicalBytes'] for artifact in manifest['artifacts']
                           for binding in artifact['bindings']
                           if binding['role'] == 'output' and binding['identifier'] == 'y']
                self.assertEqual(outputs, [2 * prod(result)])

    def test_attention_mask_input_length_sum(self):
        self.assert_lowers('reduce_sum', (1, 3000), -1, (1,))

    def test_attention_mask_minimum(self):
        self.assert_lowers('reduce_min', (1, 1, 375, 375), 2, (1, 1, 375))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('compiler', type=Path)
    parser.add_argument('--target', choices=('H16G', 'H18G'), default='H16G')
    args = parser.parse_args()
    ParakeetReductions.compiler = args.compiler.resolve()
    ParakeetReductions.target = args.target
    unittest.main(argv=[__file__], verbosity=2)
