#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from h13_reference import evaluate

MIL = '''program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<bool, [3]> cond, tensor<fp16, [3]> a, tensor<fp16, [3]> b) {
    tensor<fp16, [3]> out = select(a = a, b = b, cond = cond)[name = string("out")];
  } -> (out);
}
'''

def main():
    import struct
    a = struct.pack('<3e', 1, 2, 3)
    b = struct.pack('<3e', 4, 5, 6)
    for cond in (bytes([0, 1, 0]), bytearray([0, 1, 0]), memoryview(bytes([0, 1, 0]))):
        assert evaluate(MIL, Path('.'), {'cond': cond, 'a': a, 'b': b}) == {'out': struct.pack('<3e', 4, 2, 6)}
    for invalid in (b'', b'\x00\x01', b'\x00\x01\x00\x01', b'\x00\x02\x00', b'\x00\xff\x00'):
        try:
            evaluate(MIL, Path('.'), {'cond': invalid, 'a': a, 'b': b})
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted malformed bool input {invalid!r}')
    print('PASS dense bool input, select, malformed values and lengths')

if __name__ == '__main__':
    main()
