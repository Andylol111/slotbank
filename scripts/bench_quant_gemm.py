"""2/3/4-bit GEMM times, no 27B load."""
import time
import mlx.core as mx
import mlx.nn as nn

def sync():
    mx.synchronize()

def bench(bits, h=5120, i=17408, n=30, warmup=8):
    y = mx.random.normal((1, 1, h)).astype(mx.bfloat16)
    mx.eval(y)
    lin = nn.Linear(h, i, bias=False).to_quantized(group_size=64, bits=bits)
    mx.eval(lin.parameters())
    def run():
        mx.eval(lin(y))
    for _ in range(warmup):
        run()
        sync()
    t0 = time.perf_counter()
    for _ in range(n):
        run()
        sync()
    return (time.perf_counter() - t0) / n * 1000

for b in (2, 3, 4, 8):
    print(f"q{b}_upproj_ms={bench(b):.2f}", flush=True)
