# llama.cpp Vulkan on UMA: tensor reads from write-combined memory

Reproduction for a fixed per-request cost on hybrid/recurrent models (Qwen3.5/3.6 MoE family) with
the Vulkan backend on an integrated GPU.

## Symptom

Every request pays roughly 300 ms more than its prompt length explains, most visible on short
prompts over a cached prefix: 2 new tokens take ~320 ms, 8 take ~640 ms.

## Cause

The server saves a context checkpoint per request (the recurrent state, ~63 MiB for a 35B-A3B).
It is read back with `ggml_backend_tensor_get` -> `ggml_vk_buffer_read_2d`. On UMA that function
maps the device buffer and `memcpy`s from it whenever the buffer is `eHostVisible`. On RADV /
Strix Halo the device buffer is allocated from a `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`
memory type without `HOST_CACHED`, i.e. write-combined, and CPU reads from it run at ~210 MB/s.

`perf` during short requests (`results/perf-stock-b10723.txt`): 91% of CPU time in
`__memmove_avx512_unaligned_erms` under `ggml_backend_vk_buffer_get_tensor`.

Memory types on this device (`vulkaninfo`):

| index | heap | flags |
|---|---|---|
| 3, 4 | 1 | DEVICE_LOCAL, HOST_VISIBLE, HOST_COHERENT (device buffers land here) |
| 5, 6 | 0 | HOST_VISIBLE, HOST_COHERENT, HOST_CACHED (staging buffers) |

## Patch

`uma-hostcached-read.patch`: on UMA, `memcpy` directly only when the buffer is also
`eHostCached`; otherwise take the existing staging path (GPU copy into a host-cached staging
buffer, then `memcpy`). Allocation policy is unchanged.

## Results

Radeon 8060S (RADV STRIX_HALO), Mesa 26.0.8, kernel 7.0.0, Ryzen AI Max+ 395.
Model: Qwen3.6-35B-A3B Q4_K_M, `--ctx-size 42000 -fa on --parallel 1`, nothing else loaded on
the GPU. Prompt-eval ms from `/completion` timings.

b10723 (`results/stock-b10723.txt`, `results/patched-b10723.txt`):

| request | tokens | stock ms | patched ms |
|---|---|---|---|
| cached prefix + 1 word | 2 | 319 | 40 |
| cached prefix + 4 words | 8 | 636 | 77 |
| cached prefix + 16 words | 38 | 753 | 185 |
| cached prefix + 51 words | 143 | 861 | 313 |
| cached prefix + 200 words | 690 | 1420 | 875 |
| cached prefix + 800 words | 3090 | 4041 | 3476 |
| cold | 29 | 427 | 138 |
| cold | 92 | 505 | 231 |
| cold | 400 | 717 | 464 |
| cold | 1801 | 2275 | 1748 |

master `1a3011c` (`results/stock-master-1a3011c.txt`, `results/patched-master-1a3011c.txt`):

| request | tokens | stock ms | patched ms |
|---|---|---|---|
| cached prefix + 1 word | 2 | 326 | 42 |
| cached prefix + 4 words | 8 | 631 | 73 |
| cached prefix + 16 words | 38 | 748 | 144 |
| cached prefix + 51 words | 143 | 778 | 229 |
| cached prefix + 200 words | 690 | 1237 | 660 |
| cached prefix + 800 words | 3090 | 3320 | 2663 |
| cold | 29 | 405 | 125 |
| cold | 92 | 445 | 169 |
| cold | 400 | 624 | 359 |
| cold | 1801 | 1919 | 1403 |

Same patch on a production classification workload (Ornith-1.0-35B-A3B Q6_K + MTP, ~5 short
requests per item, 16 items serially): 5.63 -> 2.33 s/item. Logprobs of the yes/no probes are
identical to 4 decimals. Decode speed unchanged (62-64 t/s). An unpatched self-build of the same
tag matches the release binary, so the difference is the patch.

## Run it

```
./run.sh /path/to/llama-server /path/to/Qwen3.6-35B-A3B-Q4_K_M.gguf    # prints the table
./profile.sh /path/to/llama-server /path/to/model.gguf                  # perf, needs sudo
```

Apply the patch to a llama.cpp checkout: `git apply uma-hostcached-read.patch`
(tags before the `ggml-vulkan-buffers.cpp` split keep the function in `ggml-vulkan.cpp`).
