# Concurrent device-to-host copies into pageable memory corrupt a host page shared by two destinations (ROCm 7.14.1, gfx1201)

## Summary

Two host threads each copy 8 MiB from their own GPU into **pageable** host memory. When the two
destinations share one host page (one ends and the other begins inside the same 4 KiB page, as two
threads' heap allocations can), the bytes of that shared page are sometimes wrong in one of the
destinations after the copy has completed (`hipStreamSynchronize` after `hipMemcpyAsync`, or
`hipMemcpy`). The device data is correct: the same buffer read into pinned memory, or into
pageable memory that shares no page with another in-flight copy, is always right.

## Environment

- ROCm 7.14.1 (`hipRuntimeGetVersion` / `hipDriverGetVersion`: 71460850), AMD SMI 26.5.0
- 2 × AMD Radeon AI PRO R9700 (Navi 48, gfx1201, 32 GB), no peer-to-peer link between them
- Linux 7.1.8 (Debian 13), in-kernel amdgpu driver; firmware CP_MEC1 3430, RLC 12484000,
  SDMA0/1 7966358, PSP_SOSDRV 00.3A.12.14
- One process, one host thread per GPU

## Expected and actual behaviour

Expected: after `hipMemcpyAsync(dst, dev, n, hipMemcpyDeviceToHost, stream)` and
`hipStreamSynchronize(stream)` (or after `hipMemcpy(dst, dev, n, hipMemcpyDeviceToHost)`), `dst`
holds the device bytes, whatever other threads copy concurrently into other host memory.

Actual: when another thread's concurrent device-to-host copy targets memory in the same host
page, the bytes of the shared page in one destination are occasionally wrong: garbage that
matches neither the source, nor the previous contents, nor the other copy's data. The wrong
range is always within the shared page (up to the whole 4,080-byte part of it that belongs to
the destination), and it covers exactly the destination's part of that page.

## Minimal repro

`repro.hip` in this directory (plain HIP, no other library):

```
hipcc -O2 --offload-arch=gfx1201 -o repro repro.hip -lpthread
./repro async-shared 2 busy 2000
./repro sync-shared  2 busy 2000
./repro async-shared 2 idle 2000
./repro pinned       2 busy 2000
./repro async-fresh  2 busy 2000
./repro async-shared 1 busy 2000
```

Each thread fills an 8 MiB device buffer with a known pattern by a kernel, then every round
writes 4 MiB of junk to a second device buffer (pageable host-to-device) and reads the 8 MiB
back into its destination, then compares. `-shared`: one fresh `calloc`'d region per round holds
both threads' destinations back to back at byte `16 + t × 8 MiB`, so the first destination's
last (partial) page is the second destination's first page. `-fresh`: separate `calloc`'d
destinations (no shared page). `busy`: kernels on a second stream of each GPU stream through
mapped coherent host memory during the loop (host-memory traffic from both GPUs, as a
collective library exchanging data through host memory produces).

## Results (2,000 rounds per thread and variant)

| variant                                               | destination          | device 0 bad rounds | device 1 bad rounds         |
| ----------------------------------------------------- | -------------------- | ------------------- | --------------------------- |
| `hipMemcpyAsync`, 2 threads, busy                     | shared page          | 0                   | 1 (68 words, first page)    |
| `hipMemcpy`, 2 threads, busy                          | shared page          | 0                   | 2 (1,016 words, first page) |
| `hipMemcpyAsync`, 2 threads, idle                     | shared page          | 0                   | 1 (660 words, first page)   |
| `hipMemcpyAsync`, pinned destination, 2 threads, busy | pinned               | 0                   | 0                           |
| `hipMemcpyAsync`, 2 threads, busy                     | separate allocations | 0                   | 0                           |
| `hipMemcpyAsync`, 1 thread (device 0), busy           | shared page          | 0                   | —                           |

A second run gave 1/2,000 and 4/2,000 in the first two rows. The wrong rounds are among the first
rounds of a run. In our application (8 MiB reads into freshly allocated heap buffers, two
threads, with two GPUs exchanging data through host memory between the reads) the rate was much
higher: 20 of 240 reads, always the destination's last partial page, exactly `(end address mod
4096) / 4` words, i.e. the bytes of the page shared with the next heap allocation.

## Workaround

Copy through pinned (page-locked, `hipHostMalloc`) staging buffers owned by each thread and
memcpy on the host; no wrong read was seen through pinned memory in any variant.
In our two-GPU application runs (221,000 collective outputs read back and compared in a stress
test, and about 657,000 collective results cross-checked between the GPUs during data-, tensor-,
expert- and pipeline-parallel serving), copying through pinned staging gave no wrong reads.
