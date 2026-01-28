# String (Test)

String is a high-performance C++17 tes library engineered for high-throughput string manipulation and large-scale data processing. The framework integrates SIMD acceleration (AVX2/AVX-512/SWAR) with a custom memory architecture to surpass the performance limitations of standard library implementations.

Features:

- GigaPool Architecture: Utilizes `thread_local` memory pools and direct system calls (`mmap`/`VirtualAlloc`) to minimize allocation latency.
    
- Arena Allocation: Supports fast, contiguous memory allocation for high-frequency operations.
    
- Memory Alignment: Strict 64-bit alignment for compatibility with vectorized instructions.
    

Hardware Acceleration

- SIMD Intrinsics: Hand-optimized execution paths for AVX2 and AVX-512 instruction sets.
    
- Zero-Copy Operations: Implementation of `StringView` for non-owning string references and efficient move semantics.
    

Large-Scale Processing

- MmapReader: Fast memory-mapped I/O for efficient disk access.
    
- External Merge Sort: Sophisticated algorithm for sorting datasets that exceed physical RAM capacity.

Performance Benchmarks

String Sorting (Radix vs STD):

|   Scenario   |  Data Size  |  STRING (Radix)  |  STD String  |  Speedup  |
| ------------ | ----------- | ---------------- | ------------ | --------- |
|RAM Sort      |    236 MB   |     2.5181 s     |   5.6182 s   |   2.23x   |
|External Merge|    709 MB   |     18.6099 s    |   21.6920 s  |   1.16x   |

Memory Operations & Allocation:

|   Stage   |  Payload  |  Efficiency Speedup  |
| --------- | --------- | -------------------- |
|Small SSO  |    55 B   |        31.26x        |
|Small Pool |   256 B   |         2.25x        |
|Medium Pool|    4 KB   |         1.96x        |
|Large Pool |   32 KB   |         1.86x        |
|Huge System|    1 MB   |         2.02x        |

Comparison Operations (Strcmp)

|       Use Case      |   Target Size   |   Speedup   |
| ------------------- | --------------- | ----------- |
| Small ID / SSO      |       16 B      |    3.43x    |
| Asset Path          |      256 B      |    3.43x    |
| SQL Query String    |      512 B      |    2.97x    |
| Small Buffer / JSON |       2 KB      |    6.81x    |
| Script / Shader     |       4 KB      |    6.24x    |
| Database Page       |      16 KB      |    5.80x    |
| L1/L2 Boundary      |     128 KB      |    3.92x    |

 Core Operations:

- Concatenation: 1.67x faster than `std::string` when repeatedly appending to large buffers.
    
- Deep Path Comparison: 3.23x speedup for 500-byte strings differing only at the tail (SIMD optimized).
    
- Hash Map Flooding: 2.20x speedup in unordered map lookups using CRC32 hardware acceleration.
    
- SSO Copy Speed: 20.93x speedup when passing strings by value, testing small-string optimization efficiency.
    

Build:

- CMake 3.16 or higher
 
- C++17 compliant compiler (Clang 19 recommended)
    
Validation Suite:

The library includes a comprehensive validation suite ensuring data integrity and behavioral correctness.

|  Test Category  |               Component            |  Status  |
| --------------- | ---------------------------------- | -------- |
|Integrity        |SSO Integrity & Address Check       |   PASS   |
|Integrity        |Heap Data Integrity                 |   PASS   |
|Memory           |64-bit Memory Alignment             |   PASS   |
|Memory           |Move Semantics & Pointer Stability  |   PASS   |
|Logic            |Operator Overloading (==, !=, +, [])|   PASS   |
|Interface        |StringView Match & Stability        |   PASS   |
|Methods          |append() and push_back()            |   PASS   |
    
