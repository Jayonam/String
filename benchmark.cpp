#include "app.hpp"
#include "include/unordered_map.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip> // For formatting
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>


#if defined(__GNUC__) || defined(__clang__)
#define DO_NOT_OPTIMIZE(var) __asm__ volatile("" : : "g"(var) : "memory")
#else
#define DO_NOT_OPTIMIZE(var) ((void)var)
#endif

using namespace std::chrono;

// System prepare
void optimize_system_for_bench()
{
    SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
    SetProcessAffinityMask(GetCurrentProcess(), 1);
}

std::string format_size(size_t bytes)
{
    if (bytes < 1024)
        return std::to_string(bytes) + " B";
    if (bytes < 1024 * 1024)
        return std::to_string(bytes / 1024) + " KB";
    return std::to_string(bytes / 1024 / 1024) + " MB";
}

double get_median(std::vector<double> &times)
{
    if (times.empty())
        return 0;
    std::sort(times.begin(), times.end());
    if (times.size() % 2 == 0)
        return (times[times.size() / 2 - 1] + times[times.size() / 2]) / 2;
    return times[times.size() / 2];
}

double get_stable_average(std::vector<double> &times)
{
    if (times.empty())
        return 0;
    std::sort(times.begin(), times.end());

    size_t trim = times.size() / 10;
    double sum = 0;
    int count = 0;
    for (size_t i = trim; i < times.size() - trim; ++i)
    {
        sum += times[i];
        count++;
    }
    return sum / count;
}

void print_benchmark_header(int id, const std::string &title,
                            const std::string &description)
{
    std::cout << "\n[" << id << "] BENCHMARK: " << title << std::endl;
    std::cout << "Description: " << description << std::endl;
    std::cout << std::string(60, '-') << std::endl;
}

void print_result(const std::string &label, double value,
                  const std::string &unit = "ms")
{
    // Alignment: 20 characters left
    std::cout << std::left << std::setw(20) << "  " + label << ": "
              << std::right << std::setw(10) << std::fixed
              << std::setprecision(4) << value << " " << unit << std::endl;
}

void bench_strcmp_marathon()
{
    print_benchmark_header(1, "STRCMP TEST MARATHON", "STRING vs STD");

    struct Scenario
    {
        size_t size;
        std::string label;
        std::string category;
    };

    std::vector<Scenario> scenarios = {
        {16, "Small ID / SSO", "GAMEDEV"},
        {32, "UUID / Component", "GAMEDEV"},
        {64, "Single ZMM Block", "ENGINE"},
        {96, "Long Prop / Meta", "GAMEDEV"},
        {128, "Long Symbol / JSON", "GAMEDEV"},
        {256, "Asset Path", "GAMEDEV"},
        {512, "SQL Query String", "DB/SQL"},
        {2048, "Small Buffer / JSON", "GAMEDEV"},
        {4096, "Script / Shader", "GAMEDEV"},
        {8192, "Texture             ", "GAMEDEV"},
        {16 * 1024, "Database Page", "DB/SQL"},
        {128 * 1024, "L1/L2 Boundary", "ENGINE"},
        {1024 * 1024, "Message Buffer", "DB/NET"},
        {2 * 1024 * 1024, "Large Data Page", "DB/PAGE"},
        {32 * 1024 * 1024, "L3 Cache Resident", "ENGINE"},
    };

    const int RUN_COUNT = 300;

    for (const auto &sc : scenarios)
    {
        size_t current_size = sc.size;
        std::cout << "\n[STAGE] " << sc.label << " [" << sc.category << "]"
                  << std::endl;
        std::cout << "Target Size: " << format_size(current_size) << std::endl;

        std::string raw(current_size, 'A');
        if (current_size > 2)
            raw[current_size - 2] = 'X';

        String_lib::String my_s1(raw.c_str());
        String_lib::String my_s2(raw.c_str());
        if (current_size > 2)
            my_s2.GetRawData()[current_size - 2] = 'A';

        std::string std_s1 = raw;
        std::string std_s2 = raw;
        if (current_size > 2)
            std_s2[current_size - 2] = 'A';

        auto run_test_series =
            [&](auto &s1, auto &s2, size_t size, bool expected)
        {
            if ((s1 == s2) != expected)
            {
                std::cout << "  [!!!] ERROR: Result mismatch! Size: " << size
                          << std::endl;
                return std::make_pair(-1.0, 0.0);
            }

            int batch_size = 1;
            if (size <= 128)
                batch_size = 5000;
            else if (size <= 1024)
                batch_size = 1000;
            else if (size <= 32 * 1024)
                batch_size = 200;
            else if (size <= 1024 * 1024)
                batch_size = 50;
            else
                batch_size = 1;

            std::vector<double> results;
            results.reserve(RUN_COUNT);

            for (int w = 0; w < 1000; ++w)
            {
                volatile bool r = (s1 == s2);
            }

            for (int i = 0; i < RUN_COUNT; ++i)
            {

                if (size >= 1024 * 1024)
                {
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }

                for (int p = 0; p < 5; ++p)
                {
                    volatile bool r = (s1 == s2);
                }

                auto t1 = high_resolution_clock::now();
                for (int batch = 0; batch < batch_size; ++batch)
                {
                    volatile bool r = (s1 == s2);
                }
                auto t2 = high_resolution_clock::now();

                double ms = duration<double, std::milli>(t2 - t1).count();
                if (ms > 0)
                    results.push_back(ms / (double)batch_size);
            }

            std::sort(results.begin(), results.end());
            double trim_rate = 0.20;
            size_t trim = (size_t)(results.size() * trim_rate);

            if (results.size() <= trim * 2)
                return std::make_pair(results[results.size() / 2], 0.0);

            double sum = 0;
            for (size_t i = trim; i < results.size() - trim; ++i)
            {
                sum += results[i];
            }

            double avg = sum / (results.size() - 2 * trim);
            double jitter =
                ((results[results.size() - trim - 1] - results[trim]) / avg) *
                100.0;

            return std::make_pair(avg, jitter);
        };

        // Execution
        auto [my_avg, my_jitter] =
            run_test_series(my_s1, my_s2, current_size, false);
        auto [std_avg, std_jitter] =
            run_test_series(std_s1, std_s2, current_size, false);

        if (my_avg < 0 || std_avg < 0)
            continue;

        print_result("STRING Stable Avg", my_avg);
        print_result("STD Stable Avg", std_avg);

        double speedup = std_avg / my_avg;
        print_result("TOTAL SPEEDUP", speedup, "x");

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "   [Stats] Jitter: " << my_jitter << "%" << std::endl;

        if (my_jitter > 12.0)
        {
            std::cout << "   [!] WARNING: High noise detected. Results may be "
                         "inaccurate!"
                      << std::endl;
        }
        std::cout << "------------------------------------------" << std::endl;
    }
}

void bench_memcpy_marathon_v4()
{
    print_benchmark_header(3, "MEMCPY MARATHON", "STRING vs STD comparison");

    struct Scenario
    {
        size_t size;
        std::string label;
    };

    std::vector<Scenario> scenarios = {{16, "Small SSO"},
                                       {55, "M SSO (Stk)"},
                                       {128, "Pool Slab 0 (128B)"},
                                       {1024, "Pool Slab 3 (1K)"},
                                       {8192, "Pool Slab 6 (8K)"},
                                       {32760, "Pool Slab 8 (32K - M Pool)"},
                                       {65536, "System (64K - Out of Pool)"}};

    const int RUN_COUNT = 100;
    String_lib::Arena test_arena(1024 * 1024 * 256); // 256 MB

    for (const auto &sc : scenarios)
    {
        size_t size = sc.size;
        std::cout << "\n[STAGE] " << sc.label << " (" << size << " B)"
                  << std::endl;

        std::string raw_content(size, 'X');

        for (size_t i = 0; i < size; ++i)
            raw_content[i] = (char)('A' + (i % 26));

        String_lib::String engine_src(raw_content.c_str());
        std::string std_src = raw_content;

        int batch = (size <= 55) ? 50000 : (size <= 4096 ? 5000 : 500);

        auto verify_data = [&](const char *ptr)
        {
            if (std::memcmp(ptr, raw_content.c_str(), size) != 0)
            {
                std::cerr << "!!! DATA CORRUPTION !!! at size " << size
                          << std::endl;
                std::exit(-1);
            }
        };

        auto run_engine = [&]()
        {
            std::vector<double> res;
            for (int i = 0; i < RUN_COUNT; ++i)
            {
                auto t1 = std::chrono::high_resolution_clock::now();
                for (int b = 0; b < batch; ++b)
                {
                    String_lib::String copy(engine_src);
                    if (UNLIKELY(b == 0))
                        verify_data(copy.c_str());
                    __asm__ volatile("" : : "r"(copy.c_str()) : "memory");
                }
                auto t2 = std::chrono::high_resolution_clock::now();
                res.push_back(
                    std::chrono::duration<double, std::milli>(t2 - t1).count() /
                    batch);
            }
            return get_stable_average(res);
        };

        auto run_arena = [&]()
        {
            std::vector<double> res;
            for (int i = 0; i < RUN_COUNT; ++i)
            {
                test_arena.reset();
                auto t1 = std::chrono::high_resolution_clock::now();
                for (int b = 0; b < batch; ++b)
                {
                    String_lib::String copy(engine_src, &test_arena);
                    if (UNLIKELY(b == 0))
                        verify_data(copy.c_str());
                    __asm__ volatile("" : : "r"(copy.c_str()) : "memory");
                }
                auto t2 = std::chrono::high_resolution_clock::now();
                res.push_back(
                    std::chrono::duration<double, std::milli>(t2 - t1).count() /
                    batch);
            }
            return get_stable_average(res);
        };

        auto run_std = [&]()
        {
            std::vector<double> res;
            for (int i = 0; i < RUN_COUNT; ++i)
            {
                auto t1 = std::chrono::high_resolution_clock::now();
                for (int b = 0; b < batch; ++b)
                {
                    std::string copy(std_src);
                    if (UNLIKELY(b == 0))
                        verify_data(copy.c_str());
                    __asm__ volatile("" : : "r"(copy.c_str()) : "memory");
                }
                auto t2 = std::chrono::high_resolution_clock::now();
                res.push_back(
                    std::chrono::duration<double, std::milli>(t2 - t1).count() /
                    batch);
            }
            return get_stable_average(res);
        };

        double t_pool = run_engine();
        double t_arena = run_arena();
        double t_std = run_std();

        print_result("STD String     ", t_std);
        print_result("STRING (Pool)  ", t_pool);
        print_result("STRING (Arena) ", t_arena);

        std::cout << std::setprecision(2) << std::fixed;
        std::cout << "  => Pool vs STD:  " << (t_std / t_pool) << "x speedup"
                  << std::endl;
        std::cout << "  => Arena vs STD: " << (t_std / t_arena) << "x speedup"
                  << std::endl;
        std::cout << "------------------------------------------" << std::endl;
    }
}

void bench_asset_sorting()
{
    print_benchmark_header(
        2, "2M ASSET TICKETS SORT",
        "Radix (Tickets) vs std::sort (Strings) | REVERSE ORDER");

    const int N = 2000000;
    const int RUN_COUNT = 10;

    char *arena = (char *)malloc(N * 64);
    std::vector<SoftTicket> tickets_master;
    std::vector<std::string> std_strings_master;

    tickets_master.reserve(N);
    std_strings_master.reserve(N);

    srand(42);
    for (int i = 0; i < N; ++i)
    {
        char *ptr = arena + (i * 64);
        std::string raw = "Asset_Name_Stress_Test_Long_Prefix_" +
                          std::to_string(rand() % 1000) + "_" +
                          std::to_string(i);
        memcpy(ptr, raw.c_str(), raw.size());
        ptr[raw.size()] = '\0';

        tickets_master.push_back(
            {String_lib::StringView(ptr, raw.size()), (size_t)i});
        std_strings_master.push_back(raw);
    }

    std::vector<double> engine_times;
    for (int r = 0; r < RUN_COUNT; ++r)
    {
        auto test_container = tickets_master;

        auto t1 = high_resolution_clock::now();

        Radix_Internal::RadSort(test_container.data(),
                                test_container.data() + test_container.size(),
                                0);
        auto t2 = high_resolution_clock::now();

        if (r == 0)
        {
            for (size_t i = 1; i < test_container.size(); ++i)
            {

                if (LexisReverseCompare(test_container[i].view,
                                        test_container[i - 1].view))
                {
                    std::cerr << "!!! RADIX SORT ERROR AT INDEX " << i
                              << " !!!\n";
                    std::exit(-1);
                }
            }
        }
        engine_times.push_back(duration<double, std::milli>(t2 - t1).count());
    }

    std::vector<double> std_times;
    for (int r = 0; r < RUN_COUNT; ++r)
    {
        auto test_container = std_strings_master;

        auto t1 = high_resolution_clock::now();

        std::sort(test_container.begin(), test_container.end(),
                  std::greater<std::string>());
        auto t2 = high_resolution_clock::now();

        std_times.push_back(duration<double, std::milli>(t2 - t1).count());
    }

    auto get_stable_avg = [](std::vector<double> &v)
    {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    double my_avg = get_stable_avg(engine_times);
    double std_avg = get_stable_avg(std_times);

    print_result("RADIX", my_avg, "ms");
    print_result("STD Sort (String + Greater)   ", std_avg, "ms");
    print_result("TOTAL SPEEDUP", std_avg / my_avg, "x");

    std::cout
        << "  - Status: DATA INTEGRITY PASS (Verified by LexisReverseCompare)"
        << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    free(arena);
}

void bench_alignment_torture()
{
    print_benchmark_header(
        3, "MISALIGNED ACCESS (10M ITER)",
        "Measures performance when accessing unaligned data (Offset 3 bytes)");
    const char *s1 = "   Aligned_Data_Prefix_Long_String_2077";
    const char *s2 = "   Aligned_Data_Prefix_Long_String_2048";

    auto t1 = high_resolution_clock::now();
    volatile int res1 = 0;
    for (int i = 0; i < 10000000; ++i)
    {
        res1 += String_lib::c_strcmp(s1 + 3, s2 + 3, 30);
    }
    auto t2 = high_resolution_clock::now();

    std::string str1(s1 + 3, 30);
    std::string str2(s2 + 3, 30);
    auto t3 = high_resolution_clock::now();
    volatile int res2 = 0;
    for (int i = 0; i < 10000000; ++i)
    {
        res2 += (str1 == str2);
    }
    auto t4 = high_resolution_clock::now();

    double my_t = duration<double, std::milli>(t2 - t1).count();
    double std_t = duration<double, std::milli>(t4 - t3).count();
    double ratio = std_t / my_t;

    print_result("STRING", my_t);
    print_result("STD String ==", std_t);
    print_result("Win/Loss", ratio, "x");
}

// SSO
void bench_small_string_spam()
{
    const int N = 5000000;
    const int WARMUP = 100000;

    const char *test_data[] = {
        "Small_Str_1",     "Shorty",  "Test_Data_Long", "Tiny",
        "Medium_Length_S", "7Bytes!", "AnotherOne",     "HereOne"};

    auto run_engine = [&]()
    {
        for (int i = 0; i < N; ++i)
        {

            String_lib::String s(test_data[i & 7]);
            const char *p = s.c_str();
            __asm__ volatile("" : : "g"(p) : "memory");
        }
    };

    auto run_std = [&]()
    {
        for (int i = 0; i < N; ++i)
        {
            std::string s(test_data[i & 7]);
            const char *p = s.c_str();
            __asm__ volatile("" : : "g"(p) : "memory");
        }
    };

    for (int i = 0; i < WARMUP; ++i)
    {
        String_lib::String s(test_data[i & 7]);
        std::string s2(test_data[i & 7]);
    }

    auto t1 = high_resolution_clock::now();
    run_engine();
    auto t2 = high_resolution_clock::now();

    auto t3 = high_resolution_clock::now();
    run_std();
    auto t4 = high_resolution_clock::now();

    double my_t = duration<double, std::milli>(t2 - t1).count();
    double std_t = duration<double, std::milli>(t4 - t3).count();

    print_result("String", my_t);
    print_result("STD String", std_t);
    print_result("Efficiency", std_t / my_t, "x speedup");
}

void bench_concat_marathon()
{
    print_benchmark_header(5, "CONCATENATION MARATHON",
                           "Repeatedly appending small strings to a large "
                           "buffer. Tests Reserve/Realloc logic.");
    const int N = 1000000;
    const char *part = "Data_Chunk_";

    auto t1 = high_resolution_clock::now();
    String_lib::String s1;
    s1.Reserve(N * 12);
    for (int i = 0; i < N; ++i)
    {
        s1 += part;
    }
    auto t2 = high_resolution_clock::now();

    auto t3 = high_resolution_clock::now();
    std::string s2;
    s2.reserve(N * 12);
    for (int i = 0; i < N; ++i)
    {
        s2 += part;
    }
    auto t4 = high_resolution_clock::now();

    double my_t = duration<double, std::milli>(t2 - t1).count();
    double std_t = duration<double, std::milli>(t4 - t3).count();

    print_result("STRING Concat", my_t);
    print_result("STD Concat", std_t);
    print_result("Efficiency", std_t / my_t, "x speedup");
}

void bench_deep_path_cmp()
{
    print_benchmark_header(
        6, "DEEP PATH COMPARISON",
        "Compares strings that are 500 bytes long and differ only at the very "
        "e. Tests SIMD tail efficiency.");
    const int N = 1000000;
    std::string base(500, 'a');
    std::string p1 = base + "file_v1.txt";
    std::string p2 = base + "file_v2.txt";

    String_lib::String my1(p1.c_str());
    String_lib::String my2(p2.c_str());

    auto t1 = high_resolution_clock::now();
    volatile int res1 = 0;
    for (int i = 0; i < N; ++i)
    {
        res1 += (my1 == my2);
    }
    auto t2 = high_resolution_clock::now();

    auto t3 = high_resolution_clock::now();
    volatile int res2 = 0;
    for (int i = 0; i < N; ++i)
    {
        res2 += (p1 == p2);
    }
    auto t4 = high_resolution_clock::now();

    double my_t = duration<double, std::milli>(t2 - t1).count();
    double std_t = duration<double, std::milli>(t4 - t3).count();

    print_result("STRING Path Cmp", my_t);
    print_result("STD Path Cmp", std_t);
    print_result("Efficiency", std_t / my_t, "x speedup");
}

struct StringHasher
{
    using is_transparent = void;

    [[clang::always_inline]]
    static uint64_t compute_hash(const char *str, size_t len)
    {

        uint64_t h = 0x9e3779b97f4a7c15ULL;

        size_t blocks = len / 8;
        const uint64_t *p = reinterpret_cast<const uint64_t *>(str);

        for (size_t i = 0; i < blocks; ++i)
        {
            uint64_t block;

            __builtin_memcpy(&block, p + i, 8);
            h = _mm_crc32_u64(h, block);
        }

        if (len & 7)
        {
            uint64_t tail = 0;
            __builtin_memcpy(&tail, str + (len & ~7ULL), len & 7);
            h = _mm_crc32_u64(h, tail);
        }

        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;

        return h;
    }

    size_t operator()(const String_lib::String &s) const
    {
        return compute_hash(s.c_str(), s.GetLength());
    }

    size_t operator()(const std::string &s) const
    {
        return compute_hash(s.data(), s.length());
    }

    size_t operator()(const char *s) const
    {
        return compute_hash(s, std::strlen(s));
    }
};

struct StringEq
{
    using is_transparent = void;
    ALWAYS_INLINE bool operator()(const String_lib::String &lhs,
                                  const String_lib::String &rhs) const
    {
        return lhs == rhs;
    }
};

void bench_hash_map_flooding()
{
    print_benchmark_header(7, "HASH MAP FLOODING",
                           "Lookup in unordered map with CRC32.");

    const int N = 100000;
    const int M = 1000000;

    std::vector<std::string> raw_keys;
    std::vector<String_lib::String> engine_keys;
    raw_keys.reserve(N);
    engine_keys.reserve(N);

    for (int i = 0; i < N; ++i)
    {
        raw_keys.push_back("key_prefix_" + std::to_string(i) +
                           "_long_suffix_to_test_simd_and_avoid_sso");
    }

    for (int i = 0; i < N; ++i)
    {
        engine_keys.emplace_back(raw_keys[i].c_str());
    }

    ankerl::unordered_dense::map<String_lib::String, int, StringHasher,
                                 StringEq>
        my_map;

    my_map.reserve(N);
    std::unordered_map<std::string, int> std_map;

    for (int i = 0; i < N; ++i)
    {
        my_map.emplace(engine_keys[i], i);
        std_map[raw_keys[i]] = i;
    }

    auto t1 = high_resolution_clock::now();
    volatile int res1 = 0;
    for (int i = 0; i < M; ++i)
    {

        const auto &key_to_find = engine_keys[i % N];
        auto it = my_map.find(key_to_find);
        if (LIKELY(it != my_map.end()))
        {
            res1 += it->second;
        }
    }
    auto t2 = high_resolution_clock::now();

    auto t3 = high_resolution_clock::now();
    volatile int res2 = 0;
    for (int i = 0; i < M; ++i)
    {
        const std::string &key_to_find = raw_keys[i % N];
        auto it = std_map.find(key_to_find);
        if (LIKELY(it != std_map.end()))
        {
            res2 += it->second;
        }
    }
    auto t4 = high_resolution_clock::now();

    double my_t = duration<double, std::milli>(t2 - t1).count();
    double std_t = duration<double, std::milli>(t4 - t3).count();

    std::cout << "STRING Map Lookup (CRC32): " << my_t << " ms" << std::endl;
    std::cout << "STD Map Lookup: " << std_t << " ms" << std::endl;
    std::cout << "Speedup: " << std_t / my_t << "x" << std::endl;
}

void bench_threshold_bounce()
{
    print_benchmark_header(
        8, "HEAP THRESHOLD BOUNCE",
        "Strings fluctuating 54-57 bytes. Tests SSO vs Heap cost.");
    const int N = 150000;
    const char *s54 = "123456789012345678901234567890123456789012345678901234";
    const char *s57 =
        "123456789012345678901234567890123456789012345678901234567";

    auto t1 = high_resolution_clock::now();
    for (int i = 0; i < N; ++i)
    {
        {
            String_lib::String s((i & 1) ? s54 : s57);
            __asm__ volatile("" : : "g"(s.c_str()) : "memory");
        }
    }
    auto t2 = high_resolution_clock::now();

    auto t3 = high_resolution_clock::now();
    for (int i = 0; i < N; ++i)
    {
        {
            std::string s((i & 1) ? s54 : s57);
            __asm__ volatile("" : : "g"(s.c_str()) : "memory");
        }
    }
    auto t4 = high_resolution_clock::now();

    print_result("STRING (SSO/Heap mix)",
                 duration<double, std::milli>(t2 - t1).count());
    print_result("STD (Always Heap)",
                 duration<double, std::milli>(t4 - t3).count());
}

void bench_cache_jumping()
{
    print_benchmark_header(
        9, "CACHE JUMPING",
        "Random access to strings. Tests cache-line efficiency (64b).");
    const int N = 10000;
    const int M = 1000000;

    std::vector<String_lib::String> my_vec;
    std::vector<std::string> std_vec;
    for (int i = 0; i < N; ++i)
    {
        my_vec.push_back("test_string_for_cache_locality");
        std_vec.push_back("test_string_for_cache_locality");
    }

    std::vector<int> indices(M);
    for (int i = 0; i < M; ++i)
        indices[i] = rand() % N;

    auto t1 = high_resolution_clock::now();
    volatile size_t sum1 = 0;
    for (int i = 0; i < M; ++i)
    {

        if (LIKELY(i + 16 < M))
        {
            __builtin_prefetch(&my_vec[indices[i + 16]], 0, 1);
        }
        sum1 += my_vec[indices[i]].GetLength();
    }
    auto t2 = high_resolution_clock::now();

    auto t3 = high_resolution_clock::now();
    volatile size_t sum2 = 0;
    for (int i = 0; i < M; ++i)
        sum2 += std_vec[indices[i]].length();
    auto t4 = high_resolution_clock::now();

    double engine_t = duration<double, std::milli>(t2 - t1).count();
    double std_t = duration<double, std::milli>(t4 - t3).count();

    print_result("STRIMG Random Access", engine_t);
    print_result("STD Random Access", std_t);

    print_result("Efficiency", std_t / engine_t, "x speedup");
}

void bench_fragmented_concat_scenarios()
{
    print_benchmark_header(
        10, "FRAGMENTED CONCAT SCENARIOS",
        "Measuring push_back(char) performance across different target sizes.");

    struct Scenario
    {
        size_t target_size;
        std::string label;
    };

    std::vector<Scenario> scenarios = {
        {32, "Small SSO (32B)"},      {55, "Max SSO (55B)"},
        {256, "Small Pool (256B)"},   {4096, "Medium Pool (4KB)"},
        {32760, "Large Pool (32KB)"}, {1048576, "Huge System (1MB)"}};

    const int RUN_COUNT = 100;

    for (const auto &sc : scenarios)
    {
        size_t N = sc.target_size;

        int batch = (N <= 55) ? 10000 : (N <= 4096 ? 1000 : 50);

        std::cout << "\n[STAGE] " << sc.label << " | Appending " << N
                  << " chars" << std::endl;

        {
            String_lib::String prewarm;
            for (size_t i = 0; i < N; ++i)
                prewarm.push_back('W');
        }

        auto run_engine = [&]()
        {
            std::vector<double> results;
            for (int r = 0; r < RUN_COUNT; ++r)
            {
                auto t1 = high_resolution_clock::now();
                for (int b = 0; b < batch; ++b)
                {
                    String_lib::String s;
                    for (size_t i = 0; i < N; ++i)
                    {
                        s.push_back('A');
                    }

                    __asm__ volatile("" : : "r"(s.c_str()) : "memory");
                }
                auto t2 = high_resolution_clock::now();
                results.push_back(
                    duration<double, std::milli>(t2 - t1).count() / batch);
            }
            return get_stable_average(results);
        };

        auto run_std = [&]()
        {
            std::vector<double> results;
            for (int r = 0; r < RUN_COUNT; ++r)
            {
                auto t1 = high_resolution_clock::now();
                for (int b = 0; b < batch; ++b)
                {
                    std::string s;
                    for (size_t i = 0; i < N; ++i)
                    {
                        s.push_back('A');
                    }
                    __asm__ volatile("" : : "r"(s.c_str()) : "memory");
                }
                auto t2 = high_resolution_clock::now();
                results.push_back(
                    duration<double, std::milli>(t2 - t1).count() / batch);
            }
            return get_stable_average(results);
        };

        double t_engine = run_engine();
        double t_std = run_std();

        print_result("Engine", t_engine, "ms/op");
        print_result("STD String  ", t_std, "ms/op");

        std::cout << std::setprecision(2) << std::fixed;
        double speedup = t_std / t_engine;
        std::cout << "  => Efficiency: " << speedup << "x "
                  << "x speedup" << std::endl;
        std::cout << "------------------------------------------" << std::endl;
    }
}

__attribute__((noinline)) void sink_my(String_lib::String s)
{
    __asm__ volatile("" : : "g"(s.c_str()) : "memory");
}
__attribute__((noinline)) void sink_std(std::string s)
{
    __asm__ volatile("" : : "g"(s.c_str()) : "memory");
}

void bench_small_talk()
{
    print_benchmark_header(11, "SMALL-TALK INTERFACE",
                           "Passing strings by value. Tests SSO copy speed.");
    const int N = 5000000;
    String_lib::String my_s("SSO_Test_String_Content");
    std::string std_s("SSO_Test_String_Content");

    auto t1 = high_resolution_clock::now();
    for (int i = 0; i < N; ++i)
        sink_my(my_s);
    auto t2 = high_resolution_clock::now();

    auto t3 = high_resolution_clock::now();
    for (int i = 0; i < N; ++i)
        sink_std(std_s);
    auto t4 = high_resolution_clock::now();

    double my_t = duration<double, std::milli>(t2 - t1).count();
    double std_t = duration<double, std::milli>(t4 - t3).count();

    print_result("STRING Copy Cost", my_t);
    print_result("STD Copy Cost", std_t);

    print_result("Efficiency", std_t / my_t, "x speedup");
}

void bench_radix_vs_std_performance()
{
    print_benchmark_header(10, "RADIX VS STD",
                           "Full-scale sort competition (500k strings).");

    const int N = 500000;
    const int MAX_STR_LEN = 32;
    std::mt19937 rng(2026);
    std::uniform_int_distribution<int> dist(32, 126);

    std::vector<std::string> pool(N);
    std::vector<String_lib::StringView> engine_vec;
    std::vector<String_lib::StringView> std_vec;

    for (int i = 0; i < N; ++i)
    {
        int len = 8 + (rng() % MAX_STR_LEN);
        for (int j = 0; j < len; ++j)
            pool[i] += (char)dist(rng);
        engine_vec.push_back({pool[i].c_str(), pool[i].length()});
    }
    std_vec = engine_vec;

    Radix_Internal::RadSort(engine_vec.data(), engine_vec.data() + 100, 0);

    auto t1 = high_resolution_clock::now();
    Radix_Internal::RadSort(engine_vec.data(),
                            engine_vec.data() + engine_vec.size(), 0);
    auto t2 = high_resolution_clock::now();

    auto t3 = high_resolution_clock::now();
    std::sort(std_vec.begin(), std_vec.end(),
              [](const auto &a, const auto &b)
              {
                  if constexpr (std::is_same_v<decltype(a),
                                               const String_lib::StringView &>)
                      return LexisReverseCompare(a, b);
                  else
                      return LexisReverseCompare(a.view, b.view);
              });
    auto t4 = high_resolution_clock::now();

    double engine_t = duration<double, std::milli>(t2 - t1).count();
    double std_t = duration<double, std::milli>(t4 - t3).count();

    bool integrity = true;
    for (int i = 0; i < N; ++i)
    {
        if (engine_vec[i].c_str() != std_vec[i].c_str())
        {
            integrity = false;
            break;
        }
    }

    print_result("Data Integrity", integrity ? 1.0 : 0.0,
                 integrity ? "PASS" : "FAIL");
    print_result("STRING Radix Time", engine_t, "ms");
    print_result("STD Sort Time", std_t, "ms");
    print_result("Efficiency", std_t / engine_t, "x speedup");
}

int main()
{
    volatile void *ptr = &String_lib::t_gigaPool;
    optimize_system_for_bench();
    bench_radix_vs_std_performance();
    bench_memcpy_marathon_v4(); // c_memcpy not valid() builtin much more
                                // optimized -U but retain for legacy
    bench_strcmp_marathon();
    bench_asset_sorting();
    // bench_alignment_torture(); // nit valid until stringview with loadu
    bench_small_string_spam();
    bench_concat_marathon();
    bench_deep_path_cmp();
    bench_hash_map_flooding(); // erm something wrong maybe size is large try
                               // flat map -d changed to unordered dense
    bench_threshold_bounce();  // well this size strings not used anyway )
    bench_fragmented_concat_scenarios();
    bench_small_talk();
    bench_cache_jumping(); // well i have twice the volume so good

    std::cin.clear();
    while (std::cin.peek() != EOF && std::cin.get() != '\n')
        ;

    std::cin.get();
    std::cin.get();
    system("pause");
    return 0;
}