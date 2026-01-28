#include "App.hpp"
#include "String.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>


using namespace String_lib;
namespace fs = std::filesystem;
using namespace std::chrono;

struct SilenceGuard
{
    std::streambuf *old;
    std::ofstream dev_null;
    SilenceGuard() : dev_null("nul"), old(std::cout.rdbuf(dev_null.rdbuf()))
    {
    }
    ~SilenceGuard()
    {
        std::cout.rdbuf(old);
    }
};

void RunBenchmark(const std::string &mode_name, size_t target_size_bytes)
{
    const char *in_n = "bench_data.txt";
    const char *out_engine = "strings.txt";
    const char *out_std = "std_out.txt";

    size_t rows = target_size_bytes / 40;
    std::cout << "\n[SCENARIO: " << mode_name
              << "] Size: " << target_size_bytes / (1024 * 1024) << " MB\n";
    {
        std::ofstream f(in_n, std::ios::binary);
        for (size_t i = 0; i < rows; ++i)
        {
            f << "KEY_" << (rand() % 1000000) << "_DATA_CHUNK_" << i << "\n";
        }
    }

    // STRING TEST
    auto s1 = high_resolution_clock::now();
    {
        char *args[] = {(char *)"app", (char *)in_n};
        SilenceGuard sg;
        StartApp(2, args);
    }
    auto e1 = high_resolution_clock::now();
    double t_eng = duration<double>(e1 - s1).count();

    // STD STRING TEST
    auto s2 = high_resolution_clock::now();
    {
        std::ifstream f(in_n);
        std::vector<std::string> v;
        v.reserve(rows);
        std::string line;
        while (std::getline(f, line))
        {
            // like in softwrapper
            for (auto &c : line)
                if (c >= 'A' && c <= 'Z')
                    c |= 0x20;
            v.push_back(std::move(line));
        }
        std::sort(v.begin(), v.end(), std::greater<std::string>());
        std::ofstream out(out_std, std::ios::binary);
        for (const auto &s : v)
            out << s << "\n";
    }
    auto e2 = high_resolution_clock::now();
    double t_std = duration<double>(e2 - s2).count();

    size_t out_count = 0;
    bool sorted = true;
    {
        std::ifstream out(out_engine);
        std::string prev, curr;
        if (std::getline(out, prev))
        {
            out_count++;
            while (std::getline(out, curr))
            {
                out_count++;

                StringView v_prev(prev.c_str(), prev.size());
                StringView v_curr(curr.c_str(), curr.size());

                if (LexisReverseCompare(v_curr, v_prev))
                {
                    sorted = false;
                }
                prev = curr;
            }
        }
    }

    std::cout << "  - Data Integrity : "
              << (out_count == rows ? "PASS" : "FAIL") << " (" << out_count
              << "/" << rows << ")\n";
    std::cout << "  - Sort Logic     : " << (sorted ? "PASS" : "FAIL") << "\n";
    std::cout << "  - STRING Time    : " << std::fixed << std::setprecision(4)
              << t_eng << " s\n";
    std::cout << "  - STD String Time: " << t_std << " s\n";
    std::cout << "  - SPEEDUP        : " << (t_std / t_eng) << "x\n";

    fs::remove(in_n);
    fs::remove(out_std);
}

int main()
{
    std::cout << "--- STRING SORT PERFORMANCE ---\n";

    size_t ram = GetAvailableRam();
    size_t threshold = static_cast<size_t>(ram * RAM_L);

    RunBenchmark("RAM", threshold / 2);

    RunBenchmark("EM", threshold * 1.5);

    bool temp_cleaned = true;
    for (const auto &entry : fs::directory_iterator("."))
    {
        if (entry.path().filename().string().find("temp_") != std::string::npos)
            temp_cleaned = false;
    }
    std::cout << "\n[FINAL] Temp Files Cleaned: "
              << (temp_cleaned ? "YES" : "NO") << "\n";

    std::cout << "\n\033[tests completed" << std::endl;

    std::cin.clear();

    if (std::cin.rdbuf()->in_avail() > 0)
    {
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    std::cin.get();

    return 0;
}