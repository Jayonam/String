#include "String.hpp"
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>


using namespace String_lib;

void row(const char *name, bool result)
{
    std::cout << " | " << std::left << std::setw(48) << name << " | "
              << (result ? " PASS " : " FAIL ") << " |" << std::endl;
}

int main()
{
    std::cout << " STRING VALIDATION " << std::endl;

    // --- ГРУППА 1: КОНСТРУКТОРЫ И КОПИРОВАНИЕ ---
    String s_sso("SSO_Test");
    String s_copy_sso(s_sso);
    row("Copy: SSO Integrity", s_copy_sso == s_sso);

    String s_heap_src("Ivory forest played on bg 38 tm but like it so much i "
                      "will listen it once more.");
    String s_heap_cp(s_heap_src);
    row("Copy: Address check", s_heap_cp.c_str() != s_heap_src.c_str());
    row("Copy: Heap Data Integrity", s_heap_cp == s_heap_src);
    row("Memory: Alignment 64-bit check",
        ((uintptr_t)s_heap_cp.c_str() & 63) == 0);

    const char *heap_addr_old = s_heap_src.c_str();
    String s_move_heap(std::move(s_heap_src));
    row("Move: Heap Pointer Stability", s_move_heap.c_str() == heap_addr_old);
    row("Move: Source Invalidation", s_heap_src.GetLength() == 0);

    String cmp1("FANTASMIC_NIGHTWISH");
    String cmp2("FANTASMIC_NIGHTWISH");
    String cmp3("FANTASMIC_NIGHTWIsh");
    row("Operator: == ", cmp1 == cmp2);
    row("Operator: != ", cmp1 != cmp3);

    String mut("Base");
    mut.append(" + Append", 9);
    row("Method: append()", strcmp(mut.c_str(), "Base + Append") == 0);

    mut.push_back('!');
    row("Method: push_back()", mut[mut.GetLength() - 1] == '!');

    String res_test;
    res_test.Reserve(1024);
    row("Method: Reserve() alignment", ((uintptr_t)res_test.c_str() & 63) == 0);

    String op_sum = String("AVX") + String("512");
    row("Operator: + ", op_sum == "AVX512");

    StringView sv(op_sum);
    row("Interface: StringView match",
        sv.GetLength() == 6 && strcmp(sv.c_str(), "AVX512") == 0);

    char c_acc = op_sum[0];
    row("Interface: operator[] access", c_acc == 'A');

    std::cout << " tests completed " << std::endl;

    std::cin.clear();
    // Очистка буфера перед ожиданием
    if (std::cin.rdbuf()->in_avail() > 0)
    {
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    std::cin.get();

    return 0;
}