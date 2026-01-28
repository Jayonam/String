#include "String.hpp"
#include <fstream>
#include <queue>
#include <sys/stat.h>
#include <vector>

// #define _WIN32_WINNT 0x0501

#ifdef _WIN32
#include <windows.h>
#define WIN32_LEAN_AND_MEAN
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <unistd.h>

#endif

#define LOG_DEBUG(msg)                                                         \
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] " << msg             \
              << " | current: " << current                                     \
              << " | chunk_size: " << chunk.size() << std::endl

constexpr float RAM_L = .05f;

inline size_t GetAvailableRam()
{
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return status.ullAvailPhys;
#else
    struct sysinfo info;
    if (sysinfo(&info) == 0)
        return (size_t)info.freeram * info.mem_unit;
    return 8ULL * 1024 * 1024 * 1024;
#endif
}

struct MmapReader
{
    const char *m_data = nullptr;
    size_t m_size = 0;
    size_t m_offset = 0;

#ifdef _WIN32
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMapping = NULL;
#else
    int fd = -1;
#endif

    MmapReader(const char *filename)
    {
#ifdef _WIN32
        hFile = CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile != INVALID_HANDLE_VALUE)
        {
            m_size = GetFileSize(hFile, NULL);
            hMapping =
                CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
            if (hMapping)
                m_data = (const char *)MapViewOfFile(hMapping, FILE_MAP_READ, 0,
                                                     0, 0);
        }
#else
        fd = open(filename, O_RDONLY);
        struct stat st;
        if (fd != -1 && fstat(fd, &st) == 0)
        {
            m_size = st.st_size;
            m_data =
                (const char *)mmap(NULL, m_size, PROT_READ, MAP_PRIVATE, fd, 0);
        }
#endif
    }

    ~MmapReader()
    {
#ifdef _WIN32
        if (m_data)
            UnmapViewOfFile(m_data);
        if (hMapping)
            CloseHandle(hMapping);
        if (hFile != INVALID_HANDLE_VALUE)
            CloseHandle(hFile);
#else
        if (m_data)
            munmap((void *)m_data, m_size);
        if (fd != -1)
            close(fd);
#endif
    }

    // read [len_norm][data_norm][len_orig][data_orig]
    bool next_pair(String_lib::StringView &norm, String_lib::StringView &orig)
    {
        if (UNLIKELY(m_offset + sizeof(uint32_t) > m_size))
            return false;

        uint32_t n_len;
        __builtin_memcpy(&n_len, m_data + m_offset, sizeof(uint32_t));
        m_offset += sizeof(uint32_t);

        norm = String_lib::StringView(m_data + m_offset, n_len);
        m_offset += n_len;

        if (UNLIKELY(m_offset + sizeof(uint32_t) > m_size))
            return false;

        uint32_t o_len;
        __builtin_memcpy(&o_len, m_data + m_offset, sizeof(uint32_t));
        m_offset += sizeof(uint32_t);

        orig = String_lib::StringView(m_data + m_offset, o_len);
        m_offset += o_len;

        return true;
    }
};

struct SoftWrapper
{
  public:
    String_lib::String original;
    String_lib::String normalized;

    SoftWrapper(String_lib::String &&s)
        : original(String_lib::move(s)), normalized(original.GetLength())
    {
        size_t n = original.GetLength();
        const char *src = original.c_str();
        char *dst = normalized.GetRawData();

        VECTORIZE
        for (size_t i = 0; i < n; ++i)
        {
            // TODO: add ICU
            char c = src[i];
            if (c >= 'A' && c <= 'Z')
            {
                c |= 0x20;
            }
            dst[i] = c;
        }
    }
};

// TODO: Blocks -d
__attribute__((always_inline)) inline bool
LexisReverseCompare(const String_lib::StringView &lhs,
                    const String_lib::StringView &rhs)
{
    size_t len1 = lhs.GetLength();
    size_t len2 = rhs.GetLength();
    size_t min_len = (len1 < len2) ? len1 : len2;

    const char *p1 = lhs.c_str();
    const char *p2 = rhs.c_str();

    // Prepass
    if (min_len >= 8)
    {
        uint64_t u1 =
            __builtin_bswap64(*reinterpret_cast<const uint64_t *>(p1));
        uint64_t u2 =
            __builtin_bswap64(*reinterpret_cast<const uint64_t *>(p2));
        if (u1 != u2)
            return u1 > u2;

        int res = String_lib::c_strcmp_unaligned(p1 + 8, p2 + 8, min_len - 8);
        if (res != 0)
            return res > 0;
    }
    else
    {

        int res = String_lib::c_strcmp_unaligned(p1, p2, min_len);
        if (res != 0)
            return res > 0;
    }

    return len1 > len2;
}

struct SoftTicket
{
    String_lib::StringView view;
    size_t original_idx;

    bool operator>(const SoftTicket &other) const
    {
        return LexisReverseCompare(view, other.view);
    }
};

namespace Radix_Internal
{

template <typename T>
ALWAYS_INLINE uint8_t get_byte(const T &ticket, size_t offset)
{
    if constexpr (std::is_same_v<T, String_lib::StringView>)
    {
        return (offset < ticket.GetLength())
                   ? static_cast<uint8_t>(ticket.c_str()[offset])
                   : 0;
    }
    else
    {
        return (offset < ticket.view.GetLength())
                   ? static_cast<uint8_t>(ticket.view.c_str()[offset])
                   : 0;
    }
}

template <typename T> void RadSortRecursive(T *begin, T *end, size_t offset)
{
    const size_t count = end - begin;

    if (count < 32)
    {
        for (T *i = begin + 1; i < end; ++i)
        {
            T val = std::move(*i);
            T *j = i;
            while (j > begin && ([&](){
                if constexpr (std::is_same_v<T, String_lib::StringView>) return LexisReverseCompare(val, *(j - 1));
                else return LexisReverseCompare(val.view, (j - 1)->view);
            }()))
            {
                *j = std::move(*(j - 1));
                --j;
            }
            *j = std::move(val);
        }
        return;
    }

    uint8_t *bytes = nullptr;
    std::unique_ptr<uint8_t[]> heap_buffer;

    if (count <= 65536)
    {
        bytes = (uint8_t *)alloca(count);
    }
    else
    {
        heap_buffer.reset(new uint8_t[count]);
        bytes = heap_buffer.get();
    }

    uint32_t counts[256] = {0};
    for (size_t i = 0; i < count; ++i)
    {
        uint8_t b = get_byte(begin[i], offset);
        bytes[i] = b;
        counts[b]++;
    }

    uint32_t offsets[256];
    uint32_t pos = 0;
    for (int j = 255; j >= 0; --j)
    {
        offsets[j] = pos;
        pos += counts[j];
    }
    uint32_t active_offsets[256];
    std::copy(std::begin(offsets), std::end(offsets),
              std::begin(active_offsets));

    for (int j = 255; j >= 0; --j)
    {
        if (counts[j] == 0)
            continue;
        uint32_t limit = offsets[j] + counts[j];
        while (active_offsets[j] < limit)
        {
            uint32_t curr_idx = active_offsets[j];
            T current = std::move(begin[curr_idx]);
            uint8_t b = bytes[curr_idx];
            while (b != j)
            {
                uint32_t dest_idx = active_offsets[b]++;
                std::swap(current, begin[dest_idx]);
                std::swap(b, bytes[dest_idx]);
            }
            begin[active_offsets[j]++] = std::move(current);
        }
    }

    if (offset < 64)
    {
        uint32_t current_pos = 0;
        for (int j = 255; j >= 1; --j)
        {
            if (counts[j] > 1)
            {
                RadSortRecursive(begin + current_pos,
                                 begin + current_pos + counts[j], offset + 1);
            }
            current_pos += counts[j];
        }
    }
}

// interface
template <typename T> void RadSort(T *begin, T *end, size_t offset)
{
    if (end - begin < 2)
        return;
    RadSortRecursive(begin, end, offset);
}
} // namespace Radix_Internal

// DO NOT USE
/*
inline void QuickSort(std::vector<SoftTicket>& tickets, int left, int right)
{

    if(left >= right) return;

    while(left < right)
    {

        SoftTicket pivot = tickets[left + (right - left) / 2];
        int i = left;
        int j = right;


        while (i <= j)
        {

            while (tickets[i] > pivot) ++i;
            while (pivot > tickets[j]) --j;

            if(i <= j)
            {

                String_lib::swap(tickets[i], tickets[j]);
                ++i;
                --j;
            }
        }


        if(j - left < right - i)
        {
            if(left < j) QuickSort(tickets, left, j);
            left = i;
        }
        else
        {
            if(i < right) QuickSort(tickets, i, right);
            right = j;
        }
    }
}

    template<typename It, typename Comp>
inline void QuickSort(It left, It right, Comp comp) {
     while (left < right) {
        It mid = left + (right - left) / 2;


        if (comp(*mid, *left)) {
            auto t = *mid; *mid = *left; *left = t;
        }
        if (comp(*right, *left)) {
            auto t = *right; *right = *left; *left = t;
        }
        if (comp(*right, *mid)) {
            auto t = *right; *right = *mid; *mid = t;
        }

        auto pivot = *mid;
        It i = left;
        It j = right;

        while (i <= j) {
            while (comp(*i, pivot)) ++i;
            while (comp(pivot, *j)) --j;
            if (i <= j) {

                auto temp = *i;
                *i = *j;
                *j = temp;
                ++i; --j;
            }
        }


        if (j - left < right - i) {
            if (left < j) QuickSort(left, j, comp);
            left = i;
        } else {
            if (i < right) QuickSort(i, right, comp);
            right = j;
        }
    }
}
*/
// EM
struct MergeNode
{
    String_lib::StringView normalized_view;
    String_lib::StringView original_view;
    MmapReader *reader;

    MergeNode(String_lib::StringView nv, String_lib::StringView ov,
              MmapReader *r)
        : normalized_view(nv), original_view(ov), reader(r)
    {
    }

    MergeNode(MergeNode &&other) noexcept = default;
    MergeNode &operator=(MergeNode &&other) noexcept = default;
    MergeNode(const MergeNode &) = default;
    MergeNode &operator=(const MergeNode &) = default;

    bool operator<(const MergeNode &other) const
    {
        return LexisReverseCompare(other.normalized_view,
                                   this->normalized_view);
    }
};

inline void FastExternalMerge(std::vector<String_lib::String> &temp_files)
{

    {
        std::vector<std::unique_ptr<MmapReader>> readers;
        std::priority_queue<MergeNode> pq;

        for (auto &filename : temp_files)
        {
            auto reader = std::make_unique<MmapReader>(filename.c_str());
            String_lib::StringView nv, ov;

            if (reader->next_pair(nv, ov))
            {
                pq.push(MergeNode(nv, ov, reader.get()));
            }
            readers.push_back(std::move(reader));
        }

        std::ofstream outFile("strings.txt", std::ios::binary);
        std::vector<char> write_buf(256 * 1024);
        outFile.rdbuf()->pubsetbuf(write_buf.data(), write_buf.size());

        while (!pq.empty())
        {
            MergeNode top = pq.top();
            pq.pop();

            outFile.write(top.original_view.c_str(),
                          top.original_view.GetLength());
            outFile.put('\n');

            String_lib::StringView next_norm, next_orig;
            if (top.reader->next_pair(next_norm, next_orig))
            {
                pq.push(MergeNode(next_norm, next_orig, top.reader));
            }
        }
        outFile.flush();
        outFile.close();
    }

    for (auto &tf : temp_files)
    {
        std::remove(tf.c_str());
    }
}

inline void SaveChunk(std::vector<SoftWrapper> &chunk, int id,
                      std::vector<String_lib::String> &temp_files)
{
    std::vector<SoftTicket> tickets;
    tickets.reserve(chunk.size());

    for (size_t i = 0; i < chunk.size(); ++i)
    {
        tickets.push_back({String_lib::StringView(chunk[i].normalized), i});
    }

    if (!tickets.empty())
    {
        Radix_Internal::RadSort(tickets.data(), tickets.data() + tickets.size(),
                                0);
    }

    char name_buff[64];
    sprintf(name_buff, "temp_%d.bin", id);
    String_lib::String filename(name_buff);

    std::ofstream out(filename.c_str(), std::ios::binary);
    for (auto &tx : tickets)
    {
        const auto &norm = chunk[tx.original_idx].normalized;
        const auto &orig = chunk[tx.original_idx].original;

        uint32_t n_len = norm.GetLength();
        uint32_t o_len = orig.GetLength();

        out.write(reinterpret_cast<const char *>(&n_len), sizeof(n_len));
        out.write(norm.c_str(), n_len);
        out.write(reinterpret_cast<const char *>(&o_len), sizeof(o_len));
        out.write(orig.c_str(), o_len);
    }
    out.close();
    temp_files.push_back(String_lib::move(filename));
}

inline int StartApp(int argc, char *argv[])
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    if (argc < 2)
    {
        std::cerr << "Pass files as arguments. \n";
        return -1;
    }

    size_t total_cap = 0;
    for (int i = 1; i < argc; ++i)
    {
        struct stat st;
        if (stat(argv[i], &st) == 0)
            total_cap += st.st_size;
    }

    size_t ram = GetAvailableRam();

    if (total_cap < ram * RAM_L) // RAM
    {

        String_lib::String arena(total_cap + argc + 4096);
        char *write_ptr = arena.GetRawData();
        size_t actual_size = 0;

        for (int i = 1; i < argc; ++i)
        {
            std::ifstream file(argv[i], std::ios::binary);
            if (!file)
                continue;

            struct stat st;
            stat(argv[i], &st);

            file.read(write_ptr, st.st_size);
            actual_size += st.st_size;
            write_ptr += st.st_size;

            *write_ptr++ = ' ';
            actual_size++;
        }
        *write_ptr = '\0';

        std::vector<String_lib::StringView> tickets;
        tickets.reserve(total_cap / 10);

        char *start = arena.GetRawData();
        char *end = start + actual_size;
        char *curr = start;

        while (curr < end)
        {

            while (curr < end &&
                   std::isspace(static_cast<unsigned char>(*curr)))
                curr++;
            if (curr >= end || *curr == '\0')
                break;

            char *word_begin = curr;

            while (curr < end &&
                   !std::isspace(static_cast<unsigned char>(*curr)) &&
                   *curr != '\0')
                curr++;

            size_t word_len = curr - word_begin;
            if (word_len > 0)
            {
                tickets.emplace_back(word_begin, word_len);
            }
        }

        if (tickets.empty())
            return -1;

        Radix_Internal::RadSort(tickets.data(), tickets.data() + tickets.size(),
                                0);

        std::ofstream outFile("strings.txt", std::ios::binary);
        for (const auto &t : tickets)
        {
            outFile.write(t.c_str(), t.GetLength());
            outFile.put('\n');
        }
    }
    else // EXT
    {
        std::vector<String_lib::String> temp_files;
        size_t chunk_size = ram * RAM_L, current = 0;

        int count = 0;
        std::vector<SoftWrapper> chunk;
        chunk.reserve(chunk_size / 128);

        for (int i = 1; i < argc; ++i)
        {
            std::ifstream file(argv[i]);
            if (file.is_open())
            {
                String_lib::String word;
                word.Reserve(128);
                while (file >> word)
                {
                    if (chunk.size() % 10000 == 0)
                    {   // Логируем каждое 10000-е слово, чтобы не спамить
                        // LOG_DEBUG("Processing word: " << word.c_str());
                    }

                    current += (word.GetLength() * 2) + 256;
                    chunk.emplace_back(String_lib::move(word));
                    if (current > chunk_size)
                    {
                        // LOG_DEBUG("Saving chunk #" << count);
                        SaveChunk(chunk, count++, temp_files);
                        chunk.clear();
                        current = 0;
                    }
                }
                file.close();
            }
            else
            {
                String_lib::String word(argv[i]);
                current += (word.GetLength() * 2) + 256;
                chunk.emplace_back(String_lib::move(word));
                if (current > chunk_size)
                {
                    SaveChunk(chunk, count++, temp_files);
                    chunk.clear();
                    current = 0;
                }
            }
        }

        if (!chunk.empty())
            SaveChunk(chunk, count++, temp_files);

        if (!temp_files.empty())
        {
            FastExternalMerge(temp_files);
        }
    }
    return 0;
}
