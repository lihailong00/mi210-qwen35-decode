#include "candidate-features.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace scoring;
int main(int argc, char **argv) {
    if (argc != 4)
        return 1;
    std::ifstream dump(argv[1], std::ios::binary), labels(argv[2]);
    std::ofstream out(argv[3], std::ios::binary);
    if (!dump || !labels || !out)
        throw std::runtime_error("cannot open input or output");
    std::vector<std::vector<int>> refs;
    std::string line;
    while (std::getline(labels, line)) {
        std::istringstream in(line);
        std::vector<int> seq;
        int t;
        while (in >> t)
            seq.push_back(t);
        refs.push_back(std::move(seq));
    }
    std::array<int32_t, 5> hdr{};
    while (dump.read(reinterpret_cast<char *>(hdr.data()), sizeof(hdr))) {
        const int req = hdr[0], ix = hdr[1], token = hdr[2], len = hdr[3], k = hdr[4];
        if (k != K || len < 1 || len > 7 || ix < 0 || req < 0 || req >= int(refs.size()) ||
            ix + len >= int(refs[req].size()) || refs[req][ix] != token)
            throw std::runtime_error("trajectory does not align with teacher");
        std::vector<float> lattice(len * K * (K + 1));
        if (!dump.read(reinterpret_cast<char *>(lattice.data()), lattice.size() * sizeof(float)))
            throw std::runtime_error("truncated lattice");
        if (req == 0)
            continue;
        std::vector<int> prefix(refs[req].begin(), refs[req].begin() + ix + 1);
        int prev = 0;
        for (int i = 0; i < len; ++i) {
            const float *row = lattice.data() + i * K * (K + 1);
            const int target = refs[req][ix + i + 1];
            int choice = -1;
            std::array<int32_t, K> ids{};
            for (int a = 0; a < K; ++a) {
                ids[a] = int(row[a]);
                if (ids[a] == target)
                    choice = a;
            }
            std::array<int32_t, 5> key{req - 1, ix, i, choice, len};
            Block block;
            features(row, i + 1 < len ? row + K * (K + 1) : nullptr, prev, i, prefix, block);
            out.write(reinterpret_cast<const char *>(key.data()), sizeof(key));
            out.write(reinterpret_cast<const char *>(ids.data()), sizeof(ids));
            out.write(reinterpret_cast<const char *>(&block), sizeof(block));
            if (choice < 0)
                break;
            prev = choice;
            prefix.push_back(target);
        }
    }
    if (!dump.eof() || !out)
        throw std::runtime_error("trajectory I/O failed");
}
