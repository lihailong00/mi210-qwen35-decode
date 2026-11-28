#pragma once
#include "candidate-features.h"
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace scoring {
class Scorer {
    float alpha;
    std::array<Feature, H> hidden;
    std::array<float, H> bias, output;
    std::unordered_map<int, float> token_bias;
    template <class T> static void read(std::ifstream &f, T *dst, size_t count) {
        if (!f.read(reinterpret_cast<char *>(dst), sizeof(T) * count))
            throw std::runtime_error("truncated DFlash scorer");
    }

  public:
    explicit Scorer(const char *path) {
        if (!path)
            throw std::runtime_error("set DFLASH_SCORER to the trained model file");
        std::ifstream f(path, std::ios::binary);
        std::array<int, 5> shape{};
        read(f, shape.data(), 5);
        if (shape[0] != 0x44534631 || shape[1] != K || shape[2] != D || shape[3] != H || shape[4] < 0 ||
            shape[4] > 262144)
            throw std::runtime_error("invalid DFlash scorer header");
        read(f, &alpha, 1);
        read(f, hidden.data(), H);
        read(f, bias.data(), H);
        read(f, output.data(), H);
        std::vector<int> ids(shape[4]);
        std::vector<float> table(shape[4] + 1);
        read(f, ids.data(), ids.size());
        read(f, table.data(), table.size());
        if (f.peek() != std::ifstream::traits_type::eof())
            throw std::runtime_error("unexpected DFlash scorer data");
        for (size_t i = 0; i < ids.size(); ++i)
            token_bias[ids[i]] = table[i + 1];
    }
    int choose(const Block &x, const std::array<int, K> &ids) const {
        float best = -1e30f;
        int idx = 0;
        for (int a = 0; a < K; ++a) {
            auto entry = token_bias.find(ids[a]);
            float delta = entry == token_bias.end() ? 0.f : entry->second;
            for (int h = 0; h < H; ++h) {
                float z = bias[h];
                for (int d = 0; d < D; ++d)
                    z += hidden[h][d] * x[a][d];
                delta += output[h] * std::max(0.f, z);
            }
            const float score = x[a][30] + alpha * delta;
            if (score > best) {
                best = score;
                idx = a;
            }
        }
        return idx;
    }
};
} // namespace scoring
