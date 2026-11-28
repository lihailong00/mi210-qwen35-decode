#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace scoring {
constexpr int K = 16, D = 32, H = 24;
using Feature = std::array<float, D>;
using Block = std::array<Feature, K>;

inline float bounded(float x) { return std::max(-20.f, std::min(20.f, x)); }
inline float gap(const float *scores) {
    float a = -1e30f, b = -1e30f;
    for (int i = 0; i < K; ++i) {
        if (scores[i] > a) {
            b = a;
            a = scores[i];
        } else if (scores[i] > b)
            b = scores[i];
    }
    return std::min(12.f, a - b);
}
inline void features(const float *row, const float *next, int previous, int depth,
                     const std::vector<int> &prefix, Block &out) {
    const float *all = row + K, *s = all + previous * K;
    std::array<float, K> maxima{};
    for (int p = 0; p < K; ++p)
        maxima[p] = *std::max_element(all + p * K, all + (p + 1) * K);
    const float max_score = maxima[previous], margin = gap(s);
    const int end = int(prefix.size());
    for (int a = 0; a < K; ++a) {
        const int token = int(row[a]);
        int rank = 0, best_count = 0, count = 0, recency = end + 1, longest = 0, recent_match = 0;
        float sum = 0, square = 0, high = -1e30f, low = 1e30f;
        for (int j = 0; j < K; ++j) {
            rank += s[j] > s[a];
            const float d = bounded(all[j * K + a] - maxima[j]);
            sum += d;
            square += d * d;
            high = std::max(high, d);
            low = std::min(low, d);
            best_count += d == 0;
        }
        for (int j = 0; j < end; ++j) {
            if (prefix[j] != token)
                continue;
            ++count;
            recency = end - j;
            int l = 0;
            while (l < 8 && l < j && prefix[j - 1 - l] == prefix[end - 1 - l])
                ++l;
            if (l >= longest) {
                longest = l;
                recent_match = end - j;
            }
        }
        float next_gap = 0.f, next_first = 0.f, next_best_rank = 0.f;
        if (next) {
            const float *ss = next + (1 + a) * K;
            const int best = int(std::max_element(ss, ss + K) - ss);
            next_gap = gap(ss);
            next_first = bounded(ss[0] - ss[best]);
            next_best_rank = std::log1p(float(best));
        }
        const float lr = std::log1p(float(a)), cr = std::log1p(float(rank)), raw = bounded(s[a] - max_score);
        const float mean = sum / K, sd = std::sqrt(std::max(0.f, square / K - mean * mean));
        const float repeat = std::min(4, std::max(0, longest - 1));
        out[a] = {raw,
                  lr,
                  cr,
                  float(a == 0),
                  float(rank == 0),
                  mean,
                  sd,
                  high,
                  low,
                  bounded(all[a] - maxima[0]),
                  best_count / 16.f,
                  bounded(raw - mean),
                  margin,
                  next_gap,
                  next_first,
                  next_best_rank,
                  depth / 6.f,
                  float(depth == 0),
                  float(depth < 3),
                  repeat,
                  std::min(8, longest) / 8.f,
                  std::log1p(float(count)),
                  count ? 1.f / std::sqrt(float(recency)) : 0.f,
                  float(token == prefix.back()),
                  float(end >= 2 && token == prefix[end - 2]),
                  float(token < 256),
                  float(token < 2048),
                  float(token >= 100000),
                  std::log1p(float(end)) / 5.f,
                  float(longest >= 2 && recent_match <= 12),
                  raw - 2.f * lr + repeat,
                  float(next != nullptr)};
    }
}
} // namespace scoring
