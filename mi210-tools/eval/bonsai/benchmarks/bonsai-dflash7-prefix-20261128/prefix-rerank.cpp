// Single-slot greedy DFlash2 ranking experiment for the project's llama.cpp build.
#include "speculative.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <vector>

namespace {
thread_local bool observing = false;
thread_local const float * lattice = nullptr;
thread_local int row_width = 0;
thread_local int top_k = 0;
thread_local int first_pos = -1;

template <class T> T resolve(const char *symbol) {
    auto fn = reinterpret_cast<T>(dlsym(RTLD_NEXT, symbol));
    if (!fn) std::abort();
    return fn;
}

// Count previously seen suffix matches only in the generated text for this request.
// A two-token match adds one logit unit, up to four units for a five-token match.
void rank_block(common_speculative_draft_params &draft, int offset) {
    if (offset < 0 || !draft.prompt || size_t(offset) > draft.prompt->size()) std::abort();
    llama_tokens prefix(draft.prompt->end() - offset, draft.prompt->end());
    prefix.push_back(draft.id_last);
    int previous = 0;
    for (size_t i = 0; i < draft.result->size(); ++i) {
        const float *row = lattice + (i + 1) * row_width;
        const float *score = row + (1 + previous) * top_k;
        std::array<int, 32> strength{};
        int end = int(prefix.size());
        for (int next = 2; next < end; ++next) {
            int length = 0;
            while (length < 5 && length < next && prefix[next-1-length] == prefix[end-1-length]) ++length;
            if (length < 2) continue;
            for (int a = 0; a < top_k; ++a) {
                if (int(row[a]) == prefix[next]) strength[a] = std::max(strength[a], length - 1);
            }
        }
        static const auto rank_penalty = [] {
            std::array<double, 32> p{};
            for (size_t a=0; a<p.size(); ++a) p[a] = 2.0 * std::log1p(double(a));
            return p;
        }();
        double best = -INFINITY;
        for (int a = 0; a < top_k; ++a) {
            const double adjusted = score[a] - rank_penalty[a] + strength[a];
            if (adjusted > best) { best = adjusted; previous = a; }
        }
        (*draft.result)[i] = int(row[previous]);
        prefix.push_back((*draft.result)[i]);
    }
}
}

float *llama_get_embeddings_nextn(llama_context *ctx) {
    static auto next = resolve<decltype(&llama_get_embeddings_nextn)>("_Z26llama_get_embeddings_nextnP13llama_context");
    auto output = next(ctx);
    if (observing && output) {
        static auto width = resolve<int(*)(const llama_model*)>("_Z33llama_model_dflash_selector_top_kPK11llama_model");
        const auto *model = llama_get_model(ctx);
        lattice = output;
        row_width = llama_model_n_embd(model);
        top_k = width(model);
    }
    return output;
}

void common_speculative_begin(common_speculative *spec, llama_seq_id id, const llama_tokens &prompt) {
    static auto next = resolve<decltype(&common_speculative_begin)>("_Z24common_speculative_beginP18common_speculativeiRKSt6vectorIiSaIiEE");
    next(spec, id, prompt);
    first_pos = -1;
}

void common_speculative_draft(common_speculative *spec) {
    static auto next = resolve<decltype(&common_speculative_draft)>("_Z24common_speculative_draftP18common_speculative");
    static auto params = resolve<decltype(&common_speculative_get_draft_params)>("_Z35common_speculative_get_draft_paramsP18common_speculativei");
    auto &draft = params(spec, 0);
    observing = true;
    lattice = nullptr;
    next(spec);
    observing = false;
    if (!draft.result || draft.result->empty()) return;
    if (!lattice || top_k != 16 || row_width < top_k*(top_k+1) || draft.temp != 0) std::abort();
    if (first_pos < 0) first_pos = draft.pos0;
    rank_block(draft, draft.pos0 - first_pos);
}
