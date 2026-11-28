#include "speculative.h"
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
namespace { int request=-1,first=-1; }
void common_speculative_begin(common_speculative *s,llama_seq_id id,const llama_tokens &p) {
    static auto next=(decltype(&common_speculative_begin))dlsym(RTLD_NEXT,"_Z24common_speculative_beginP18common_speculativeiRKSt6vectorIiSaIiEE");
    next(s,id,p);++request;first=-1;
}
void common_speculative_draft(common_speculative *s) {
    static auto next=(decltype(&common_speculative_draft))dlsym(RTLD_NEXT,"_Z24common_speculative_draftP18common_speculative");
    static auto params=(decltype(&common_speculative_get_draft_params))dlsym(RTLD_NEXT,"_Z35common_speculative_get_draft_paramsP18common_speculativei");
    next(s);auto &dp=params(s,0);
    if(!dp.result||dp.result->empty())return;
    if(first<0) first=dp.pos0;
    static FILE *f=std::fopen(std::getenv("DRAFT_LOG"),"w");
    if(!f)std::abort();
    std::fprintf(f,"%d %d %d %zu",request,dp.pos0-first,dp.id_last,dp.result->size());
    for(int t:*dp.result) std::fprintf(f," %d",t);
    std::fputc('\n',f);
}
