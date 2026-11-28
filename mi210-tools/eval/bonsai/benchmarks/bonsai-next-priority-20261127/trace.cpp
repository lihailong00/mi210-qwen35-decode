#include "llama.h"
#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>
#include <vector>
#include <chrono>
struct common_sampler;
struct common_speculative;
static thread_local bool drafting;
static FILE * trace() { static auto * f=fopen(getenv("BONSAI_TRACE"),"w");return f; }
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void common_speculative_draft(common_speculative * spec) {
    static auto next=(decltype(&common_speculative_draft))dlsym(RTLD_NEXT,"_Z24common_speculative_draftP18common_speculative");
    drafting=true; next(spec); drafting=false;
}
int common_sampler_sample(common_sampler * s,llama_context * ctx,int idx,bool gf) {
    static auto next=(decltype(&common_sampler_sample))dlsym(RTLD_NEXT,"_Z21common_sampler_sampleP14common_samplerP13llama_contextib");
    static auto cand=(llama_token_data_array *(*)(common_sampler*,bool))dlsym(RTLD_NEXT,"_Z29common_sampler_get_candidatesP14common_samplerb");
    int token=next(s,ctx,idx,gf);
    if(drafting) {
        auto * arr=cand(s,true); FILE *f=trace();
        fprintf(f,"D %.8f %zu",now(),arr->size);
        for(size_t i=0;i<3 && i<arr->size;i++) fprintf(f," %d %.7f",arr->data[i].id,arr->data[i].p);
        fprintf(f,"\n");
    }
    return token;
}
using ids=std::vector<int>;
ids common_sampler_sample_and_accept_n(common_sampler *s,llama_context *ctx,const ids& idx,const ids& draft,bool gf) {
    static auto next=(decltype(&common_sampler_sample_and_accept_n))dlsym(RTLD_NEXT,"_Z34common_sampler_sample_and_accept_nP14common_samplerP13llama_contextRKSt6vectorIiSaIiEES7_b");
    ids result=next(s,ctx,idx,draft,gf);FILE *f=trace();
    fprintf(f,"V %.8f %zu",now(),draft.size());for(int t:draft)fprintf(f," %d",t);
    fprintf(f," %zu",result.size());for(int t:result)fprintf(f," %d",t);fprintf(f,"\n");fflush(f);
    return result;
}
