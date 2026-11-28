#include <cstring>
#include "speculative.h"
#include <dlfcn.h>
#include <cmath>
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
static thread_local bool inside=false;
static thread_local const float *ls=nullptr;
static thread_local int hidden=0,top=0;
float *llama_get_embeddings_nextn(llama_context *ctx){
 static auto fn=(decltype(&llama_get_embeddings_nextn))dlsym(RTLD_NEXT,"_Z26llama_get_embeddings_nextnP13llama_context");
 auto ptr=fn(ctx);
 if(inside && ptr){ls=ptr;auto model=llama_get_model(ctx);hidden=llama_model_n_embd(model);static auto gettop=(int(*)(const llama_model*))dlsym(RTLD_NEXT,"_Z33llama_model_dflash_selector_top_kPK11llama_model");top=gettop(model);}
 return ptr;
}
static int req=-1,first=-1;
#ifdef FRONT_REFERENCE
static std::vector<llama_tokens> refs;
#endif
void common_speculative_begin(common_speculative*sp,llama_seq_id id,const llama_tokens &pr){
 static auto fn=(decltype(&common_speculative_begin))dlsym(RTLD_NEXT,"_Z24common_speculative_beginP18common_speculativeiRKSt6vectorIiSaIiEE");
 fn(sp,id,pr);++req;first=-1;
#ifdef FRONT_REFERENCE
 if(refs.empty()){std::ifstream f(getenv("SELECT_REFS"));std::string l;while(std::getline(f,l)){std::istringstream s(l);llama_tokens ts;int t;while(s>>t)ts.push_back(t);refs.push_back(ts);}}
#endif
}
void common_speculative_draft(common_speculative *sp){
 static auto fn=(decltype(&common_speculative_draft))dlsym(RTLD_NEXT,"_Z24common_speculative_draftP18common_speculative");
 static auto par=(decltype(&common_speculative_get_draft_params))dlsym(RTLD_NEXT,"_Z35common_speculative_get_draft_paramsP18common_speculativei");
 auto &dp=par(sp,0);
 ls=nullptr;inside=true;fn(sp);inside=false;
 if(!dp.result || dp.result->empty())return;
 if(!ls||top<=1||top>32||hidden<top*(top+1))abort();

int offset = dp.pos0 - (first < 0 ? (first = dp.pos0) : first);
const char *mode = getenv("PREFIX_MODE");
const int n = int(dp.result->size());
static FILE *file = getenv("PREFIX_DUMP") ? fopen(getenv("PREFIX_DUMP"), "wb") : nullptr;
int prev=0;
for (int i=0; i<n; ++i) {
    const float *row = ls + (1+i)*hidden;
    const float *scores = row + top*(1+prev);
    prev = int(std::max_element(scores,scores+top)-scores);
    if ((*dp.result)[i] != int(row[prev])) abort();
}
if (file) {
    int header[] = {req, offset, dp.id_last, n, top};
    if (fwrite(header,sizeof(int),5,file)!=5) abort();
    for (int i=0;i<n;++i) {
        const size_t width=top*(1+top);
        if (fwrite(ls+(1+i)*hidden,sizeof(float),width,file)!=width) abort();
    }
}
if (!strcmp(mode,"dense")) dp.result->clear();
}
