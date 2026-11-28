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
 auto &v=*dp.result;const int n=v.size();
 std::vector<int> policy(n*top);
 std::vector<double> next(top,0),value(top);
 int original=0;
 for(int i=0;i<n;++i){
     const float *row=ls+(i+1)*hidden;
     const float *scores=row+top+original*top;
     original=int(std::max_element(scores,scores+top)-scores);
     if(v[i]!=int(row[original]))abort();
 }
#ifdef FRONT_REFERENCE
 if(first<0)first=dp.pos0;int ix=dp.pos0-first;
 const auto &ref=refs.at(req);if(ix<0||ix>=int(ref.size())||dp.id_last!=ref[ix])abort();
 int pred=0;
 for(int i=0;i<n;i++){
   const float *row=ls+(i+1)*hidden;const float *score=row+top+pred*top;
   int chosen=int(std::max_element(score,score+top)-score);
   if(ix+1+i<int(ref.size())) for(int a=0;a<top;a++)if(int(row[a])==ref[ix+1+i]){chosen=a;break;}
   v[i]=int(row[chosen]);pred=chosen;
 }
#else
 for(int i=n-1;i>=0;i--){
   const float *row=ls+(i+1)*hidden;
   for(int prev=0;prev<top;prev++){
      const float *score=row+top+prev*top;double norm=0,best=-1;int chosen=0;
      double max=*std::max_element(score,score+top);
      for(int a=0;a<top;a++){
        double exp_score=std::exp(score[a]-max);norm+=exp_score;
        double gain=exp_score*(1+next[a]);if(gain>best){best=gain;chosen=a;}
      }
      value[prev]=best/norm;policy[i*top+prev]=chosen;
   }
   next.swap(value);
 }
 int pred=0;
 for(int i=0;i<n;i++){pred=policy[i*top+pred];v[i]=int(ls[(i+1)*hidden+pred]);}
#endif
    // The block is always proposed in full. Its verified prefix may be shorter.
    int width=n;
    double best=-1.;
    for(int k : {3,4,7}){
        if(k>n)continue;
        double expected=1.;
#ifdef FRONT_REFERENCE
        for(int i=0;i<k && ix+i+1<int(ref.size()) && v[i]==ref[ix+i+1];i++)expected+=1.;
#else
        int prev=0;double survival=1.;
        for(int i=0;i<k;i++){
            const float *row=ls+(i+1)*hidden,*score=row+top+prev*top;
            const float mx=*std::max_element(score,score+top);
            int a=0;for(;a<top;a++)if(int(row[a])==v[i])break;
            if(a==top)abort();
            double denom=0.;for(int j=0;j<top;j++)denom+=std::exp(double(score[j]-mx));
            survival*=std::exp(double(score[a]-mx))/denom;expected+=survival;prev=a;
        }
#endif
        const double ms=4.8+(k==3?17.3:k==4?20.2:29.6);
        if(expected/ms>best){best=expected/ms;width=k;}
    }
    v.resize(width);
    static FILE *f=fopen(getenv("ADAPT_LOG"),"w");fprintf(f,"%d %d %d\\n",req,n,width);
}
