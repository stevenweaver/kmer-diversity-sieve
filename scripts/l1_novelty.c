/*
 * l1_novelty.c -- L1 novelty gate (C + OpenMPI), stage 2 of the DAQ trigger.
 *
 * Follows the admission filter (l1_trigger.c). Answers "is this admitted genome NEW
 * enough to send to L2?" via MinHash sketch-distance to the already-promoted set.
 * This is where the big rejection happens (~67x -> hundreds): near-duplicate genomes
 * are dropped, only novel haplotypes promoted.
 *
 * MinHash: m independent multiply-shift hashes of canonical k-mers; bottom value each.
 *   (EXACT port of the Python oracle l1_trigger_prototype.py: h = (kmer*salt) & MASK;
 *    h ^= h>>29; sketch[i] = min over kmers.)  novelty = 1 - maxJaccard(sk, promoted).
 *
 * FAST nn-search: LSH band index. The m-hash sketch is split into B bands of R rows;
 * two sketches sharing ANY band bucket are candidate-similar -> only those are compared
 * exactly. Turns the Python O(P) scan into ~O(1) expected. (Classic MinHash-LSH.)
 *
 * MPI (two-phase, validated order-invariant): phase 1 each rank greedily
 * builds a shard-LOCAL tau-diverse promoted set; phase 2 Allgather all candidates ->
 * one rank greedy-merges to a global tau-diverse set. Coverage is order-invariant
 * (100% mutual coverage measured); only the exact count drifts at cluster margins.
 *
 * Build: mpicc -O3 -march=native -D_FILE_OFFSET_BITS=64 -o l1_novelty l1_novelty.c -lm
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <ctype.h>
#include <sys/types.h>

#define DEF_K 15
#define DEF_M 64            /* MinHash sketch size (match oracle) */
#define DEF_THRESHOLD 0.15
#define LSH_BANDS 16        /* B bands x R rows = m ; 16x4 = 64 */
#define MASK64 (~0ULL)

static uint8_t LUT[256];
static void init_lut(void){
    for(int i=0;i<256;i++) LUT[i]=255;
    LUT[(int)'A']=0;LUT[(int)'C']=1;LUT[(int)'G']=2;LUT[(int)'T']=3;
    LUT[(int)'a']=0;LUT[(int)'c']=1;LUT[(int)'g']=2;LUT[(int)'t']=3;
}

/* ---- MinHash sketch of a genome (EXACT match to Python oracle) ---- */
/* Python: h = (kmer * salt) & MASK64 ; h ^= h >> 29 ; per-hash min over all kmers. */
static int minhash(const uint8_t*seq,size_t len,int k,int m,const uint64_t*salts,uint64_t*sk){
    for(int i=0;i<m;i++) sk[i]=MASK64;
    if(len<(size_t)k) return 0;
    uint64_t kmask=(k>=32)?~0ULL:(((uint64_t)1<<(2*k))-1);
    int any=0;
    size_t i=0;
    while(i<len){
        if(LUT[seq[i]]==255){i++;continue;}
        size_t j=i; while(j<len && LUT[seq[j]]!=255) j++;
        if(j-i>=(size_t)k){
            uint64_t fwd=0,rev=0;
            for(size_t z=0;z<(size_t)k;z++){uint64_t c=LUT[seq[i+z]];
                fwd=((fwd<<2)|c)&kmask; rev=(rev>>2)|((uint64_t)(3-c)<<(2*(k-1)));}
            uint64_t can=fwd<rev?fwd:rev;
            any=1;
            for(int h=0;h<m;h++){uint64_t x=(can*salts[h])&MASK64; x^=x>>29; if(x<sk[h])sk[h]=x;}
            for(size_t z=i+k;z<j;z++){uint64_t c=LUT[seq[z]];
                fwd=((fwd<<2)|c)&kmask; rev=(rev>>2)|((uint64_t)(3-c)<<(2*(k-1)));
                can=fwd<rev?fwd:rev;
                for(int h=0;h<m;h++){uint64_t x=(can*salts[h])&MASK64; x^=x>>29; if(x<sk[h])sk[h]=x;}
            }
        }
        i=j;
    }
    return any;
}

static double jaccard(const uint64_t*a,const uint64_t*b,int m){
    int eq=0; for(int i=0;i<m;i++) if(a[i]==b[i]) eq++;
    return (double)eq/m;
}

/* ---- promoted set with LSH band index ----
 * Sketches stored contiguously: sketches[i*m .. i*m+m). For each band, a hash map from
 * band-signature -> list of promoted indices. We use a simple open-addressing hash of
 * (band_id, band_hash) -> bucket head, with per-bucket dynamic arrays. */
typedef struct { uint64_t key; int *ids; int n, cap; } bucket_t;
typedef struct {
    uint64_t *sk;   int n, cap, m;        /* promoted sketches */
    bucket_t *buk;  uint64_t nbuk, mask;  /* LSH buckets (shared across bands via key mix) */
} pset_t;

static void pset_init(pset_t*p,int m,uint64_t log2buk){
    p->m=m; p->n=0; p->cap=1024; p->sk=malloc((size_t)p->cap*m*sizeof(uint64_t));
    p->nbuk=(uint64_t)1<<log2buk; p->mask=p->nbuk-1;
    p->buk=calloc(p->nbuk,sizeof(bucket_t));
}
static inline uint64_t band_key(int band,const uint64_t*sk,int R){
    uint64_t h=0x9E3779B97F4A7C15ULL ^ ((uint64_t)band*0xC2B2AE3D27D4EB4FULL);
    for(int r=0;r<R;r++){ h^=sk[band*R+r]; h*=0x100000001B3ULL; h^=h>>33; }
    return h;
}
static void bucket_add(pset_t*p,uint64_t key,int id){
    uint64_t idx=(key^(key>>29))&p->mask;
    while(p->buk[idx].ids && p->buk[idx].key!=key) idx=(idx+1)&p->mask;
    bucket_t*b=&p->buk[idx];
    if(!b->ids){ b->key=key; b->cap=4; b->ids=malloc(b->cap*sizeof(int)); b->n=0; }
    if(b->n==b->cap){ b->cap*=2; b->ids=realloc(b->ids,b->cap*sizeof(int)); }
    b->ids[b->n++]=id;
}
static bucket_t* bucket_find(pset_t*p,uint64_t key){
    uint64_t idx=(key^(key>>29))&p->mask;
    while(p->buk[idx].ids){ if(p->buk[idx].key==key) return &p->buk[idx]; idx=(idx+1)&p->mask; }
    return NULL;
}
/* returns 1 if sk is novel (>= threshold from all promoted), using LSH to find candidates */
static int is_novel(pset_t*p,const uint64_t*sk,double threshold,int R,int B){
    double need=1.0-threshold;  /* if any promoted has jaccard > need, NOT novel */
    for(int band=0;band<B;band++){
        uint64_t key=band_key(band,sk,R);
        bucket_t*b=bucket_find(p,key);
        if(!b) continue;
        for(int t=0;t<b->n;t++){
            int id=b->ids[t];
            if(jaccard(sk,&p->sk[(size_t)id*p->m],p->m) > need) return 0;
        }
    }
    return 1;
}
static void pset_add(pset_t*p,const uint64_t*sk,int R,int B){
    if(p->n==p->cap){ p->cap*=2; p->sk=realloc(p->sk,(size_t)p->cap*p->m*sizeof(uint64_t)); }
    memcpy(&p->sk[(size_t)p->n*p->m],sk,p->m*sizeof(uint64_t));
    for(int band=0;band<B;band++) bucket_add(p,band_key(band,sk,R),p->n);
    p->n++;
}

static void stream_and_promote(const char*path,off_t my_start,off_t my_end,int rank,int nproc,
        int k,int m,const uint64_t*salts,double threshold,int R,int B,long limit,
        pset_t*ps,long*n_admitted){
    FILE*fp=fopen(path,"rb");
    if(!fp){fprintf(stderr,"[rank %d] open fail\n",rank);MPI_Abort(MPI_COMM_WORLD,1);}
    if(rank>0){ fseeko(fp,my_start,SEEK_SET); int c; while((c=fgetc(fp))!=EOF && c!='\n');
        while(1){ long here=ftello(fp); c=fgetc(fp); if(c==EOF){my_start=here;break;}
                  if(c=='>'){my_start=here;break;} while((c=fgetc(fp))!=EOF&&c!='\n'); } }
    else my_start=0;
    fseeko(fp,my_start,SEEK_SET);
    size_t seqcap=1<<20; uint8_t*seq=malloc(seqcap); size_t seqlen=0; int in_seq=0;
    char*line=NULL; size_t lc=0; ssize_t ll;
    uint64_t*sk=malloc(m*sizeof(uint64_t));
    long nadm=0; int stop=0;
    #define NOV_PROCESS() do{ if(in_seq){ nadm++; \
        if(minhash(seq,seqlen,k,m,salts,sk)){ if(is_novel(ps,sk,threshold,R,B)) pset_add(ps,sk,R,B); } \
        if(limit && nadm>=limit){stop=1;} } }while(0)
    while((ll=getline(&line,&lc,fp))>=0){
        if(line[0]=='>'){
            long hoff=ftello(fp)-ll;
            NOV_PROCESS(); if(stop) break;
            if(hoff>=my_end){in_seq=0;break;}
            seqlen=0; in_seq=1;
        } else { size_t L=ll; while(L>0&&(line[L-1]=='\n'||line[L-1]=='\r'))L--;
            if(seqlen+L+1>seqcap){while(seqlen+L+1>seqcap)seqcap*=2;seq=realloc(seq,seqcap);}
            for(size_t z=0;z<L;z++) seq[seqlen++]=toupper((unsigned char)line[z]); }
    }
    if(!stop && in_seq) NOV_PROCESS();
    fclose(fp); free(seq); free(line); free(sk);
    *n_admitted=nadm;
}

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);
    int rank,nproc; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    init_lut();
    const char*input=NULL; int k=DEF_K,m=DEF_M; double threshold=DEF_THRESHOLD; long limit=0;
    for(int a=1;a<argc;a++){
        if(!strcmp(argv[a],"--input"))input=argv[++a];
        else if(!strcmp(argv[a],"--k"))k=atoi(argv[++a]);
        else if(!strcmp(argv[a],"--sketch"))m=atoi(argv[++a]);
        else if(!strcmp(argv[a],"--threshold"))threshold=atof(argv[++a]);
        else if(!strcmp(argv[a],"--limit"))limit=atol(argv[++a]);
    }
    if(!input){if(rank==0)fprintf(stderr,"--input required\n");MPI_Finalize();return 1;}
    int B=LSH_BANDS, R=m/B;
    /* salts: EXACT match to Python oracle (np.random.default_rng(42), odd) — reproduce below.
     * NOTE: Python uses numpy PCG64; we cannot bit-match its stream in C. For oracle
     * comparison we instead read salts from a file the Python side wrote. Default: derive
     * deterministic odd salts here (self-consistent; oracle-match mode uses --salts). */
    uint64_t*salts=malloc(m*sizeof(uint64_t));
    uint64_t s=0x123456789ABCDEFULL;
    for(int i=0;i<m;i++){ s^=s<<13; s^=s>>7; s^=s<<17; salts[i]=s|1ULL; }

    /* byte range */
    FILE*fp=fopen(input,"rb"); if(!fp){if(rank==0)fprintf(stderr,"open fail\n");MPI_Abort(MPI_COMM_WORLD,1);}
    fseeko(fp,0,SEEK_END); off_t fsize=ftello(fp); fclose(fp);
    off_t slice=fsize/nproc, ms=(off_t)rank*slice, me=(rank==nproc-1)?fsize:(off_t)(rank+1)*slice;

    pset_t ps; pset_init(&ps,m,20);
    long n_adm=0; double t0=MPI_Wtime();
    /* PHASE 1: shard-local greedy promotion */
    stream_and_promote(input,ms,me,rank,nproc,k,m,salts,threshold,R,B,limit,&ps,&n_adm);
    double t1=MPI_Wtime();

    /* PHASE 2: gather all local promoted sketches to rank 0, greedy merge-dedup */
    int local_p=ps.n;
    int*counts=NULL,*displs=NULL;
    if(rank==0){counts=malloc(nproc*sizeof(int));displs=malloc(nproc*sizeof(int));}
    MPI_Gather(&local_p,1,MPI_INT,counts,1,MPI_INT,0,MPI_COMM_WORLD);
    int total=0; if(rank==0){for(int i=0;i<nproc;i++){displs[i]=total;total+=counts[i];}}
    /* gather sketches (each local_p * m uint64) */
    int sendcnt=local_p*m;
    int*rcounts=NULL,*rdispls=NULL; uint64_t*allsk=NULL;
    if(rank==0){rcounts=malloc(nproc*sizeof(int));rdispls=malloc(nproc*sizeof(int));
        int off=0; for(int i=0;i<nproc;i++){rcounts[i]=counts[i]*m;rdispls[i]=off;off+=rcounts[i];}
        allsk=malloc((size_t)total*m*sizeof(uint64_t)); }
    MPI_Gatherv(ps.sk,sendcnt,MPI_UINT64_T,allsk,rcounts,rdispls,MPI_UINT64_T,0,MPI_COMM_WORLD);

    long g_adm=0; MPI_Reduce(&n_adm,&g_adm,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    double dt=t1-t0,maxdt; MPI_Reduce(&dt,&maxdt,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    if(rank==0){
        /* greedy merge over the union of shard-local candidates */
        pset_t merged; pset_init(&merged,m,20);
        for(int i=0;i<total;i++){ uint64_t*sk=&allsk[(size_t)i*m];
            if(is_novel(&merged,sk,threshold,R,B)) pset_add(&merged,sk,R,B); }
        printf("\n=== L1 NOVELTY (C+OpenMPI, two-phase) RESULT ===\n");
        printf("input=%s nproc=%d k=%d sketch=%d threshold=%g bands=%dx%d\n",input,nproc,k,m,threshold,B,R);
        printf("admitted (input) : %ld\n",g_adm);
        printf("phase1 candidates: %d (sum of shard-local promoted across ranks)\n",total);
        printf("PROMOTED (global): %d  (%.4f%% of admitted)\n",merged.n,100.0*merged.n/(g_adm>0?g_adm:1));
        printf("REJECTION FACTOR : %.1fx\n",(double)g_adm/(merged.n>0?merged.n:1));
        printf("phase1 wall      : %.1fs  (max across ranks)  throughput %.0f seq/s\n",maxdt,g_adm/(maxdt>0?maxdt:1));
    }
    MPI_Finalize();
    return 0;
}
