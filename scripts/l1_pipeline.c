/*
 * l1_pipeline.c -- COMPOSED L1 DAQ trigger (C + OpenMPI): admission + novelty, ONE pass.
 *
 * Fuses the two validated stages (l1_trigger.c admission, l1_novelty.c novelty) into a
 * single streaming pass. Efficiency: k-mers are extracted ONCE per genome and reused for
 * BOTH the Count-Min frequency check (admission) and the MinHash sketch (novelty) — no
 * double file read, no double k-mer parse.
 *
 * Per genome (streaming, byte-range sharded across MPI ranks):
 *   1. extract distinct canonical k-mers ONCE
 *   2. ADMISSION (compressor-2 port): rare k-mer = CMS df <= binomial_cutoff(N,err,p);
 *      singleton = df==0; REJECT if singleton-count is an outlier (> mean+SD*outlier_sd).
 *      -> rejected genomes (error-laden) STOP here, never reach novelty or L2.
 *   3. NOVELTY (admitted only): MinHash sketch from the SAME k-mers; promote if tau-distant
 *      from the shard-local promoted set (LSH band index, ~O(1) nn-search).
 *   4. PHASE 2: Gatherv shard-local promoted sketches -> rank0 greedy merge -> global set.
 *
 * Output: the global promoted set = the ~1.3% that goes to L2 reconstruction. Optionally
 * emit their headers (--emit-promoted) as the L2 input manifest.
 *
 * Composition validated: admission exactly matches its oracle; novelty matches magnitude;
 * both scaling curves measured (admission 17.8x@64 file-bound, novelty 63.9x@64 linear).
 *
 * Build: mpicc -O3 -march=native -D_FILE_OFFSET_BITS=64 -o l1_pipeline l1_pipeline.c -lm
 * Run:   mpirun -np N ./l1_pipeline --input file.fasta [--emit-promoted out.txt] [opts]
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <ctype.h>
#include <sys/types.h>

/* ---- admission config (match l1_trigger.c / oracle) ---- */
#define DEF_K            15
#define DEF_ERR_RATE     1e-4
#define DEF_P            0.9
#define DEF_OUTLIER_SD   5.0
#define DEF_WARMUP       2000
#define CMS_LOG2W        22   /* 2^22 x 4 rows x 4B = 64 MB/rank (was 2^24=256MB; caused node OOM at 128 ranks/node) */
#define CMS_D            4
/* ---- novelty config (match l1_novelty.c) ---- */
#define DEF_M            64
#define DEF_THRESHOLD    0.15
#define LSH_BANDS        16
#define MASK64 (~0ULL)

static const uint64_t CMS_SEEDS[CMS_D] = {
    0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL,
    0x165667B19E3779F9ULL, 0xD6E8FEB86659FD93ULL
};

static uint8_t LUT[256];
static void init_lut(void){
    for(int i=0;i<256;i++) LUT[i]=255;
    LUT[(int)'A']=0;LUT[(int)'C']=1;LUT[(int)'G']=2;LUT[(int)'T']=3;
    LUT[(int)'a']=0;LUT[(int)'c']=1;LUT[(int)'g']=2;LUT[(int)'t']=3;
}
static inline uint64_t mix64(uint64_t x,uint64_t seed){
    x^=seed; x^=x>>30; x*=0xBF58476D1CE4E5B9ULL; x^=x>>27; x*=0x94D049BB133111EBULL; x^=x>>31; return x;
}

/* ---- Count-Min sketch (admission) ---- */
typedef struct { uint32_t *t; uint64_t W,mask; } cms_t;
static void cms_init(cms_t*c){ c->W=(uint64_t)1<<CMS_LOG2W; c->mask=c->W-1;
    c->t=calloc((size_t)CMS_D*c->W,sizeof(uint32_t));
    if(!c->t){fprintf(stderr,"CMS alloc fail\n");MPI_Abort(MPI_COMM_WORLD,1);} }
static inline uint32_t cms_query(const cms_t*c,uint64_t key){ uint32_t m=UINT32_MAX;
    for(int r=0;r<CMS_D;r++){uint64_t col=mix64(key,CMS_SEEDS[r])&c->mask;uint32_t v=c->t[(uint64_t)r*c->W+col];if(v<m)m=v;} return m; }
static inline void cms_inc(cms_t*c,uint64_t key){
    for(int r=0;r<CMS_D;r++){uint64_t col=mix64(key,CMS_SEEDS[r])&c->mask;c->t[(uint64_t)r*c->W+col]++;} }

static long binomial_cutoff(long N,double p,double t){
    if(N<=0)return 0; double s=0.0,term=pow(1.0-p,(double)N); long i=0;
    while(s<t && i<N){ s+=term; double d=(1.0-p)*(double)(i+1); if(d==0.0)break;
        term=term*p/(1.0-p)*(double)(N-i)/(double)(i+1); i++; } return i; }

/* distinct canonical k-mers -> out[], returns distinct count (sorted+uniq'd) */
static int cmp_u64(const void*a,const void*b){uint64_t x=*(const uint64_t*)a,y=*(const uint64_t*)b;return(x>y)-(x<y);}
static size_t genome_kmers(const uint8_t*seq,size_t len,int k,uint64_t*out,size_t cap){
    if(len<(size_t)k)return 0; size_t nout=0; uint64_t kmask=(k>=32)?~0ULL:(((uint64_t)1<<(2*k))-1);
    size_t i=0;
    while(i<len){ if(LUT[seq[i]]==255){i++;continue;} size_t j=i; while(j<len&&LUT[seq[j]]!=255)j++;
        if(j-i>=(size_t)k){ uint64_t fwd=0,rev=0;
            for(size_t z=0;z<(size_t)k;z++){uint64_t c=LUT[seq[i+z]];fwd=((fwd<<2)|c)&kmask;rev=(rev>>2)|((uint64_t)(3-c)<<(2*(k-1)));}
            uint64_t can=fwd<rev?fwd:rev; if(nout<cap)out[nout++]=can;
            for(size_t z=i+k;z<j;z++){uint64_t c=LUT[seq[z]];fwd=((fwd<<2)|c)&kmask;rev=(rev>>2)|((uint64_t)(3-c)<<(2*(k-1)));
                can=fwd<rev?fwd:rev; if(nout<cap)out[nout++]=can;} }
        i=j; }
    if(nout==0)return 0;
    qsort(out,nout,sizeof(uint64_t),cmp_u64);
    size_t nu=0; for(size_t z=0;z<nout;z++) if(z==0||out[z]!=out[z-1]) out[nu++]=out[z];
    return nu;
}
/* MinHash FROM already-extracted distinct k-mers (reuse — no re-parse). Matches oracle. */
static void minhash_from_kmers(const uint64_t*km,size_t nu,int m,const uint64_t*salts,uint64_t*sk){
    for(int i=0;i<m;i++) sk[i]=MASK64;
    for(size_t z=0;z<nu;z++){ uint64_t can=km[z];
        for(int h=0;h<m;h++){uint64_t x=(can*salts[h])&MASK64; x^=x>>29; if(x<sk[h])sk[h]=x;} }
}
static double jaccard(const uint64_t*a,const uint64_t*b,int m){int e=0;for(int i=0;i<m;i++)if(a[i]==b[i])e++;return(double)e/m;}
static double n_fraction(const uint8_t*s,size_t l){if(l==0)return 1.0;size_t a=0;for(size_t i=0;i<l;i++)if(LUT[s[i]]!=255)a++;return 1.0-(double)a/l;}

/* ---- novelty promoted-set + LSH band index (from l1_novelty.c) ---- */
typedef struct { uint64_t key; int *ids; int n,cap; } bucket_t;
typedef struct { uint64_t*sk; int n,cap,m; bucket_t*buk; uint64_t nbuk,mask;
                 char**hdrs; } pset_t;    /* hdrs: promoted headers (for --emit) */
static void pset_init(pset_t*p,int m,uint64_t log2buk){ p->m=m;p->n=0;p->cap=1024;
    p->sk=malloc((size_t)p->cap*m*sizeof(uint64_t)); p->hdrs=malloc(p->cap*sizeof(char*));
    p->nbuk=(uint64_t)1<<log2buk; p->mask=p->nbuk-1; p->buk=calloc(p->nbuk,sizeof(bucket_t)); }
/* Date-partitioned novelty: fold the sample's month bucket into the LSH
 * band key so novelty is judged WITHIN a date window. A genome only ever compares against
 * promoted genomes from the SAME month, so undersampled months keep their own tau-diverse
 * representatives instead of being starved by redundancy from sequence-dense months.
 * mbucket = year*12+month (or -1 = undated; undated genomes share one bucket). */
static inline uint64_t band_key(int mbucket,int band,const uint64_t*sk,int R){
    uint64_t h=0x9E3779B97F4A7C15ULL^((uint64_t)band*0xC2B2AE3D27D4EB4FULL);
    h^=(uint64_t)(mbucket+1)*0x94D049BB133111EBULL; h*=0x100000001B3ULL; h^=h>>33;  /* date in signature */
    for(int r=0;r<R;r++){h^=sk[band*R+r];h*=0x100000001B3ULL;h^=h>>33;} return h; }
static void bucket_add(pset_t*p,uint64_t key,int id){ uint64_t idx=(key^(key>>29))&p->mask;
    while(p->buk[idx].ids&&p->buk[idx].key!=key)idx=(idx+1)&p->mask; bucket_t*b=&p->buk[idx];
    if(!b->ids){b->key=key;b->cap=4;b->ids=malloc(b->cap*sizeof(int));b->n=0;}
    if(b->n==b->cap){b->cap*=2;b->ids=realloc(b->ids,b->cap*sizeof(int));} b->ids[b->n++]=id; }
static bucket_t* bucket_find(pset_t*p,uint64_t key){ uint64_t idx=(key^(key>>29))&p->mask;
    while(p->buk[idx].ids){if(p->buk[idx].key==key)return &p->buk[idx];idx=(idx+1)&p->mask;} return NULL; }
static int is_novel(pset_t*p,int mbucket,const uint64_t*sk,double threshold,int R,int B){
    double need=1.0-threshold;
    /* band_key mixes in mbucket => bucket_find only returns SAME-month promoted genomes,
     * so the jaccard (pure-sequence) comparison is automatically within-month. */
    for(int band=0;band<B;band++){ bucket_t*b=bucket_find(p,band_key(mbucket,band,sk,R)); if(!b)continue;
        for(int t=0;t<b->n;t++){ if(jaccard(sk,&p->sk[(size_t)b->ids[t]*p->m],p->m)>need) return 0; } }
    return 1;
}
static void pset_add(pset_t*p,int mbucket,const uint64_t*sk,int R,int B,const char*hdr){
    if(p->n==p->cap){p->cap*=2;p->sk=realloc(p->sk,(size_t)p->cap*p->m*sizeof(uint64_t));p->hdrs=realloc(p->hdrs,p->cap*sizeof(char*));}
    memcpy(&p->sk[(size_t)p->n*p->m],sk,p->m*sizeof(uint64_t));
    p->hdrs[p->n]= hdr?strdup(hdr):NULL;
    for(int band=0;band<B;band++) bucket_add(p,band_key(mbucket,band,sk,R),p->n); p->n++;
}
/* parse month bucket (year*12+month0) from a GISAID header: field 2 of '|' split = YYYY-MM-DD.
 * returns -1 if undated/unparseable (all such genomes share one bucket). */
static int month_bucket(const char*hdr){
    const char*p=strchr(hdr,'|'); if(!p) return -1; p++;              /* -> collection date */
    if(!(p[0]>='0'&&p[0]<='9'&&p[1]>='0'&&p[1]<='9'&&p[2]>='0'&&p[2]<='9'&&p[3]>='0'&&p[3]<='9')) return -1;
    int y=(p[0]-'0')*1000+(p[1]-'0')*100+(p[2]-'0')*10+(p[3]-'0');
    if(p[4]!='-'||!(p[5]>='0'&&p[5]<='9')||!(p[6]>='0'&&p[6]<='9')) return -1;
    int mo=(p[5]-'0')*10+(p[6]-'0'); if(mo<1||mo>12) return -1;
    return y*12+(mo-1);
}

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);
    int rank,nproc; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    init_lut();
    const char*input=NULL,*emit=NULL; int k=DEF_K,m=DEF_M,warmup=DEF_WARMUP; long limit=0;
    long min_len=27000; double max_n_frac=0.05;   /* completeness gate (§2.22): drop partial/gappy */
    double err_rate=DEF_ERR_RATE,pcut=DEF_P,outlier_sd=DEF_OUTLIER_SD,threshold=DEF_THRESHOLD;
    for(int a=1;a<argc;a++){
        if(!strcmp(argv[a],"--input"))input=argv[++a];
        else if(!strcmp(argv[a],"--emit-promoted"))emit=argv[++a];
        else if(!strcmp(argv[a],"--k"))k=atoi(argv[++a]);
        else if(!strcmp(argv[a],"--sketch"))m=atoi(argv[++a]);
        else if(!strcmp(argv[a],"--threshold"))threshold=atof(argv[++a]);
        else if(!strcmp(argv[a],"--err-rate"))err_rate=atof(argv[++a]);
        else if(!strcmp(argv[a],"--p"))pcut=atof(argv[++a]);
        else if(!strcmp(argv[a],"--outlier-sd"))outlier_sd=atof(argv[++a]);
        else if(!strcmp(argv[a],"--warmup"))warmup=atoi(argv[++a]);
        else if(!strcmp(argv[a],"--limit"))limit=atol(argv[++a]);
        else if(!strcmp(argv[a],"--min-len"))min_len=atol(argv[++a]);
        else if(!strcmp(argv[a],"--max-n-frac"))max_n_frac=atof(argv[++a]);
    }
    if(!input){if(rank==0)fprintf(stderr,"--input required\n");MPI_Finalize();return 1;}
    int B=LSH_BANDS,R=m/B;
    uint64_t*salts=malloc(m*sizeof(uint64_t)); uint64_t s=0x123456789ABCDEFULL;
    for(int i=0;i<m;i++){s^=s<<13;s^=s>>7;s^=s<<17;salts[i]=s|1ULL;}

    cms_t cms; cms_init(&cms);
    pset_t ps; pset_init(&ps,m,20);

    /* byte-range shard */
    FILE*fp=fopen(input,"rb"); if(!fp){if(rank==0)fprintf(stderr,"open fail\n");MPI_Abort(MPI_COMM_WORLD,1);}
    fseeko(fp,0,SEEK_END); off_t fsize=ftello(fp);
    off_t slice=fsize/nproc, ms=(off_t)rank*slice, me=(rank==nproc-1)?fsize:(off_t)(rank+1)*slice;
    if(rank>0){ fseeko(fp,ms,SEEK_SET); int c; while((c=fgetc(fp))!=EOF&&c!='\n');
        while(1){long here=ftello(fp);c=fgetc(fp);if(c==EOF){ms=here;break;}if(c=='>'){ms=here;break;}while((c=fgetc(fp))!=EOF&&c!='\n');} }
    else ms=0;
    fseeko(fp,ms,SEEK_SET);

    size_t seqcap=1<<20,hdrcap=1<<12,kcap=1<<16;
    uint8_t*seq=malloc(seqcap); char*hdr=malloc(hdrcap); uint64_t*kbuf=malloc(kcap*sizeof(uint64_t));
    uint64_t*sk=malloc(m*sizeof(uint64_t));
    size_t seqlen=0; int in_seq=0; char*line=NULL; size_t lc=0; ssize_t ll;
    long n_mine=0,n_admit=0,n_reject=0,n_skip=0,n_lowqual=0; double sc_sum=0,sc_sqsum=0; long sc_n=0;
    double t0=MPI_Wtime(); int stop=0;

    #define PIPE_PROCESS() do{ if(in_seq){ n_mine++; \
        /* COMPLETENESS GATE (§2.22): drop partial/gappy genomes BEFORE novelty can promote them for \
         * being k-mer-sparse. Cheap length + N-fraction check, before k-mer extraction. */ \
        if((long)seqlen<min_len || n_fraction(seq,seqlen)>max_n_frac){ n_lowqual++; } else { \
        size_t nu=genome_kmers(seq,seqlen,k,kbuf,kcap); \
        if(nu==0){ n_skip++; } else { \
            /* ADMISSION */ \
            long n_singleton=0; for(size_t z=0;z<nu;z++){uint32_t d=cms_query(&cms,kbuf[z]);if(d==0)n_singleton++;} \
            int reject=0; \
            if(sc_n>=warmup && n_mine>warmup){ double mean=sc_sum/sc_n,var=sc_sqsum/sc_n-mean*mean; if(var<0)var=0; \
                if((double)n_singleton>mean+outlier_sd*sqrt(var)+0.5) reject=1; } \
            sc_sum+=n_singleton; sc_sqsum+=(double)n_singleton*n_singleton; sc_n++; \
            if(reject){ n_reject++; } else { n_admit++; \
                for(size_t z=0;z<nu;z++) cms_inc(&cms,kbuf[z]); \
                /* NOVELTY on admitted only, reusing the SAME k-mers. Date-partitioned: \
                 * novelty judged within the sample's month bucket. */ \
                minhash_from_kmers(kbuf,nu,m,salts,sk); \
                int mb=month_bucket(hdr); \
                if(is_novel(&ps,mb,sk,threshold,R,B)) pset_add(&ps,mb,sk,R,B,hdr); \
            } } } \
        if(limit && n_mine>=limit) stop=1; } }while(0)

    while((ll=getline(&line,&lc,fp))>=0){
        if(line[0]=='>'){ long hoff=ftello(fp)-ll; PIPE_PROCESS(); if(stop)break;
            if(hoff>=me){in_seq=0;break;}
            size_t L=ll; while(L>0&&(line[L-1]=='\n'||line[L-1]=='\r'))L--;
            if(L>=hdrcap){hdrcap=L+1;hdr=realloc(hdr,hdrcap);} memcpy(hdr,line+1,L-1); hdr[L-1]=0;
            seqlen=0; in_seq=1;
        } else { size_t L=ll; while(L>0&&(line[L-1]=='\n'||line[L-1]=='\r'))L--;
            if(seqlen+L+1>seqcap){while(seqlen+L+1>seqcap)seqcap*=2;seq=realloc(seq,seqcap);}
            for(size_t z=0;z<L;z++) seq[seqlen++]=toupper((unsigned char)line[z]); }
    }
    if(!stop && in_seq) PIPE_PROCESS();
    fclose(fp);
    double t1=MPI_Wtime();

    /* ---- PHASE 2: gather shard-local promoted (sketches + headers) -> rank0 merge ----
     * Headers gathered as fixed-width (HW) buffers so MPI_Gatherv works on plain bytes.
     * Sequences are NOT moved (they'd be huge); instead rank0 broadcasts the SURVIVING
     * header set and each rank RE-EMITS its own promoted genomes by re-scanning its shard. */
    #define HW 256
    int local_p=ps.n; int*counts=NULL;
    if(rank==0) counts=malloc(nproc*sizeof(int));
    MPI_Gather(&local_p,1,MPI_INT,counts,1,MPI_INT,0,MPI_COMM_WORLD);
    int total=0; int*rc=NULL,*rd=NULL,*hc=NULL,*hd=NULL; uint64_t*allsk=NULL; char*allhdr=NULL;
    if(rank==0){ rc=malloc(nproc*sizeof(int)); rd=malloc(nproc*sizeof(int));
        hc=malloc(nproc*sizeof(int)); hd=malloc(nproc*sizeof(int)); int off=0,hoff=0;
        for(int i=0;i<nproc;i++){rc[i]=counts[i]*m;rd[i]=off;off+=rc[i];
            hc[i]=counts[i]*HW; hd[i]=hoff; hoff+=hc[i]; total+=counts[i];}
        allsk=malloc((size_t)total*m*sizeof(uint64_t)); allhdr=malloc((size_t)total*HW); }
    /* pack this rank's headers into fixed-width buffer */
    char*myhdr=calloc((size_t)local_p*HW,1);
    for(int i=0;i<local_p;i++){ if(ps.hdrs[i]){ strncpy(&myhdr[(size_t)i*HW],ps.hdrs[i],HW-1); } }
    MPI_Gatherv(ps.sk,local_p*m,MPI_UINT64_T,allsk,rc,rd,MPI_UINT64_T,0,MPI_COMM_WORLD);
    MPI_Gatherv(myhdr,local_p*HW,MPI_CHAR,allhdr,hc,hd,MPI_CHAR,0,MPI_COMM_WORLD);

    long g_mine=0,g_admit=0,g_reject=0,g_skip=0,g_lowqual=0;
    MPI_Reduce(&n_mine,&g_mine,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&n_admit,&g_admit,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&n_reject,&g_reject,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&n_skip,&g_skip,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&n_lowqual,&g_lowqual,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    double dt=t1-t0,maxdt; MPI_Reduce(&dt,&maxdt,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    /* rank0 merges; builds the SURVIVING header list, broadcast to all ranks for re-emit */
    int n_promoted=0; char*promoted_hdrs=NULL;   /* n_promoted*HW, on all ranks after bcast */
    if(rank==0){
        pset_t merged; pset_init(&merged,m,20);
        promoted_hdrs=malloc((size_t)total*HW);   /* upper bound */
        for(int i=0;i<total;i++){ uint64_t*msk=&allsk[(size_t)i*m];
            /* cross-shard merge must ALSO partition by month (else same-seq/different-month
             * candidates collapse), so recompute the month from the gathered header. */
            int mb=month_bucket(&allhdr[(size_t)i*HW]);
            if(is_novel(&merged,mb,msk,threshold,R,B)){ pset_add(&merged,mb,msk,R,B,NULL);
                memcpy(&promoted_hdrs[(size_t)n_promoted*HW],&allhdr[(size_t)i*HW],HW); n_promoted++; } }
        printf("\n=== L1 PIPELINE (composed: admission + novelty, one pass) ===\n");
        printf("input=%s nproc=%d k=%d sketch=%d threshold=%g\n",input,nproc,k,m,threshold);
        printf("seen (input genomes) : %ld\n",g_mine);
        printf("  LOW-QUAL (partial) : %ld  (%.3f%%)  [completeness gate: <%ld nt or >%.0f%% N]\n",
               g_lowqual,100.0*g_lowqual/(g_mine>0?g_mine:1),min_len,100*max_n_frac);
        printf("  skip(no kmers)     : %ld\n",g_skip);
        printf("  REJECT (error)     : %ld  (%.3f%%)  [admission]\n",g_reject,100.0*g_reject/(g_mine>0?g_mine:1));
        printf("  admitted           : %ld  (%.3f%%)\n",g_admit,100.0*g_admit/(g_mine>0?g_mine:1));
        printf("PROMOTED to L2       : %d  (%.4f%% of input, %.4f%% of admitted)\n",
               n_promoted,100.0*n_promoted/(g_mine>0?g_mine:1),100.0*n_promoted/(g_admit>0?g_admit:1));
        printf("overall rejection    : %.1fx (input/promoted)\n",(double)g_mine/(n_promoted>0?n_promoted:1));
        printf("throughput           : %.0f seq/s (aggregate, %d ranks)  wall %.1fs\n",g_mine/(maxdt>0?maxdt:1),nproc,maxdt);
    }

    /* free the big per-rank structures before the emit re-scan — CMS/pset no longer needed;
     * halves peak memory during emit (matters at scale — this class of over-allocation OOM'd nodes). */
    free(cms.t); cms.t=NULL;
    free(ps.sk); free(kbuf); free(sk);

    /* ---- EMIT: broadcast surviving headers, each rank re-scans its shard + writes owned promoted genomes ---- */
    if(emit){
        MPI_Bcast(&n_promoted,1,MPI_INT,0,MPI_COMM_WORLD);
        if(rank!=0) promoted_hdrs=malloc((size_t)n_promoted*HW);
        MPI_Bcast(promoted_hdrs,(size_t)n_promoted*HW,MPI_CHAR,0,MPI_COMM_WORLD);
        /* build a hash set of promoted headers for O(1) lookup */
        /* (simple: linear scan is fine for a few thousand; use qsort+bsearch for scale) */
        qsort(promoted_hdrs,n_promoted,HW,(int(*)(const void*,const void*))strcmp);
        char outpath[4096]; snprintf(outpath,sizeof outpath,"%s.rank%d.fasta",emit,rank);
        FILE*of=fopen(outpath,"w");
        FILE*fp2=fopen(input,"rb"); fseeko(fp2,ms,SEEK_SET);
        char*ln=NULL; size_t lcp=0; ssize_t l2; int emit_seq=0; long ehoff;
        char curh[HW]={0}; uint8_t*ebuf=malloc(seqcap); size_t ebl=0; int have=0;
        #define FLUSH_EMIT() do{ if(have){ char key[HW]; strncpy(key,curh,HW-1); key[HW-1]=0; \
            if(bsearch(key,promoted_hdrs,n_promoted,HW,(int(*)(const void*,const void*))strcmp)){ \
                fprintf(of,">%s\n",curh); fwrite(ebuf,1,ebl,of); fputc('\n',of); } } }while(0)
        while((l2=getline(&ln,&lcp,fp2))>=0){
            if(ln[0]=='>'){ ehoff=ftello(fp2)-l2; FLUSH_EMIT();
                if(ehoff>=me){have=0;break;}
                size_t L=l2; while(L>0&&(ln[L-1]=='\n'||ln[L-1]=='\r'))L--;
                strncpy(curh,ln+1,L-1<HW-1?L-1:HW-1); curh[(L-1<HW-1?L-1:HW-1)]=0;
                ebl=0; have=1;
            } else { size_t L=l2; while(L>0&&(ln[L-1]=='\n'||ln[L-1]=='\r'))L--;
                if(ebl+L+1>seqcap){while(ebl+L+1>seqcap)seqcap*=2;ebuf=realloc(ebuf,seqcap);}
                for(size_t z=0;z<L;z++) ebuf[ebl++]=toupper((unsigned char)ln[z]); }
        }
        if(have) FLUSH_EMIT();
        fclose(fp2); fclose(of); free(ebuf); free(ln);
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank==0) fprintf(stderr,"[emit] promoted genomes written to %s.rank*.fasta (concat for L2 input)\n",emit);
    }
    MPI_Finalize();
    return 0;
}
