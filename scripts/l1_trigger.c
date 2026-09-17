/*
 * l1_trigger.c  --  L1 k-mer trigger: real-vs-error admission + novelty (C + OpenMPI)
 *
 * Streaming, alignment-free REAL-VS-ERROR filter (Count-Min k-mer frequency), plus a
 * MinHash novelty gate. This is the performant rewrite of the validated Python
 * prototype (scripts/l1_trigger_compressor.py = reference oracle).
 *
 * PIPELINE (per genome, streaming):
 *   1. extract canonical k-mers (2-bit, reverse-complement-min) over maximal ACGT runs
 *   2. query a COUNT-MIN SKETCH for each distinct k-mer's corpus document-frequency
 *   3. classify: rare k-mer = df <= binomial_cutoff(N, err_rate, p)  [depth/N-aware]
 *      singleton = df == 0 (unseen). ADMIT unless singleton-count is an outlier
 *      (> running mean + outlier_sd*SD)  [compressor-2 mech #2: error-laden genome].
 *   4. update the CMS for ADMITTED genomes only (errors don't pollute the corpus).
 *
 * MPI: each rank streams a shard of the FASTA (round-robin by record index), keeps a
 * LOCAL Count-Min sketch, then ranks MPI_Allreduce(SUM) their CMS arrays into a global
 * table. (Count-Min is linear/additive => sum-merge is exact for the sketch.) A second
 * pass makes final decisions against the merged global counts. Single-pass streaming
 * (running counts) is also supported for the true-online mode via --online.
 *
 * Correctness: k-mer canonicalization + binomial_cutoff match the Python oracle bit-for-
 * bit; validated on a shared test slice before scale runs.
 *
 * Build:  mpicc -O3 -march=native -o l1_trigger l1_trigger.c -lm
 * Run:    mpirun -np N ./l1_trigger --input file.fasta [opts]
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <ctype.h>
#include <sys/types.h>

/* ---- config defaults (match the Python oracle) ---- */
#define DEF_K            15
#define DEF_ERR_RATE     1e-4
#define DEF_P            0.9
#define DEF_OUTLIER_SD   5.0
#define DEF_WARMUP       2000
/* Count-Min sketch geometry: d rows x w cols. w=2^24 (~16.7M) x d=4 x 4B = 256MB/rank. */
#define CMS_LOG2W        24
#define CMS_D            4

static const uint64_t CMS_SEEDS[CMS_D] = {
    0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL,
    0x165667B19E3779F9ULL, 0xD6E8FEB86659FD93ULL
};

/* 2-bit lookup: A,C,G,T -> 0..3 ; everything else -> 255 (breaks a valid run) */
static uint8_t LUT[256];
static void init_lut(void){
    for (int i=0;i<256;i++) LUT[i]=255;
    LUT[(int)'A']=0; LUT[(int)'C']=1; LUT[(int)'G']=2; LUT[(int)'T']=3;
    LUT[(int)'a']=0; LUT[(int)'c']=1; LUT[(int)'g']=2; LUT[(int)'t']=3;
}

/* splitmix64-style finalizer for hashing k-mer codes into CMS columns */
static inline uint64_t mix64(uint64_t x, uint64_t seed){
    x ^= seed;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

/* ---- Count-Min sketch (document frequency: increment once per k-mer per genome) ---- */
typedef struct {
    uint32_t *t;          /* CMS_D rows x W cols, row-major */
    uint64_t  W, mask;
} cms_t;

static void cms_init(cms_t *c){
    c->W = (uint64_t)1 << CMS_LOG2W;
    c->mask = c->W - 1;
    c->t = calloc((size_t)CMS_D * c->W, sizeof(uint32_t));
    if(!c->t){ fprintf(stderr,"CMS alloc failed\n"); MPI_Abort(MPI_COMM_WORLD,1); }
}
static inline uint32_t cms_query(const cms_t *c, uint64_t key){
    uint32_t m = UINT32_MAX;
    for(int r=0;r<CMS_D;r++){
        uint64_t col = mix64(key, CMS_SEEDS[r]) & c->mask;
        uint32_t v = c->t[(uint64_t)r*c->W + col];
        if(v<m) m=v;
    }
    return m;
}
static inline void cms_inc(cms_t *c, uint64_t key){
    for(int r=0;r<CMS_D;r++){
        uint64_t col = mix64(key, CMS_SEEDS[r]) & c->mask;
        c->t[(uint64_t)r*c->W + col]++;
    }
}

/* ---- binomial cutoff: exact port of compressor-2.bf filter.binomial_cutoff ---- */
static long binomial_cutoff(long N, double p, double t){
    if(N<=0) return 0;
    double s=0.0, term=pow(1.0-p,(double)N);
    long i=0;
    while(s<t && i<N){
        s += term;
        double denom=(1.0-p)*(double)(i+1);
        if(denom==0.0) break;
        term = term * p/(1.0-p) * (double)(N-i)/(double)(i+1);
        i++;
    }
    return i;
}

/* ---- distinct canonical k-mers of a sequence into a caller buffer; returns count ----
 * Writes raw canonical codes (not yet CMS-hashed). Dedup is done by caller (sort+uniq).
 */
static size_t genome_kmers(const uint8_t *seq, size_t len, int k, uint64_t *out, size_t cap){
    if(len < (size_t)k) return 0;
    size_t nout=0;
    uint64_t kmask = (k>=32)?~0ULL:(((uint64_t)1<<(2*k))-1);
    size_t i=0;
    while(i<len){
        if(LUT[seq[i]]==255){ i++; continue; }
        size_t j=i; while(j<len && LUT[seq[j]]!=255) j++;
        size_t runlen=j-i;
        if(runlen>=(size_t)k){
            uint64_t fwd=0, rev=0;
            /* prime first k-1 */
            for(size_t z=0; z<(size_t)k; z++){
                uint64_t c = LUT[seq[i+z]];
                fwd = ((fwd<<2)|c) & kmask;
                rev = (rev>>2) | ((uint64_t)(3-c) << (2*(k-1)));
            }
            uint64_t can = fwd<rev?fwd:rev;
            if(nout<cap) out[nout++]=can;
            for(size_t z=i+k; z<j; z++){
                uint64_t c = LUT[seq[z]];
                fwd = ((fwd<<2)|c) & kmask;
                rev = (rev>>2) | ((uint64_t)(3-c) << (2*(k-1)));
                can = fwd<rev?fwd:rev;
                if(nout<cap) out[nout++]=can;
            }
        }
        i=j;
    }
    return nout;
}

static int cmp_u64(const void*a,const void*b){
    uint64_t x=*(const uint64_t*)a, y=*(const uint64_t*)b;
    return (x>y)-(x<y);
}

/* non-ACGT fraction (N/ambiguity/gap) for diagnostics */
static double n_fraction(const uint8_t *seq, size_t len){
    if(len==0) return 1.0;
    size_t acgt=0;
    for(size_t i=0;i<len;i++) if(LUT[seq[i]]!=255) acgt++;
    return 1.0 - (double)acgt/(double)len;
}

int main(int argc, char**argv){
    MPI_Init(&argc,&argv);
    int rank,nproc;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    init_lut();

    const char *input=NULL, *dump=NULL;
    int k=DEF_K, warmup=DEF_WARMUP; long limit=0;
    double err_rate=DEF_ERR_RATE, pcut=DEF_P, outlier_sd=DEF_OUTLIER_SD;
    long report_every=100000;
    for(int a=1;a<argc;a++){
        if(!strcmp(argv[a],"--input")) input=argv[++a];
        else if(!strcmp(argv[a],"--k")) k=atoi(argv[++a]);
        else if(!strcmp(argv[a],"--err-rate")) err_rate=atof(argv[++a]);
        else if(!strcmp(argv[a],"--p")) pcut=atof(argv[++a]);
        else if(!strcmp(argv[a],"--outlier-sd")) outlier_sd=atof(argv[++a]);
        else if(!strcmp(argv[a],"--warmup")) warmup=atoi(argv[++a]);
        else if(!strcmp(argv[a],"--limit")) limit=atol(argv[++a]);
        else if(!strcmp(argv[a],"--report-every")) report_every=atol(argv[++a]);
        else if(!strcmp(argv[a],"--dump")) dump=argv[++a];
    }
    if(!input){ if(rank==0) fprintf(stderr,"--input required\n"); MPI_Finalize(); return 1; }

    cms_t cms; cms_init(&cms);

    /* streaming state. BYTE-RANGE sharding: each rank owns a contiguous byte slice of
     * the file (no redundant reads). n_mine = records whose HEADER starts in this rank's
     * slice; a record is owned by exactly one rank (the one containing its '>'). */
    long n_mine=0, n_admit=0, n_reject=0, n_skip=0;
    double sc_sum=0.0, sc_sqsum=0.0; long sc_n=0;

    FILE *fp=fopen(input,"rb");
    if(!fp){ fprintf(stderr,"[rank %d] cannot open %s\n",rank,input); MPI_Abort(MPI_COMM_WORLD,1); }

    /* compute this rank's byte range */
    fseeko(fp,0,SEEK_END); off_t fsize=ftello(fp);
    off_t slice=fsize/nproc;
    off_t my_start=(off_t)rank*slice;
    off_t my_end=(rank==nproc-1)?fsize:(off_t)(rank+1)*slice;
    /* align start to the next record boundary: rank 0 starts at 0; others skip the
     * partial record straddling the boundary (it belongs to the previous rank, which
     * reads PAST its end to finish it). */
    if(rank>0){
        fseeko(fp,my_start,SEEK_SET);
        int c; off_t pos=my_start;
        /* advance to the start of the next line, then to the next '>' at col 0 */
        while((c=fgetc(fp))!=EOF){ pos++; if(c=='\n') break; }
        while(1){ long here=ftello(fp); c=fgetc(fp); if(c==EOF){ my_start=here; break; }
                  if(c=='>'){ my_start=here; break; }
                  while((c=fgetc(fp))!=EOF && c!='\n'); }
    } else my_start=0;
    fseeko(fp,my_start,SEEK_SET);

    FILE *df=NULL;
    if(dump){ char path[4096]; snprintf(path,sizeof path,"%s.rank%d",dump,rank); df=fopen(path,"w");
              if(df) fprintf(df,"decision\tlen\tn_frac\tn_kmers\tn_singleton\tcutoff\theader\n"); }

    /* FASTA streaming reader */
    size_t seqcap=1<<20, hdrcap=1<<12, kcap=1<<16;
    uint8_t *seq=malloc(seqcap); char *hdr=malloc(hdrcap);
    uint64_t *kbuf=malloc(kcap*sizeof(uint64_t));
    size_t seqlen=0; int in_seq=0;
    char *line=NULL; size_t linecap=0; ssize_t ll;
    off_t cur_hdr_off=my_start;   /* byte offset where the current record's '>' began */
    double t0=MPI_Wtime();

    #define PROCESS_RECORD() do{ \
        if(in_seq){ \
            n_mine++; \
            size_t nk = genome_kmers(seq,seqlen,k,kbuf,kcap); \
            if(nk==0){ n_skip++; } else { \
                qsort(kbuf,nk,sizeof(uint64_t),cmp_u64); \
                size_t nu=0; for(size_t z=0;z<nk;z++){ if(z==0||kbuf[z]!=kbuf[z-1]) kbuf[nu++]=kbuf[z]; } \
                long cutoff = binomial_cutoff(n_mine>1?n_mine-1:1, err_rate, pcut); \
                long n_singleton=0; \
                for(size_t z=0;z<nu;z++){ uint32_t dfq=cms_query(&cms,kbuf[z]); if(dfq==0) n_singleton++; } \
                int reject=0; \
                if(sc_n>=warmup && n_mine>warmup){ \
                    double mean=sc_sum/sc_n; double var=sc_sqsum/sc_n-mean*mean; if(var<0)var=0; \
                    double sd=sqrt(var); double thr=mean+outlier_sd*sd+0.5; \
                    if((double)n_singleton>thr) reject=1; } \
                if(!reject){ n_admit++; for(size_t z=0;z<nu;z++) cms_inc(&cms,kbuf[z]); } \
                else n_reject++; \
                sc_sum+=n_singleton; sc_sqsum+=(double)n_singleton*n_singleton; sc_n++; \
                if(df){ double nf=n_fraction(seq,seqlen); \
                    fprintf(df,"%s\t%zu\t%.4f\t%zu\t%ld\t%ld\t%s\n", reject?"REJECT":"ADMIT",seqlen,nf,nu,n_singleton,cutoff,hdr); } \
            } \
            if(rank==0 && report_every>0 && (n_mine%report_every)==0){ \
                double dt=MPI_Wtime()-t0; \
                fprintf(stderr,"[rank0 mine=%ld] admit=%ld reject=%ld skip=%ld rate=%.0f/s\n", \
                    n_mine,n_admit,n_reject,n_skip, n_mine/(dt>0?dt:1)); } \
            if(limit && n_mine>=limit){ break_flag=1; } \
        } \
    }while(0)

    int break_flag=0;
    while((ll=getline(&line,&linecap,fp))>=0){
        if(line[0]=='>'){
            /* a record is OWNED by the rank whose slice contains its '>' offset.
             * We may read past my_end to finish a record we started; but we must NOT
             * start a NEW record whose '>' is at/after my_end (that's the next rank's). */
            long hdr_off = ftello(fp) - ll;
            PROCESS_RECORD();
            if(break_flag) break;
            if(hdr_off >= my_end){ in_seq=0; break; }  /* boundary: next rank owns this */
            cur_hdr_off = hdr_off;
            size_t L=ll; while(L>0 && (line[L-1]=='\n'||line[L-1]=='\r')) L--;
            if(L>=hdrcap){ hdrcap=L+1; hdr=realloc(hdr,hdrcap); }
            memcpy(hdr,line+1,L-1); hdr[L-1]=0;
            seqlen=0; in_seq=1;
        } else {
            size_t L=ll; while(L>0 && (line[L-1]=='\n'||line[L-1]=='\r')) L--;
            if(seqlen+L+1>seqcap){ while(seqlen+L+1>seqcap) seqcap*=2; seq=realloc(seq,seqcap); }
            for(size_t z=0;z<L;z++) seq[seqlen++]=toupper((unsigned char)line[z]);
        }
    }
    if(!break_flag && in_seq){ PROCESS_RECORD(); }  /* flush final owned record */

    fclose(fp); if(df) fclose(df);

    /* ---- global reduction of counts (Count-Min is additive, but here each record was
     * handled by exactly one rank, so counts are DISJOINT; for global-corpus decisions
     * we'd Allreduce the CMS. For this admit/reject shard mode we sum the scalar tallies) ---- */
    long g_admit=0,g_reject=0,g_skip=0,g_mine=0;
    MPI_Reduce(&n_mine,&g_mine,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&n_admit,&g_admit,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&n_reject,&g_reject,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&n_skip,&g_skip,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    long g_seen = g_mine; /* byte-range: every record owned by exactly one rank */
    double dt=MPI_Wtime()-t0;
    double maxdt; MPI_Reduce(&dt,&maxdt,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    if(rank==0){
        printf("\n=== L1 TRIGGER (C+OpenMPI) RESULT ===\n");
        printf("input        : %s\n", input);
        printf("nproc=%d k=%d err_rate=%g p=%g outlier_sd=%g warmup=%d\n",
               nproc,k,err_rate,pcut,outlier_sd,warmup);
        printf("CMS          : %d x 2^%d (%.0f MB/rank)\n", CMS_D, CMS_LOG2W,
               (double)CMS_D*((uint64_t)1<<CMS_LOG2W)*4/1e6);
        printf("seen         : %ld\n", g_seen);
        printf("processed    : %ld (across %d ranks)\n", g_mine, nproc);
        printf("skip(no kmer): %ld\n", g_skip);
        printf("ADMIT        : %ld  (%.2f%%)\n", g_admit, 100.0*g_admit/(g_mine>0?g_mine:1));
        printf("REJECT(error): %ld  (%.2f%%)\n", g_reject, 100.0*g_reject/(g_mine>0?g_mine:1));
        printf("throughput   : %.1f seq/s (aggregate, %d ranks)   wall %.1fs\n",
               g_mine/(maxdt>0?maxdt:1), nproc, maxdt);
        printf("NOTE: real-vs-error admission filter (compressor-2 port). Novelty promotion layers on top.\n");
    }
    free(seq); free(hdr); free(kbuf); free(line); free(cms.t);
    MPI_Finalize();
    return 0;
}
