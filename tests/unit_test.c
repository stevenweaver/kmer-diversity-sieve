/* unit_test.c -- unit tests for the pure primitives in scripts/l1_pipeline.c.
 *
 * We #include the implementation directly (with -DL1_NO_MAIN so its main() is
 * excluded) to reach its `static` functions without duplicating code. Build:
 *
 *   cc -O2 -DL1_NO_MAIN -D_FILE_OFFSET_BITS=64 -I../scripts \
 *      -o unit_test unit_test.c -lm
 *
 * (the Makefile `test` target does this for you). Exit non-zero on first failure.
 */
#include "l1_pipeline.c"

#include <stdio.h>

static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
           fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } \
} while (0)

/* ---- helpers ---- */
static size_t kmers_of(const char *s, int k, uint64_t *out, size_t cap) {
    return genome_kmers((const uint8_t *)s, strlen(s), k, out, cap);
}

/* ================= k-mer extraction / canonicalization ================= */
static void test_genome_kmers(void) {
    uint64_t buf[256];

    /* too-short sequence -> zero k-mers */
    CHECK(kmers_of("ACGT", 5, buf, 256) == 0, "k>len should yield 0 kmers");

    /* exactly-k-length sequence -> exactly 1 distinct k-mer */
    CHECK(kmers_of("ACGTA", 5, buf, 256) == 1, "len==k should yield 1 distinct kmer");

    /* canonical form: a sequence and its reverse-complement share the SAME
     * canonical k-mer set (that's the whole point of canonicalization). */
    uint64_t a[256], b[256];
    size_t na = kmers_of("ACGTTGCA", 5, a, 256);       /* palindrome-ish */
    size_t nb = kmers_of("TGCAACGT", 5, b, 256);       /* revcomp of above */
    CHECK(na == nb, "fwd and revcomp must have equal distinct-kmer counts (%zu vs %zu)", na, nb);
    /* both are sorted+uniq'd, so a byte compare of the arrays proves set-equality */
    CHECK(na > 0 && memcmp(a, b, na * sizeof(uint64_t)) == 0,
          "fwd and revcomp must produce the SAME canonical kmer set");

    /* output is sorted and unique (genome_kmers post-sorts+dedups) */
    size_t n = kmers_of("AAAAAAAAAA", 3, buf, 256);     /* all identical kmers */
    CHECK(n == 1, "homopolymer should collapse to 1 distinct kmer, got %zu", n);

    /* N / non-ACGT breaks runs: "ACGTNACGT" with k=5 has no run of length>=5 */
    CHECK(kmers_of("ACGTNACGT", 5, buf, 256) == 0, "N should break the k-mer run");
    CHECK(kmers_of("ACGTNACGTA", 5, buf, 256) == 1, "second run of len>=k should yield kmers");
}

/* ================= n_fraction (completeness gate input) ================= */
static void test_n_fraction(void) {
    init_lut();
    CHECK(n_fraction((const uint8_t *)"ACGT", 4) == 0.0, "all-ACGT -> 0 N-fraction");
    CHECK(n_fraction((const uint8_t *)"NNNN", 4) == 1.0, "all-N -> 1.0 N-fraction");
    double h = n_fraction((const uint8_t *)"ACNN", 4);
    CHECK(h > 0.49 && h < 0.51, "half-N -> ~0.5, got %f", h);
    CHECK(n_fraction((const uint8_t *)"", 0) == 1.0, "empty -> 1.0 (treated as all-missing)");
}

/* ================= month_bucket (date-partition key parsing) ================= */
static void test_month_bucket(void) {
    /* GISAID-style header: name|date|... ; date = YYYY-MM-DD ; bucket = year*12 + (month-1) */
    CHECK(month_bucket("hCoV-19/x|2021-01-15|foo") == 2021 * 12 + 0, "2021-01 -> 2021*12+0");
    CHECK(month_bucket("x|2020-12-31|y") == 2020 * 12 + 11, "2020-12 -> 2020*12+11");
    /* undated / unparseable -> -1 (shared bucket) */
    CHECK(month_bucket("no-pipe-here") == -1, "no '|' -> -1");
    CHECK(month_bucket("x|notadate|y") == -1, "non-numeric date -> -1");
    CHECK(month_bucket("x|2021-13-01|y") == -1, "month>12 -> -1");
    CHECK(month_bucket("x|2021-00-01|y") == -1, "month 0 -> -1");
}

/* ================= Count-Min sketch (admission frequency table) ================= */
static void test_cms(void) {
    cms_t c; cms_init(&c);
    /* a never-inserted key reads df==0 (the "singleton" signal admission relies on) */
    CHECK(cms_query(&c, 0x1234567ULL) == 0, "unseen key -> df 0");
    cms_inc(&c, 0x1234567ULL);
    CHECK(cms_query(&c, 0x1234567ULL) >= 1, "after one inc, df >= 1");
    for (int i = 0; i < 41; i++) cms_inc(&c, 0x1234567ULL);
    CHECK(cms_query(&c, 0x1234567ULL) >= 42, "count-min never under-counts");
    free(c.t);
}

/* ================= MinHash + Jaccard (novelty distance) ================= */
static void test_minhash_jaccard(void) {
    init_lut();
    int m = 64;
    uint64_t salts[64], s = 0x123456789ABCDEFULL;
    for (int i = 0; i < m; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; salts[i] = s | 1ULL; }

    uint64_t ka[256], kb[256], ska[64], skb[64];
    /* identical sequences -> identical sketches -> Jaccard 1.0 */
    size_t na = kmers_of("ACGTACGTACGTACGTACGTACGT", 7, ka, 256);
    minhash_from_kmers(ka, na, m, salts, ska);
    minhash_from_kmers(ka, na, m, salts, skb);
    CHECK(jaccard(ska, skb, m) == 1.0, "identical inputs -> Jaccard 1.0");

    /* a totally different sequence -> estimated Jaccard well below 1 */
    size_t nb = kmers_of("TTTTAAAACCCCGGGGTTTTAAAACCCC", 7, kb, 256);
    minhash_from_kmers(kb, nb, m, salts, skb);
    double j = jaccard(ska, skb, m);
    CHECK(j < 0.9, "disjoint-ish inputs -> Jaccard < 0.9, got %f", j);
    CHECK(j >= 0.0 && j <= 1.0, "Jaccard estimate must be in [0,1], got %f", j);
}

/* ================= binomial-ish sanity: warmup outlier math is elsewhere ================= */

int main(void) {
    init_lut();
    test_genome_kmers();
    test_n_fraction();
    test_month_bucket();
    test_cms();
    test_minhash_jaccard();

    fprintf(stderr, "\nunit_test: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
