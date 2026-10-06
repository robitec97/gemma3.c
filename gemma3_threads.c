/*
 * gemma3_threads.c - POSIX thread pool implementation for Linux and macOS
 *
 * A forward pass dispatches several hundred small parallel jobs per token, so
 * dispatch latency matters as much as throughput. Workers therefore spin on an
 * atomic generation counter for a short while after each job and only fall
 * back to sleeping on a condition variable when the pool goes idle. The
 * calling thread executes a share of every job instead of just waiting.
 */

#include "gemma3_threads.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <pthread/qos.h>
#endif

/* How long an idle worker keeps spinning before it goes to sleep. Long enough
 * to bridge the gaps between jobs within a token (sampling, small serial ops),
 * short enough not to burn a core while waiting for user input. */
#define SPIN_NS 2000000LL  /* 2 ms */

static inline void cpu_relax(void) {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause" ::: "memory");
#endif
}

static long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

struct gemma3_thread_pool {
    int num_threads;              /* total, including the calling thread */
    pthread_t *threads;           /* num_threads - 1 workers */
    int num_started;
    pthread_mutex_t mutex;
    pthread_cond_t cond;

    gemma3_task_fn fn;            /* published before generation is bumped */
    void *arg;

    atomic_uint generation;       /* incremented once per job */
    atomic_int pending;           /* workers that have not finished the job */
    atomic_int sleeping;          /* workers blocked on cond */
    atomic_int shutdown;
};

typedef struct {
    gemma3_thread_pool *pool;
    int thread_idx;
} worker_arg;

static void *worker_func(void *param) {
    worker_arg *wa = (worker_arg *)param;
    gemma3_thread_pool *pool = wa->pool;
    int idx = wa->thread_idx;
    free(wa);

#ifdef __APPLE__
    /* Prefer performance cores for compute threads. */
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    unsigned seen = 0;
    for (;;) {
        unsigned gen;
        long long spin_start = now_ns();
        int spins = 0;
        while ((gen = atomic_load(&pool->generation)) == seen) {
            if (atomic_load_explicit(&pool->shutdown, memory_order_relaxed)) return NULL;
            cpu_relax();
            if ((++spins & 1023) != 0 || now_ns() - spin_start < SPIN_NS) continue;

            /* Idle: sleep until the next job. The seq_cst increment of
             * `sleeping` followed by the generation re-check pairs with the
             * submitter's generation bump followed by its `sleeping` check,
             * so a wake-up can never be missed. */
            pthread_mutex_lock(&pool->mutex);
            atomic_fetch_add(&pool->sleeping, 1);
            while (atomic_load(&pool->generation) == seen && !atomic_load(&pool->shutdown)) {
                pthread_cond_wait(&pool->cond, &pool->mutex);
            }
            atomic_fetch_sub(&pool->sleeping, 1);
            pthread_mutex_unlock(&pool->mutex);
            spin_start = now_ns();
            spins = 0;
        }
        if (atomic_load(&pool->shutdown)) return NULL;
        seen = gen;

        pool->fn(pool->arg, idx, pool->num_threads);
        atomic_fetch_sub_explicit(&pool->pending, 1, memory_order_release);
    }
}

int gemma3_num_cpus(void) {
#if defined(__APPLE__)
    int n = 0;
    size_t len = sizeof(n);
    if (sysctlbyname("hw.ncpu", &n, &len, NULL, 0) == 0 && n > 0) return n;
    return 1;
#elif defined(_SC_NPROCESSORS_ONLN)
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 1;
#else
    return 1;
#endif
}

gemma3_thread_pool *gemma3_thread_pool_create(int num_threads) {
    if (num_threads <= 0) {
        const char *env = getenv("GEMMA3_THREADS");
        if (env && atoi(env) > 0) num_threads = atoi(env);
    }
    if (num_threads <= 0) num_threads = gemma3_num_cpus();
    if (num_threads > 256) num_threads = 256;

    gemma3_thread_pool *pool = (gemma3_thread_pool *)calloc(1, sizeof(gemma3_thread_pool));
    if (!pool) return NULL;

    pool->num_threads = num_threads;
    atomic_init(&pool->generation, 0);
    atomic_init(&pool->pending, 0);
    atomic_init(&pool->sleeping, 0);
    atomic_init(&pool->shutdown, 0);
    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->cond, NULL);

    if (num_threads > 1) {
        pool->threads = (pthread_t *)calloc((size_t)num_threads - 1, sizeof(pthread_t));
        if (!pool->threads) {
            gemma3_thread_pool_destroy(pool);
            return NULL;
        }
        for (int i = 1; i < num_threads; i++) {
            worker_arg *wa = (worker_arg *)malloc(sizeof(worker_arg));
            if (!wa) {
                gemma3_thread_pool_destroy(pool);
                return NULL;
            }
            wa->pool = pool;
            wa->thread_idx = i;
            if (pthread_create(&pool->threads[i - 1], NULL, worker_func, wa) != 0) {
                free(wa);
                gemma3_thread_pool_destroy(pool);
                return NULL;
            }
            pool->num_started++;
        }
    }

    return pool;
}

void gemma3_thread_pool_destroy(gemma3_thread_pool *pool) {
    if (!pool) return;

    pthread_mutex_lock(&pool->mutex);
    atomic_store(&pool->shutdown, 1);
    pthread_cond_broadcast(&pool->cond);
    pthread_mutex_unlock(&pool->mutex);

    for (int i = 0; i < pool->num_started; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    free(pool->threads);

    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->cond);
    free(pool);
}

int gemma3_thread_pool_size(const gemma3_thread_pool *pool) {
    return pool ? pool->num_threads : 1;
}

void gemma3_thread_pool_run(gemma3_thread_pool *pool, gemma3_task_fn fn, void *arg) {
    if (!fn) return;
    if (!pool || pool->num_threads <= 1) {
        fn(arg, 0, 1);
        return;
    }

    pool->fn = fn;
    pool->arg = arg;
    atomic_store(&pool->pending, pool->num_threads - 1);
    atomic_fetch_add(&pool->generation, 1);

    if (atomic_load(&pool->sleeping) > 0) {
        pthread_mutex_lock(&pool->mutex);
        pthread_cond_broadcast(&pool->cond);
        pthread_mutex_unlock(&pool->mutex);
    }

    fn(arg, 0, pool->num_threads);

    while (atomic_load_explicit(&pool->pending, memory_order_acquire) > 0) {
        cpu_relax();
    }
}

/* ---- parallel_for with dynamic chunked scheduling ---- */

typedef struct {
    gemma3_range_fn fn;
    void *arg;
    int n;
    int chunk;
    atomic_int next;
} pfor_task;

static void pfor_worker(void *a, int thread_idx, int num_threads) {
    (void)thread_idx;
    (void)num_threads;
    pfor_task *t = (pfor_task *)a;
    for (;;) {
        int start = atomic_fetch_add_explicit(&t->next, t->chunk, memory_order_relaxed);
        if (start >= t->n) break;
        int end = start + t->chunk;
        if (end > t->n) end = t->n;
        t->fn(t->arg, start, end);
    }
}

void gemma3_parallel_for(gemma3_thread_pool *pool, int n, int chunk,
                         gemma3_range_fn fn, void *arg) {
    if (n <= 0 || !fn) return;
    if (chunk < 1) chunk = 1;
    if (!pool || pool->num_threads <= 1 || n <= chunk) {
        fn(arg, 0, n);
        return;
    }
    pfor_task t;
    t.fn = fn;
    t.arg = arg;
    t.n = n;
    t.chunk = chunk;
    atomic_init(&t.next, 0);
    gemma3_thread_pool_run(pool, pfor_worker, &t);
}
