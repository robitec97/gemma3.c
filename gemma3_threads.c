/*
 * gemma3_threads.c - Cross-platform thread pool implementation
 *
 * Uses Win32 threads on Windows, pthreads elsewhere.
 */

#include "gemma3_threads.h"
#include <stdlib.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

struct gemma3_thread_pool {
    int num_threads;
    HANDLE *threads;
    /* Synchronization: barrier-style using events */
    HANDLE start_event;    /* Manual-reset event: signals workers to start */
    HANDLE *done_events;   /* Auto-reset events: each worker signals when done */
    volatile gemma3_task_fn current_fn;
    volatile void *current_arg;
    volatile int shutdown;
};

typedef struct {
    gemma3_thread_pool *pool;
    int thread_idx;
} worker_arg;

static DWORD WINAPI worker_func(LPVOID param) {
    worker_arg *wa = (worker_arg *)param;
    gemma3_thread_pool *pool = wa->pool;
    int idx = wa->thread_idx;
    free(wa);

    while (1) {
        WaitForSingleObject(pool->start_event, INFINITE);
        if (pool->shutdown) break;

        if (pool->current_fn) {
            pool->current_fn((void *)pool->current_arg, idx, pool->num_threads);
        }
        SetEvent(pool->done_events[idx]);
    }
    return 0;
}

gemma3_thread_pool *gemma3_thread_pool_create(int num_threads) {
    if (num_threads <= 0) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        num_threads = (int)si.dwNumberOfProcessors;
        if (num_threads < 1) num_threads = 1;
    }

    gemma3_thread_pool *pool = (gemma3_thread_pool *)calloc(1, sizeof(gemma3_thread_pool));
    if (!pool) return NULL;

    pool->num_threads = num_threads;
    pool->shutdown = 0;
    pool->current_fn = NULL;
    pool->current_arg = NULL;

    pool->start_event = CreateEvent(NULL, TRUE, FALSE, NULL); /* Manual reset */
    pool->threads = (HANDLE *)calloc(num_threads, sizeof(HANDLE));
    pool->done_events = (HANDLE *)calloc(num_threads, sizeof(HANDLE));
    if (!pool->start_event || !pool->threads || !pool->done_events) {
        gemma3_thread_pool_destroy(pool);
        return NULL;
    }

    for (int i = 0; i < num_threads; i++) {
        pool->done_events[i] = CreateEvent(NULL, FALSE, FALSE, NULL); /* Auto reset */
        worker_arg *wa = (worker_arg *)malloc(sizeof(worker_arg));
        if (!wa || !pool->done_events[i]) {
            free(wa);
            pool->num_threads = i; /* Only destroy created threads */
            gemma3_thread_pool_destroy(pool);
            return NULL;
        }
        wa->pool = pool;
        wa->thread_idx = i;
        pool->threads[i] = CreateThread(NULL, 0, worker_func, wa, 0, NULL);
        if (!pool->threads[i]) {
            free(wa);
            pool->num_threads = i;
            gemma3_thread_pool_destroy(pool);
            return NULL;
        }
    }

    return pool;
}

void gemma3_thread_pool_destroy(gemma3_thread_pool *pool) {
    if (!pool) return;

    pool->shutdown = 1;
    if (pool->start_event) SetEvent(pool->start_event);

    if (pool->threads) {
        for (int i = 0; i < pool->num_threads; i++) {
            if (pool->threads[i]) {
                WaitForSingleObject(pool->threads[i], INFINITE);
                CloseHandle(pool->threads[i]);
            }
        }
        free(pool->threads);
    }
    if (pool->done_events) {
        for (int i = 0; i < pool->num_threads; i++) {
            if (pool->done_events[i]) CloseHandle(pool->done_events[i]);
        }
        free(pool->done_events);
    }
    if (pool->start_event) CloseHandle(pool->start_event);
    free(pool);
}

int gemma3_thread_pool_size(const gemma3_thread_pool *pool) {
    return pool ? pool->num_threads : 0;
}

void gemma3_thread_pool_run(gemma3_thread_pool *pool, gemma3_task_fn fn, void *arg) {
    if (!pool || !fn) return;

    pool->current_fn = fn;
    pool->current_arg = arg;

    /* Reset done events, then signal all workers */
    ResetEvent(pool->start_event);
    for (int i = 0; i < pool->num_threads; i++) {
        ResetEvent(pool->done_events[i]);
    }
    SetEvent(pool->start_event);

    /* Wait for all workers to finish */
    WaitForMultipleObjects(pool->num_threads, pool->done_events, TRUE, INFINITE);

    /* Reset start event so workers block again */
    ResetEvent(pool->start_event);
    pool->current_fn = NULL;
}

#else /* POSIX */

#include <pthread.h>
#ifdef __linux__
#include <unistd.h>
#endif
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

struct gemma3_thread_pool {
    int num_threads;
    pthread_t *threads;
    pthread_mutex_t mutex;
    pthread_cond_t cond_start;
    pthread_cond_t cond_done;
    volatile gemma3_task_fn current_fn;
    volatile void *current_arg;
    volatile int tasks_remaining;
    volatile int generation;  /* Incremented each run to avoid spurious wakeups */
    volatile int shutdown;
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

    int last_gen = 0;
    while (1) {
        pthread_mutex_lock(&pool->mutex);
        while (pool->generation == last_gen && !pool->shutdown) {
            pthread_cond_wait(&pool->cond_start, &pool->mutex);
        }
        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }
        last_gen = pool->generation;
        gemma3_task_fn fn = pool->current_fn;
        void *arg = (void *)pool->current_arg;
        int nt = pool->num_threads;
        pthread_mutex_unlock(&pool->mutex);

        if (fn) {
            fn(arg, idx, nt);
        }

        pthread_mutex_lock(&pool->mutex);
        pool->tasks_remaining--;
        if (pool->tasks_remaining == 0) {
            pthread_cond_signal(&pool->cond_done);
        }
        pthread_mutex_unlock(&pool->mutex);
    }
    return NULL;
}

static int get_num_cpus(void) {
#ifdef __linux__
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 1;
#elif defined(__APPLE__)
    int n;
    size_t len = sizeof(n);
    if (sysctlbyname("hw.ncpu", &n, &len, NULL, 0) == 0 && n > 0) return n;
    return 1;
#else
    return 1;
#endif
}

gemma3_thread_pool *gemma3_thread_pool_create(int num_threads) {
    if (num_threads <= 0) {
        num_threads = get_num_cpus();
    }

    gemma3_thread_pool *pool = (gemma3_thread_pool *)calloc(1, sizeof(gemma3_thread_pool));
    if (!pool) return NULL;

    pool->num_threads = num_threads;
    pool->shutdown = 0;
    pool->generation = 0;
    pool->tasks_remaining = 0;
    pool->current_fn = NULL;
    pool->current_arg = NULL;

    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->cond_start, NULL);
    pthread_cond_init(&pool->cond_done, NULL);

    pool->threads = (pthread_t *)calloc(num_threads, sizeof(pthread_t));
    if (!pool->threads) {
        gemma3_thread_pool_destroy(pool);
        return NULL;
    }

    for (int i = 0; i < num_threads; i++) {
        worker_arg *wa = (worker_arg *)malloc(sizeof(worker_arg));
        if (!wa) {
            pool->num_threads = i;
            gemma3_thread_pool_destroy(pool);
            return NULL;
        }
        wa->pool = pool;
        wa->thread_idx = i;
        if (pthread_create(&pool->threads[i], NULL, worker_func, wa) != 0) {
            free(wa);
            pool->num_threads = i;
            gemma3_thread_pool_destroy(pool);
            return NULL;
        }
    }

    return pool;
}

void gemma3_thread_pool_destroy(gemma3_thread_pool *pool) {
    if (!pool) return;

    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->cond_start);
    pthread_mutex_unlock(&pool->mutex);

    if (pool->threads) {
        for (int i = 0; i < pool->num_threads; i++) {
            pthread_join(pool->threads[i], NULL);
        }
        free(pool->threads);
    }

    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->cond_start);
    pthread_cond_destroy(&pool->cond_done);
    free(pool);
}

int gemma3_thread_pool_size(const gemma3_thread_pool *pool) {
    return pool ? pool->num_threads : 0;
}

void gemma3_thread_pool_run(gemma3_thread_pool *pool, gemma3_task_fn fn, void *arg) {
    if (!pool || !fn) return;

    pthread_mutex_lock(&pool->mutex);
    pool->current_fn = fn;
    pool->current_arg = arg;
    pool->tasks_remaining = pool->num_threads;
    pool->generation++;
    pthread_cond_broadcast(&pool->cond_start);

    /* Wait for all workers to finish */
    while (pool->tasks_remaining > 0) {
        pthread_cond_wait(&pool->cond_done, &pool->mutex);
    }
    pool->current_fn = NULL;
    pthread_mutex_unlock(&pool->mutex);
}

#endif /* _WIN32 / POSIX */
