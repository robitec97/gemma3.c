/*
 * gemma3_threads.h - Thread pool for parallel computation (POSIX threads)
 */

#ifndef GEMMA3_THREADS_H
#define GEMMA3_THREADS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gemma3_thread_pool gemma3_thread_pool;

/* Create a thread pool with the given total number of threads (the calling
 * thread counts as one of them and takes part in every run).
 * If num_threads <= 0, uses $GEMMA3_THREADS or the number of CPU cores.
 * Returns NULL on failure. */
gemma3_thread_pool *gemma3_thread_pool_create(int num_threads);

/* Destroy the thread pool, joining all worker threads. */
void gemma3_thread_pool_destroy(gemma3_thread_pool *pool);

/* Get the number of threads that execute each task (including the caller). */
int gemma3_thread_pool_size(const gemma3_thread_pool *pool);

/* Task function type: called with (task_arg, thread_index, num_threads) */
typedef void (*gemma3_task_fn)(void *arg, int thread_idx, int num_threads);

/* Run fn once on every thread (thread_idx 0 is the caller) and wait for all
 * of them to finish. Workers spin briefly before sleeping so back-to-back
 * runs (hundreds per token) have microsecond-level dispatch latency. */
void gemma3_thread_pool_run(gemma3_thread_pool *pool, gemma3_task_fn fn, void *arg);

/* Range task: process items [start, end). */
typedef void (*gemma3_range_fn)(void *arg, int start, int end);

/* Parallel for-loop over [0, n) with dynamic scheduling in chunks of
 * `chunk` items, so faster cores (e.g. Apple P-cores) take more work.
 * Runs inline when pool is NULL or n is small. */
void gemma3_parallel_for(gemma3_thread_pool *pool, int n, int chunk,
                         gemma3_range_fn fn, void *arg);

/* Number of online CPU cores (>= 1). */
int gemma3_num_cpus(void);

#ifdef __cplusplus
}
#endif

#endif /* GEMMA3_THREADS_H */
