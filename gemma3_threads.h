/*
 * gemma3_threads.h - Cross-platform thread pool for parallel computation
 */

#ifndef GEMMA3_THREADS_H
#define GEMMA3_THREADS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gemma3_thread_pool gemma3_thread_pool;

/* Create a thread pool with the given number of worker threads.
 * If num_threads <= 0, uses the number of available CPU cores.
 * Returns NULL on failure. */
gemma3_thread_pool *gemma3_thread_pool_create(int num_threads);

/* Destroy the thread pool, joining all worker threads. */
void gemma3_thread_pool_destroy(gemma3_thread_pool *pool);

/* Get the number of worker threads in the pool. */
int gemma3_thread_pool_size(const gemma3_thread_pool *pool);

/* Task function type: called with (task_arg, thread_index, num_threads) */
typedef void (*gemma3_task_fn)(void *arg, int thread_idx, int num_threads);

/* Submit a parallel-for task and wait for all threads to complete.
 * The function fn is called once on each worker thread. */
void gemma3_thread_pool_run(gemma3_thread_pool *pool, gemma3_task_fn fn, void *arg);

#ifdef __cplusplus
}
#endif

#endif /* GEMMA3_THREADS_H */
