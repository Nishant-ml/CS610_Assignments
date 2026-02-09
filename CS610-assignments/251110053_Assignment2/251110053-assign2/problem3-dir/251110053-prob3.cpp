#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <pthread.h>
#include <x86intrin.h>
#include <atomic>
using std::cerr;
using std::cout;
using std::endl;

using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::milliseconds;

#define N (1000)
#define CACHELINE 64
#define NUM_THREADS 64

// Shared variables
uint64_t var1 = 0, var2 = (N * NUM_THREADS + 1);

// Abstract base class
class LockBase {
public:
  virtual void acquire(uint16_t tid) = 0;
  virtual void release(uint16_t tid) = 0;
  virtual ~LockBase() {}
};

typedef struct thr_args {
  uint16_t m_id;
  LockBase* m_lock;
} ThreadArgs;

/** Pthread mutex */
class PthreadMutex : public LockBase {
public:
  void acquire(uint16_t tid) override { pthread_mutex_lock(&lock); }
  void release(uint16_t tid) override { pthread_mutex_unlock(&lock); }
private:
  pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
};

/** SpinLock */
class SpinLock : public LockBase {
  alignas(CACHELINE) volatile uint64_t value = 0;
public:
  void acquire(uint16_t tid) override {
    __asm__ __volatile__("" ::: "memory");
    __asm__ __volatile__("mfence" ::: "memory");
    uint64_t expected, desired;
    do {
      expected = 0; desired = 1;
      __asm__ volatile("lock cmpxchgq %2, %1"
                       : "=a"(expected), "+m"(value)
                       : "r"(desired), "0"(expected)
                       : "memory");
    } while (expected != 0);
  }
  void release(uint16_t tid) override {
    __asm__ __volatile__("" ::: "memory");
    __asm__ __volatile__("mfence" ::: "memory");
    uint64_t zero = 0;
    __asm__ volatile("xchgq %0, %1" : "+r"(zero), "+m"(value) :: "memory");
  }
};

/** TicketLock  */
class TicketLock : public LockBase {
  alignas(CACHELINE) volatile uint64_t next_ticket = 0;
  alignas(CACHELINE) volatile uint64_t now_serving = 0;
public:
  void acquire(uint16_t tid) override {
    __asm__ __volatile__("" ::: "memory");
    __asm__ __volatile__("mfence" ::: "memory");
    uint64_t my_ticket;
    uint64_t one = 1;
    __asm__ volatile("lock xaddq %0, %1"
                     : "=r"(my_ticket), "+m"(next_ticket)
                     : "0"(one)
                     : "memory");
    while (true) {
      if (now_serving == my_ticket) break;
      _mm_pause();
    }
  }
  void release(uint16_t tid) override {
    __asm__ __volatile__("" ::: "memory");
    __asm__ __volatile__("mfence" ::: "memory");
    __asm__ volatile("lock incq %0" : "+m"(now_serving) :: "memory");
  }
};

class BakeryLock : public LockBase {
  uint16_t n;
  volatile uint64_t* choosing;
  volatile uint64_t* ticket;
public:
  BakeryLock(uint16_t num_threads = NUM_THREADS) {
    n = num_threads;
    choosing = new uint64_t[n]();
    ticket   = new uint64_t[n]();
  }
  ~BakeryLock() {
    delete[] choosing;
    delete[] ticket;
  }

  void acquire(uint16_t id) override {
    choosing[id] = 1;
    __asm__ __volatile__("mfence" ::: "memory");

    uint64_t maxnum = 0;
    for (uint16_t i = 0; i < n; i++) {
      uint64_t t = ticket[i];   // atomic on x86_64 if naturally aligned
      if (t > maxnum) maxnum = t;
    }
    ticket[id] = maxnum + 1;

    __asm__ __volatile__("mfence" ::: "memory");
    choosing[id] = 0;
    __asm__ __volatile__("mfence" ::: "memory");

    for (uint16_t j = 0; j < n; j++) {
      if (j == id) continue;

      while (choosing[j]) {
        _mm_pause();
        __asm__ __volatile__("" ::: "memory");
      }

      while (ticket[j] != 0 &&
            (ticket[j] < ticket[id] ||
             (ticket[j] == ticket[id] && j < id))) {
        _mm_pause();
        __asm__ __volatile__("" ::: "memory");
      }
    }
  }

  void release(uint16_t id) override {
    ticket[id] = 0;
    __asm__ __volatile__("mfence" ::: "memory");
  }
};


/** FilterLock */
class FilterLock : public LockBase {
  uint16_t n;
  volatile uint16_t* level;
  volatile uint16_t* victim;
public:
  FilterLock(uint16_t num_threads = NUM_THREADS) {
    n = num_threads;
    level = new uint16_t[n]();
    victim = new uint16_t[n]();
  }
  ~FilterLock() {
    delete[] level;
    delete[] victim;
  }
  void acquire(uint16_t id) override {
    for (uint16_t L = 1; L < n; L++) {
      level[id] = L;
          __asm__ __volatile__("" ::: "memory");
          __asm__ __volatile__("mfence" ::: "memory");
      victim[L] = id;
          __asm__ __volatile__("" ::: "memory");
          __asm__ __volatile__("mfence" ::: "memory");
      for (uint16_t k = 0; k < n; k++) {
        if (k == id) continue;
        while (level[k] >= L && victim[L] == id) {
          _mm_pause(); 
          __asm__ __volatile__("" ::: "memory");
          __asm__ __volatile__("mfence" ::: "memory");
          
        }
      }
    }
  }
  void release(uint16_t id) override {
    __asm__ __volatile__("" ::: "memory");
    __asm__ __volatile__("mfence" ::: "memory");
    level[id] = 0;
  }
};

/** Anderson Array Queue Lock */
class ArrayQLock : public LockBase {
  static const uint16_t QSIZE = 1024;
  volatile bool* flags;
  volatile uint64_t tail;
  thread_local static uint16_t myslot;
public:
  ArrayQLock() {
    flags = new bool[QSIZE]();
    flags[0] = true;
    tail = 0;
  }
  ~ArrayQLock() { delete[] flags; }

  void acquire(uint16_t tid) override {
    __asm__ __volatile__("" ::: "memory");
    __asm__ __volatile__("mfence" ::: "memory");
    uint64_t slot;
    uint64_t one = 1;
    __asm__ volatile("lock xaddq %0, %1"
                     : "=r"(slot), "+m"(tail)
                     : "0"(one)
                     : "memory");
    myslot = slot % QSIZE;
    while (!flags[myslot]) { _mm_pause(); }
  }

  void release(uint16_t tid) override {
    __asm__ __volatile__("" ::: "memory");
    __asm__ __volatile__("mfence" ::: "memory");
    flags[myslot] = false;
    flags[(myslot + 1) % QSIZE] = true;
  }
};
thread_local uint16_t ArrayQLock::myslot = 0;

/** Benchmark infra */
inline void critical_section() {
  var1++;
  var2--;
}

std::atomic_uint64_t sync_time = 0;
pthread_barrier_t g_barrier;

void* thrBody(void* arguments) {
  ThreadArgs* tmp = static_cast<ThreadArgs*>(arguments);
  pthread_barrier_wait(&g_barrier);

  HRTimer start = HR::now();
  for (uint64_t i = 0; i < N; i++) {
    tmp->m_lock->acquire(tmp->m_id);
    critical_section();
    tmp->m_lock->release(tmp->m_id);
  }
  HRTimer end = HR::now();
  auto duration = duration_cast<milliseconds>(end - start).count();
  sync_time.fetch_add(duration);
  pthread_exit(NULL);
}

int main() {
   int error = pthread_barrier_init(&g_barrier, NULL, NUM_THREADS);
   if (error != 0) {
    cerr << "Error in barrier init.\n";
    exit(EXIT_FAILURE);
  }

  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  pthread_t tid[64];
  ThreadArgs args[64] = {{0}};

    cout<<"For Threads="<<NUM_THREADS<<endl;

    // Pthread mutex
    LockBase* lock_obj = new PthreadMutex();
    uint16_t i = 0;
    while (i < NUM_THREADS) {
      args[i].m_id = i;
      args[i].m_lock = lock_obj;

      error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
      if (error != 0) {
        cerr << "\nThread cannot be created : " << strerror(error) << "\n";
        exit(EXIT_FAILURE);
      }
      i++;
    }

    i = 0;
    void* status;
    while (i < NUM_THREADS) {
      error = pthread_join(tid[i], &status);
      if (error) {
        cerr << "ERROR: return code from pthread_join() is " << error << "\n";
        exit(EXIT_FAILURE);
      }
      i++;
    }

    assert(var1 == N * NUM_THREADS && var2 == 1);
    // cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
    cout << "Pthread mutex: Time taken (us): " << sync_time << "\n";

    // Filter lock
    var1 = 0;
    var2 = (N * NUM_THREADS + 1);
    sync_time = 0;

    lock_obj = new FilterLock();
    i = 0;
    while (i < NUM_THREADS) {
      args[i].m_id = i;
      args[i].m_lock = lock_obj;

      error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
      if (error != 0) {
        printf("\nThread cannot be created : [%s]", strerror(error));
        exit(EXIT_FAILURE);
      }
      i++;
    }

    i = 0;
    while (i < NUM_THREADS) {
      error = pthread_join(tid[i], &status);
      if (error) {
        printf("ERROR: return code from pthread_join() is %d\n", error);
        exit(EXIT_FAILURE);
      }
      i++;
    }

    cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
    assert(var1 == N * NUM_THREADS && var2 == 1);
    cout << "Filter lock: Time taken (us): " << sync_time << "\n";

    // Bakery lock
    var1 = 0;
    var2 = (N * NUM_THREADS + 1);
    sync_time = 0;

    lock_obj = new BakeryLock();
    i = 0;
    while (i < NUM_THREADS) {
      args[i].m_id = i;
      args[i].m_lock = lock_obj;

      error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
      if (error != 0) {
        printf("\nThread cannot be created : [%s]", strerror(error));
        exit(EXIT_FAILURE);
      }
      i++;
    }

    i = 0;
    while (i < NUM_THREADS) {
      error = pthread_join(tid[i], &status);
      if (error) {
        printf("ERROR: return code from pthread_join() is %d\n", error);
        exit(EXIT_FAILURE);
      }
      i++;
    }

    cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
    assert(var1 == N * NUM_THREADS && var2 == 1);
    cout << "Bakery lock: Time taken (us): " << sync_time << "\n";

    // Spin lock
    var1 = 0;
    var2 = (N * NUM_THREADS + 1);
    sync_time = 0;

    lock_obj = new SpinLock();
    i = 0;
    while (i < NUM_THREADS) {
      args[i].m_id = i;
      args[i].m_lock = lock_obj;

      error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
      if (error != 0) {
        printf("\nThread cannot be created : [%s]", strerror(error));
        exit(EXIT_FAILURE);
      }
      i++;
    }

    i = 0;
    while (i < NUM_THREADS) {
      error = pthread_join(tid[i], &status);
      if (error) {
        printf("ERROR: return code from pthread_join() is %d\n", error);
        exit(EXIT_FAILURE);
      }
      i++;
    }

    cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
    assert(var1 == N * NUM_THREADS && var2 == 1);
    cout << "Spin lock: Time taken (us): " << sync_time << "\n";

    // Ticket lock
    var1 = 0;
    var2 = (N * NUM_THREADS + 1);
    sync_time = 0;

    lock_obj = new TicketLock();
    i = 0;
    while (i < NUM_THREADS) {
      args[i].m_id = i;
      args[i].m_lock = lock_obj;

      error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
      if (error != 0) {
        printf("\nThread cannot be created : [%s]", strerror(error));
        exit(EXIT_FAILURE);
      }
      i++;
    }

    i = 0;
    while (i < NUM_THREADS) {
      error = pthread_join(tid[i], &status);
      if (error) {
        printf("ERROR: return code from pthread_join() is %d\n", error);
        exit(EXIT_FAILURE);
      }
      i++;
    }

    cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
    assert(var1 == N * NUM_THREADS && var2 == 1);
    cout << "Ticket lock: Time taken (us): " << sync_time << "\n";

    // Array Q lock
    var1 = 0;
    var2 = (N * NUM_THREADS + 1);
    sync_time = 0;

    lock_obj = new ArrayQLock();
    i = 0;
    while (i < NUM_THREADS) {
      args[i].m_id = i;
      args[i].m_lock = lock_obj;

      error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
      if (error != 0) {
        printf("\nThread cannot be created : [%s]", strerror(error));
        exit(EXIT_FAILURE);
      }
      i++;
    }

    i = 0;
    while (i < NUM_THREADS) {
      error = pthread_join(tid[i], &status);
      if (error) {
        printf("ERROR: return code from pthread_join() is %d\n", error);
        exit(EXIT_FAILURE);
      }
      i++;
    }

    cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
    assert(var1 == N * NUM_THREADS && var2 == 1);
    cout << "Array Q lock: Time taken (us): " << sync_time << "\n";


    pthread_barrier_destroy(&g_barrier);
    pthread_attr_destroy(&attr);
 
    pthread_exit(NULL);
}
