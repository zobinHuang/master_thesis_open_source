#pragma once

#include <sys/time.h>
#include <stdint.h>
#include <algorithm>
#include <chrono>

inline double get_time_s() {
  double tseconds = 0;
  struct timeval t;
  gettimeofday(&t, NULL);
  tseconds = (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
  return tseconds;
}

inline uint64_t get_time_us() {
    uint64_t t_us = 0;
    struct timeval t;
    gettimeofday(&t, NULL);
    t_us = (uint64_t)t.tv_sec * 1.0e6 + (uint64_t)t.tv_usec;
    return t_us;
}

inline uint64_t get_time_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>
              (std::chrono::steady_clock::now().time_since_epoch()).count();
}