// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef LIB_JXL_ENC_LOSSLESS_PROBE_POOL_H_
#define LIB_JXL_ENC_LOSSLESS_PROBE_POOL_H_

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "lib/jxl/base/data_parallel.h"

namespace jxl {

// One invocation of the caller's runner owns all execution. Independent probe
// encoders may fork work into this arena, without recursively entering that
// runner or creating extra OS threads. Waiting callbacks help only their own
// child group: helping an ancestor's sibling could overwrite thread scratch
// retained by the suspended callback.
class LosslessProbePool {
 public:
  template <typename Task>
  Status Run(size_t count, ThreadPool* caller, size_t workers,
             size_t max_encoders, const Task& task) {
    JXL_ENSURE(workers > 0 && max_encoders > 0);
    const char* quiet = std::getenv("JXL_LOSSLESS_ARENA_QUIET");
    quiet_ = quiet && std::strcmp(quiet, "1") == 0;
    size_t next = 0, active = 0;
    bool success = true;
    const auto init = [&](size_t threads) -> Status {
      threads_ = threads;
      return true;
    };
    const auto work = [&](uint32_t, size_t thread) -> Status {
      const Local previous = local_;
      local_ = {this, thread};
      for (;;) {
        Job job;
        size_t probe = count;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          for (;;) {
            job = TakeAny();
            if (job.group) break;
            if (success && next < count && active < max_encoders) {
              probe = next++;
              ++active;
              break;
            }
            if (active == 0 && (!success || next == count)) {
              local_ = previous;
              return true;
            }
            changed_.wait(lock);
          }
        }
        if (job.group) {
          Execute(job);
        } else {
          ThreadPool nested(Runner, this);
          const bool ok = !!task(probe, &nested);
          {
            std::lock_guard<std::mutex> lock(mutex_);
            success &= ok;
            --active;
          }
          changed_.notify_all();
        }
      }
    };
    JXL_RETURN_IF_ERROR(RunOnPool(caller, 0, workers, init, work,
                                  "LosslessCooperativeProbes"));
    JXL_ENSURE(success);
    return true;
  }

 private:
  struct Group {
    void* opaque;
    JxlParallelRunFunction function;
    uint32_t next, end;
    size_t remaining;
    uint32_t quantum;
  };
  struct Job {
    Group* group = nullptr;
    uint32_t begin = 0, end = 0;
  };
  struct Local { LosslessProbePool* owner; size_t thread; };
  inline static thread_local Local local_{nullptr, 0};

  Job Take(Group& group) {
    if (group.next == group.end) return {};
    const uint32_t begin = group.next;
    group.next += std::min(group.quantum, group.end - begin);
    return {&group, begin, group.next};
  }
  Job TakeAny() {
    // Descendants are published last. Prefer them to keep suspended encoder
    // states short-lived, rather than accumulating more root work.
    for (auto it = groups_.rbegin(); it != groups_.rend(); ++it) {
      const Job job = Take(**it);
      if (job.group) return job;
    }
    return {};
  }
  void Execute(Job job) {
    for (uint32_t i = job.begin; i < job.end; ++i)
      job.group->function(job.group->opaque, i, local_.thread);
    bool complete;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      job.group->remaining -= job.end - job.begin;
      complete = job.group->remaining == 0;
    }
    // A waiting parent has already exhausted its queue. Only its final child
    // can unblock it; new queues and root completions publish separately.
    if (!quiet_ || complete) changed_.notify_all();
  }
  static JxlParallelRetCode Runner(void* opaque, void* context,
                                  JxlParallelRunInit init,
                                  JxlParallelRunFunction function,
                                  uint32_t begin, uint32_t end) {
    auto& self = *static_cast<LosslessProbePool*>(opaque);
    if (local_.owner != &self || begin > end)
      return JXL_PARALLEL_RET_RUNNER_ERROR;
    const auto result = init(context, self.threads_);
    if (result != JXL_PARALLEL_RET_SUCCESS) return result;
    if (self.quiet_ && end - begin <= 1) {
      if (begin != end) function(context, begin, local_.thread);
      return JXL_PARALLEL_RET_SUCCESS;
    }
    Group group{context, function, begin, end, size_t{end - begin},
                static_cast<uint32_t>(std::max<size_t>(1, (end - begin) / (4 * self.threads_)))};
    {
      std::lock_guard<std::mutex> lock(self.mutex_);
      self.groups_.push_back(&group);
    }
    self.changed_.notify_all();
    for (;;) {
      Job job;
      {
        std::unique_lock<std::mutex> lock(self.mutex_);
        if (!group.remaining) {
          const auto found = std::find(self.groups_.begin(), self.groups_.end(), &group);
          self.groups_.erase(found);
          break;
        }
        job = self.Take(group);
        if (!job.group) {
          self.changed_.wait(lock);
          continue;
        }
      }
      self.Execute(job);
    }
    return JXL_PARALLEL_RET_SUCCESS;
  }
  size_t threads_ = 1;
  bool quiet_ = false;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::vector<Group*> groups_;
};

}  // namespace jxl

#endif  // LIB_JXL_ENC_LOSSLESS_PROBE_POOL_H_
