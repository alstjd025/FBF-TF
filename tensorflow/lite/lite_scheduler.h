#pragma once
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <queue>
#include <utility>
#include <vector>

#include "condition_variable"
#include "future"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/util.h"
#include "thread"

// WARNING! This class is deprecatied.
// WARNING! This class is deprecatied.
// WARNING! This class is deprecatied.

namespace tflite {

class Interpreter;  // forward declaire
class Subgraph;
class InterpreterBuilder;

class LiteScheduler {
 public:
  LiteScheduler();
  LiteScheduler(Interpreter* interpreter);
  ~LiteScheduler();

  void FlushAndEnqueueJobs();
  void Profile();
  TfLiteStatus RebuildSubgraphsAndJobs();
  void SchedulerSpin();
  void Wake();
  void Join();
  void NeedReschedule();
  TfLiteStatus RegisterInterpreterBuilder(InterpreterBuilder* builder);

  bool need_reschedule = false;
  bool stop_scheduler = false;  // if false, scheduler stops.
  SchedulerStatus state;

  Interpreter* interpreter_;
  std::vector<InterpreterBuilder*> builders;
  std::shared_ptr<std::vector<tflite::Job*>> jobs_enqueued;

  std::thread scheduler_thread;
  std::mutex scheduler_lock;
  std::condition_variable scheduler_cv_;
};

}  // namespace tflite
