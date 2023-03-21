#pragma once
#include <ctime>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdarg>
#include <vector>
#include <utility>
#include <queue>
#include "condition_variable"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/c/common.h"

#include <functional>
#include "thread"
#include "future"

namespace tflite{


class Scheduler
{
  // TODO : Consider a better relationship
  friend class ModelFactory;

  public:
    Scheduler();
    Scheduler(std::shared_ptr<tflite::Interpreter> interpreter);
    ~Scheduler();

    void SchedulerSpin(); 
    void notify();

    // Check schedulability
    bool CheckSchedulability();

    void NeedReschedule();

    // Reschedule whole jobs
    TfLiteStatus Reschedule();

    // Ready workers that needed for scheduled jobs.
    // Clean up the existing workers and create needed ones.
    TfLiteStatus ReadyWorkers();

    TfLiteStatus DoInvoke();

    bool need_reschedule = false;

    SchedulerStatus state;
    std::shared_ptr<tflite::Interpreter> interpreter_;  
    std::shared_ptr<std::vector<tflite::Job*>> jobs_enqueued;

    std::thread scheduler_thread;
    std::mutex scheduler_lock;
    std::condition_variable scheduler_cv_;
};


// ModelFactory doesn't own a scheduler.
class ModelFactory 
{
  public:
    ModelFactory();
    ModelFactory(std::shared_ptr<tflite::Interpreter> interpreter);
    ~ModelFactory();

  void ModelFactorySpin();

  // Creates an invokable context and profile it by invoking several times.
  // Uses layer, subgraph partitioning for optimized scheduling and utilization.
  TfLiteStatus CreateAndProfileModel(const char* model);
  TfLiteStatus GiveSubgraphtoInterperter(std::shared_ptr<Scheduler> scheduler);

  TfLiteStatus GiveModel(const char* model);

  // Map container for interpreterBuilders and model_id
  // key : model_id, value : interpreterbuilder
  std::map<int, tflite::InterpreterBuilder*> builder_and_id;

  std::shared_ptr<tflite::Interpreter> interpreter_; 

  std::thread modelfactory_thread;

};


} //namespace tflite

