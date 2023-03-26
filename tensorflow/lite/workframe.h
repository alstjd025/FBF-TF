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

#include "tensorflow/lite/worker_core.h"
#include "tensorflow/lite/scheduler.h"

/*
Author : Minsung Kim
Contact : alstjd025@gmail.com


This is a WorkFrame source for worker in tensorflow lite.

Below is basic structure of WorkFrame

/////////////////////////////////////////////////////////
/                       [WorkFrame]                     /
/  |--------Scheduler---------------------------------| /
/  |                                                  | /
/  | |-interpreter------------------------------------| /
/  | |     |-----------|     |-------------|          | /
/  | |     |  [Worker] |     |   [Worker]  |   . . .  | /
/  | |     |    CPU 1  |     |   CPU2 GPU1 |          | /
/  | |     |-----------|     |-------------|          | /
/  | |                                                | /
/  | |                                                | /
/  | |                                                | /
/  | |  JobQueue - [job1, job2, job3,, ]              | /
/  | |------------------------------------------------| /
/  |                                                  | /
/  |--------------------------------------------------| /
/                                                       /
/                                                       /
/  |--------ModelFactory------------------------------| /
/  |   interpreterbuilder                             | /
/  |   -> build from model (model name)               | /
/  |   -> profile model (maybe invoke?)               | /
/  |   -> make jobs (layer, subgraph partitioning)    | /
/  |   -> give jobs to scheduler                      | /
/  |--------------------------------------------------| /
/////////////////////////////////////////////////////////




*/

namespace tflite{
class Worker; // forward declare of Worker class

class WorkFrame{
  public:
    WorkFrame();

    // A WorkerAPI needs at least one model and one worker.

    
    ~WorkFrame();

    // Create worker of a given worker type and number of cpu
    // The number of cpu means how many cpus a single worker will use.
    TfLiteStatus CreateWorker(ResourceType wType, int cpu_num);
  
    // Allocates a task to specific worker
    TfLiteStatus AllocateTask();


    // Creates a profiler and a scheduler
    // Call this function at only initial phase.
    TfLiteStatus CreateProfilerandScheduler();

    // Gives a job to scheduler.
    // This task will profile a whole model with profiler
    // and partitions with layers and subgraphs.
    // A job is a subset of subgraphs.
    // TODO : consider quantized model input
    TfLiteStatus CreateAndGiveJob(const char* model);

    void JoinAll();

    // Tests invoke with a single job and a single worker
    // Subject to remove
    TfLiteStatus TestInvoke();

  private:
  
    // Scheduler for workers.
    // Scheduler owns the interpreter alone and workers can own its
    // weak ptr only.
    std::shared_ptr<Scheduler> scheduler_;

    // Profiler
    std::shared_ptr<ModelFactory> factory_;

    // Interpreter
    // An interpreter object is shared by scheduler and profiler and only
    // exists in single onject at runtime.
    // Initialized and shared to scheduler and profiler when 
    // CreateProfilerandScheduler() called.
    std::shared_ptr<tflite::Interpreter> interpreter;

    // mutex lock for interpreter // WARNING
    // Interpreter can only accessed by a single thread.
    // For now, we don't use this function since Proflier, Scheduler, workFrame works
    // in one way, not multi-threaded.
    std::mutex interpreter_lock;

    // condition variable for scheduler
    // notifies scheduler when a new job created and registered to interpreter
    std::condition_variable scheduler_cv;
};



} //namespace tflite
