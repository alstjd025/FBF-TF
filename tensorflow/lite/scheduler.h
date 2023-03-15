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
  friend class Profiler;

  public:
    Scheduler();
    ~Scheduler();

    tflite::Interpreter* interpreter;  
};

class Profiler 
{
  public:
    Profiler();
    Profiler(const char* model, std::shared_ptr<Scheduler> scheduler);
    ~Profiler();

  // Creates a invokable context and profile it by invoking several times.
  // Uses layer, subgraph partitioning for optimized scheduling and utilization.
  TfLiteStatus CreateAndProfileModel(const char* model, std::shared_ptr<Scheduler> scheduler);
  TfLiteStatus GiveJobtoInterperter(std::shared_ptr<Scheduler> scheduler);

  tflite::InterpreterBuilder* builder;

};


} //namespace tflite

