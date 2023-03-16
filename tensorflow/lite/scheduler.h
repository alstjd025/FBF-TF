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

    TfLiteStatus Reschedule();
    SchedulerStatus state;
    std::shared_ptr<tflite::Interpreter> interpreter_;  
};


// ModelFactory doesn't own a scheduler.
class ModelFactory 
{
  public:
    ModelFactory();
    ModelFactory(std::shared_ptr<tflite::Interpreter> interpreter);
    ~ModelFactory();

  // Creates an invokable context and profile it by invoking several times.
  // Uses layer, subgraph partitioning for optimized scheduling and utilization.
  TfLiteStatus CreateAndProfileModel(const char* model);
  TfLiteStatus GiveSubgraphtoInterperter(std::shared_ptr<Scheduler> scheduler);

  TfLiteStatus GiveModel(const char* model);

  // Vector container for interpreterBuilders
  std::vector<tflite::InterpreterBuilder*> builders;
  std::shared_ptr<tflite::Interpreter> interpreter_; 

};


} //namespace tflite

