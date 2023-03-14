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

/*
Author : Minsung Kim
Contact : alstjd025@gmail.com


This is a worker api source for worker in tensorflow lite.

/////////////////////////////////////////////////////////
/                        WorkerAPI                      /
/                                                       /
/  ///////////////      ///////////////                 /
/  /   [Worker]  /      /   [Worker]  /                 /
/  /     CPU1    /      /  CPU2  GPU1 /                 /
/  / interperter /      / interperter /     .  .  .     /     
/  ///////////////      / interpreter /                 /
/                       ///////////////                 /
/                                                       /
/////////////////////////////////////////////////////////


*/

namespace tflite{
class Worker; // forward declare of Worker class

class WorkerAPI{
  public:
    WorkerAPI();
    WorkerAPI(const char* model_name);
    ~WorkerAPI();

    // Create worker of a given worker type
    TfLiteStatus CreateWorker(workerType wType);
  
    // Allocates a task to specific worker
    TfLiteStatus AllocateTask();

    // vector container of worker_ids
    std::vector<int> worker_ids;

    // vector container of workers
    std::vector<Worker*> workers;

    // tflite InterpreterBuilder container
    std::vector<tflite::InterpreterBuilder*> interpreterbuilders;
    

};



} //namespace tflite
