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

This is a worker api source for woker in tensorflow lite.
*/

namespace tflite{
class Worker; // forward declare of Worker class

class WorkerAPI{
  public:
    WorkerAPI();
    ~WorkerAPI();

    // vector container of workers
    std::vector<Worker*> workers;


};



} //namespace tflite
