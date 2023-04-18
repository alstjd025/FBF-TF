#pragma once
#include <ctime>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdarg>
#include <vector>
#include <utility>
#include <queue>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <pack.h>
#include <unistd.h>
#include <functional>

#include "condition_variable"
#include "opencv2/opencv.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/c/common.h"
#include "thread"
#include "future"

/*
Author : Minsung Kim
This class is re-writed for IPC with scheduler and 1:1 relationship for interpreter
and interpreterbuilder. (commit b56faa4981)

*/


namespace tflite{

class LiteScheduler;

class TfLiteRuntime{
  public:
    TfLiteRuntime();
    ~TfLiteRuntime();

    // Creates Interpreter
    TfLiteStatus CreateTfLiteRuntime();

    TfLiteStatus AddModelToRuntime(const char* new_model);

    TfLiteStatus DebugInvoke();

    void FeedInputToModel(const char* model, std::vector<cv::Mat>& input,
                          INPUT_TYPE input_type);
    
    // For debugging only
    void FeedInputToInterpreter(std::vector<cv::Mat>& mnist, 
                                  std::vector<cv::Mat>& imagetnet);

    void WakeScheduler();
    void JoinScheduler();

  private:
    tflite::Interpreter* interpreter;
    tflite::InterpreterBuilder* interpreter_builder;
    LiteScheduler* scheduler;
};

} // namespace tflite