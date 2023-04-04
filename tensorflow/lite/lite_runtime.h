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
#include "opencv2/opencv.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/c/common.h"
#include <functional>
#include "thread"
#include "future"


/*
Author : Minsung Kim

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

    void FeedInputToModel(const char* model, std::vector<cv::Mat>& input);

    void WakeScheduler();

  private:
    std::shared_ptr<tflite::Interpreter> interpreter;
    LiteScheduler* scheduler;

    // Map container for interpreterBuilders and model_id
    // key : model_id, value : interpreterbuilder
    std::map<int, tflite::InterpreterBuilder*> builder_and_id;
    int builders_created = 0;

};

} // namespace tflite