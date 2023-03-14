#pragma once

#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdarg>
#include <string>
#include <queue>
#include <functional>
#include "condition_variable"
#include "opencv2/opencv.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/c/common.h"
#include "mutex"
#include "condition_variable"
#include "thread"
#include "future"

/*
Author : Minsung Kim
Contact : alstjd025@gmail.com

This is a worker core source for tensorflow lite.
*/

namespace tflite{

typedef struct job{
  
} job;


class Worker
{
  public:
    Worker();
    Worker(workerType wType, int w_id);
    
    ~Worker();

    // worker id
    int worker_id;
    workerType type;

    // threads, mutex and condition variables for synchronization and multi-
    // threading
    std::mutex mtx_lock;
    std::condition_variable cond_variable;
    std::thread working_thread;

    // queue for data sharing
    std::queue<sharedcontext*>* share_queue;
    
    // interpreter 
    std::unique_ptr<tflite::Interpreter>* interpreter;
    

};

} //namspace tflite
