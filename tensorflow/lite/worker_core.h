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
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/core/subgraph.h"
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

class Interpreter; // pre-declairation of interpreter

class Worker
{
  public:
    Worker();
    Worker(WorkerType wType, int w_id, Interpreter* interpreter);

    // Give a worker new Job
    void GiveJob(tflite::Job* new_job);

    void Work();

    bool JobTryLock();
    void JobUnlock();

    Worker* returnThis() {return this;}
    
    ~Worker();

    // worker id
    int worker_id;
    WorkerType type;

    // threads, mutex and condition variables for synchronization and multi-
    // threading
    std::mutex mtx_lock;
    std::condition_variable worker_cv;
    std::thread working_thread;

    // queue for data sharing
    std::queue<sharedcontext*>* share_queue;

    // interpre†er
    // CAREFULLY USE
    // NEEDS BETTER IMPLEMENTATION
    Interpreter* interpreter_;

};

} //namspace tflite
