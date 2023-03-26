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
    Worker(ResourceType wType, int w_id, Interpreter* interpreter);

    // Give a worker new Job
    void GiveJob(tflite::Job* new_job);

    void Work();
    
    void WakeWorker();

    void ChangeStateTo(WorkerState new_state);

    WorkerState returnState() { return state; }

    Worker* returnThis() {return this;}

    bool HaveJob() { return have_job; }
    
    ~Worker();

    // worker id
    int worker_id;

    // type of resource this worker uses
    ResourceType type;

    // current state of this worker
    WorkerState state;

    // worker exit flag
    bool stop_working = true;

    bool have_job = false;

    bool input_refresh = false;

    // threads, mutex and condition variables for synchronization and multi-
    // threading
    std::mutex worker_lock;
    std::condition_variable worker_cv;
    std::thread working_thread;

    // queue for data sharing
    std::queue<sharedcontext*>* share_queue;

    std::vector<tflite::Job*> jobs;

    // interpre†er
    // CAREFULLY USE
    // NEEDS BETTER IMPLEMENTATION
    Interpreter* interpreter_;

};

// Worker class for CPU-GPU co-execution
class CoWorker : public Worker
{
  public:
    CoWorker();
    ~CoWorker();
};

} //namspace tflite
