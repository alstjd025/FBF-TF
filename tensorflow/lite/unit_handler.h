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
#include "tensorflow/lite/unit.h"

/*
Unit handler class
test
 */

namespace tflite
{


class UnitHandler
{
private:
    ///  Mutex Lock For Queue
    std::mutex mtx_lock; 

    /// Mutex Lock For Queue empty check
    std::mutex mtx_lock_;

    /// Mutex Lock For GPU&CPU Timing Control
    std::mutex mtx_lock_timing;

    /// Condition Variable to control Units in Subgraph::Invoke
    std::condition_variable Ucontroller;

    /// Condition Variable to control Units in Unit::Invoke
    std::condition_variable Outcontroller;

    /// Contains every Units
    std::vector<Unit*> vUnitContainer;

    /// Queue for Tensor Sharing Between Units (this case CPU & GPU)
    std::queue<SharedContext*>* qSharedData;

    /// Pointer of InterpreterBuilder (Single Object)
    tflite::InterpreterBuilder* builder_;

    /// Pointer for CPU InterpreterBuilder (Quantized Model)
    tflite::InterpreterBuilder* CPUBuilder_;
    
    /// Pointer for GPU InterpreterBuilder (Original Model)
    tflite::InterpreterBuilder* GPUBuilder_;
    
    bool bUseTwoModel;
    int iUnitCount; 
    int numThreads;
    std::vector<cv::Mat> inputData;
    const char* fileNameOriginal;
    const char* fileNameQuantized;
    std::vector<std::function<void>*> workers;
    int C_Counter = 0;
    int G_Counter = 0;
    UnitSettings setting;

public:
    UnitHandler();
    UnitHandler(const char* filename);
    UnitHandler(const char* OriginalModel, const char* QuantizedModel);

    TfLiteStatus CreateUnitCPU(UnitType eType, std::vector<cv::Mat> input, int partitioning);
    TfLiteStatus CreateUnitGPU(UnitType eType, std::vector<cv::Mat> input, int partitioning);
    TfLiteStatus Invoke(UnitType eType, UnitType eType_, std::vector<cv::Mat> input);

    TfLiteStatus CreateAndInvokeCPU(UnitType eType, std::vector<cv::Mat> input);
    TfLiteStatus CreateAndInvokeGPU(UnitType eType, std::vector<cv::Mat> input);

    void DeleteSharedContext(SharedContext* dataTobeCleared);

    void PrintInterpreterStatus();
    void PrintMsg(const char* msg);

    ~UnitHandler() {};
    //tflite::Interpreter* GetInterpreter();
};

} //End of Namespace tflite

