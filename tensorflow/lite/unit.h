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
#include "thread"
#include "future"


#define Image_x 28
#define Image_y 28
#define Image_ch 1


#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

/*
Unit Class for Tflite Distribute Stradegy

Class Constructor
    args -> (tflite::interpreterBuilder *builder,
            input Data(cv::Mat),
            Num of Devices,
            Device Name(ex. GPU1), 
            Device Type(GPU or CPU),
            .
            .
            )
    Make Interpreter & TfLiteDelegate Object for each Device
    Delegate Interpreter & Allocate Tensors
    
    **IMPORTANT**
    YOU BASICALLY HAVE ONE CPU TfLiteInterpreter Object

*/

namespace tflite{

class UnitHandler; //Forward declare of UnitHandler since UnitHander Uses Unit

class Unit 
{   
    public:
        virtual Interpreter* GetInterpreter() = 0;
        virtual TfLiteStatus Invoke(UnitType eType, std::mutex& mtx_lock,
                                    std::mutex& mtx_lock_,
                                    std::mutex& mtx_lock_timing,
                                    std::mutex& mtx_lock_debug,
                                    std::condition_variable& Ucontroller,
                                    std::condition_variable& Outcontroller,
                                    std::queue<SharedContext*>* qSharedData,
                                    int* C_Count, int* G_Count) = 0;
        virtual void SetInput(std::vector<cv::Mat> input_) = 0;
        virtual UnitType GetUnitType() = 0;

        UnitType eType;
        std::vector<cv::Mat> input;
        std::thread myThread;
        std::unique_ptr<tflite::Interpreter> interpreter;
        std::string name;
        int partition;
};

//Unit Class for CPU
class UnitCPU : public Unit
{
    public:
        UnitCPU();
        UnitCPU(UnitType eType_, std::unique_ptr<tflite::Interpreter>* interpreter);
        ~UnitCPU() {};
        TfLiteStatus Invoke(UnitType eType, std::mutex& mtx_lock,
                                    std::mutex& mtx_lock_,
                                    std::mutex& mtx_lock_timing,
                                    std::mutex& mtx_lock_debug,
                                    std::condition_variable& Ucontroller,
                                    std::condition_variable& Outcontroller,
                                    std::queue<SharedContext*>* qSharedData,
                                    int* C_Count, int* G_Count);
        Interpreter* GetInterpreter();
        UnitType GetUnitType();
        void SetInput(std::vector<cv::Mat> input_);

        UnitType eType;
        std::vector<cv::Mat> input;
        std::thread myThread;
        std::unique_ptr<tflite::Interpreter>* interpreterCPU;
        std::string name;
        int partition;
};

//Unit Class for GPU
class UnitGPU : public Unit
{
    public:
        UnitGPU();
        UnitGPU(UnitType eType_, std::unique_ptr<tflite::Interpreter>* interpreter);
        ~UnitGPU() {};
        TfLiteStatus Invoke(UnitType eType, std::mutex& mtx_lock,
                                    std::mutex& mtx_lock_,
                                    std::mutex& mtx_lock_timing,
                                    std::mutex& mtx_lock_debug,
                                    std::condition_variable& Ucontroller,
                                    std::condition_variable& Outcontroller,
                                    std::queue<SharedContext*>* qSharedData,
                                    int* C_Count, int* G_Count);
        Interpreter* GetInterpreter();
        UnitType GetUnitType();
        void SetInput(std::vector<cv::Mat> input_);
        void PrintTest(std::vector<double> b_delegation_optimizer); // HOON : save each loop's latency & accuracy
        UnitType eType;
        std::vector<cv::Mat> input;
        std::thread myThread;
        std::unique_ptr<tflite::Interpreter>* interpreterGPU;
        std::string name;
        int partition;
};

} // End of namespace tflite