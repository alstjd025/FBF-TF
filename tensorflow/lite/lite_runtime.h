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
    TfLiteRuntime(char* uds_runtime, char* uds_scheduler,
                      const char* model, INPUT_TYPE type);
    TfLiteRuntime(char* uds_runtime, char* uds_scheduler,
                      const char* f_model, const char* i_model, INPUT_TYPE type);

    ~TfLiteRuntime();

    TfLiteStatus AddModelToRuntime(const char* new_model);
  
    // An overloaded function for Co-execution
    TfLiteStatus AddModelToRuntime(const char* f_model, const char* i_model);
    
    TfLiteStatus RegisterModeltoScheduler();
    TfLiteStatus PartitionSubgraphs();

    // Partitions subgraph in both Float & Int.
    TfLiteStatus PartitionCoSubgraphs();

    // Binds the subgraphs in co-execution subgraphs.
    TfLiteStatus BindCoExecutionSubgraphs();

    void FeedInputToModel(const char* model, std::vector<cv::Mat>& input,
                          INPUT_TYPE input_type);
    void FeedInputToModel(const char* model, cv::Mat& input,
                          INPUT_TYPE input_type);

    
    /// For debugging only ==
    void FeedInputToInterpreter(std::vector<cv::Mat>& mnist, 
                                  std::vector<cv::Mat>& imagetnet);

    // Debug invoke (for single interpreter invoke test) 
    TfLiteStatus DebugInvoke();

    // Debug invoke (for co-execution invoke test api) 
    TfLiteStatus DebugCoInvoke();

    // Debug invoke (for co-execution invoke synchronization test)
    // Call after DebugCoInvoke()
    void DebugSyncInvoke(ThreadType type);


    void FeedInputToModelDebug(const char* model, cv::Mat& input,
                          INPUT_TYPE input_type);
    void PrintOutput(Subgraph* subgraph);
    void PrintTensor(TfLiteTensor& tensor, bool is_output);
    void PrintTensorSerial(TfLiteTensor& tensor);
    ////// ==

    void WakeScheduler();
    void JoinScheduler();

    TfLiteStatus Invoke();
    TfLiteStatus InvokeCoExecution();
    TfLiteStatus InvokeSingleExecution();

    // Merge output(which is intermediate in the view of whole task)
    // data from previous subgraph.
    void CopyIntermediateDataIfNeeded(Subgraph* subgraph);
    
    // Merge output of sub-subgraph(for co-execution) to main subgraph's input.
    void MergeCoExecutionData(Subgraph* cpu_source, Subgraph* gpu_source);

    //// IPC functions
    // Initialize UDS and check communication with scheduler.
    TfLiteStatus InitializeUDS();
    TfLiteStatus ChangeStatewithPacket(tf_packet& rx_p);
    TfLiteStatus SendPacketToScheduler(tf_packet& tx_p);
    TfLiteStatus ReceivePacketFromScheduler(tf_packet& rx_p);
    //////

  private:
    RuntimeState state;
    int runtime_id = -1;
    tflite::Interpreter* interpreter;
    tflite::Interpreter* quantized_interpreter;
    tflite::InterpreterBuilder* interpreter_builder;
    tflite::InterpreterBuilder* quantized_builder;

    //// Co-execution
    bool co_execution = false;
    
    std::thread c_thread;
    std::thread g_thread;

    std::condition_variable invoke_sync_cv;
    std::condition_variable data_sync_cv;
    std::mutex data_sync_mtx;
    std::mutex invoke_sync_mtx;
    bool is_execution_done = false;
    bool invoke_cpu = false;
    Subgraph* co_execution_graph = nullptr;
    ////

    // Subgraph partitioning
    int partitioning_plan[1000][4];

    // IPC
    char* uds_runtime_filename;
    char* uds_scheduler_filename;
    int runtime_sock;
    size_t addr_size;
    struct sockaddr_un runtime_addr;
    struct sockaddr_un scheduler_addr;

    bool output_correct = false;

};

} // namespace tflite