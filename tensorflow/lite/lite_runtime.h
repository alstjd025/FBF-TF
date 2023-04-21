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
                                        const char* model);
    ~TfLiteRuntime();

    TfLiteStatus AddModelToRuntime(const char* new_model);
    TfLiteStatus RegisterModeltoScheduler();
    TfLiteStatus PartitionSubgraphs();

    TfLiteStatus DebugInvoke();

    void FeedInputToModel(const char* model, std::vector<cv::Mat>& input,
                          INPUT_TYPE input_type);
    
    // For debugging only
    void FeedInputToInterpreter(std::vector<cv::Mat>& mnist, 
                                  std::vector<cv::Mat>& imagetnet);

    void WakeScheduler();
    void JoinScheduler();

    TfLiteStatus Invoke();
    void CopyIntermediateDataIfNeeded(Subgraph* subgraph);
    void PrintOutput(Subgraph* subgraph);
    void PrintTensor(TfLiteTensor& tensor, bool is_output);

    //// IPC functions
    // Initialize UDS and check communication with scheduler.
    TfLiteStatus InitializeUDS();
    TfLiteStatus ChangeStatewithPacket(tf_packet& rx_p);
    TfLiteStatus SendPacketToScheduler(tf_packet& tx_p);
    TfLiteStatus ReceivePacketFromScheduler(tf_packet& rx_p);

  private:
    RuntimeState state;
    int runtime_id = -1;
    tflite::Interpreter* interpreter;
    tflite::InterpreterBuilder* interpreter_builder;

    // Subgraph partitioning
    int partitioning_plan[1000][3];

    // IPC
    char* uds_runtime_filename;
    char* uds_scheduler_filename;
    int runtime_sock;
    size_t addr_size;
    struct sockaddr_un runtime_addr;
    struct sockaddr_un scheduler_addr;

};

} // namespace tflite