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
#include "tensorflow/lite/kernels/internal/cppmath.h"
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

    // Prepares co-execution for intermediate & shared tensors between interpreters.
    TfLiteStatus PrepareCoExecution();

    void FeedInputToModel(const char* model, std::vector<cv::Mat>& input,
                          INPUT_TYPE input_type);
    void FeedInputToModel(const char* model, cv::Mat& input,
                          INPUT_TYPE input_type);

    
    /// For debugging only ==
    void InitLogFile();
    void WriteVectorLog(std::vector<double>& latency, int n);
    std::ofstream logFile; 
    std::ofstream logFile_; 

    void FeedInputToInterpreter(std::vector<cv::Mat>& mnist, 
                                  std::vector<cv::Mat>& imagetnet);

    // Debug invoke (for single interpreter invoke test) 
    TfLiteStatus DebugInvoke();

    // Debug invoke (for co-execution invoke test api) 
    TfLiteStatus DebugCoInvoke();

    // Debug invoke (for co-execution invoke synchronization test)
    // Call after DebugCoInvoke()
    void DebugSyncInvoke(PrecisionType type);

    void FeedInputToModelDebug(const char* model, cv::Mat& input,
                               cv::Mat& input_quant, INPUT_TYPE input_type);
    void PrintOutput(Subgraph* subgraph);
    void PrintTensor(TfLiteTensor& tensor, bool is_output);
    void PrintTensorSerial(TfLiteTensor& tensor);
    void PrintyoloOutput(TfLiteTensor& tensor);
    std::vector<std::vector<float>*>* GetFloatOutputInVector();
    std::vector<std::vector<uint8_t>*>* GetUintOutputInVector();
    ////// ==

    void WakeScheduler();
    void JoinScheduler();

    TfLiteStatus Invoke();
    TfLiteStatus InvokeCoExecution();
    TfLiteStatus InvokeSingleExecution();

    // Merge output(which is intermediate in the view of whole task)
    // data from previous subgraph.
    void CopyIntermediateDataIfNeeded(Subgraph* subgraph);

    // Merge output(which is intermediate in the view of whole task)
    // data from full precision subgraph.
    void CopyIntermediateDataIfNeeded(Subgraph* co_subgraph, Subgraph* subgraph);

    
    
    // Merge output of sub-subgraph(for co-execution) to main subgraph's input.
    void MergeCoExecutionData(Subgraph* min_precision_subgraph
                            , Subgraph* max_precision_subgraph);

    // Quantize given tensor
    // (This function changes the entire metadata to uint8)
    TfLiteStatus QuantizeGivenTensor(TfLiteTensor* tensor);

    // Quantize given tensor and return buffer pointer which contains quantized
    // values.
    // (This function does not )
    void* QuantizeGivenTensorandReturnBuffer(TfLiteTensor* tensor);

    TfLiteStatus QuantizeGivenTensorandCopy(TfLiteTensor* source_tensor,
                                            TfLiteTensor* dest_tensor);
    
    // This function is deprecated.
    // Use Quant/dequantOnCopy() instead. 
    // Dequantize given tensor to float32 and return the swaped buffer.
    // Recommended to use with RestoreOriginalBuffer.
    void* DequantizeGivenTensor(TfLiteTensor* tensor);

    // This function is deprecated.
    // Use Quant/dequantOnCopy() instead.
    // Dequantize given tensor to float32 and return the swaped buffer.
    // Recommended to use with RestoreOriginalBuffer. 
    // Calculate dequantization parameters from given refenrece tensor.
    void* DequantizeGivenTensorWithReference(
                        TfLiteTensor* tensor, TfLiteTensor* ref_tensor);

    // Restore original buffer of given tensor and buffer
    // Use after dequantization
    void RestoreOriginalBuffer(TfLiteTensor* tensor, void* buffer);

    void QuantizeFloats(const float* float_data_ptr, int n_batch,
                                int n_data, int8_t* quantized_data_ptr,
                                float* scaling_factors, int32_t* zero_points,
                                bool do_asymmetric);

    void QuantizeSymFloatsMain(const float* values, const int size,
                                     int8_t* quantized_values, float min_value,
                                     float max_value, float* scaling_factor);

    void QuantizeSymFloats(const float* values, const int size,
                                     int8_t* quantized_values, float* min_value,
                                     float* max_value, float* scaling_factor);

    TfLiteAffineQuantization* CalcQuantizationParamsFromTensor(TfLiteTensor* tensor);

    TfLiteStatus QuantizeOnCopy(TfLiteTensor* source, TfLiteTensor* dest);
    TfLiteStatus DequantizeOnCopy(TfLiteTensor* source, TfLiteTensor* dest);

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

    TfLiteTensor* global_output_tensor = nullptr;

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

    // must do readonly works on this object.
    Subgraph* co_execution_graph = nullptr;

    // must do readonly works on this object.
    Subgraph* main_execution_graph = nullptr;
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