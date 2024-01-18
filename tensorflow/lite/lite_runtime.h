#pragma once
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <queue>
#include <random>
#include <utility>
#include <vector>

#include "condition_variable"
#include "future"
#include "opencv2/opencv.hpp"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/latency_predictor.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "thread"

/*
Author : Minsung Kim
This class is re-writed for IPC with scheduler and 1:1 relationship for
interpreter and interpreterbuilder. (commit b56faa4981)

*/

namespace tflite {

typedef struct TfLiteMergeTensor {
  TfLiteTensor* tensor;
  PartitioningType partition_type;
  int tensor_idx;
  bool is_used = false;
} TfLiteMergeTensor;

class LiteScheduler;
// class YOLO_Parser;

class TfLiteRuntime {
 public:
  TfLiteRuntime(char* uds_runtime, char* uds_scheduler, const char* model,
                INPUT_TYPE type);
  TfLiteRuntime(char* uds_runtime, char* uds_scheduler, const char* f_model,
                const char* i_model, INPUT_TYPE type, DEVICE_TYPE d_type,
                bool use_predictor);

  ~TfLiteRuntime();

  TfLiteStatus AddModelToRuntime(const char* new_model);

  // An overloaded function for Co-execution
  TfLiteStatus AddModelToRuntime(const char* f_model, const char* i_model);

  TfLiteStatus RegisterModeltoScheduler();
  TfLiteStatus PartitionSubgraphs();

  TfLiteStatus PredictSubgraphPartitioning();

  // Partitions subgraph in both Float & Int.
  TfLiteStatus PartitionCoSubgraphs();

  // Prepares co-execution for intermediate & shared tensors between
  // interpreters.
  TfLiteStatus PrepareCoExecution();

  /// For debugging only ==
  void InitLogFile();
  void WriteVectorLog(std::vector<double>& latency, int n);
  void WriteVectorLog(std::vector<double>& latency, std::vector<string>& label,
                      int n);
  void WriteInitStateLog();

  // Fill dummy input (random 0~1.0) to given tensor buffer.
  void FeedDummyInputToTensor(TfLiteTensor* tensor);

  // contains name of test sequence
  void SetTestSequenceName(std::string name);
  void SetLogPath(std::string path);
  std::string sequence_name;
  std::string log_path;

  std::ofstream m_interpreter_lat_log;
  std::ofstream s_interpreter_lat_log;
  std::ofstream m_interpreter_t_stamp_log;
  std::ofstream s_interpreter_t_stamp_log;

  std::vector<double> latency_main_interpreter;
  std::vector<double> timestamp_main_interpreter;
  std::vector<double> latency_sub_interpreter;
  std::vector<double> timestamp_sub_interpreter;
  std::vector<string> timestamp_label_main_interpreter;
  std::vector<string> timestamp_label_sub_interpreter;

  // Debug invoke (for co-execution invoke test api)
  TfLiteStatus Invoke();

  // Debug invoke (for co-execution invoke synchronization test)
  // Call after DebugCoInvoke()
  void DoInvoke(InterpreterType type, TfLiteStatus& return_state);

  void SetInputType(INPUT_TYPE input_type_);

  void SetDeviceType(DEVICE_TYPE device_type_);

  DEVICE_TYPE GetDeviceType();

  INPUT_TYPE GetInputTypeFromString(string input_type);

  void CopyInputToInterpreter(const char* model, cv::Mat& input,
                              cv::Mat& input_quant);

  void PrintOutput(Subgraph* subgraph);

  void PrintTensor(TfLiteTensor& tensor, bool is_output);

  void PrintTensorSerial(TfLiteTensor& tensor);

  void PrintyoloOutput(TfLiteTensor& tensor);

  std::vector<std::vector<float>*>* GetFloatOutputInVector();

  std::vector<std::vector<uint8_t>*>* GetUintOutputInVector();

  // Copy output(which is intermediate in the view of whole task)
  // data from previous subgraph (with id).
  // used in main subgraph.
  TfLiteStatus CopyIntermediateDataIfNeeded(Subgraph* subgraph,
                                            int prev_subgraph_id,
                                            TfLiteMergeTensor* buffer_tensor);

  // Copy output(which is intermediate in the view of whole task)
  // data from full precision subgraph.
  // used in sub-subgraph.
  TfLiteStatus CopyIntermediateDataIfNeeded(Subgraph* co_subgraph,
                                            Subgraph* subgraph,
                                            TfLiteMergeTensor* buffer_tensor);

  // Merge output of sub-subgraph(for co-execution) to main subgraph's input.
  TfLiteStatus MergeCoExecutionData(int prev_sub_subgraph,
                                    int prev_main_subgraph, int dest_subgraph_,
                                    TfLiteMergeTensor* buffer_tensor);

  // Quantize given tensor
  // (This function changes the entire metadata to uint8)
  TfLiteStatus QuantizeGivenTensor(TfLiteTensor* tensor);

  // Quantize given tensor and return buffer pointer which contains quantized
  // values.
  // (This function does not )
  void* QuantizeGivenTensorandReturnBuffer(
      TfLiteTensor* tensor, TfLiteAffineQuantization* quant_params);

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
  void* DequantizeGivenTensorWithReference(TfLiteTensor* tensor,
                                           TfLiteTensor* ref_tensor);

  // Restore original buffer of given tensor and buffer
  // Use after dequantization
  void RestoreOriginalBuffer(TfLiteTensor* tensor, void* buffer);

  void QuantizeFloats(const float* float_data_ptr, int n_batch, int n_data,
                      int8_t* quantized_data_ptr, float* scaling_factors,
                      int32_t* zero_points, bool do_asymmetric);

  void QuantizeSymFloatsMain(const float* values, const int size,
                             int8_t* quantized_values, float min_value,
                             float max_value, float* scaling_factor,
                             int32_t* zero_points);

  void QuantizeSymFloats(const float* values, const int size,
                         int8_t* quantized_values, float* min_value,
                         float* max_value, float* scaling_factor,
                         int32_t* zero_points);

  TfLiteAffineQuantization* CalcQuantizationParamsFromTensor(
      TfLiteTensor* tensor);

  TfLiteStatus QuantizeOnCopy(TfLiteTensor* source, TfLiteTensor* dest);
  TfLiteStatus DequantizeOnCopy(TfLiteTensor* source, TfLiteTensor* dest);

  //// IPC functions
  // Initialize UDS and check communication with scheduler.
  TfLiteStatus InitializeUDS();
  TfLiteStatus ChangeStatewithPacket(tf_packet& rx_p);
  RuntimeState GetRuntimeState() { return state; };
  TfLiteStatus SendPacketToScheduler(tf_packet& tx_p);
  TfLiteStatus ReceivePacketFromScheduler(tf_packet& rx_p);
  void ShutdownScheduler();
  //////

 private:
  RuntimeState state;
  int runtime_id = -1;
  tflite::Interpreter* interpreter;
  tflite::Interpreter* sub_interpreter;
  tflite::InterpreterBuilder* interpreter_builder;
  tflite::InterpreterBuilder* sub_builder;

  TfLiteTensor* global_output_tensor = nullptr;

  INPUT_TYPE input_type;

  DEVICE_TYPE device_type;
  MODEL_TYPE model_type;

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
  int co_subgraph_id = -1;

  // must do readonly works on this object.
  Subgraph* main_execution_graph = nullptr;
  ////

  // used to merge co-execution data if extra scratch buffer needed.
  TfLiteMergeTensor* merge_tensor = nullptr;

  // Subgraph partitioning
  int partitioning_plan[TF_P_PLAN_LENGTH];

  // sj
  std::vector<TfLiteDelegate*> delegate;

  // IPC
  char* uds_runtime_filename;
  char* uds_scheduler_filename;
  int runtime_sock;
  size_t addr_size;
  struct sockaddr_un runtime_addr;
  struct sockaddr_un scheduler_addr;

  bool output_correct = false;
};

}  // namespace tflite