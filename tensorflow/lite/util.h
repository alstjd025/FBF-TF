/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file provides general C++ utility functions in TFLite.
// For example: Converting between `TfLiteIntArray`, `std::vector` and
// Flatbuffer vectors. These functions can't live in `context.h` since it's pure
// C.

#ifndef TENSORFLOW_LITE_UTIL_H_
#define TENSORFLOW_LITE_UTIL_H_

// Minsung
// CPU setups
#define DEFAULT_AFFINITY 0
#define DEFAULT_THREADS  1

// Minsung
// packet predefines
//[VLS Todo] change this parameter for appropiate size of runtime and init packet.
#define TF_P_PLAN_LENGTH     10000 

//[VLS Todo] maybe use these params later?
#define TF_P_PLAN_RUNTIME_LENGTH 10
#define TF_P_PLAN_INIT_LENGTH 10000

#define TF_P_PLAN_SIZE       4

// Partitioning parameter seperators.
// See tf_scheduler.h for detailed descriptions.
#define PART_PARM_SEP_OP   -1   
#define PART_PARM_SEP_RESROURCE   -2
#define PART_PARM_SEP_SUBG   -3
#define PART_PARM_SEP_ENDP   -4

// Packet partitioning plan resource types
#define TF_P_PLAN_CPU        0
#define TF_P_PLAN_GPU        1
#define TF_P_PLAN_CO_E       2
#define TF_P_PLAN_CPU_XNN    3
#define TF_P_PLAN_CO_E_XNN   4

// packet partitioning plan end flag
#define TF_P_END_PLAN       -1

// packet master partitioning plan end flag
#define TF_P_END_MASTER        -2

// packet predefines

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"

namespace tflite {


// Memory allocation parameter used by ArenaPlanner.
// Clients (such as delegates) might look at this to ensure interop between
// TFLite memory & hardware buffers.
// NOTE: This only holds for tensors allocated on the arena.
constexpr int kDefaultTensorAlignment = 64;

// The prefix of Flex op custom code.
// This will be matched agains the `custom_code` field in `OperatorCode`
// Flatbuffer Table.
// WARNING: This is an experimental API and subject to change.
constexpr char kFlexCustomCodePrefix[] = "Flex";

// Checks whether the prefix of the custom name indicates the operation is an
// Flex operation.
bool IsFlexOp(const char* custom_name);

// Converts a `std::vector` to a `TfLiteIntArray`. The caller takes ownership
// of the returned pointer.
TfLiteIntArray* ConvertVectorToTfLiteIntArray(const std::vector<int>& input);

// Converts an array (of the given size) to a `TfLiteIntArray`. The caller
// takes ownership of the returned pointer, and must make sure 'dims' has at
// least 'rank' elements.
TfLiteIntArray* ConvertArrayToTfLiteIntArray(const int rank, const int* dims);

// Checks whether a `TfLiteIntArray` and an int array have matching elements.
// The caller must guarantee that 'b' has at least 'b_size' elements.
bool EqualArrayAndTfLiteIntArray(const TfLiteIntArray* a, const int b_size,
                                 const int* b);

size_t CombineHashes(std::initializer_list<size_t> hashes);

struct TfLiteIntArrayDeleter {
  void operator()(TfLiteIntArray* a) {
    if (a) TfLiteIntArrayFree(a);
  }
};

// Helper for Building TfLiteIntArray that is wrapped in a unique_ptr,
// So that it is automatically freed when it goes out of the scope.
std::unique_ptr<TfLiteIntArray, TfLiteIntArrayDeleter> BuildTfLiteIntArray(
    const std::vector<int>& data);

// Populates the size in bytes of a type into `bytes`. Returns kTfLiteOk for
// valid types, and kTfLiteError otherwise.
TfLiteStatus GetSizeOfType(TfLiteContext* context, const TfLiteType type,
                           size_t* bytes);

// Creates a stub TfLiteRegistration instance with the provided
// `custom_op_name`. The op will fail if invoked, and is useful as a
// placeholder to defer op resolution.
// Note that `custom_op_name` must remain valid for the returned op's lifetime..
TfLiteRegistration CreateUnresolvedCustomOp(const char* custom_op_name);

// Checks whether the provided op is an unresolved custom op.
bool IsUnresolvedCustomOp(const TfLiteRegistration& registration);

// Returns a descriptive name with the given op TfLiteRegistration.
std::string GetOpNameByRegistration(const TfLiteRegistration& registration);

// Minsung
// Get parameters used for HW partitioning.
bool GetParamsForPartitioning(const TfLiteRegistration* registration,
                              const TfLiteNode* node, TfLiteContext* context,
                              int& filter_size, int& stride, int& padding_type,
                              int& padding_height, int& padding_width,
                              int& padding_height_offset, int& padding_width_offset);


typedef struct SharedTensorsInGraphs{
  int model_id; // identifier

  // <tensor id, subgraph ids>
  std::vector<std::pair<int, std::vector<int>>> pair_tensor_graph;
} SharedTensorsInGraphs;

namespace HW{
  // overlap equation from CoDL (Mobisys '23)
  // S  : stride
  // K  : filter size
  // Hi : Input Height
  // Ho : Output Height
  int GetOverlapConv(int S, int K, int Hi, int Ho);

  // padding equation for conv
  // S  : stride
  // K  : filter size
  // Hi : Input Height
  // Ho : Output Height
  int GetZeroPaddingConv(int S, int K, int Hi, int Ho);

  // Output height equation for conv
  // S  : stride
  // P  : padding
  // K  : filter size
  // Hi : Input Height
  int GetOutputHeightConv(int S, int K, int P, int Hi);

  int GetOutputHeightofSameFeatureConv(int Hi_origin, int S);

  int GetInputHeightofSameFeatureConv(int Ho_origin, int S);

  int GetOverlapPool(int S, int K, int Hi, int Ho);
  
  int GetPaddingPool(int S, int K, int Hi);

  int GetOutputHeightPool(int S, int K, int P, int Hi);
} // namespace HW


// Minsung
// Scheduler status
typedef enum SchedulerState{
  INIT_SCHED,           // Initial phase.
  SCHEDULING,    // Scheduling, means interpreter and jobs are ready.
  NSCHEDULABLE,         // Not schedulable.
  RESCHEDULE,     // Need reschedule
  STOP
} SchedulerStatus;

//WARNING! This struct is deprecatied.
typedef enum JobState{
  INIT_JOB,       // Initial phase. 
  READY,        // Job is ready to be scheduled.(allocation & op mapping done)
  INVOKE,      // Job is currently scheduled and invoking.
  SLEEP,       // Job is stopped by scheduler.
  DONE        // Job done and can be erased from mememory.
} JobState;

//WARNING! This struct is deprecatied.
typedef enum WorkerState{
  INIT_WORK,
  WORKING,
  BLOCKED
} WorkerState;

typedef enum InterpreterType{
  MAIN_INTERPRETER,
  SUB_INTERPRETER
} InterpreterType;

// Partitioning types for co-execution
typedef enum PartitioningType{
  NO_PARTITIONING,
  HEIGHT_PARTITIONING,
  CHANNEL_PARTITIONING,
  BRANCH_PARALLEL
} PartitioningType;

// ResourceType for subgraph
// NEED_REFACTOR (02634) : Better use clear name. (like SubgraphType)
typedef enum ResourceType{
  CPU,
  GPU,
  CO_CPU,
  CO_GPU,
  CPU_XNN,
  CO_CPU_XNN,
  NONE
} ResourceType;

// packet partitioning plan resource types
// #define TF_P_PLAN_CPU        0
// #define TF_P_PLAN_GPU        1
// #define TF_P_PLAN_CO_E       2
// #define TF_P_PLAN_CPU_XNN    3
// #define TF_P_PLAN_CO_E_XNN   4

typedef enum DelegateType{
  XNN_DELEGATE,
  GPU_DELEGATE,
  NO_DELEGATE,
} DelegateType;

typedef enum RuntimeState{
  INITIALIZE,
  NEED_PROFILE,
  SUBGRAPH_CREATE,
  INVOKE_,
  BLOCKED_,
  TERMINATE
} RuntimeState;

//WARNING! This struct is deprecatied.
typedef enum InvokeType{ // An invoke type of job
  CONTINOUS,          
  PROFILING
} InvokeType;

//WARNING! This struct is deprecatied.
typedef struct Job{
  int model_id = -1;      // 
  int job_id = -1;        //
  float time_slot;      //ms
  float dead_line;      //ms
  time_t start;
  time_t end;
  bool input_refreshed;
  JobState state = JobState::INIT_JOB;
  InvokeType invoke_type = InvokeType::PROFILING;
  ResourceType resource_type = ResourceType::CPU;
  std::vector<std::pair<int, int>> subgraphs;
  std::vector<int> cpu_affinity;
} Job;


typedef struct ProfileData{
  std::vector<float> latency_by_layers;
  std::vector<std::vector<int>> layer_subsets;
  std::vector<enum ResourceType> subset_resource;
  std::vector<std::vector<int>> partitioning_ratios;
  std::vector<int> gpu_layers;
  bool is_valid;
} ProfileData;

typedef enum INPUT_TYPE{
  MNIST,
  IMAGENET224,
  IMAGENET256,
  IMAGENET300,
  IMAGENET416,
  COCO416,
  LANENET_FRAME,
  CENTERNET_FRAME,
  USER
}INPUT_TYPE;


// TODO : Consider including partitioning ratios per layer.
typedef struct tf_packet{
  short runtime_id;
  short runtime_current_state;
  short runtime_next_state;
  int cur_subgraph;
  int cur_graph_resource; // 0 for cpu, 1 for gpu
  int partitioning_plan[1000];
  int subgraph_ids[2][100]; 
  float latency[1000];
  float gpu_utilization;
  float cpu_utilization;
}tf_packet;

// To minimize communication overhead at runtime,
// we use different packets for runtime phase and init phase.
// At init phase, need big ary for paramters while runtime needs small ary.
typedef struct tf_runtime_packet{ // runtime packet(use at invoke)
  short runtime_id;
  short runtime_current_state;
  short runtime_next_state;
  int cur_subgraph;
  int cur_graph_resource; // 0 for cpu, 1 for gpu
  int partitioning_plan[1000]; // [VLS Todo] deprecate later.
  int subgraph_ids[2][100]; 
  float latency[1000]; // [VLS Todo] deprecate later.
  float gpu_utilization;
  float cpu_utilization;
}tf_runtime_packet;

typedef struct tf_initialization_packet{// runtime packet(use at init)
  short runtime_id;
  short runtime_current_state;
  short runtime_next_state;
  int cur_subgraph;
  int cur_graph_resource; // 0 for cpu, 1 for gpu
  int partitioning_plan[TF_P_PLAN_LENGTH];
  int subgraph_ids[2][1000];
  float latency[1000];
  float gpu_utilization;
  float cpu_utilization;
}tf_initialization_packet;
//

////////////////////////////////////////////////////////////////////
// HOON : utility funcs for parsing Yolo output
class YOLO_Parser{
  public:
    YOLO_Parser();
    ~YOLO_Parser();
    static std::vector<std::vector<float>> real_bbox_cls_vector; 
    static std::vector<int> real_bbox_cls_index_vector;
    static std::vector<std::vector<int>> real_bbox_loc_vector;
    std::vector<int> get_cls_index(std::vector<std::vector<float>>& real_bbox_cls_vector);
    void make_real_bbox_cls_vector(TfLiteTensor* cls_tensor, std::vector<int>& real_bbox_index_vector, std::vector<std::vector<float>>& real_bbox_cls_vector);
    void make_real_bbox_loc_vector(TfLiteTensor* loc_tensor, std::vector<int>& real_bbox_index_vector,std::vector<std::vector<int>>& real_bbox_loc_vector);
    void SOFTMAX(std::vector<float>& real_bbox_cls_vector);
    void NMS(const std::vector<int>& real_bbox_cls_index_vector, const std::vector<std::vector<float>>& real_bbox_cls_vector, const std::vector<std::vector<int>>& real_bbox_loc_vector);
    struct BoundingBox {
      float left, top, right, bottom;
      float score;
      int class_id;
    };
    static std::vector<YOLO_Parser::BoundingBox> result_boxes;
    static bool CompareBoxesByScore(const BoundingBox& box1, const BoundingBox& box2);
    float CalculateIoU(const BoundingBox& box1, const BoundingBox& box2);
    void NonMaximumSuppression(std::vector<BoundingBox>& boxes, float iou_threshold);
    void PerformNMSUsingResults(
    const std::vector<int>& real_bbox_index_vector,
    const std::vector<std::vector<float>>& real_bbox_cls_vector,
    const std::vector<std::vector<int>>& real_bbox_loc_vector,
    float iou_threshold, const std::vector<int> real_bbox_cls_index_vector);
    std::string yolo_labels[80] =
  {
   "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
   "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird",
   "cat", "dog", "horse", "sheep", "cow",  "elephant", "bear", "zebra", "giraffe", "backpack",
   "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
   "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
   "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
   "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
   "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
   "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
   "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
  };
}; 

typedef enum DEVICE_TYPE{
  ODROID,
  XAVIER
} DEVICE_TYPE;

typedef enum MODEL_TYPE{
  MOBILENET,
  EFFICIENTNET,
  YOLO,
  LANENET,
  MOVENET,
  CENTERNET,
} MODEL_TYPE;

typedef struct DelegateWrapper{
  TfLiteDelegate* delegate;
  DelegateType delegate_type = DelegateType::NO_DELEGATE;
  int prefered_utilization = 0;
} DelegateWrapper;

////////////////////////////////////////////////////////////////////
}  // namespace tflite

#endif  // TENSORFLOW_LITE_UTIL_H_
