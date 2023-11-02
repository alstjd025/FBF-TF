#include <iostream>
#include <vector>
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/optional_debug_tools.h"

namespace Predictor{
  typedef enum Latency_Term{
    CPF,
    CPT,
    KD,
    FW,
    IVS,
    MG,
    CP
  } Latency_Term;

  typedef struct SubgraphCandidate{
    tflite::ResourceType resource_type;
    int start_node;
    int end_node;
    TfLiteIntArray* in_dim;
    TfLiteIntArray* origin_in_dim;
    TfLiteIntArray* out_dim; 
    TfLiteIntArray* origin_out_dim; 
    
    int input_size = 1;
    int origin_input_size = 1;

    int output_size = 1;
    int origin_output_size = 1;

    float flops;
    float origin_flops;

    // Expected CopyFromExternalObject latency
    float CPF = 0;
    float CPF_origin = 0;
    // Expected CopyToExternalObject latency
    float CPT = 0;
    float CPT_origin = 0;
    // Expected kernelDispatch latency
    float KD = 0;
    float KD_origin = 0;
    // Expected FLush latency
    float FW = 0;
    float FW_origin = 0;
    // Expected Total latency;
    float SUM = 0; 
    float SUM_origin = 0; 
    // Expected subgraph scope invoke latency
    float IVS = 0;
    float IVS_origin = 0;
    // Expected merge latency
    float MG = 0;
    float MG_origin = 0;
    // Expected copy(between subgraphs) latency
    float CP = 0;
    float CP_origin = 0;

    // Expected max_latency
  } SubgraphCandidate;

  typedef struct PartitioningPlan{
    std::vector<std::pair<SubgraphCandidate*, SubgraphCandidate*>> subgraphs;
    std::vector<std::pair<int, int>> partitioning_points;
  } PartitioningPlan;


  class PartitioningPredictor
  {
    public:
      PartitioningPredictor(tflite::DEVICE_TYPE device_type_,
                            tflite::MODEL_TYPE model_type_);
      ~PartitioningPredictor();
      // layer index starts at 0
      std::vector<PartitioningPlan*> total_plans;
      std::vector<int> yolo_partitioning_cadidates = {8, 20, 32, 55};
      std::vector<int> mobilenet_partitioning_cadidates = {3, 7, 11, 23, 27, 30};
      std::vector<int> efficientnet_partitioning_cadidates = {
                  5, 9, 13, 17, 20, 24, 28, 32, 39, 43, 47, 51, 55, 62, 66, 70, 74, 78, 85, 89, 93,
                  97, 101, 105, 109};

      void StartPredictor(tflite::Subgraph* origin_subgraph);

      void SimulateSubgraphPartitioning(tflite::Subgraph* origin_subgraph,
                                      std::vector<PartitioningPlan*>& new_plan);

      void SimulateHeightPartitioning(tflite::Subgraph* origin_subgraph, 
                                      SubgraphCandidate* new_subgraph);

      float LatencyPredict(Latency_Term term, tflite::ResourceType r_type ,int x_value);

      void CopyTensorsFromContext(TfLiteContext* context);
      TfLiteTensor* CopyNoBufferTensor(TfLiteTensor& tensor);
      TfLiteTensor* GetTensor(int tensor_idx);
      void ResizeTensorNaive(int tensor_idx, std::vector<int>& new_dim);

      void GetTotalFlopsforGivenSubgraph(tflite::Subgraph* origin_subgraph,
                                          SubgraphCandidate* new_subgraph);

      void GetOriginFlopsforGivenSubgraph(tflite::Subgraph* origin_subgraph,
                                          SubgraphCandidate* new_subgraph);

      void PrintPredictionResult();

    private:
    std::vector<TfLiteTensor*> copied_tensors;
    tflite::DEVICE_TYPE d_type;
    tflite::MODEL_TYPE m_type;
    int partitioning_ratio_gpu;
    int partitioning_ratio_cpu;
    std::vector<std::vector<int>> efficient_points {
      {55},
      {28, 55},
      {28, 55, 85},
      {13, 28, 55, 86},
      {13, 28 ,43 ,55 ,85},
      {13, 28 ,43 ,55 ,70 ,85},
      {13, 28 ,43 ,55 ,70 ,85 ,97},
      {9 , 13 ,28 ,43 ,55 ,70 ,85 ,97},
      {9 , 13 ,20 ,28 ,43 ,55 ,70 ,85 ,97},
      {9 , 13 ,20 ,28 ,39 ,43 ,55 ,70 ,85 ,97},
      {9 , 13 ,20 ,28 ,39 ,43 ,51 ,55 ,70 ,85 ,97},
      {9 , 13 ,20 ,28 ,39 ,43 ,51 ,55 ,66 ,70 ,85 ,97},
      {9 , 13 ,20 ,28 ,39 ,43 ,51 ,55 ,66 ,70 ,78 ,85 ,97},
      {9 , 13 ,20 ,28 ,39 ,43 ,51 ,55 ,66 ,70 ,78 ,85 ,93 ,97},
      {9 , 13 ,20 ,28 ,39 ,43 ,51 ,55 ,66 ,70 ,78 ,85 ,93 ,97 ,105},
      {5 , 9  ,13 ,20 ,28 ,39 ,43 ,51 ,55 ,66 ,70 ,78 ,85 ,93 ,97 ,105},
      {5 , 9  ,13 ,17 ,20 ,28 ,39 ,43 ,51 ,55 ,66 ,70 ,78 ,85 ,93 ,97 ,105},
      {5 , 9  ,13 ,17 ,20 ,24 ,28 ,39 ,43 ,51 ,55 ,66 ,70 ,78 ,85 ,93 ,97 ,105},
      {5 , 9  ,13 ,17 ,20 ,24 ,28 ,32 ,39 ,43 ,51 ,55 ,66 ,70 ,78 ,85 ,93 ,97 ,105},
      {5 , 9  ,13 ,17 ,20 ,24 ,28 ,32 ,39 ,43 ,47 ,51 ,55 ,66 ,70 ,78 ,85 ,93 ,97 ,105},
      {5 , 9  ,13 ,17 ,20 ,24 ,28 ,32 ,39 ,43 ,47 ,51 ,55 ,62 ,66 ,70 ,78 ,85 ,93 ,97 ,105},
      {5 , 9  ,13 ,17 ,20 ,24 ,28 ,32 ,39 ,43 ,47 ,51 ,55 ,62 ,66 ,70 ,74 ,78 ,85 ,93 ,97 ,105},
      {5 , 9  ,13 ,17 ,20 ,24 ,28 ,32 ,39 ,43 ,47 ,51 ,55 ,62 ,66 ,70 ,74 ,78 ,85 ,89 ,93 ,97 ,105},
      {5 , 9  ,13 ,17 ,20 ,24 ,28 ,32 ,39 ,43 ,47 ,51 ,55 ,62 ,66 ,70 ,74 ,78 ,85 ,89 ,93 ,97 ,101 ,105},
      {5 , 9  ,13 ,17 ,20 ,24 ,28 ,32 ,39 ,43 ,47 ,51 ,55 ,62 ,66 ,70 ,74 ,78 ,85 ,89 ,93 ,97 ,101 ,105 ,109}
    };
    std::vector<std::vector<int>> yolo_points{
      {13, 28, 55, 86}
    };
    std::vector<std::vector<int>> mobilenet_points{
      {11},
      {3, 11},
      {3, 7, 11},
      {3, 7, 11, 23},
      {3, 7, 11, 17, 23}
    };
  };

} // namespace Predictor
