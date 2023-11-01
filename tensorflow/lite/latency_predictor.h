#include <iostream>
#include <vector>
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/schema/schema_generated.h"

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
    std::vector<int> in_dim;
    int input_size = 1;
    std::vector<int> out_dim; 
    int output_size = 1;
    float flops;

    // Expected CopyFromExternalObject latency
    float CPF = 0;
    // Expected CopyToExternalObject latency
    float CPT = 0;
    // Expected kernelDispatch latency
    float KD = 0;
    // Expected FLush latency
    float FW = 0;
    // Expected Total latency;
    float SUM = 0; 
    // Expected subgraph scope invoke latency
    float IVS = 0;
    // Expected merge latency
    float MG = 0;
    // Expected copy(between subgraphs) latency
    float CP = 0;

  } SubgraphCandidate;

  typedef struct PartitioningPlan{
    std::vector<std::pair<SubgraphCandidate*, SubgraphCandidate*>> subgraphs;
    std::vector<std::pair<int, int>> partitioning_points;
  } PartitioningPlan;


  class PartitioningPredictor
  {
    PartitioningPredictor();
    
    PartitioningPredictor(tflite::DEVICE_TYPE device_type_,
                          tflite::MODEL_TYPE model_type_);
    ~PartitioningPredictor();
    public:
      // layer index starts at 0
      std::vector<std::vector<PartitioningPlan*>> total_plans;
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

    private:
    std::vector<TfLiteTensor*> copied_tensors;
    tflite::DEVICE_TYPE d_type;
    tflite::MODEL_TYPE m_type;
    int partitioning_ratio_gpu;
    int partitioning_ratio_cpu;
  };

} // namespace Predictor

/*
55
28 55
28 55 85
13 28 55 85
13 28 43 55 85
13 28 43 55 70 85
13 28 43 55 70 85 97
9 13 28 43 55 70 85 97
9 13 20 28 43 55 70 85 97
9 13 20 28 39 43 55 70 85 97
9 13 20 28 39 43 51 55 70 85 97
9 13 20 28 39 43 51 55 66 70 85 97
9 13 20 28 39 43 51 55 66 70 78 85 97
9 13 20 28 39 43 51 55 66 70 78 85 93 97
9 13 20 28 39 43 51 55 66 70 78 85 93 97 105
5 9 13 20 28 39 43 51 55 66 70 78 85 93 97 105
5 9 13 17 20 28 39 43 51 55 66 70 78 85 93 97 105
5 9 13 17 20 24 28 39 43 51 55 66 70 78 85 93 97 105
5 9 13 17 20 24 28 32 39 43 51 55 66 70 78 85 93 97 105
5 9 13 17 20 24 28 32 39 43 47 51 55 66 70 78 85 93 97 105
5 9 13 17 20 24 28 32 39 43 47 51 55 62 66 70 78 85 93 97 105
5 9 13 17 20 24 28 32 39 43 47 51 55 62 66 70 74 78 85 93 97 105
5 9 13 17 20 24 28 32 39 43 47 51 55 62 66 70 74 78 85 89 93 97 105
5 9 13 17 20 24 28 32 39 43 47 51 55 62 66 70 74 78 85 89 93 97 101 105
5 9 13 17 20 24 28 32 39 43 47 51 55 62 66 70 74 78 85 89 93 97 101 105 109

*/
