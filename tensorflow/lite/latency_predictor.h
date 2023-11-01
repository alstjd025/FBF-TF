#include <iostream>
#include <vector>
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/core/subgraph.h"

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
    std::vector<int> out_dim; 
    float flops;

    // Expected CopyFromExternalObject latency
    float CPF;
    // Expected CopyToExternalObject latency
    float CPT;
    // Expected kernelDispatch latency
    float KD;
    // Expected FLush latency
    float FW;
    // Expected Total latency;
    float SUM; 
    // Expected subgraph scope invoke latency
    float IVS;
    // Expected merge latency
    float MG;
    // Expected copy(between subgraphs) latency
    float CP;

  } SubgraphCandidate;

  typedef struct PartitioningPlan{
    std::vector<SubgraphCandidate*> subgraphs;
    std::vector<int> partitioning_cadidate;
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

      void SimulateHeightPartitioning(tflite::Subgraph* origin_subgraph, tflite::ResourceType r_type,
                                      int start_node, int end_node);

      float LatencyPredict(Latency_Term term, int x_value);
      // CopyFromExternalObject

      // CopyToExternalObject

      // KernelDispatch

      // Flush and Wait

      // Merge

      // Copy

      // CPU dispatch

      // XNN dispatch

    private:
    tflite::DEVICE_TYPE d_type;
    tflite::MODEL_TYPE m_type;
  };

} // namespace Predictor
