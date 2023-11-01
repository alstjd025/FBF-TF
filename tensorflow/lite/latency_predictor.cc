#include "tensorflow/lite/latency_predictor.h"

namespace Predictor{
PartitioningPredictor::PartitioningPredictor(tflite::DEVICE_TYPE device_type_,
                                             tflite::MODEL_TYPE model_type_):
                                              d_type(device_type_),
                                              m_type(model_type_)
                                              {
  std::cout << "Initialized Partitioning Predictor" << "\n";
}

void PartitioningPredictor::StartPredictor(tflite::Subgraph* origin_subgraph){
  std::vector<int> partitioning_candidates;
  switch (m_type)
  {
  case tflite::MODEL_TYPE::EFFICIENTNET :
    partitioning_candidates = efficientnet_partitioning_cadidates;
    break;
  case tflite::MODEL_TYPE::MOBILENET :
    partitioning_candidates = mobilenet_partitioning_cadidates;
    break;
  case tflite::MODEL_TYPE::YOLO :
    partitioning_candidates = yolo_partitioning_cadidates;
    break;
  default:
    break;
  }
  

}

void PartitioningPredictor::SimulateSubgraphPartitioning(
                                    tflite::Subgraph* origin_subgraph,
                                    std::vector<PartitioningPlan*>& new_plan){
  // iterate SimulateHeightPartitioning
  return;
}

void PartitioningPredictor::SimulateHeightPartitioning(
                                  tflite::Subgraph* origin_subgraph, tflite::ResourceType r_type,
                                  int start_node, int end_node){
  
  return;
}

float PartitioningPredictor::LatencyPredict(Latency_Term term, int x_value){ 
  float output;
  switch (term)
  {
  case Latency_Term::CPF :
    /* code */
    break;
  case Latency_Term::CPT :
    /* code */
    break;
  case Latency_Term::KD :
    /* code */
    break;
  case Latency_Term::FW :
    /* code */
    break;
  case Latency_Term::IVS :
    /* code */
    break;
  case Latency_Term::MG :
    /* code */
    break;
  case Latency_Term::CP :
    /* code */
    break;
  default:
    break;
  }

  return output;
}


} // namespace Predictor