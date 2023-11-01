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
  if(d_type == tflite::DEVICE_TYPE::ODROID){
    partitioning_ratio_gpu = 14;
    partitioning_ratio_cpu = 16;
  }else if(d_type == tflite::DEVICE_TYPE::XAVIER){
    partitioning_ratio_gpu = 17;
    partitioning_ratio_cpu = 13;

  }
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
  
  // Select a partitioning point
  std::vector<std::vector<int>> partitioning_points;

  
  // Create subgraph partitioning plan from seleceted partitioing point 
  std::vector<std::pair<int, int>> new_graphs;
  PartitioningPlan* new_plan = new PartitioningPlan;
  new_plan->partitioning_points = new_graphs;
  // SimulateSubgraphPartitioning with created plan


}

void PartitioningPredictor::SimulateSubgraphPartitioning(
                                    tflite::Subgraph* origin_subgraph,
                                    std::vector<PartitioningPlan*>& new_plan){
  // iterate SimulateHeightPartitioning
  for(int plan_idx=0; plan_idx < new_plan.size(); ++plan_idx){
    std::cout << "Plan idx " << plan_idx << "\n";
    PartitioningPlan* working_plan = new_plan[plan_idx];
    for(int partition_idx=0; partition_idx < working_plan->partitioning_points.size();
        ++partition_idx){
      std::cout << "Partitioning idx " << partition_idx << "\n";
      int start_node = working_plan->partitioning_points[partition_idx].first;
      int end_node = working_plan->partitioning_points[partition_idx].first;
      std::cout << "start : " << start_node << " end : " << end_node << "\n";
      // GPU subgraph predict
      SubgraphCandidate* gpu_subgraph = new SubgraphCandidate;
      gpu_subgraph->start_node = start_node;
      gpu_subgraph->end_node = end_node;
      gpu_subgraph->resource_type = tflite::ResourceType::CO_GPU;
      SimulateHeightPartitioning(origin_subgraph, gpu_subgraph);
      // CPU subgrapg predict 
      SubgraphCandidate* cpu_subgraph = new SubgraphCandidate;
      cpu_subgraph->start_node = start_node;
      cpu_subgraph->end_node = end_node;
      cpu_subgraph->resource_type = tflite::ResourceType::CO_CPU_XNN;
      SimulateHeightPartitioning(origin_subgraph, cpu_subgraph);
    }
  }
  return;
}

void PartitioningPredictor::SimulateHeightPartitioning(
                                  tflite::Subgraph* origin_subgraph,
                                  SubgraphCandidate* new_subgraph){
  // Calculate padding for given subgraph partitioing plan
  tflite::ResourceType resource_type = new_subgraph->resource_type;
  // create new execution plan from new subgraph.
  std::vector<int> execution_plan_;
  int start_node = new_subgraph->start_node;
  int end_node = new_subgraph->end_node;
  int total_execution_plan_size = origin_subgraph->execution_plan().size();
  int start_tensor_idx, end_tensor_idx;
  int nodes_in_subgraph = end_node - start_node + 1;
  if(nodes_in_subgraph < 1){
    std::cout << "nodes_in_subgraph ERROR " << "n";
  }
  for(int i=0; i<nodes_in_subgraph; ++i){
    execution_plan_.push_back(start_node);
    start_node++;
  }
  std::vector<std::pair<TfLiteNode, TfLiteRegistration>> nodes_and_registration_ =
    origin_subgraph->nodes_and_registration();
  CopyTensorsFromContext(origin_subgraph->context());
  // Need to modify simulate height partition.
  // use dummy tensors.
  // no data pointer allocation.
  if(resource_type == tflite::ResourceType::CO_CPU || 
      resource_type == tflite::ResourceType::CO_CPU_XNN){
    int partitioning_ratio = partitioning_ratio_cpu;
    std::vector<int> tensors_already_partitioned;
    std::cout << "Sub subgraph partitioning" << "\n";
    std::cout << "partitioning ratio : " << partitioning_ratio << "\n";
    for(int execution_plan_idx = execution_plan_.size() - 1;
            execution_plan_idx >= 0; execution_plan_idx--){
      bool is_output_feature_same = false;
      // Calculate overlap and zero_padding_overlap.
      // And change dims of given tensor.
      int node_index = execution_plan_[execution_plan_idx];
      int input_tensor_idx, output_tensor_idx;
      std::vector<int> input_tensor_indices;
      std::vector<int> new_input_dim, new_output_dim;
      TfLiteNode& node = nodes_and_registration_[node_index].first;
      const TfLiteRegistration& registration = nodes_and_registration_[node_index].second;
      if(node.inputs->size > 0 && node.outputs->size > 0){
        if(strcmp(GetOpName(registration), "CONCATENATION") == 0 ||
            strcmp(GetOpName(registration), "ADD") == 0){
          // Need to change dims of both inputs for concatenation layer.
          input_tensor_indices.push_back(node.inputs->data[1]);
        } // else just change first input tensor which is actual input of node.
        input_tensor_indices.push_back(node.inputs->data[0]);
        output_tensor_idx = node.outputs->data[0];
        if(execution_plan_idx == 0 && !input_tensor_indices.empty())
          start_tensor_idx = input_tensor_indices[0];
        if(execution_plan_idx == execution_plan_.size() - 1)
          end_tensor_idx = output_tensor_idx;
      }else{
        std::cout << "ERROR Node " << GetOpName(registration) 
              << " input, output size not > 0" << "\n";
        return;
      }
      int zero_padding_overlap = 0;
      int padding_overlap = 0;
      int padding_layer_placeholder = 0; // we have to consider padding layer manually.
      bool is_last_output = false;
      for(int idx=0; idx<input_tensor_indices.size(); ++idx){
        std::cout << "==================================" << "\n";
        std::cout << GetOpName(registration) << "\n";
        input_tensor_idx = input_tensor_indices[idx];
        TfLiteTensor* input_tensor = nullptr;
        TfLiteTensor* output_tensor = nullptr;
        input_tensor = GetTensor(input_tensor_idx);
        output_tensor = GetTensor(output_tensor_idx);
        ///milestone
        int origin_output_height, origin_input_height, new_input_height,
            new_output_height, filter, stride, padding_type, padding_height,
            padding_width, padding_height_offset, padding_width_offset;
        // Get parameters of current looking layer.
        if(!tflite::GetParamsForPartitioning(&registration, &node, origin_subgraph->context(),
                                    filter, stride, padding_type, padding_height,
                                    padding_width, padding_height_offset,
                                    padding_width_offset)){
          std::cout << "GetParamsForPartitioning returned FALSE" << "\n";
          return;
        }
        bool is_output_same = false;
        if(padding_type == 1) // !! Important flag
          is_output_same = true;

        // Divide last node's input and output tensor height.
        origin_output_height = output_tensor->dims->data[1];
        if(execution_plan_idx == execution_plan_.size() - 1){
          is_last_output = true;
          // divide output tensor's dimension in last node.
          new_output_height = std::round((origin_output_height * 0.1) * partitioning_ratio);
        }else{
          is_last_output = false;
          new_output_height = origin_output_height;
        }
        // divide input tensor's dimension in node.
        origin_input_height = input_tensor->dims->data[1];
        bool need_to_be_partitioned = true;
        for(int i=0; i<tensors_already_partitioned.size(); ++i){
          if(tensors_already_partitioned[i] == input_tensor_idx){
            need_to_be_partitioned = false; // in case of tensor already partitioned by primary concate layer.
          }
        }
        if(need_to_be_partitioned)
          new_input_height = std::round((origin_input_height  * 0.1) * partitioning_ratio);
        else  
          new_input_height = origin_input_height;
        // Get parameters(filter size, stride) of node.
        // padding info
        // same == 1
        // valid == 2
        // output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
        // Use different function to calculate zero_padding for Conv and Pool.
        switch (registration.builtin_code) //
        {
          
        case kTfLiteBuiltinPad:
        case kTfLiteBuiltinPadv2:
          padding_layer_placeholder = origin_output_height - origin_input_height;
          if(padding_layer_placeholder < 0){
            std::cout << "padding layer placeholder error" << "\n";
            padding_layer_placeholder = 1;
          }
          break;
        case kTfLiteBuiltinDepthwiseConv2d:
        case kTfLiteBuiltinConv2d:
          zero_padding_overlap = 0;
          if(!is_output_same)
            padding_overlap = tflite::HW::GetOverlapConv(stride, filter, new_input_height,
                                                                  new_output_height);
          break;
        case kTfLiteBuiltinMaxPool2d:
        case kTfLiteBuiltinAveragePool2d:
          std::cout << "Pool padding calc" << "\n";
          zero_padding_overlap = 0; // no 'same' option in pooling layer
          padding_overlap = tflite::HW::GetOverlapPool(stride, filter, new_input_height,
                                                                new_output_height);
          break;
        default:
          break;
        }
        // Calculate padding 
        std::cout << "Params in layer : " << "\n" <<
              " filter : " << filter << "\n" <<
              " stride : " << stride << "\n" <<
              " padding_type : " << padding_type << "\n" <<
              " padding_height : " << padding_height << "\n" <<
              " padding_width : " << padding_width << "\n" <<
              " padding_height_offset : " << padding_height_offset << "\n" <<
              " padding_witdh_offset : " << padding_width_offset << "\n" <<
              " zero padding calculated : " << zero_padding_overlap << "\n";
        int padding_to_add = (padding_overlap - padding_layer_placeholder);
        new_input_height += padding_to_add;
        std::cout << "-----------------------------------" << "\n";
        std::cout << "tensor : " << input_tensor_idx << "\n" <<
                    " origin input_height : " << origin_input_height <<  "\n" <<
                    " origin output_height : " << origin_output_height << "\n" <<
                    " new input height : " << new_input_height << "\n" <<
                    " new output height : " << new_output_height << "\n" <<
                    " overlap to add : " << padding_overlap <<  "\n" <<
                    " added zero padding : " << zero_padding_overlap <<  "\n";
        // Change height of input tensor
        for(int i=0; i<input_tensor->dims->size; ++i){
          new_input_dim.push_back(input_tensor->dims->data[i]);
        }
        if( (strcmp(GetOpName(registration), "PAD") == 0)){
          new_input_dim[1] = origin_output_height + padding_to_add;
        }else if(is_last_output ||
            (new_input_height <= origin_input_height && new_input_height >= origin_output_height)){
          new_input_dim[1] = new_input_height;
          std::cout << "a1 " << new_input_dim[1] << "\n";
        }else if( new_input_height < origin_output_height){ 
          new_input_dim[1] = origin_output_height + padding_to_add;
          std::cout << "a2 "<< new_input_dim[1] << "\n";
        }
        else{
          std::cout << "calculated in height too big " << new_input_height <<
                      " " <<  origin_input_height <<  "\n";
          new_input_dim[1] = new_input_height;
          // return kTfLiteError; (no return)
        }
        // Change height of output tensor if padding is 'same'
        // TODO : Fix duplicate code (newdims... Resize())
        if(is_last_output || (new_input_height == new_output_height)){
          for(int i=0; i<output_tensor->dims->size; ++i){
            new_output_dim.push_back(output_tensor->dims->data[i]);
          }
          if(is_last_output || 
              (new_input_height <= origin_input_height && new_input_height >= origin_output_height)){
            new_output_dim[1] = new_input_height;
            std::cout << "b1 " << new_output_dim[1] << "\n";
          }else if(new_input_height <= origin_input_height && new_input_height < origin_output_height){
            new_output_dim[1] = origin_output_height;
            std::cout << "b2 " << new_output_dim[1] << "\n";
          }
          else if(new_input_height > origin_input_height){
            std::cout << "calculated out height too big " << new_input_height <<
                        " " <<  new_output_dim[1] <<  "\n";
            new_output_dim[1] = new_input_height;
            // return kTfLiteError;
          }
          std::cout << "resize output " << "\n";
          ResizeTensorNaive(output_tensor_idx, new_output_dim);
        }
        std::cout << "resize input " << "\n";
        ResizeTensorNaive(input_tensor_idx, new_input_dim);
        new_input_dim.clear();
        new_output_dim.clear();
      }
      is_output_feature_same = false;
      zero_padding_overlap = 0;
      if(registration.builtin_code == kTfLiteBuiltinConcatenation ||
          registration.builtin_code == kTfLiteBuiltinAdd){
        tensors_already_partitioned.push_back(node.inputs->data[1]);
        tensors_already_partitioned.push_back(node.inputs->data[0]);
      }
    }
    for(int execution_plan_idx = 0; execution_plan_idx<execution_plan_.size();
          ++execution_plan_idx){
      int node_index = execution_plan_[execution_plan_idx];
      TfLiteNode& node = nodes_and_registration_[node_index].first;
      const TfLiteRegistration& registration = nodes_and_registration_[node_index].second;
      if (registration.custom_name != nullptr) {
        printf("Node %3zu %s ", node_index, registration.custom_name);
      } else {
        printf("Node %3zu %s ", node_index, tflite::EnumNamesBuiltinOperator()[registration.builtin_code]);
      }
      TfLiteIntArray* outputs = node.outputs;
      for(int i=0; i<outputs->size; ++i){
        int tensor_index = outputs->data[i];
        TfLiteTensor* tensor_ = GetTensor(tensor_index);
        printf("Tensor %3zu %10zu bytes (%4.1f MB) ", tensor_index,
            tensor_->bytes,
            (static_cast<float>(tensor_->bytes) / (1 << 20)));
        for(int k=0; k<tensor_->dims->size; k++){
          std::cout << tensor_->dims->data[k] << " ";
        } 
        std::cout << "\n";
      }
    }
    
  }else{

    // Below is main interpreter side.
    int partitioning_ratio = partitioning_ratio_gpu;
    std::cout << "Main subgraph partitioning" << "\n";
    std::cout << "partitioning ratio : " << partitioning_ratio << "\n";
    std::vector<int> tensors_already_partitioned;
    for(int execution_plan_idx = execution_plan_.size() - 1;
            execution_plan_idx >= 0; execution_plan_idx--){
      bool is_output_feature_same = false;
      // Calculate overlap and zero_padding_overlap.
      // And change dims of given tensor.
      int node_index = execution_plan_[execution_plan_idx];
      int input_tensor_idx, output_tensor_idx;
      std::vector<int> input_tensor_indices;
      std::vector<int> new_input_dim, new_output_dim;
      TfLiteNode& node = nodes_and_registration_[node_index].first;
      const TfLiteRegistration& registration = nodes_and_registration_[node_index].second;
      if(node.inputs->size > 0 && node.outputs->size > 0){
        if(strcmp(GetOpName(registration), "CONCATENATION") == 0
            || strcmp(GetOpName(registration), "ADD") == 0){
          // Need to change dims of both inputs for concatenation layer.
          input_tensor_indices.push_back(node.inputs->data[1]);
        } // else just change first input tensor which is actual input of node.
        input_tensor_indices.push_back(node.inputs->data[0]);
        output_tensor_idx = node.outputs->data[0];
        if(execution_plan_idx == 0 && !input_tensor_indices.empty())
          start_tensor_idx = input_tensor_indices[0];
        if(execution_plan_idx == execution_plan_.size() - 1)
          end_tensor_idx = output_tensor_idx;
      }else{
        std::cout << "ERROR Node " << GetOpName(registration) 
              << " input, output size not > 0" << "\n";
        return;
      }
      int zero_padding_overlap = 0;
      int padding_overlap = 0;
      int padding_layer_placeholder = 0; // we have to consider padding layer manually.
      bool is_last_output = false;
      for(int idx=0; idx<input_tensor_indices.size(); ++idx){
        std::cout << "==================================" << "\n";
        std::cout << GetOpName(registration) << "\n";
        input_tensor_idx = input_tensor_indices[idx];
        TfLiteTensor* input_tensor = nullptr;
        TfLiteTensor* output_tensor = nullptr;
        input_tensor = GetTensor(input_tensor_idx);
        output_tensor = GetTensor(output_tensor_idx);
        int origin_output_height, origin_input_height, new_input_height,
            new_output_height, filter, stride, padding_type, padding_height,
            padding_width, padding_height_offset, padding_width_offset;
        // Get parameters(filter size, stride) of node.
        if(!tflite::GetParamsForPartitioning(&registration, &node, origin_subgraph->context(),
                                    filter, stride, padding_type, padding_height,
                                    padding_width, padding_height_offset,
                                    padding_width_offset)){
          std::cout << "GetParamsForPartitioning returned FALSE" << "\n";
          return;
        }
        bool need_to_be_partitioned = true;

        // Check for 1x1 conv and output feature type ('same', 'valid')
        if(padding_type == 1){
          if(stride == 1 && filter == 1){
            is_output_feature_same = false; // case of 1x1 conv
          }else{
            is_output_feature_same = true;
          }  
        }

        // First, divide last node's input and output tensor height.
        origin_output_height = output_tensor->dims->data[1];
        if(execution_plan_idx == execution_plan_.size() - 1){
          is_last_output = true;
          // divide output tensor's dimension in last node.
          new_output_height = std::round((origin_output_height * 0.1) * partitioning_ratio);
        }else{
          is_last_output = false;
          new_output_height = origin_output_height;
        }
        // divide input tensor's dimension in node.
        origin_input_height = input_tensor->dims->data[1];
        for(int i=0; i<tensors_already_partitioned.size(); ++i){
          if(tensors_already_partitioned[i] == input_tensor_idx){
            // in case of tensor already partitioned by primary concate layer.
            need_to_be_partitioned = false; 
          }
        }
        if(need_to_be_partitioned)
          if(is_output_feature_same){ // 'same'
            new_input_height = tflite::HW::GetInputHeightofSameFeatureConv(new_output_height, stride);
          }else{ // 'valid'
            new_input_height = std::round((origin_input_height  * 0.1) * partitioning_ratio);
          }
        else{ // no need to be partitioned (input height already partitoned)
          new_input_height = origin_input_height;
        }  
        
        switch (registration.builtin_code) 
        {
        case kTfLiteBuiltinPad:
        case kTfLiteBuiltinPadv2:
          padding_layer_placeholder = origin_output_height - origin_input_height;
          if(padding_layer_placeholder < 0){
            std::cout << "padding layer placeholder error" << "\n";
            padding_layer_placeholder = 0;
          }
          break;
        case kTfLiteBuiltinDepthwiseConv2d:
        case kTfLiteBuiltinConv2d:
          std::cout << "Conv padding calc" << "\n";
          // padding info
          // same == 1
          // valid == 2
          if(padding_type == 1){
            // if(new_input_height != origin_output_height && !is_last_output)
            //   new_input_height = origin_output_height;
            zero_padding_overlap = tflite::HW::GetZeroPaddingConv(stride, filter, new_input_height,
                                                                      new_output_height);
          }
          padding_overlap = tflite::HW::GetOverlapConv(stride, filter, new_input_height,
                                                                new_output_height);
          break;
        case kTfLiteBuiltinMaxPool2d:
        case kTfLiteBuiltinAveragePool2d:
          std::cout << "Pool padding calc" << "\n";
          zero_padding_overlap = 0; // no 'same' option in pooling layer
          padding_overlap = tflite::HW::GetOverlapPool(stride, filter, new_input_height,
                                                                new_output_height);
          break;
        default:
          break;
        }
      
        std::cout << "Params in layer : " << "\n" <<
                    " filter : " << filter << "\n" <<
                    " stride : " << stride << "\n" <<
                    " padding_type : " << padding_type << "\n" <<
                    " padding_height : " << padding_height << "\n" <<
                    " padding_width : " << padding_width << "\n" <<
                    " padding_height_offset : " << padding_height_offset << "\n" <<
                    " padding_witdh_offset : " << padding_width_offset << "\n" <<
                    " zero padding calculated : " << zero_padding_overlap << "\n";
        // Calculate padding 
        
        if(zero_padding_overlap != 0)
          padding_overlap = 0;
        int padding_to_add = (padding_overlap + zero_padding_overlap - padding_layer_placeholder);
        new_input_height += padding_to_add;
        if(is_output_feature_same)
          new_output_height += padding_to_add;
        std::cout << "-----------------------------------" << "\n";
        std::cout << "tensor : " << input_tensor_idx << "\n" <<
                    " origin input_height : " << origin_input_height <<  "\n" <<
                    " origin output_height : " << origin_output_height << "\n" <<
                    " new input height : " << new_input_height << "\n" <<
                    " new output height : " << new_output_height << "\n" <<
                    " overlap to add : " << padding_overlap <<  "\n" <<
                    " added zero padding : " << zero_padding_overlap <<  "\n";

        // Change height of input tensor
        for(int i=0; i<input_tensor->dims->size; ++i){
          new_input_dim.push_back(input_tensor->dims->data[i]);
        }
        if( (strcmp(GetOpName(registration), "PAD") == 0)){
          new_input_dim[1] = origin_output_height + padding_to_add;
        }else if(is_last_output ||
            (new_input_height <= origin_input_height && new_input_height >= origin_output_height)){
          new_input_dim[1] = new_input_height;
          std::cout << "a1 " << new_input_dim[1] << "\n";
        }else if( new_input_height < origin_output_height){ 
          new_input_dim[1] = origin_output_height + padding_to_add;
          std::cout << "a2 "<< new_input_dim[1] << "\n";
        }
        else{ // TODO : Need better logic here
          std::cout << "calculated in height too big " << new_input_height <<
                      " " <<  origin_input_height <<  "\n";
          new_input_dim[1] = new_input_height;
          // return kTfLiteError; (no return)
        }
        // Change height of output tensor if padding is 'same'
        // TODO : Fix duplicate code (newdims... Resize())
        // worked in yolo? ()
        // if(is_last_output || (is_output_feature_same && new_input_height == new_output_height)){
        if(is_last_output || is_output_feature_same){
          for(int i=0; i<output_tensor->dims->size; ++i){
            new_output_dim.push_back(output_tensor->dims->data[i]);
          }
          if(is_last_output || 
              (new_input_height <= origin_input_height && new_input_height >= origin_output_height)){
            new_output_dim[1] = new_input_height;
            std::cout << "b1 " << new_output_dim[1] << "\n";
          }else if(new_input_height <= origin_input_height && new_input_height < origin_output_height){
            new_output_dim[1] = origin_output_height;
            std::cout << "b2 " << new_output_dim[1] << "\n";
          }
          else if(new_input_height > origin_input_height){ // TODO : Need better logic here
            std::cout << "calculated out height too big " << new_input_height <<
                        " " <<  new_output_dim[1] <<  "\n";
            new_output_dim[1] = new_input_height;
            // return kTfLiteError; (no return)
          }
          std::cout << "resize output (dim" << new_output_dim.size() << ")\n";
          ResizeTensorNaive(output_tensor_idx, new_output_dim);
        }
        std::cout << "resize input (dim" << new_input_dim.size() << ")\n";
        ResizeTensorNaive(input_tensor_idx, new_input_dim);
        new_input_dim.clear();
        new_output_dim.clear();
      }
      // See execution plan from current watching exectuion plan + 1 to end.
      // Then propagate dims which changed by zero_padding_overlap.
      // TODO : Fix duplicate code (consider better implementation)
      if(is_output_feature_same){ // Case of padding type 'same'
        for(int execution_plan_idx_inner = execution_plan_idx+1;
            execution_plan_idx_inner < execution_plan_.size(); execution_plan_idx_inner++){
          std::cout << "Propagate zero_padding_overlap "<< zero_padding_overlap <<
                      " to node " << execution_plan_idx_inner << "\n";
          int node_index = execution_plan_[execution_plan_idx_inner];
          int input_tensor_idx_, output_tensor_idx_;
          std::vector<int> input_tensor_indices_;
          TfLiteNode& node = nodes_and_registration_[node_index].first;
          const TfLiteRegistration& registration = nodes_and_registration_[node_index].second;
          if(strcmp(GetOpName(registration), "CONCATENATION") == 0 ||
              strcmp(GetOpName(registration), "ADD") == 0){
            // Need to change dims of both inputs for concatenation layer.
            input_tensor_indices_.push_back(node.inputs->data[1]);
          } // else just change first input tensor which is actual input of node.
          input_tensor_indices_.push_back(node.inputs->data[0]);
          output_tensor_idx_ = node.outputs->data[0];    
          TfLiteTensor* input_tensor;
          TfLiteTensor* output_tensor;
          // change input tensor dim. (add zero_padding_overlap)
          for(int idx=0; idx < input_tensor_indices_.size(); ++idx){
            std::cout << "tensor : " << input_tensor_indices_[idx] << "\n";
            input_tensor = GetTensor(input_tensor_indices_[idx]);
            std::vector<int> new_dim;
            for(int i=0; i<input_tensor->dims->size; ++i){
              new_dim.push_back(input_tensor->dims->data[i]);
            }
            if(new_output_dim[1] > new_dim[1]){
              new_dim[1] = new_dim[1] + zero_padding_overlap;
            }
            std::cout << "resize input zero padding (dim" << new_dim.size() << ")\n";
            ResizeTensorNaive(input_tensor_indices_[idx], new_dim);
          }
          // change output tensor dim. (add zero_padding_overlap)
          output_tensor = GetTensor(output_tensor_idx_);
          std::vector<int> new_dim_;
          for(int i=0; i<output_tensor->dims->size; ++i){
            new_dim_.push_back(output_tensor->dims->data[i]);
          }
          new_dim_[1] = new_dim_[1] + zero_padding_overlap;
          std::cout << "resize output zero_padding (dim" << new_dim_.size() << ")\n";
          ResizeTensorNaive(output_tensor_idx_, new_dim_);
          std::cout << "dddd22" << "\n";
        }
      }
      new_input_dim.clear();
      new_output_dim.clear();
      is_output_feature_same = false;
      zero_padding_overlap = 0;
      if(registration.builtin_code == kTfLiteBuiltinConcatenation ||
                      registration.builtin_code == kTfLiteBuiltinAdd){
        tensors_already_partitioned.push_back(node.inputs->data[1]);
        tensors_already_partitioned.push_back(node.inputs->data[0]);
      }
    }
    std::cout << "Height partitioning done" << "\n";
    for(int execution_plan_idx = 0; execution_plan_idx<execution_plan_.size();
          ++execution_plan_idx){
      int node_index = execution_plan_[execution_plan_idx];
      TfLiteNode& node = nodes_and_registration_[node_index].first;
      const TfLiteRegistration& registration = nodes_and_registration_[node_index].second;
      if (registration.custom_name != nullptr) {
        printf("Node %3zu %s ", node_index, registration.custom_name);
      } else {
        printf("Node %3zu %s ", node_index, tflite::EnumNamesBuiltinOperator()[registration.builtin_code]);
      }
      TfLiteIntArray* outputs = node.outputs;
      for(int i=0; i<outputs->size; ++i){
        int tensor_index = outputs->data[i];
        TfLiteTensor* tensor_ = GetTensor(tensor_index);
        printf("Tensor %3zu %10zu bytes (%4.1f MB) ", tensor_index,
            tensor_->bytes,
            (static_cast<float>(tensor_->bytes) / (1 << 20)));
        for(int k=0; k<tensor_->dims->size; k++){
          std::cout << tensor_->dims->data[k] << " ";
        } 
        std::cout << "\n";
      }
    }  
  }
  // Set input and output dims for given subgraph
  TfLiteTensor* input_tensor = GetTensor(start_tensor_idx);
  for(int i=0;i<input_tensor->dims->size; ++i){
    new_subgraph->in_dim.push_back(input_tensor->dims->data[i]);
    new_subgraph->input_size *= input_tensor->dims->data[i];
  }
  TfLiteTensor* output_tensor = GetTensor(end_tensor_idx);
  for(int i=0;i<output_tensor->dims->size; ++i){
    new_subgraph->out_dim.push_back(output_tensor->dims->data[i]);
    new_subgraph->output_size *= output_tensor->dims->data[i];
  }
  // Calculcate latency terms for given subgraph partitiong plan
  // Get Flops
  GetTotalFlopsforGivenSubgraph(origin_subgraph, new_subgraph);
  // Get latency terms
  if(end_node == total_execution_plan_size-1){ // means that this is final subgraph
    //if gpu
      //CPF, CPT, KD, FW, SUM, IVS, MG

    //if cpu
      //SUM, IVS, MG
  }else if(start_node == 0){ // means that this is first subgraph
    //if gpu
      //CPF, CPT, KD, FW, SUM, IVS, MG
    //if cpu
      //SUM, IVS, MG
  }else{
    //if gpu
      //CPF, CPT, KD, FW, SUM, IVS, MG
    //if cpu
      //SUM, IVS, MG
  }
  return;
}

void PartitioningPredictor::CopyTensorsFromContext(TfLiteContext* context){
  for(int i=0; i<copied_tensors.size(); ++i){
    TfLiteIntArrayFree(copied_tensors[i]->dims);
  }
  copied_tensors.clear();

  int num_tensors = context->tensors_size;
  std::cout << "Copy " << num_tensors << " from original context" << "\n";
  for(int i=0; i<num_tensors; ++i){
    TfLiteTensor* new_tensor = CopyNoBufferTensor(context->tensors[i]);
    copied_tensors.push_back(new_tensor);
  }
  std::cout << "Copied " << copied_tensors.size() << " from original context" << "\n";
}

TfLiteTensor* PartitioningPredictor::CopyNoBufferTensor(TfLiteTensor& tensor){
  TfLiteTensor* new_tensor = new TfLiteTensor;
  TfLiteIntArrayEqualsArray(new_tensor->dims, tensor.dims->size,
                            tensor.dims->data);
  new_tensor->bytes = tensor.bytes;
}

TfLiteTensor* PartitioningPredictor::GetTensor(int tensor_idx){
  if(tensor_idx <= copied_tensors.size())
    return copied_tensors[tensor_idx];
  else{
    std::cout << "No tensor " << tensor_idx << " in copied_tensors" << "\n";
    return nullptr;
  }
}

void PartitioningPredictor::ResizeTensorNaive(int tensor_idx, std::vector<int>& new_dim){
  TfLiteTensor* working_tensor = GetTensor(tensor_idx);
  TfLiteIntArray* new_dim_ = tflite::ConvertVectorToTfLiteIntArray(new_dim);
  TfLiteIntArrayFree(working_tensor->dims);
  working_tensor->dims = new_dim_;
}

void PartitioningPredictor::GetTotalFlopsforGivenSubgraph(tflite::Subgraph* origin_subgraph,
                                                          SubgraphCandidate* new_subgraph){
  float total_flops = 0;
  int start_node = new_subgraph->start_node;
  int end_node = new_subgraph->end_node;
  std::vector<std::pair<TfLiteNode, TfLiteRegistration>> nodes_and_registration_ =
    origin_subgraph->nodes_and_registration();
  for(int node_idx = start_node; node_idx < end_node+1; ++node_idx){
    TfLiteNode& node = nodes_and_registration_[node_idx].first;
    TfLiteRegistration& reg = nodes_and_registration_[node_idx].second;
    TfLiteIntArray* outputs = node.outputs;
    TfLiteIntArray* inputs = node.inputs;
    int origin_output_height, origin_input_height, new_input_height,
      new_output_height, filter, stride, padding_type, padding_height,
      padding_width, padding_height_offset, padding_width_offset;
    TfLiteContext* context_ = origin_subgraph->context();
    if(strcmp(GetOpName(reg), "CONV_2D") == 0 || strcmp(GetOpName(reg), "DEPTHWISE_CONV_2D") == 0){
      tflite::GetParamsForPartitioning(&reg, &node, context_,
                                  filter, stride, padding_type, padding_height, padding_width, 
                                  padding_height_offset, padding_width_offset);
    }
    for(int i=0; i<outputs->size; ++i){
      int output_tensor_idx = outputs->data[i];
      int input_tensor_idx = inputs->data[0];
      float flops = 0;
      TfLiteTensor* tensor = GetTensor(output_tensor_idx);
      TfLiteTensor* i_tensor = GetTensor(input_tensor_idx);
      if(strcmp(GetOpName(reg), "CONV_2D") == 0){
        double mac = tensor->dims->data[1] * tensor->dims->data[2] * 
                     tensor->dims->data[3] * i_tensor->dims->data[3] * filter * filter;
        flops = 2*mac/1000000;
        total_flops += flops;
        printf("\033[0;31mFLOPs : %.1f\033[0m\n", flops);
      }
      if(strcmp(GetOpName(reg), "DEPTHWISE_CONV_2D") == 0){
        double mac = tensor->dims->data[1] * tensor->dims->data[2] * 
                     tensor->dims->data[3] * filter * filter;
        flops = 2*mac/1000000;
        total_flops += flops;
        printf("\033[0;31mFLOPs : %.1f\033[0m\n", flops);
      }
    }
  }
  new_subgraph->flops = total_flops;
}

float PartitioningPredictor::LatencyPredict(Latency_Term term,
                                            tflite::ResourceType r_type,
                                            int x_value){ 
  float output;
  switch (term)
  {
  case Latency_Term::CPF :
    if(d_type == tflite::DEVICE_TYPE::ODROID)
      switch (m_type){
        case tflite::MODEL_TYPE::EFFICIENTNET:
          output = (6e-09) * static_cast<float>(x_value) + 0.00018;
          break;
        case tflite::MODEL_TYPE::MOBILENET:
          break;
        case tflite::MODEL_TYPE::YOLO:
          break;
        default:
          break;
      }
    else if(d_type == tflite::DEVICE_TYPE::XAVIER)
      switch (m_type){
        case tflite::MODEL_TYPE::EFFICIENTNET:
          output = (1e-09) * static_cast<float>(x_value) + 0.00014;
          break;
        case tflite::MODEL_TYPE::MOBILENET:
          output = (1e-09) * static_cast<float>(x_value) + 0.00011;
          break;
        case tflite::MODEL_TYPE::YOLO:
          break;
        default:
          break;
      }
    break;
  case Latency_Term::CPT :
    if(d_type == tflite::DEVICE_TYPE::ODROID)
      switch (m_type){
          output = (8e-09) * static_cast<float>(x_value) + 0.00055;
        case tflite::MODEL_TYPE::EFFICIENTNET:
          break;
        case tflite::MODEL_TYPE::MOBILENET:
          break;
        case tflite::MODEL_TYPE::YOLO:
          break;
        default:
          break;
      }
    else if(d_type == tflite::DEVICE_TYPE::XAVIER)
      switch (m_type){
        case tflite::MODEL_TYPE::EFFICIENTNET:
          output = (2e-09) * static_cast<float>(x_value) + 0.00026;
          break;
        case tflite::MODEL_TYPE::MOBILENET:
          output = (2e-09) * static_cast<float>(x_value) + 0.0003;
          break;
        case tflite::MODEL_TYPE::YOLO:
          break;
        default:
          break;
      }
    /* code */
    break;
  case Latency_Term::KD :
    if(d_type == tflite::DEVICE_TYPE::ODROID)
      switch (m_type){
        case tflite::MODEL_TYPE::EFFICIENTNET:
          output = (9.8151e-05) * static_cast<float>(x_value) + 0.00046;
          break;
        case tflite::MODEL_TYPE::MOBILENET:
          break;
        case tflite::MODEL_TYPE::YOLO:
          break;
        default:
          break;
      }
    else if(d_type == tflite::DEVICE_TYPE::XAVIER)
      switch (m_type){
        case tflite::MODEL_TYPE::EFFICIENTNET:
          output = (8.395e-06) * static_cast<float>(x_value) + 0.00043;
          break;
        case tflite::MODEL_TYPE::MOBILENET:
          output = (4.545e-06) * static_cast<float>(x_value) + 0.00079;
          break;
        case tflite::MODEL_TYPE::YOLO:
          break;
        default:
          break;
      }
    /* code */
    break;
  case Latency_Term::FW :
    if(d_type == tflite::DEVICE_TYPE::ODROID){
      output = 0;
    }
    else if(d_type == tflite::DEVICE_TYPE::XAVIER)
      switch (m_type){
        case tflite::MODEL_TYPE::EFFICIENTNET:
          output = (1e-09) * static_cast<float>(x_value) + (1e-04);
          break;
        case tflite::MODEL_TYPE::MOBILENET:
          output = (1e-09) * static_cast<float>(x_value) + 0.00012;
          break;
        case tflite::MODEL_TYPE::YOLO:
          break;
        default:
          break;
      }
    /* code */
    break;
  case Latency_Term::IVS :
    if(d_type == tflite::DEVICE_TYPE::ODROID)
      switch (m_type){
        case tflite::MODEL_TYPE::EFFICIENTNET:
          if(r_type == tflite::ResourceType::CO_CPU_XNN){
            
          }else{
            output = (0.00010528) * static_cast<float>(x_value) + 0.00051;
          }
          break;
        case tflite::MODEL_TYPE::MOBILENET:
          if(r_type == tflite::ResourceType::CO_CPU_XNN){

          }else { }
          break;
        case tflite::MODEL_TYPE::YOLO:
          if(r_type == tflite::ResourceType::CO_CPU_XNN){

          } else { }
          break;
        default:
          break;
      }
    else if(d_type == tflite::DEVICE_TYPE::XAVIER)
      switch (m_type){
        case tflite::MODEL_TYPE::EFFICIENTNET:
          if(r_type == tflite::ResourceType::CO_CPU_XNN){
            
          }else{ 
            output = (9.948e-06) * static_cast<float>(x_value) + 0.00119;
          }
          break;
        case tflite::MODEL_TYPE::MOBILENET:
          if(r_type == tflite::ResourceType::CO_CPU_XNN){

          }else { 
            output = (4.089e-06) * static_cast<float>(x_value) + 0.00198;
          }
          break;
        case tflite::MODEL_TYPE::YOLO:
          if(r_type == tflite::ResourceType::CO_CPU_XNN){

          } else { }
        default:
          break;
      }
    /* code */
    break;
  case Latency_Term::MG :
    if(d_type == tflite::DEVICE_TYPE::ODROID){
      output = (2e-09) * static_cast<float>(x_value) + (3e-05);
    }
    else if(d_type == tflite::DEVICE_TYPE::XAVIER){
      output = (1e-09) * static_cast<float>(x_value) + (1e-05);
    }
    /* code */
    break;
  case Latency_Term::CP :
    if(d_type == tflite::DEVICE_TYPE::ODROID){
      output = (2e-09) * static_cast<float>(x_value);
      }
    else if(d_type == tflite::DEVICE_TYPE::XAVIER){
      output = (1e-09) * static_cast<float>(x_value) + (1e-05);
    }
    break;
  default:
    break;
  }
  return output;
}


} // namespace Predictor