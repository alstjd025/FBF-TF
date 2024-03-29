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
#include "tensorflow/lite/optional_debug_tools.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"
namespace tflite {

// cascade operator overloading for debug message.
std::ostream& operator<<(std::ostream& out, const tflite::ResourceType value){
  const char* s = 0;
#define PROCESS_VAL(p) case(p): s = #p; break;
  switch(value){
    PROCESS_VAL(CPU);     
    PROCESS_VAL(GPU);     
    PROCESS_VAL(CO_CPU);
    PROCESS_VAL(CO_GPU);
    PROCESS_VAL(CPU_XNN);
    PROCESS_VAL(CO_CPU_XNN);
    PROCESS_VAL(NONE);
  }
#undef PROCESS_VAL
  return out << s;
}

// cascade operator overloading for debug message.
std::ostream& operator<<(std::ostream& out, const tflite::PartitioningType value){
  const char* s = 0;
#define PROCESS_VAL(p) case(p): s = #p; break;
  switch(value){
    PROCESS_VAL(NO_PARTITIONING);     
    PROCESS_VAL(HEIGHT_PARTITIONING);     
    PROCESS_VAL(CHANNEL_PARTITIONING);
  }
#undef PROCESS_VAL
  return out << s;
}

void PrintIntVector(const std::vector<int>& v) {
  for (const auto& it : v) {
    printf(" %d", it);
  }
  printf("\n");
}

void PrintTfLiteIntVector(const TfLiteIntArray* v) {
  if (!v) {
    printf(" (null)\n");
    return;
  }
  for (int k = 0; k < v->size; k++) {
    printf(" %d", v->data[k]);
  }
  printf("\n");
}

const char* TensorTypeName(TfLiteType type) {
  switch (type) {
    case kTfLiteNoType:
      return "kTfLiteNoType";
    case kTfLiteFloat32:
      return "kTfLiteFloat32";
    case kTfLiteInt32:
      return "kTfLiteInt32";
    case kTfLiteUInt8:
      return "kTfLiteUInt8";
    case kTfLiteInt8:
      return "kTfLiteInt8";
    case kTfLiteInt64:
      return "kTfLiteInt64";
    case kTfLiteString:
      return "kTfLiteString";
    case kTfLiteBool:
      return "kTfLiteBool";
    case kTfLiteInt16:
      return "kTfLiteInt16";
    case kTfLiteComplex64:
      return "kTfLiteComplex64";
    case kTfLiteComplex128:
      return "kTfLiteComplex128";
    case kTfLiteFloat16:
      return "kTfLiteFloat16";
    case kTfLiteFloat64:
      return "kTfLiteFloat64";
  }
  return "(invalid)";
}

const char* AllocTypeName(TfLiteAllocationType type) {
  switch (type) {
    case kTfLiteMemNone:
      return "kTfLiteMemNone";
    case kTfLiteMmapRo:
      return "kTfLiteMmapRo";
    case kTfLiteDynamic:
      return "kTfLiteDynamic";
    case kTfLiteArenaRw:
      return "kTfLiteArenaRw";
    case kTfLiteArenaRwPersistent:
      return "kTfLiteArenaRwPersistent";
    case kTfLitePersistentRo:
      return "kTfLitePersistentRo";
    case kTfLiteCustom:
      return "kTfLiteCustom";
  }
  return "(invalid)";
}

// Prints a dump of what tensors and what nodes are in the interpreter.
void PrintInterpreterState(Interpreter* interpreter) {
  printf("Interpreter has %zu tensors and %zu nodes\n",
         interpreter->tensors_size(), interpreter->nodes_size());
  printf("Inputs:");
  PrintIntVector(interpreter->inputs());
  printf("Outputs:");
  PrintIntVector(interpreter->outputs());
  printf("\n");
  for (size_t tensor_index = 0; tensor_index < interpreter->tensors_size();
       tensor_index++) {
    TfLiteTensor* tensor = interpreter->tensor(static_cast<int>(tensor_index));
    printf("Tensor %3zu %-20s %10s %15s %10zu bytes (%4.1f MB) ", tensor_index,
           tensor->name, TensorTypeName(tensor->type),
           AllocTypeName(tensor->allocation_type), tensor->bytes,
           (static_cast<float>(tensor->bytes) / (1 << 20)));
    PrintTfLiteIntVector(tensor->dims);
  }
  printf("\n");
  for (size_t node_index = 0; node_index < interpreter->nodes_size();
       node_index++) {
    const std::pair<TfLiteNode, TfLiteRegistration>* node_and_reg =
        interpreter->node_and_registration(static_cast<int>(node_index));
    const TfLiteNode& node = node_and_reg->first;
    const TfLiteRegistration& reg = node_and_reg->second;
    if (reg.custom_name != nullptr) {
      printf("Node %3zu Operator Custom Name %s\n", node_index,
             reg.custom_name);
    } else {
      printf("Node %3zu Operator Builtin Code %3d %s\n", node_index,
             reg.builtin_code, EnumNamesBuiltinOperator()[reg.builtin_code]);
    }
    printf("  Inputs:");
    PrintTfLiteIntVector(node.inputs);
    printf("  Outputs:");
    PrintTfLiteIntVector(node.outputs);
    if (node.intermediates && node.intermediates->size) {
      printf("  Intermediates:");
      PrintTfLiteIntVector(node.intermediates);
    }
    if (node.temporaries && node.temporaries->size) {
      printf("  Temporaries:");
      PrintTfLiteIntVector(node.temporaries);
    }
  }
}

// Minsung
// Prints a dump of what tensors and what nodes are in the interpreter.
void PrintInterpreterStateV2(Interpreter* interpreter) {
  int subgraph_size = interpreter->subgraphs_size();
  printf("Interpreter has %d subgraphs\n", subgraph_size);
  //interpreter->PrintSubgraphInfo();
  for(int subgraph_index=0; subgraph_index < subgraph_size; ++subgraph_index){
    std::cout << "======================================" << "\n";
    int subgraph_id = interpreter->subgraph(subgraph_index)->GetGraphid();
    int tensor_size = interpreter->subgraph_id(subgraph_id)->tensors_size();
    int node_size = interpreter->nodes_size(subgraph_id);
    printf("Subgraph ID %d has %d tensors and %d nodes\n", subgraph_id,
        tensor_size, node_size);
    printf("Model ID : %d\n", interpreter->subgraph_id(subgraph_id)->GetModelid());
    std::cout << "Resource type : " 
          << interpreter->subgraph_id(subgraph_id)->GetResourceType() << "\n";
    std::cout<< "Partitioning type : " 
          << interpreter->subgraph_id(subgraph_id)->GetPartitioningType() << "\n";
    if(interpreter->subgraph_id(subgraph_id)->IsInvokable())
      std::cout << "State : Invokable" << "\n";
    else
      std::cout << "State : Not Invokable" << "\n";
    for (size_t node_index = 0; node_index < node_size;
        node_index++) {
      const std::pair<TfLiteNode, TfLiteRegistration>* node_and_reg =
          interpreter->node_and_registration(static_cast<int>(node_index), subgraph_id);
      const TfLiteNode& node = node_and_reg->first;
      const TfLiteRegistration& reg = node_and_reg->second;
      if (reg.custom_name != nullptr) {
        printf("Node %3zu Operator Custom Name %s\n", node_index,
              reg.custom_name);
      } else {
        printf("Node %3zu Operator Builtin Code %3d %s\n", node_index,
              reg.builtin_code, EnumNamesBuiltinOperator()[reg.builtin_code]);
      }
      printf("  Inputs:");
      PrintTfLiteIntVector(node.inputs);
      printf("  Outputs:");
      PrintTfLiteIntVector(node.outputs);
      if (node.intermediates && node.intermediates->size) {
        printf("  Intermediates:");
        PrintTfLiteIntVector(node.intermediates);
      }
      if (node.temporaries && node.temporaries->size) {
        printf("  Temporaries:");
        PrintTfLiteIntVector(node.temporaries);
      }
    }
    std::cout << "======================================" << "\n";
    printf("Inputs:");
    PrintIntVector(interpreter->inputs(subgraph_id));
    printf("Outputs:");
    PrintIntVector(interpreter->outputs(subgraph_id));
    printf("\n");
    printf("Tensor size : %d\n", tensor_size);
    for (size_t tensor_index = 0; tensor_index < tensor_size-1;
       tensor_index++) {
      TfLiteTensor* tensor = interpreter->tensor(subgraph_id, static_cast<int>(tensor_index));
      printf("Tensor %3zu %-20s %10s %15s %10zu bytes (%4.1f MB) ", tensor_index,
           tensor->name, TensorTypeName(tensor->type),
           AllocTypeName(tensor->allocation_type), tensor->bytes,
           (static_cast<float>(tensor->bytes) / (1 << 20)));
      PrintTfLiteIntVector(tensor->dims);
    }
    printf("\n");
  }
}

// Minsung
// Prints a dump of what tensors and what nodes are in the interpreter.
// Simplified version of PrintInterpreterStateV2
void PrintInterpreterStateV3(Interpreter* interpreter) {
  int subgraph_size = interpreter->subgraphs_size();
  int TOTAL_buffer_size = 0;
  printf("Interpreter has %d subgraphs\n", subgraph_size);
  interpreter->PrintSubgraphInfo();
  for(int subgraph_index=0; subgraph_index < subgraph_size; ++subgraph_index){
    std::cout << "======================================" << "\n";
    int subgraph_id = interpreter->subgraph(subgraph_index)->GetGraphid();
    int tensor_size = interpreter->subgraph_id(subgraph_id)->tensors_size();
    int node_size = interpreter->nodes_size(subgraph_id);
    printf("Subgraph ID %d has %d tensors and %d nodes\n", subgraph_id,
        tensor_size, node_size);
    printf("RW buffer size : %dbytes\n",interpreter->subgraph_id(subgraph_id)->GetArenaRWBufferSize());
    printf("Persistent buffer size : %dbytes\n",interpreter->subgraph_id(subgraph_id)->GetArenaPersistentBufferSize());
    int overall_buffer_size = interpreter->subgraph_id(subgraph_id)->GetArenaRWBufferSize() + interpreter->subgraph_id(subgraph_id)->GetArenaPersistentBufferSize();
    printf("[BEFOR] Get memory size allocated by arena buffer : %.2f MB\n", static_cast<float>(overall_buffer_size) / (1 << 20));
    printf("\033[0;31m[BEFORE] Subgraph[%d]'s memory overhead : %4.1f MB \033[0m\n",subgraph_id,
        static_cast<float>(overall_buffer_size) / (1 << 20)); // EZE
    printf("Model ID : %d\n", interpreter->subgraph_id(subgraph_id)->GetModelid());
    std::cout << "Resource type : " 
          << interpreter->subgraph_id(subgraph_id)->GetResourceType() << "\n";
    std::cout<< "Partitioning type : " 
          << interpreter->subgraph_id(subgraph_id)->GetPartitioningType() << "\n";
    if(interpreter->subgraph_id(subgraph_id)->IsInvokable())
      std::cout << "State : Invokable" << "\n";
    else
      std::cout << "State : Not Invokable" << "\n";
    for (size_t node_index = 0; node_index < node_size;
        node_index++) {
      const std::pair<TfLiteNode, TfLiteRegistration>* node_and_reg =
          interpreter->node_and_registration(static_cast<int>(node_index), subgraph_id);
      const TfLiteNode& node = node_and_reg->first;
      const TfLiteRegistration& reg = node_and_reg->second;
      if (reg.custom_name != nullptr) {
        printf("Node %3zu Operator Custom Name %s\n", node_index,
              reg.custom_name);
      } else {
        printf("Node %3zu Operator Builtin Code %3d %s\n", node_index,
              reg.builtin_code, EnumNamesBuiltinOperator()[reg.builtin_code]);
      }
      printf("  Inputs:");
      PrintTfLiteIntVector(node.inputs);
      printf("  Outputs:");
      PrintTfLiteIntVector(node.outputs);
      if (node.intermediates && node.intermediates->size) {
        printf("  Intermediates:");
        PrintTfLiteIntVector(node.intermediates);
      }
      if (node.temporaries && node.temporaries->size) {
        printf("  Temporaries:");
        PrintTfLiteIntVector(node.temporaries);
      }
    }
    std::cout << "======================================" << "\n";
    printf("Inputs:");
    PrintIntVector(interpreter->inputs(subgraph_id));
    printf("Outputs:");
    PrintIntVector(interpreter->outputs(subgraph_id));
    printf("\n");
    printf("Tensor size : %d\n", tensor_size);
    for (size_t tensor_index = 0; tensor_index < tensor_size;
       tensor_index++) {
      TfLiteTensor* tensor = interpreter->tensor(subgraph_id, static_cast<int>(tensor_index));
      printf("Tensor %3zu %10s %15s %10zu bytes (%4.1f MB) ", tensor_index,
           TensorTypeName(tensor->type),
           AllocTypeName(tensor->allocation_type), tensor->bytes,
           (static_cast<float>(tensor->bytes) / (1 << 20)));
      printf("%p ", tensor->data.data);
      PrintTfLiteIntVector(tensor->dims);
      if(tensor->allocation_type == 1){ // FOR kTfLiteMmapRo [weight, bias, etc_params]
        overall_buffer_size += tensor->bytes;
      }
    }
    printf("\033[0;31m[AFTER] Subgraph[%d]'s memory overhead : %4.1f MB \033[0m\n",subgraph_id,
          static_cast<float>(overall_buffer_size) / (1 << 20)); // EZE
    TOTAL_buffer_size += overall_buffer_size;
  }
 printf("\033[0;32m[AFTER] TOTAL memory overhead : %4.1f MB \033[0m\n", static_cast<float>(TOTAL_buffer_size) / (1 << 20)); // EZE
}

void PrintInterpreterStateSimple(Interpreter* interpreter,
                                 Interpreter* sub_interpreter
                                , std::string& buf){
  int subgraph_size = interpreter->subgraphs_size();
  buf = std::string("Interpreter has ") + std::to_string(subgraph_size)
                                         + std::string(" subgraphs \n");
  //interpreter->PrintSubgraphInfo();
  for(int subgraph_index=0; subgraph_index < subgraph_size; ++subgraph_index){
    buf += std::string("======================================\n");
    int subgraph_id = interpreter->subgraph(subgraph_index)->GetGraphid();
    int tensor_size = interpreter->subgraph_id(subgraph_id)->tensors_size();
    int node_size = interpreter->nodes_size(subgraph_id);
    buf += std::string("Subgraph ID ") + std::to_string(subgraph_id) +
           std::string(" has ") + std::to_string(tensor_size) + 
           std::string(" tensors and ") + std::to_string(node_size) + std::string("\n");
    buf += std::string("subgraph type ") + \
           std::to_string(interpreter->subgraph_id(subgraph_id)->GetResourceType()) +
           std::string("\n");
    buf += std::string("partitioning type ") + \
           std::to_string(interpreter->subgraph_id(subgraph_id)->GetPartitioningType()) +
           std::string("\n");
    buf += std::string("partitioning ratio ") + \
           std::to_string(interpreter->subgraph_id(subgraph_id)->GetPartitioningRatio()) +
           std::string("\n");
  }
  subgraph_size = sub_interpreter->subgraphs_size();
  buf += std::string("Sub Interpreter has ") + std::to_string(subgraph_size)
                                         + std::string(" subgraphs \n");
  //interpreter->PrintSubgraphInfo();
  for(int subgraph_index=0; subgraph_index < subgraph_size; ++subgraph_index){
    buf += std::string("======================================\n");
    int subgraph_id = sub_interpreter->subgraph(subgraph_index)->GetGraphid();
    int tensor_size = sub_interpreter->subgraph_id(subgraph_id)->tensors_size();
    int node_size = sub_interpreter->nodes_size(subgraph_id);
    buf += std::string("Subgraph ID ") + std::to_string(subgraph_id) +
           std::string(" has ") + std::to_string(tensor_size) + 
           std::string(" tensors and ") + std::to_string(node_size) + std::string("\n");
    buf += std::string("subgraph type ") + \
           std::to_string(sub_interpreter->subgraph_id(subgraph_id)->GetResourceType()) +
           std::string("\n");
    buf += std::string("partitioning type ") + \
           std::to_string(sub_interpreter->subgraph_id(subgraph_id)->GetPartitioningType()) +
           std::string("\n");
    buf += std::string("partitioning ratio ") + \
           std::to_string(interpreter->subgraph_id(subgraph_id)->GetPartitioningRatio()) +
           std::string("\n");
  }
  buf += std::string("======================================\n");
  buf += std::string("LOG_START\n");
}

void PrintInterpreterStateDimandSize(Interpreter* interpreter){
  int subgraph_size = interpreter->subgraphs_size();
  double tot = 0;
  printf("Interpreter has %d subgraphs\n", subgraph_size);
  //interpreter->PrintSubgraphInfo();
  for(int subgraph_index=0; subgraph_index < subgraph_size; ++subgraph_index){
    std::cout << "======================================" << "\n";
    int subgraph_id = interpreter->subgraph(subgraph_index)->GetGraphid();
    int tensor_size = interpreter->subgraph_id(subgraph_id)->tensors_size();
    int node_size = interpreter->nodes_size(subgraph_id);
    printf("Subgraph ID %d has %d tensors and %d nodes\n", subgraph_id,
        tensor_size, node_size);
    for (size_t node_index = 0; node_index < node_size;
        node_index++) {
      const std::pair<TfLiteNode, TfLiteRegistration>* node_and_reg =
          interpreter->node_and_registration(static_cast<int>(node_index), subgraph_id);
      const TfLiteNode& node = node_and_reg->first;
      const TfLiteRegistration& reg = node_and_reg->second;
      int origin_output_height, origin_input_height, new_input_height,
        new_output_height, filter, stride, padding_type, padding_height,
        padding_width, padding_height_offset, padding_width_offset;
      TfLiteContext* context_ = interpreter->subgraph(subgraph_index)->context();
      if(strcmp(GetOpName(reg), "CONV_2D") == 0 || strcmp(GetOpName(reg), "DEPTHWISE_CONV_2D") == 0){
        GetParamsForPartitioning(&reg, &node, context_,
                                    filter, stride, padding_type, padding_height, padding_width, 
                                    padding_height_offset, padding_width_offset);
      }

      if (reg.custom_name != nullptr) {
        printf("Node %3zu %s ", node_index, reg.custom_name);
      } else {
        printf("Node %3zu %s ", node_index, EnumNamesBuiltinOperator()[reg.builtin_code]);
      }
      TfLiteIntArray* outputs = node.outputs;
      TfLiteIntArray* inputs = node.inputs;

      
      for(int i=0; i<outputs->size; ++i){
        int tensor_index = outputs->data[i];
        int i_tensor_idx = inputs->data[0];
        double flops = 0;
        TfLiteTensor* tensor = interpreter->tensor(subgraph_id, static_cast<int>(tensor_index));
        TfLiteTensor* i_tensor = interpreter->tensor(subgraph_id, static_cast<int>(i_tensor_idx));
        printf("Tensor %3zu %10zu bytes(%4.1f MB) ", tensor_index, tensor->bytes,(static_cast<float>(tensor->bytes) / (1 << 20)));
        PrintTfLiteIntVector(tensor->dims);
        if(strcmp(GetOpName(reg), "CONV_2D") == 0){
          double mac = tensor->dims->data[1] * tensor->dims->data[2] * tensor->dims->data[3] * i_tensor->dims->data[3] * filter * filter;
          flops = 2*mac/1000000;
          tot += flops;
          printf("\033[0;31mFLOPs : %.1f\033[0m\n", flops);
        }
        if(strcmp(GetOpName(reg), "DEPTHWISE_CONV_2D") == 0){
          double mac = tensor->dims->data[1] * tensor->dims->data[2] * tensor->dims->data[3] * filter * filter;
          flops = 2*mac/1000000;
          tot += flops;
          printf("\033[0;31mFLOPs : %.1f\033[0m\n", flops);
        }
      }
    }
    printf("\n");
  }
  printf("\033[0;32mTotal Flops : %.1f\033[0m\n", tot);
}

}  // namespace tflite
