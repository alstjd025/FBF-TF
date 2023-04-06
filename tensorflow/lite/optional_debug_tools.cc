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
  for(int subgraph_index=0; subgraph_index < subgraph_size; ++subgraph_index){
    std::cout << "======================================" << "\n";
    int subgraph_id = interpreter->subgraph(subgraph_index)->GetGraphid();
    int tensor_size = interpreter->subgraph_id(subgraph_id)->tensors_size();
    int node_size = interpreter->nodes_size(subgraph_id);
    printf("Subgraph ID %d has %d tensors and %d nodes\n", subgraph_id,
        tensor_size, node_size);
    printf("Model ID : %d\n", interpreter->subgraph_id(subgraph_id)->GetModelid());
    
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

}  // namespace tflite
