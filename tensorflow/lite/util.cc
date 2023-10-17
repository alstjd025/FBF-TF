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
#include "tensorflow/lite/util.h"

#include <complex>
#include <cstring>
#include <iostream>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/builtin_op_data.h"

namespace tflite {
namespace {

TfLiteStatus UnresolvedOpInvoke(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context,
                       "Encountered an unresolved custom op. Did you miss "
                       "a custom op or delegate?");
  return kTfLiteError;
}

}  // namespace

bool IsFlexOp(const char* custom_name) {
  return custom_name && strncmp(custom_name, kFlexCustomCodePrefix,
                                strlen(kFlexCustomCodePrefix)) == 0;
}

std::unique_ptr<TfLiteIntArray, TfLiteIntArrayDeleter> BuildTfLiteIntArray(
    const std::vector<int>& data) {
  std::unique_ptr<TfLiteIntArray, TfLiteIntArrayDeleter> result(
      TfLiteIntArrayCreate(data.size()));
  std::copy(data.begin(), data.end(), result->data);
  return result;
}

TfLiteIntArray* ConvertVectorToTfLiteIntArray(const std::vector<int>& input) {
  return ConvertArrayToTfLiteIntArray(static_cast<int>(input.size()),
                                      input.data());
}

TfLiteIntArray* ConvertArrayToTfLiteIntArray(const int rank, const int* dims) {
  TfLiteIntArray* output = TfLiteIntArrayCreate(rank);
  for (size_t i = 0; i < rank; i++) {
    output->data[i] = dims[i];
  }
  return output;
}

bool EqualArrayAndTfLiteIntArray(const TfLiteIntArray* a, const int b_size,
                                 const int* b) {
  if (!a) return false;
  if (a->size != b_size) return false;
  for (int i = 0; i < a->size; ++i) {
    if (a->data[i] != b[i]) return false;
  }
  return true;
}

size_t CombineHashes(std::initializer_list<size_t> hashes) {
  size_t result = 0;
  // Hash combiner used by TensorFlow core.
  for (size_t hash : hashes) {
    result = result ^
             (hash + 0x9e3779b97f4a7800ULL + (result << 10) + (result >> 4));
  }
  return result;
}

TfLiteStatus GetSizeOfType(TfLiteContext* context, const TfLiteType type,
                           size_t* bytes) {
  // TODO(levp): remove the default case so that new types produce compilation
  // error.
  switch (type) {
    case kTfLiteFloat32:
      *bytes = sizeof(float);
      break;
    case kTfLiteInt32:
      *bytes = sizeof(int);
      break;
    case kTfLiteUInt8:
      *bytes = sizeof(uint8_t);
      break;
    case kTfLiteInt64:
      *bytes = sizeof(int64_t);
      break;
    case kTfLiteBool:
      *bytes = sizeof(bool);
      break;
    case kTfLiteComplex64:
      *bytes = sizeof(std::complex<float>);
      break;
    case kTfLiteComplex128:
      *bytes = sizeof(std::complex<double>);
      break;
    case kTfLiteInt16:
      *bytes = sizeof(int16_t);
      break;
    case kTfLiteInt8:
      *bytes = sizeof(int8_t);
      break;
    case kTfLiteFloat16:
      *bytes = sizeof(TfLiteFloat16);
      break;
    case kTfLiteFloat64:
      *bytes = sizeof(double);
      break;
    default:
      if (context) {
        context->ReportError(
            context,
            "Type %d is unsupported. Only float32, int8, int16, int32, int64, "
            "uint8, bool, complex64 supported currently.",
            type);
      }
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteRegistration CreateUnresolvedCustomOp(const char* custom_op_name) {
  return TfLiteRegistration{nullptr,
                            nullptr,
                            nullptr,
                            /*invoke*/ &UnresolvedOpInvoke,
                            nullptr,
                            BuiltinOperator_CUSTOM,
                            custom_op_name,
                            1};
}

bool IsUnresolvedCustomOp(const TfLiteRegistration& registration) {
  return registration.builtin_code == tflite::BuiltinOperator_CUSTOM &&
         registration.invoke == &UnresolvedOpInvoke;
}

std::string GetOpNameByRegistration(const TfLiteRegistration& registration) {
  auto op = registration.builtin_code;
  std::string result =
      EnumNameBuiltinOperator(static_cast<BuiltinOperator>(op));
  if ((op == kTfLiteBuiltinCustom || op == kTfLiteBuiltinDelegate) &&
      registration.custom_name) {
    result += " " + std::string(registration.custom_name);
  }
  return result;
}


bool GetParamsForPartitioning(const TfLiteRegistration* registration,
                              const TfLiteNode* node, TfLiteContext* context,
                              int& filter_size, int& stride, int& padding_type,
                              int& padding_height, int& padding_width,
                              int& padding_height_offset, int& padding_width_offset){
  switch (registration->builtin_code)
  {
  case kTfLiteBuiltinConv2d:{
    const TfLiteConvParams* conv_params = 
        reinterpret_cast<const TfLiteConvParams*>(node->builtin_data);
    if(node->inputs->size != 3){
      std::cout << "GetParamsForPartitioning ERROR" << "\n";
      std::cout << "Node input tensor size is not 3" << "\n";
      return false;
    }
    if(context->tensors[node->inputs->data[1]].dims->size != 4){
      std::cout << "GetParamsForPartitioning ERROR" << "\n";
      std::cout << "Tensor dimension is not 4" << "\n";
      return false;
    }
    // get filter size from filter tensor
    filter_size = context->tensors[node->inputs->data[1]].dims->data[1];  
    // get stride and padding from params
    stride = conv_params->stride_height;
    // padding info
    // same == 1
    // valid == 2
    padding_type = conv_params->padding;
    padding_height = 0;
    padding_width = 0;
    padding_height_offset = 0;
    padding_width_offset = 0;
    break;
    }
  case kTfLiteBuiltinDepthwiseConv2d:{
    const TfLiteDepthwiseConvParams* depth_conv_params = 
        reinterpret_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data);
    if(node->inputs->size != 3){
      std::cout << "GetParamsForPartitioning ERROR" << "\n";
      std::cout << "Node input tensor size is not 3" << "\n";
      return false;
    }
    if(context->tensors[node->inputs->data[1]].dims->size != 4){
      std::cout << "GetParamsForPartitioning ERROR" << "\n";
      std::cout << "Tensor dimension is not 4" << "\n";
      return false;
    }
    // get filter size from filter tensor
    filter_size = context->tensors[node->inputs->data[1]].dims->data[1];  
    // get stride and padding from params
    stride = depth_conv_params->stride_height;
    // padding info
    // same == 1
    // valid == 2
    padding_type = depth_conv_params->padding;
    padding_height = 0;
    padding_width = 0;
    padding_height_offset = 0;
    padding_width_offset = 0;
    break;
    }
  case kTfLiteBuiltinMaxPool2d:{
    const TfLitePoolParams* pool_params = 
        reinterpret_cast<const TfLitePoolParams*>(node->builtin_data);
    
    // get filter size from params
    filter_size = pool_params->filter_height;  
    // get stride and padding from params
    stride = pool_params->stride_height;
    padding_type = pool_params->padding;
    // Pooling needs paading values.
    /* Consider params below.
    typedef struct {
      int width;
      int height;
      int width_offset;
      int height_offset;
    } TfLitePaddingValues;
    */
    padding_height = pool_params->computed.padding.height;
    padding_width = pool_params->computed.padding.width;
    padding_height_offset = pool_params->computed.padding.height_offset;
    padding_width_offset = pool_params->computed.padding.width_offset;
    break;
    }
  case kTfLiteBuiltinAveragePool2d:{
    const TfLitePoolParams* pool_params = 
        reinterpret_cast<const TfLitePoolParams*>(node->builtin_data);

    // get filter size from params
    filter_size = pool_params->filter_height;  
    // get stride and padding from params
    stride = pool_params->stride_height;
    // padding info
    // same == 1
    // valid == 2
    padding_type = pool_params->padding;
    padding_height = pool_params->computed.padding.height;
    padding_width = pool_params->computed.padding.width;
    padding_height_offset = pool_params->computed.padding.height_offset;
    padding_width_offset = pool_params->computed.padding.width_offset;
    break;
    }
  default:
    filter_size = 0;
    stride = 0;
    padding_type = 0;
    break;
  }
  return true;
}


int HW::GetOverlapConv(int S, int K, int Hi, int Ho){
  if(S == 0)
    S = 1;
  int padding = S * (Ho - 1) - Hi + K;
  if(padding < 0){
    padding = 0;
  }
  if(S == 1 && K == 1) // case of 1x1 conv
    padding = 0;
  return padding;
}

int HW::GetZeroPaddingConv(int S, int K, int Hi, int Ho){
  int padding = S * (Ho - 1) - Hi + K;
  if(padding < 0){
    padding = 0;
  }
  return padding;
}

int HW::GetOutputHeightConv(int S, int K, int P, int Hi){
  int output_height = (((Hi - K + 2 * P)/S) + 1);
  if (output_height <= 0)
    output_height = -1;
  return output_height;
}

// Must divide in even numbers.
int HW::GetOverlapPool(int S, int K, int Hi, int Ho){
  if(S == 0)
    S = 1;
  int padding = S * (Ho - 1) - Hi + K;
  if(padding < 0){
    padding = 0;
  }
  return padding;
}

int HW::GetPaddingPool(int S, int K, int Hi){

}

int HW::GetOutputHeightPool(int S, int K, int P, int Ho){

}

////////////////////////////////////////////////////////////////////
// HOON : utility funcs for parsing Yolo output

YOLO_Parser::YOLO_Parser() {};

YOLO_Parser::~YOLO_Parser() {};

bool YOLO_Parser::CompareBoxesByScore(const BoundingBox& box1, const BoundingBox& box2) {
    return box1.score > box2.score; }   

float YOLO_Parser::CalculateIoU(const BoundingBox& box1, const BoundingBox& box2) {
    float x1 = std::max(box1.left, box2.left);
    float y1 = std::max(box1.top, box2.top);
    float x2 = std::min(box1.right, box2.right);
    float y2 = std::min(box1.bottom, box2.bottom);
    float area_box1 = (box1.right - box1.left) * (box1.bottom - box1.top);
    float area_box2 = (box2.right - box2.left) * (box2.bottom - box2.top);
    float intersection_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float union_area = area_box1 + area_box2 - intersection_area;
  
    if (union_area > 0.0f) {
        return intersection_area / union_area;
    } else {
        return 0.0f; // Avoid division by zero
    }
  }
void YOLO_Parser::NonMaximumSuppression(std::vector<BoundingBox>& boxes, float iou_threshold) {
  std::sort(boxes.begin(), boxes.end(), CompareBoxesByScore);
  std::vector<BoundingBox> selected_boxes;
  while (!boxes.empty()) {
      BoundingBox current_box = boxes[0];
      selected_boxes.push_back(current_box);
      boxes.erase(boxes.begin());
      for (auto it = boxes.begin(); it != boxes.end();) {
          float iou = CalculateIoU(current_box, *it);
          if (iou > iou_threshold) {
              it = boxes.erase(it);
          } else {
              ++it;
          }
      }
  }
  boxes = selected_boxes;
}
void YOLO_Parser::PerformNMSUsingResults(
    const std::vector<int>& real_bbox_index_vector,
    const std::vector<std::vector<float>>& real_bbox_cls_vector,
    const std::vector<std::vector<int>>& real_bbox_loc_vector,
    float iou_threshold, const std::vector<int> real_bbox_cls_index_vector)
  {
    std::vector<BoundingBox> bounding_boxes;

    for (size_t i = 0; i < real_bbox_index_vector.size(); ++i) {
        BoundingBox box;
        box.left = static_cast<float>(real_bbox_loc_vector[i][0]);
        box.top = static_cast<float>(real_bbox_loc_vector[i][1]);
        box.right = static_cast<float>(real_bbox_loc_vector[i][2]);
        box.bottom = static_cast<float>(real_bbox_loc_vector[i][3]);
        box.score = static_cast<float>(real_bbox_cls_vector[i][real_bbox_cls_index_vector[i]]); // Using the class score
        box.class_id = real_bbox_cls_index_vector[i];
        bounding_boxes.push_back(box);
    }

    NonMaximumSuppression(bounding_boxes, iou_threshold);

    printf("\033[0;32mAfter NMS : \033[0m");
    printf("Number of bounding boxes after NMS: %d\n",bounding_boxes.size());
    result_boxes = bounding_boxes;
    bounding_boxes.clear();
  }

void YOLO_Parser::SOFTMAX(std::vector<float>& row) {
    const float threshold = 0.999999; 
    float maxElement = *std::max_element(row.begin(), row.end());
    float sum = 0.0;
    const float scalingFactor = 20.0; //20
    for (auto& i : row)
        sum += std::exp(scalingFactor * (i - maxElement));
    for (int i = 0; i < row.size(); ++i) {
        row[i] = std::exp(scalingFactor * (row[i] - maxElement)) / sum;
        if (row[i] > threshold)
            row[i] = threshold; 
    }
}

////////////////////////////////////////////////////////////////////

}  // namespace tflite
