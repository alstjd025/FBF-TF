#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <utility>
#include <vector>
#include <queue>
#include <mutex>
#include "condition_variable"
#include "cstring"

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/core/macros.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/experimental/resource/resource_base.h"
#include "tensorflow/lite/memory_planner.h"
#include "tensorflow/lite/util.h"

namespace tflite{
    namespace {
        extern std::vector<std::vector<float>> real_bbox_cls_vector; 
        extern std::vector<int> real_bbox_cls_index_vector;
        extern std::vector<std::vector<float>> real_bbox_loc_vector;
    }
}
