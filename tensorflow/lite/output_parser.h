#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/c/common.h"

#include <vector>
#include <iostream>
#include <cmath>
#include <string>

// Minsung
// parsing class for yolo

// yolov4 output tensor : 948[1, 10647, 80], 969[1, 10647, 4]

struct DetectedObject{
  int class_id;
  int box_idx;
  float confidence;
  float x, y, w, h;
};

class OutputParser
{
  public:
    OutputParser();
    OutputParser(int intput_w, int intput_h, int anchor_size,
                int num_class, float conf_threshold);

    // parsing body function
    bool setOutputTensors(TfLiteTensor* output_tensor);
    bool ParseOutput();
    void PrintOutput();

    ~OutputParser();

    int input_w_;
    int input_h_;
    int anchor_size_;
    int num_class_;
    float conf_threshhold_;
    TfLiteTensor* output_tensor_;
    std::vector<DetectedObject> output_objects;
};