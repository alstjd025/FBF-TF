#include <vector>
#include "tensorflow/lite/c/common.h"

#define KMCONTEXT() kmcontext.setContext(&context_)
#define KMNODE() kmcontext.setNode(&node)

class KmContext {
  public:
    void setContext(TfLiteContext* c);
    void setNode(TfLiteNode* n);
    TfLiteContext* context_;
    std::vector<TfLiteNode*> node_;
};

extern KmContext kmcontext;