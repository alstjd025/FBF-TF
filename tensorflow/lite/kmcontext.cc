#include <iostream>

#include "tensorflow/lite/kmcontext.h"
using namespace std;

KmContext kmcontext;

void KmContext::setContext(TfLiteContext* c) {
    context_ = c;
    cout << "setContext : " << *(float*)context_->tensors[0].data.data << endl;
}

void KmContext::setNode(TfLiteNode* n) {
    node_.push_back(n);
}