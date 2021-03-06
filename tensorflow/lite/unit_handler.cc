#include "unit_handler.h"
#include "kmcontext.h"
#include <typeinfo>
//#define MULTITHREAD
#define CPUONLY


extern std::mutex mtx_lock;

namespace tflite
{

UnitHandler::UnitHandler() :  fileNameOriginal(nullptr), builder_(nullptr) {}

UnitHandler::UnitHandler(const char* OriginalModel)
                                        :fileNameOriginal(OriginalModel)
{
    PrintMsg("Create InterpreterBuilder Using One Model");
    std::cout << "You have " << std::thread::hardware_concurrency() <<
                 " Processors " << "\n";
    vUnitContainer.reserve(10);
    bUseTwoModel = false;
    std::unique_ptr<tflite::FlatBufferModel>* model;
    model = new std::unique_ptr<tflite::FlatBufferModel>\
    (tflite::FlatBufferModel::BuildFromFile(fileNameOriginal));
    TFLITE_MINIMAL_CHECK(model != nullptr);
    // Build the interpreter with the InterpreterBuilder.
    tflite::ops::builtin::BuiltinOpResolver* resolver;
    resolver = new tflite::ops::builtin::BuiltinOpResolver;
    builder_ = new tflite::InterpreterBuilder(**model, *resolver);
    PrintMsg("Create InterpreterBuilder");
    if(builder_ == nullptr){
        PrintMsg("InterpreterBuilder nullptr ERROR");
    }
    qSharedData = new std::queue<SharedContext*>;
}

UnitHandler::UnitHandler(const char* OriginalModel, const char* QuanizedModel)
                                        :fileNameOriginal(OriginalModel),\
                                         fileNameQuantized(QuanizedModel)

{
    PrintMsg("Create Original, Quantized Model InterpreterBuilder");
    PrintMsg(fileNameOriginal);
    PrintMsg(fileNameQuantized);
    std::cout << "You have " << std::thread::hardware_concurrency() <<
                 " Processors " << "\n";
    vUnitContainer.reserve(10);
    bUseTwoModel = true;
    std::unique_ptr<tflite::FlatBufferModel>* OriginalModelFlatBuffer;
    std::unique_ptr<tflite::FlatBufferModel>* QuantizedModelFlatBuffer;
    OriginalModelFlatBuffer = new std::unique_ptr<tflite::FlatBufferModel>\
    (tflite::FlatBufferModel::BuildFromFile(fileNameOriginal));
    QuantizedModelFlatBuffer = new std::unique_ptr<tflite::FlatBufferModel>\
    (tflite::FlatBufferModel::BuildFromFile(fileNameQuantized));
    TFLITE_MINIMAL_CHECK(OriginalModelFlatBuffer != nullptr);
    TFLITE_MINIMAL_CHECK(QuantizedModelFlatBuffer != nullptr);
    
    // Build the interpreter with the InterpreterBuilder.
    tflite::ops::builtin::BuiltinOpResolver* OriginalResolver;
    OriginalResolver = new tflite::ops::builtin::BuiltinOpResolver;
    GPUBuilder_ = new tflite::InterpreterBuilder(**OriginalModelFlatBuffer,\
                                                             *OriginalResolver);

                                      
    tflite::ops::builtin::BuiltinOpResolver* QuantizedResolver;
    QuantizedResolver = new tflite::ops::builtin::BuiltinOpResolver;
    CPUBuilder_ = new tflite::InterpreterBuilder(**QuantizedModelFlatBuffer,\
                                                             *QuantizedResolver);

                                      
    PrintMsg("Create InterpreterBuilder");
    if(GPUBuilder_ == nullptr){
        PrintMsg("GPU InterpreterBuilder nullptr ERROR");
    }
    if(CPUBuilder_ == nullptr){
        PrintMsg("CPU InterpreterBuilder nullptr ERROR");
    }
    qSharedData = new std::queue<SharedContext*>;
}

TfLiteStatus UnitHandler::CreateUnitCPU(UnitType eType,
                                         std::vector<cv::Mat> input, int partitioning){
    std::unique_ptr<tflite::Interpreter>* interpreter;
    if(bUseTwoModel){
        if(CPUBuilder_ ==nullptr){
            PrintMsg("CPU InterpreterBuilder nullptr ERROR");
            return kTfLiteError;
        }
        interpreter = new std::unique_ptr<tflite::Interpreter>;
        (*CPUBuilder_)(interpreter, 4);
    }
    else{
        if(builder_ == nullptr){
            PrintMsg("InterpreterBuilder nullptr ERROR");
            return kTfLiteError;
        }
        interpreter = new std::unique_ptr<tflite::Interpreter>;
        (*builder_)(interpreter, 4);
    }
    #ifdef MULTITHREAD
    TFLITE_MINIMAL_CHECK(interpreter->get()->SetPartitioning(2, eType) == kTfLiteOk);  
    #endif 
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);
    TFLITE_MINIMAL_CHECK(interpreter->get()->AllocateTensors() == kTfLiteOk);  
    UnitCPU* temp;
    temp = new UnitCPU(eType, std::move(interpreter));
    temp->SetInput(input);
    vUnitContainer.push_back(temp);
    iUnitCount++;    
    PrintMsg("Build CPU Interpreter");
    #ifdef MULTITHREAD
    kmcontext.channelPartitioning("CONV_2D", 0.2);
    #endif
    #ifdef QUANTIZE
    if(interpreter->get()->QuantizeSubgraph() != kTfLiteOk){
        std::cout << "Quantization Error \n";
        return kTfLiteError;  
    }
    #endif
    std::cout << "[CPU INTERETER STATE] \n";
    tflite::PrintInterpreterState(interpreter->get());
    return kTfLiteOk;
}


TfLiteStatus UnitHandler::CreateAndInvokeCPU(UnitType eType,
                                             std::vector<cv::Mat> input){ 
    mtx_lock.lock();
    if (CreateUnitCPU(eType, input, 2) != kTfLiteOk){
        PrintMsg("CreateUnitCPUError");
        return kTfLiteError;
    }
    mtx_lock.unlock();
    std::vector<Unit*>::iterator iter;
    for(iter = vUnitContainer.begin(); iter != vUnitContainer.end(); ++iter){
        if((*iter)->GetUnitType() == eType){
            if((*iter)->Invoke(eType, mtx_lock, mtx_lock_,
                                    mtx_lock_timing, Ucontroller, Outcontroller,
                                    qSharedData, &C_Counter, &G_Counter) != kTfLiteOk)
                return kTfLiteError;
        }
    }
    return kTfLiteOk;
}

TfLiteStatus UnitHandler::CreateUnitGPU(UnitType eType,
                                         std::vector<cv::Mat> input, int partitioning){
    std::unique_ptr<tflite::Interpreter>* interpreter;
    if(bUseTwoModel){
        if(GPUBuilder_ ==nullptr){
            PrintMsg("GPU InterpreterBuilder nullptr ERROR");
            return kTfLiteError;
        }
        interpreter = new std::unique_ptr<tflite::Interpreter>;
        (*GPUBuilder_)(interpreter);
    }
    else{
        if(builder_ == nullptr){
            PrintMsg("InterpreterBuilder nullptr ERROR");
            return kTfLiteError;
        }
        interpreter = new std::unique_ptr<tflite::Interpreter>;
        (*builder_)(interpreter);
    }
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);
    TfLiteDelegate *MyDelegate = NULL;
    const TfLiteGpuDelegateOptionsV2 options = {
        .is_precision_loss_allowed = 1, 
        .inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
        .inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY,
        .inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
        .inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
        .experimental_flags = 1,
        .max_delegated_partitions = 30,
    };
    #ifdef MULTITHREAD
    //Set Partitioning Value : GPU Side Filters
    TFLITE_MINIMAL_CHECK(interpreter->get()->SetPartitioning(8, eType) == kTfLiteOk); 
    TFLITE_MINIMAL_CHECK(interpreter->get()->PrepareTensorsSharing(eType) == kTfLiteOk); 
    #endif
    //tflite::PrintInterpreterState(interpreter->get());
    MyDelegate = TfLiteGpuDelegateV2Create(&options);
    if(interpreter->get()->ModifyGraphWithDelegate(MyDelegate) != kTfLiteOk) {
        PrintMsg("Unable to Use GPU Delegate");
        return kTfLiteError;
    }
    TFLITE_MINIMAL_CHECK(interpreter->get()->AllocateTensors() == kTfLiteOk);
    UnitGPU* temp;
    temp = new UnitGPU(eType, std::move(interpreter));
    temp->SetInput(input);
    //Set ContextHandler Pointer
    vUnitContainer.push_back(temp);
    iUnitCount++;
    PrintMsg("Build GPU Interpreter");
    PrintMsg("GPU Interpreter Pre Invoke State");
    tflite::PrintInterpreterState(interpreter->get());
    return kTfLiteOk;
}


TfLiteStatus UnitHandler::CreateAndInvokeGPU(UnitType eType,
                                             std::vector<cv::Mat> input){
    mtx_lock.lock();
    if (CreateUnitGPU(eType, input, 8) != kTfLiteOk){
        PrintMsg("CreateUnitGPUError");
        return kTfLiteError;
    }
    mtx_lock.unlock();
    std::vector<Unit*>::iterator iter;
    for(iter = vUnitContainer.begin(); iter != vUnitContainer.end(); ++iter){
        if((*iter)->GetUnitType() == eType){
           if((*iter)->Invoke(eType, mtx_lock, mtx_lock_
                                ,mtx_lock_timing, Ucontroller, Outcontroller
                                ,qSharedData, &C_Counter, &G_Counter) != kTfLiteOk)
               return kTfLiteError;
        }
    }
    return kTfLiteOk;
}

void UnitHandler::PrintMsg(const char* msg){
    std::cout << "UnitHandler : \"" << msg << "\"\n";
    return;
}

void UnitHandler::PrintInterpreterStatus(){
    std::vector<Unit*>::iterator iter;
    for(iter = vUnitContainer.begin(); iter != vUnitContainer.end(); ++iter){
        //std::cout << "Device ====== " << (*iter)->eType << " ======\n";
        PrintInterpreterState((*iter)->GetInterpreter());
    }
    return;
}

TfLiteStatus UnitHandler::Invoke(UnitType eType, UnitType eType_,
                                 std::vector<cv::Mat> input){
    //eType -> CPU
    //eType_ -> GPU
    PrintMsg("Invoke");
    #ifdef MULTITHREAD
    std::thread cpu;
    std::thread gpu;
    cpu = std::thread(&UnitHandler::CreateAndInvokeCPU, this, eType, input);
    gpu = std::thread(&UnitHandler::CreateAndInvokeGPU, this, eType_, input);
    cpu.join();
    gpu.join();
    #endif
    
    #ifdef GPUONLY
    std::thread gpu;
    gpu = std::thread(&UnitHandler::CreateAndInvokeGPU, this, eType_, input);
    gpu.join();
    #endif

    #ifdef CPUONLY
    std::thread cpu;
    cpu = std::thread(&UnitHandler::CreateAndInvokeCPU, this, eType, input);
    cpu.join();
    #endif
    
    PrintMsg("ALL Jobs Done");
}
} // End of namespace tflite

