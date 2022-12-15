/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <TensorFlowLite.h>

#include "main_functions.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "constants.h"
#include "model.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "math.h"
//float x_val[3]={0,0,0};

double y_val[3]={0,0,0};
double diff0,diff1,diff2,diff0sqrd,diff1sqrd,diff2sqrd;

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int input_length=2251;
// Keep track of how many inferences we have performed.
int inference_count = 0;
int add1,add2; 
double max_var;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 60 * 1024;
alignas(16)  uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  

  Serial.begin(9600);
  while (!Serial);
  Serial.println("Started");


  
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  
  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;
  
  
  
  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  

  
  // Allocate memory from the tensor_arena for the model's tensors.
  interpreter->AllocateTensors();

   

    // Obtain pointer to the model's input tensor.
  input = interpreter->input(0);
  output = interpreter->output(0);


  //Serial.println(input->dims->size);
  //Serial.println(input->type);
  //Serial.println(input->dims->data[0]);
  //Serial.println(input->dims->data[1]);

  if ((input->dims->size != 3) || (input->type != kTfLiteFloat32)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    return;
  }

  
       



   for(inference_count=0; inference_count<input_length;inference_count++){

//     Place our value in the model's input tensor
       input->data.f[0] = (double)(save_test_data0[inference_count]);
       input->data.f[1] = (double)(save_test_data1[inference_count]); 
       input->data.f[2] = (double)(save_test_data2[inference_count]);
       
       
       //Serial.println(inference_count);
       
       //Serial.println("Inp:");
       
       
       //Serial.println( input->data.f[0],11);
       
       //Serial.println( input->data.f[1],11);
       //Serial.println( input->data.f[2],11);

       
       //TF_LITE_REPORT_ERROR(error_reporter, "%f\n",static_cast<double>(save_test_data0[0]));
       //TF_LITE_REPORT_ERROR(error_reporter, "%f\n",static_cast<double>(save_test_data1[0]));
       //TF_LITE_REPORT_ERROR(error_reporter, "%f\n",static_cast<double>(save_test_data2[0]));
 


    // Run inference, and report any error
       TfLiteStatus invoke_status = interpreter->Invoke();
       if (invoke_status != kTfLiteOk) {
            TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x_val: %f\n",
                      (double)(input->data.f[0]));
            return;
        }
  
       y_val[0]=(double)(output->data.f[0]);
       y_val[1]=(double)(output->data.f[1]);
       y_val[2]=(double)(output->data.f[2]);

       
       //Serial.println( "Op:");
       
       //Serial.println( y_val[0],11);
       
       //Serial.println( y_val[1],11);
       //Serial.println( y_val[2],11);

      
       //TF_LITE_REPORT_ERROR(error_reporter, "Diff:\n");
       
       diff0=(double)(y_val[0])-((double)(save_test_data0[inference_count]));
       diff1=(double)(y_val[1])-((double)(save_test_data1[inference_count]));
       diff2=(double)(y_val[2])-((double)(save_test_data2[inference_count]))
       
       //Serial.println(diff0,11);
       //Serial.println(diff1,11);
       //Serial.println(diff2,11);
dist

       //TF_LITE_REPORT_ERROR(error_reporter, "Diffsqrd:\n");

       
       diff0sqrd=sq(diff0);
       diff1sqrd=sq(diff1);
       diff2sqrd =sq(diff2)

       //Serial.println(diff0sqrd,11);
       //Serial.println(diff1sqrd,11);
       //Serial.println(diff2sqrd,11);
       
       
       dist[inference_count]=sqrt(diff0sqrd+diff1sqrd+diff2sqrd);
       
       //delay(300);
        
       
       

                   
       //Serial.println( "Dist:");
       
       //Serial.println(dist[inference_count],11);


             
   }  


   max_var = dist[0];
   //Serial.println(max_var,11);
   for(inference_count=0; inference_count<(input_length-1);inference_count++){

      if(dist[inference_count+1]>=max_var)
      {
         max_var=dist[inference_count+1];
      }
   }
  
   Serial.println( "Max Dist:");
       
   Serial.println(max_var,11);

   for(inference_count=0; inference_count<(input_length);inference_count++){

      dist[inference_count]=double((dist[inference_count])/(max_var));


      //Serial.println( "Dist:");
       
      //Serial.println(dist[inference_count],11);



      if(dist[inference_count]>threshold && inference_count<anomaly){
        FN++;
        }

     if(dist[inference_count]<threshold && inference_count>anomaly){
        FP++;
        }

     if(dist[inference_count]<threshold && inference_count<anomaly){
        TN++;
        }

     if(dist[inference_count]>threshold && inference_count>anomaly){
        TP++;
       }
      
   }


      
  
       Serial.println("TP");
       Serial.println(TP);
       Serial.println("TN");
       Serial.println(TN);
       Serial.println("FP");
       Serial.println(FP);
       Serial.println("FN");
       Serial.println(FN);

       add1=(TP+TN);
       Serial.println("TP+TN:");
       Serial.println(add1);
       
       add2=(TP+TN+FP+FN);
       Serial.println("TP+TN+FP+FN:");
       Serial.println(add2);
       
       accuracy = (double)((double)(add1)/(double)(add2))*((double)(100));
       Serial.println("Accuracy:");
       Serial.println(accuracy,8); 
       
       Serial.println("Precision:");
       precision = (double)((double)(TP)/(double)(TP+FP))*((double)(100));
       Serial.println(precision,8);

       Serial.println("Recall:");
       recall = (double)((double)(TP)/(double)(TP+FN))*((double)(100));
       Serial.println(recall,8);

       
       Serial.println("F1:"); 
       F1 = 2* (double)(recall*precision)/(double)(recall+precision);
       Serial.println(F1,8); 
       
       Serial.println("Done");

}  
   
  


  

// The name of this function is important for Arduino compatibility.
void loop() {

/*

  // Place our value in the model's input tensor
       input->data.f[0] = (double)(save_test_data0[inference_count]);
       input->data.f[1] = (double)(save_test_data1[inference_count]);
       input->data.f[2] = (double)(save_test_data2[inference_count]);
       TF_LITE_REPORT_ERROR(error_reporter, "%d\n",inference_count);
       TF_LITE_REPORT_ERROR(error_reporter, "Inp:\n");
       
       //TF_LITE_REPORT_ERROR(error_reporter, "%f\n",(double)(input->data.f[0]));
       //TF_LITE_REPORT_ERROR(error_reporter, "%f\n",(double)(input->data.f[1]));
       //TF_LITE_REPORT_ERROR(error_reporter, "%f\n",(double)(input->data.f[2]));
       
       TF_LITE_REPORT_ERROR(error_reporter, "%f\n",(save_test_data0[inference_count]));
       TF_LITE_REPORT_ERROR(error_reporter, "%f\n",(save_test_data1[inference_count]));
       TF_LITE_REPORT_ERROR(error_reporter, "%f\n",(save_test_data2[inference_count]));
 


    // Run inference, and report any error
       TfLiteStatus invoke_status = interpreter->Invoke();
       if (invoke_status != kTfLiteOk) {
            TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x_val: %f\n",
                      (double)(input->data.f[0]));
            return;
        }
  
       y_val[0]=(double)(output->data.f[0]);
       y_val[1]=(double)(output->data.f[1]);
       y_val[2]=(double)(output->data.f[2]);
       diff0=(y_val[0])-((double)(save_test_data0[inference_count]));
       diff1=(y_val[1])-((double)(save_test_data1[inference_count]));
       diff2=(y_val[2])-((double)(save_test_data2[inference_count]));
       diff0sqrd=sq(diff0);
       diff1sqrd=sq(diff1);
       diff2sqrd=sq(diff2);
       dist=sqrt(diff0sqrd+diff1sqrd+diff2sqrd);
       delay(150);
        
       while (!Serial);
       TF_LITE_REPORT_ERROR(error_reporter, "Op:\n");
       TF_LITE_REPORT_ERROR(error_reporter, "%f\n",(double)(y_val[0]));
       TF_LITE_REPORT_ERROR(error_reporter, "%f\n",(double)(y_val[1]));
       TF_LITE_REPORT_ERROR(error_reporter, "%f\n",(double)(y_val[2]));
       

       while (!Serial);
            
       TF_LITE_REPORT_ERROR(error_reporter, "dist:\n");
       TF_LITE_REPORT_ERROR(error_reporter, "%f\n",(double)(dist));
       //Serial.println("%f",(double)(dist));

       
       
 /*
  // Output the results. A custom HandleOutput function can be implemented
  // for each supported hardware target.
  HandleOutput(error_reporter, x_val, y_val);

  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
  if (inference_count > input_length) 
  inference_count = 0;
*/
  
}
