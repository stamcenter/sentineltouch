
/*********************************************************************************************************************** 
*
* @author: Nges Brian, Njungle 
*
* MIT License
* Copyright (c) 2025 Secure, Trusted and Assured Microelectronics, Arizona State University

* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.

* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
********************************************************************************************************************/



#include <iostream>
#include <sys/stat.h>

#include "FHEController.h"
#include "ANNController.h"



using namespace std;

CryptoContext<DCRTPoly> context;
FHEController fheController(context);

#ifndef INDEX_VALUE
#define INDEX_VALUE 0
#endif

#ifndef NUM_TO_READ
#define NUM_TO_READ 10
#endif

#ifndef NUM_USERS
#define NUM_USERS 400
#endif

#ifndef EMBEDDING_SPACE
#define EMBEDDING_SPACE 16
#endif

#ifndef MODEL
#define MODEL "lenet5"
#endif

#ifndef DATASET
#define DATASET "polyu"
#endif

#ifndef OPERATION
#define OPERATION "register"
#endif

vector<int> measuringTime;
vector<int> operationTime;
vector<int> enInTime;
auto startIn = get_current_time();

int numUsers = NUM_USERS;

int main(int argc, char *argv[]) {

    auto begin_time = startTime();
    printWelcomeMessage();
    /*** Generate the context of the project in the FHEController and pass it to the SnnController */
    int ringDegree = 13;
    int numSlots = 12;
    int circuitDepth = 11;
    int dcrtBits = 50;
    int firstMod = 54;
    double initVal = 0.501;
    int digitSize = 4;
    vector<uint32_t> levelBudget = {3, 3};
    int serialize = true;
    fheController.setNumUsers(numUsers);
    fheController.generate_context(ringDegree, numSlots, circuitDepth, dcrtBits, firstMod, digitSize, initVal, levelBudget, serialize);
    context = fheController.getContext();
    ANNController annController(context);
    printDuration(begin_time, "Context Generated and Keys Serialization", false);

    vector<vector<int>> rotation_keys;
    int kernelWidth = 5;
    int poolSize = 2;
    int strideLen = 1;
    int paddingLen = 0;
    int rotPositions = 16;
    int embedding_space = EMBEDDING_SPACE;
    vector<int> imgWidth = {28, 24, 12, 8, 4};
    vector<int> channels = {1, 6, 16, 256, 120, 84, embedding_space};
   
    //** generate rotation keys*/
    auto conv1_keys = annController.generate_conv_rotation_positions(imgWidth[0], channels[0], channels[1],  kernelWidth, paddingLen, strideLen);
    auto avg1_keys = annController.generate_avgpool_optimized_rotation_positions(imgWidth[1],channels[1],  poolSize, poolSize);
    auto conv2_keys = annController.generate_conv_rotation_positions(imgWidth[2], channels[1], channels[2], kernelWidth, paddingLen, strideLen);
    auto avg2_keys = annController.generate_avgpool_optimized_rotation_positions(imgWidth[3],channels[2], poolSize, poolSize);
    auto fc_keys = annController.generate_fullyconnected_rotation_positions(channels[4], rotPositions);

    auto user_positions = annController.generate_users_rotation_positions(numUsers, embedding_space);
    
    
    rotation_keys.push_back(conv1_keys);
    rotation_keys.push_back(avg1_keys);
    rotation_keys.push_back(conv2_keys);
    rotation_keys.push_back(avg2_keys);
    rotation_keys.push_back(fc_keys);
    rotation_keys.push_back(user_positions);

    /*** join all keys and generate unique values only */
    vector<int> rotation_positions;
    for (const auto& vec : rotation_keys) {
        rotation_positions.insert(rotation_positions.end(), vec.begin(), vec.end());
    }

    std::sort(rotation_positions.begin(), rotation_positions.end());
    auto new_end = std::remove(rotation_positions.begin(), rotation_positions.end(), 0);
    new_end = std::unique(rotation_positions.begin(), rotation_positions.end());
    unique(rotation_positions.begin(), rotation_positions.end());
    rotation_positions.erase(new_end, rotation_positions.end());
    std::sort(rotation_positions.begin(), rotation_positions.end());
    
    /*** Generate the rotation positions, generate rotation keys, and load rotation keys */
    auto begin_rotkeygen_time = startTime();
    cout << "This is the rotation positions (" << rotation_positions.size() <<"): " << rotation_positions << endl;
    fheController.generate_rotation_keys(rotation_positions, "rotation_keys.bin",  true);
    cout << "*****************************************************" << endl;
    cout << "User Rotation Indices (" << user_positions.size() <<"): " << user_positions << endl;
    printDuration(begin_rotkeygen_time, "Rotation KeyGen (position, gen, and load) Time", false);

    /*************************************************** Prepare Weights for the network **************************************************/
    /*** 1st Convolution */
    // cout << "Load Weights" << endl;
    auto wloading_time = startTime();
    string modelName = std::string(MODEL) + "_" + DATASET + "_" + std::to_string(embedding_space);
    string dataPath = "./../weights/"+modelName+"/";

    auto conv1_biasVector = load_bias(dataPath+"conv1_bias.csv");
    auto conv1_rawKernel = load_weights(dataPath+"conv1_weight.csv", channels[1], channels[0], kernelWidth, kernelWidth);
    int conv1WidthSq = pow(imgWidth[0], 2);
    vector<vector<Ptext>> conv1_kernelData;
    for(int i=0; i<channels[1]; i++){
        auto encodeKernel = fheController.encode_kernel(conv1_rawKernel[i], conv1WidthSq);

        conv1_kernelData.push_back(encodeKernel);
    }
    auto conv1biasEncoded = fheController.encode_baisVector(conv1_biasVector, (imgWidth[1] * imgWidth[1]));

    auto conv2_rawKernel = load_weights(dataPath+"conv2_weight.csv", channels[2], channels[1], kernelWidth, kernelWidth);
    auto conv2_biasVector = load_bias(dataPath+"conv2_bias.csv");
    int conv2WidthSq = pow(imgWidth[2], 2);
    vector<vector<Ptext>> conv2_kernelData;
    for(int i=0; i<channels[2]; i++){
        auto encodeKernel = fheController.encode_kernel(conv2_rawKernel[i], conv2WidthSq);
        conv2_kernelData.push_back(encodeKernel);
    }
    auto conv2biasEncoded = fheController.encode_baisVector(conv2_biasVector, (imgWidth[3]* imgWidth[3]));

     /*** first fully layer connected kernel and bias */
    auto fc1_biasVector = load_bias(dataPath+"fc1_bias.csv");
    auto fc1_rawKernel = load_fc_weights(dataPath+"fc1_weight.csv", channels[4], channels[3]);
    vector<Ptext> fc1_kernelData;
    for(int i=0; i < channels[4]; i++){
        auto encodeWeights = fheController.encode_packedVector(fc1_rawKernel[i]);
        fc1_kernelData.push_back(encodeWeights);
    }
    Ptext fc1baisVector = context->MakeCKKSPackedPlaintext(fc1_biasVector, 1);
    
     /*** second fully layer connected weights and bias */
    auto fc2_biasVector = load_bias(dataPath+"fc2_bias.csv");
    auto fc2_rawKernel = load_fc_weights(dataPath+"fc2_weight.csv", channels[5], channels[4]);
    vector<Ptext> fc2_kernelData;
    for(int i=0; i<channels[5]; i++){
        auto encodeWeights = fheController.encode_packedVector(fc2_rawKernel[i]);
        fc2_kernelData.push_back(encodeWeights);
    }
    Ptext fc2baisVector = context->MakeCKKSPackedPlaintext(fc2_biasVector, 1);

     /*** third fully layer connected weights and bias */
    auto fc3_biasVector = load_bias(dataPath+"fc3_bias.csv");
    auto fc3_rawKernel = load_fc_weights(dataPath+"fc3_weight.csv", channels[6], channels[5]);
    vector<Ptext> fc3_kernelData;
    for(int i=0; i<channels[6]; i++){
        auto encodeWeights = fheController.encode_packedVector(fc3_rawKernel[i]);
        fc3_kernelData.push_back(encodeWeights);
    }
    Ptext fc3baisVector = context->MakeCKKSPackedPlaintext(fc3_biasVector, 1);

    printDuration(wloading_time, "Weights Loading Time", false);

    /************************************************************************************************ */
    int reluScale = 10;
    vector<int> dataSizeVector;
    dataSizeVector.push_back((channels[1]*pow(imgWidth[1], 2)));
    dataSizeVector.push_back(( channels[2]*pow(imgWidth[3], 2)));
    std::ofstream outFile;
    outFile.open("./../results/lenet5/fhelenet5Predictions.txt", std::ios_base::app);

    string filesDirName = "./../exported_images/"+string(DATASET)+"/"+string(MODEL);

    vector<string> files;
    for (const auto& entry : fs::directory_iterator(filesDirName)) {
        files.push_back(entry.path().filename().string()); 
    }

   
    /****
    * Registered Users data*/
    string operation = string(OPERATION);
    int operationCount = 2; 
    cout << "Number of Users: " <<numUsers << endl; 
    if(operation == "authenticate"){
        operationCount = 1; 
    }

   for (int outer = 0; outer < operationCount; outer++) {
        string imagesFld = filesDirName+"/"+operation;
        for (int idx = INDEX_VALUE; idx < NUM_TO_READ; idx++) {
            /************************************************************************************************ */
            vector<double> readInput = load_user_input(imagesFld, idx);
            // auto enIn = get_current_time();
            Ctext encryptedInput = fheController.encrypt_inputData(readInput);
            // enInTime.push_back(measure_time(enIn, get_current_time()));
            // totalTime(enInTime, "Encryption");
            cout << endl << idx+1 << " - Read, Normalized and Encrypt"<< endl;
            /************************************************************************************************ */
            
            /***** The first Convolution Layer takes  image=(1,28,28), kernel=(6,1,5,5) 
             * stride=1, pooling=0 output= (6,24,24) = 3456 vals */
            // cout << "Layer 1: " << endl;
            // auto inference_time = startTime();
            startIn = get_current_time();
            auto convData = annController.secure_conv2D(encryptedInput, conv1_kernelData, conv1biasEncoded, imgWidth[0], channels[0], channels[1], kernelWidth);
            measuringTime.push_back(measure_time(startIn, get_current_time()));

            reluScale = fheController.read_scalingValue(convData, dataSizeVector[0]);


            startIn = get_current_time();
            convData = annController.secure_relu(convData, reluScale, dataSizeVector[0], 59);
            convData = annController.secure_optimzed_AvgPool(convData, imgWidth[1], channels[1], poolSize, poolSize);
             
            // cout << "Layer 2: " << endl;
            convData = annController.secure_conv2D(convData, conv2_kernelData, conv2biasEncoded, imgWidth[2], channels[1], channels[2], kernelWidth);
            measuringTime.push_back(measure_time(startIn, get_current_time()));
            
            convData = fheController.bootstrap_function(convData);
            reluScale = fheController.read_scalingValue(convData, dataSizeVector[1]);

            startIn = get_current_time();
            convData = annController.secure_relu(convData, reluScale, dataSizeVector[1]);
            measuringTime.push_back(measure_time(startIn, get_current_time()));
            
            convData = fheController.bootstrap_function(convData);

            startIn = get_current_time();
            convData = annController.secure_optimzed_AvgPool(convData, imgWidth[3], channels[2], poolSize, poolSize);
            // fheController.read_minmaxValue(convData, channels[4]);
            
            // cout << "FC 1: " << endl;
            convData = annController.secure_flinear(convData, fc1_kernelData, fc1baisVector, channels[3], channels[4], rotPositions);
            measuringTime.push_back(measure_time(startIn, get_current_time()));

            reluScale = fheController.read_scalingValue(convData, channels[4]);
            convData = fheController.bootstrap_function(convData);
            // fheController.read_minmaxValue(convData, channels[4]);
            
            startIn = get_current_time();
            convData = annController.secure_relu(convData, reluScale, channels[4]);

            // cout << "FC 2: " << endl;
            convData = annController.secure_flinear(convData, fc2_kernelData, fc2baisVector, channels[4], channels[5], rotPositions);
            measuringTime.push_back(measure_time(startIn, get_current_time()));

            reluScale = fheController.read_scalingValue(convData, channels[5]);
            convData = fheController.bootstrap_function(convData);
            startIn = get_current_time();
            convData = annController.secure_relu(convData, reluScale, channels[5]);

            // cout << "FC 3: " << endl;
            convData = annController.secure_flinear(convData, fc3_kernelData, fc3baisVector, channels[5], channels[6], rotPositions);
            measuringTime.push_back(measure_time(startIn, get_current_time()));
            totalTime(measuringTime, "Feature Extraction");

            /*****
             * carry out the user operation which is either identification or registration *******/            
            if (strcmp(operation.c_str(), "authenticate") == 0) {

                convData = fheController.bootstrap_function(convData);
                // cout << "Generated Embeddings: " << endl; 
                // fheController.decryptAndPrint(convData, embedding_space);

                auto operationIn = get_current_time();
                Ctext results = fheController.secure_cosine_similarity(convData, modelName, numUsers, embedding_space, rotPositions);
                operationTime.push_back(measure_time(operationIn, get_current_time()));
                measuringTime.push_back(measure_time(operationIn, get_current_time()));
                // fheController.decryptAndPrint(results, NUM_TO_READ);
                cout << "------ Identified User: " << endl;

                // auto enIn = get_current_time();
                fheController.resultsInterpreter(results, NUM_TO_READ, idx);
                // enInTime.push_back(measure_time(enIn, get_current_time()));
                // totalTime(enInTime, "Interpret");

                
                totalTime(operationTime, "Identification");
                operationTime.clear();
            } 

            else if (strcmp(operation.c_str(), "register") == 0) {
                // cout << "Start Registration: " << endl;

                convData = fheController.bootstrap_function(convData);
                // cout << "Generated Embeddings: " << endl; 
                // fheController.decryptAndPrint(convData, embedding_space);

                auto operationIn = get_current_time();
                int result = fheController.secure_user_registration(convData, idx, modelName, embedding_space, rotPositions);                
                operationTime.push_back(measure_time(operationIn, get_current_time()));
                measuringTime.push_back(measure_time(operationIn, get_current_time()));
                cout << "------ Registration status: "<< result << endl;
                totalTime(operationTime, "register");
                operationTime.clear();
                
            }
            totalTime(measuringTime, "");
            measuringTime.clear();
        }
        // cout << "Show number of users" << endl;
        // vector<Ctext> regUsers = fheController.reeadRegisteredUsers(modelName, embedding_space);
        // for(int i=0; i<embedding_space; i++){
        //     fheController.decryptAndPrint(regUsers[i], 20);
        // }

        operation = "authenticate";
        cout << "\n##################################### IDENTIFICATION  ##########################\n" << endl;
    }
    outFile.close();
    cout << "All predicted results printed to File." << endl;
   return 0;
}


