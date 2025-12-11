
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
    int ringDegree = 12;
    int numSlots = 11;
    int circuitDepth = 0;
    int dcrtBits = 20;
    int firstMod = 26;
    int digitSize = 2;
    double initVal = 0.4523;
    vector<uint32_t> levelBudget = {3, 3};
    int serialize = true;
     fheController.setNumUsers(numUsers);
    fheController.generate_context(ringDegree, numSlots, circuitDepth, dcrtBits, firstMod, digitSize, initVal, levelBudget, serialize);
    context = fheController.getContext();
    ANNController annController(context);
    printDuration(begin_time, "Context Generated and Keys Serialization", false);
    int rotPositions = 8;
    int embedding_space = EMBEDDING_SPACE;

    /*** Generate the rotation positions, generate rotation keys, and load rotation keys */
    auto user_positions = annController.generate_users_rotation_positions(numUsers, embedding_space);
    auto begin_rotkeygen_time = startTime();
    cout << "This is the rotation positions (" << user_positions.size() <<"): " << user_positions << endl;
    fheController.generate_rotation_keys(user_positions, "rotation_keys.bin",  true);
    cout << "*****************************************************" << endl;
    printDuration(begin_rotkeygen_time, "Rotation KeyGen (position, gen, and load) Time", false);

    string modelName = "hybrid_"+std::string(MODEL) + "_" + DATASET + "_" + std::to_string(embedding_space);

    string filesDirName = "./../embedding_csv/"+string(DATASET)+"/"+string(MODEL)+"/"+to_string(embedding_space);
    
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
            
            // auto normIn = get_current_time();
            // Ctext normCtext = fheController.secure_batch_norm(encryptedInput, 1<<numSlots);
            // operationTime.push_back(measure_time(normIn, get_current_time()));
            // totalTime(operationTime, "Batch Norm");
            // operationTime.clear();

            /*****
             * carry out the user operation which is either identification or registration *******/            
            if (strcmp(operation.c_str(), "authenticate") == 0) {
                auto operationIn = get_current_time();
                // Ctext results = fheController.secure_cosine_similarity(encryptedInput, modelName, embedding_space, rotPositions);
                Ctext results = fheController.secure_cosine_similarity_hybrid(encryptedInput, modelName, embedding_space, rotPositions);
                operationTime.push_back(measure_time(operationIn, get_current_time()));
                measuringTime.push_back(measure_time(operationIn, get_current_time()));
                cout << "------ Identified User: " << endl;

                auto enIn = get_current_time();
                fheController.resultsInterpreter(results, NUM_TO_READ, idx);
                enInTime.push_back(measure_time(enIn, get_current_time()));
                totalTime(enInTime, "Interpret");
                
                totalTime(operationTime, "Identification");
                operationTime.clear();
            }

            else if (strcmp(operation.c_str(), "register") == 0) {
                auto operationIn = get_current_time();
                int result = fheController.secure_user_registration(encryptedInput, idx, modelName, embedding_space, rotPositions);                
                operationTime.push_back(measure_time(operationIn, get_current_time()));
                measuringTime.push_back(measure_time(operationIn, get_current_time()));
                cout << "------ Registration status: "<< result << endl;
                totalTime(operationTime, "register");
                operationTime.clear();
            }

            totalTime(measuringTime, "");
            measuringTime.clear();
        }
        operation = "authenticate";
        cout << "\n##################################### IDENTIFICATION  ##########################\n" << endl;
    }
   return 0;
}


