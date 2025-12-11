
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

#ifndef FHEON_FHECONTROLLER_H
#define FHEON_FHECONTROLLER_H

#include <thread>
#include <openfhe.h>

#include <ciphertext-ser.h>
#include <cryptocontext-ser.h>
#include <key/key-ser.h>
#include <scheme/ckksrns/ckksrns-ser.h>
#include "Utils.h"
#include "UtilsData.h"

using namespace lbcrypto;
using namespace std;
using namespace std::chrono;
using namespace utils;
using namespace utilsdata;

using Ptext = Plaintext;
using Ctext = Ciphertext<DCRTPoly>;

class FHEController {

protected:
    CryptoContext<DCRTPoly> context;

public:

    int circuit_depth;
    int num_slots;
    int pLWE;
    int mult_depth = 10;
    string keys_folder = "./../HEkeys/";
    string rotation_prefix = "rotation_keys_"; 
    string mult_prefix = "mult_keys_";
    string sum_prefix = "sum_keys_";
    string users_prefix = "./../encrypted_users/";
    int numUsers = 100;
    Ctext encnewtonInit; 

    FHEController(CryptoContext<DCRTPoly> ctx) : context(ctx) {}
    CryptoContext<DCRTPoly> getContext() const {
        return context;
    }

    void setNumUsers(int num_users){
        numUsers = num_users;
    }

    /*
     * Generating context, bootstrapping keys, rotation keys and loading them */
    void generate_context(int ringDim=15, int numSlots=14, int mlevelBootstrap=10, int dcrtBits=55, int firstMod=56,   
                        int numDigits=3, double initVal = 0.4523, vector<uint32_t> levelBudget ={4, 4}, bool serialize=true);

    void generate_bootstrapping_keys(int bootstrap_slots, string filename, bool serialize);
    void generate_rotation_keys(vector<int> rotations, string filename = "", bool serialize = true);
    void generate_bootstrapping_and_rotation_keys(vector<int> rotations, int bootstrap_slots, bool serialize, const string& filename);
    
    void load_context(bool verbose = true);
    void load_rotation_keys(const string& filename, bool verbose=true);
    void load_bootstrapping_and_rotation_keys(const string& filename, int bootstrap_slots, bool verbose);

    void clear_rotation_keys();
    void clear_context(int bootstrapping_key_slots);
    void clear_bootstrapping_and_rotation_keys(int bootstrap_num_slots);

    Ctext bootstrap_function(Ctext& ciphertext, int level = 2);
    Ctext encrypt_packedVector(vector<double> inputData);
    Ctext reencrypt_data(Ptext plaintextVector);
    Ptext encode_packedVector(vector<double> packedVector);
    Ptext encode_packedVector(vector<double> packedVector, int encode_level, int noSlots);
    Ptext decrypt_packedVector(Ctext encryptedpackedVector, int cols);
    
    vector<vector<Ctext>> encrypt_kernel(vector<vector<vector<double>>> kernelData, int colsSquare);
    vector<Ptext> encode_kernel(vector<vector<vector<double>>> kernelData, int colsSquare);
    vector<Ptext> encode_kernel(vector<double> kernelData, int colsSquare);
    Ptext encode_shortcut_packedVector(vector<double> packedVector, int colsSquare);
    Ptext encode_baisVector(vector<double> packedVector, int colsSquare, int level=1);

    Ctext change_numSlots(Ctext& encryptedInput, uint32_t numSlots);
    int read_inferencedValue(Ctext& inferencedData, int noElements, ofstream* outFile = nullptr);
    int read_minmaxValue(Ctext inferencedData, int noElements);
    int read_scalingValue(Ctext inferencedData, int noElements);
    int resultsInterpreter(Ctext& inferencedData, int noElements, int label);

    /***************************************    SENTINALTOUCH    ***********************************/
    int secure_user_registration(Ctext& user_embeddings, int user_id, string model_name, int embedding_space, int rotation_index=8);
    int secure_user_deletion(int user_id, string modelName, int embeddings_space);
    vector<Ctext> reeadRegisteredUsers(string modelName, int embedding_space);
    int storeUsers(vector<Ctext>& registeredUsers, string modelName, int embedding_space);
    Ctext encrypt_inputData(vector<double> inputData);
    void decryptAndPrint(Ctext encryptedpackedVector, int cols);
    Ctext approx_sqrt(Ctext encryptedInput);
    Ctext approx_rsqrt(Ctext encryptedInput);
    
    Ctext secure_cosine_similarity(Ctext& encryptedInput, string model_name, int embedding_space=16, int rotation_index=8);
    Ctext secure_cosine_similarity(Ctext& encryptedInput, string model_name, int num_users, int embedding_space=16, int rotation_index=8);
    Ctext secure_cosine_similarity_hybrid(Ctext& encryptedInput, string model_name, int embedding_space, int rotation_index);
    Ctext secure_batch_norm(Ctext &encryptedInput, int noElements);

    Ctext readEncryptedUserIndex(Ctext &encryptedInput, int numUsers);

private:
    KeyPair<DCRTPoly> keyPair;
    vector<uint32_t> level_budget = {4, 4};
    vector<uint32_t> bsgsDim = {0, 0};
    
    vector<Ctext>users;
    
    void keys_serialization();
    Ptext create_index_mask(int index, int vectorLen);
};


#endif //FHEON_FHECONTROLLER_H
