
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

/**
 * @brief FHE Controller for defining and managing homomorphic ANN functions.
 *
 * This class provides different methods for convolution, pooling, fully connected
 *  relu, etc for neural network development on encrypted data using FHE.
 */


#include <fstream>
#include <filesystem> 

namespace fs = std::filesystem;

#include "FHEController.h"

/***
 * Use to calculte the PQ which determines the security level of application
 */
double getlogPQ(const DCRTPoly& poly) {
    int n = poly.GetNumOfElements();
    double logPQ = 0;
    for (int i = 0; i < n; i++) {
        auto qi = poly.GetParams()->GetParams()[i]->GetModulus(); 
        // std::cout << log(qi.ConvertToDouble()) / log(2) << std::endl;
        logPQ += log(qi.ConvertToDouble()) / log(2);
    }
    
    return logPQ;
}


/******************************************************************************************************************
 * Author: Nges Brian,
 * Date: July 11, 2024
 * This function is used to generate the context of the project. For a Start, we are using Ring 
 * 65536 which is the most standard version of CKKS that can be used and everything is set to work with it
 ***************************************************************************************************************/
void FHEController::generate_context(int ringDim, int numSlots, int mlevelBootstrap, 
                        int dcrtBits, int firstMod,   int numDigits, 
                        double initVal, vector<uint32_t> levelBudget, bool serialize) {

    CCParams<CryptoContextCKKSRNS> parameters;
    auto secretKeyDist = SPARSE_TERNARY;

    // ScalingTechnique rescaleTech = FLEXIBLEAUTO;
    // ScalingTechnique rescaleTech = FLEXIBLEAUTOEXT;
    ScalingTechnique rescaleTech = FIXEDAUTO;
    level_budget = levelBudget;
    num_slots = 1 << numSlots;
    mult_depth = mlevelBootstrap;

    parameters.SetRingDim(1 << ringDim);
    parameters.SetBatchSize(num_slots);
    parameters.SetScalingModSize(dcrtBits);
    parameters.SetFirstModSize(firstMod);
    parameters.SetNumLargeDigits(numDigits);
    
    parameters.SetSecretKeyDist(secretKeyDist);
    parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
    parameters.SetScalingTechnique(rescaleTech);
    //  parameters.SetSecurityLevel(lbcrypto::HEStd_128_classic);
    // parameters.SetDecryptionNoiseMode(FIXED_NOISE_DECRYPT);
    
    circuit_depth = mult_depth + FHECKKSRNS::GetBootstrapDepth(level_budget, secretKeyDist);
    parameters.SetMultiplicativeDepth(circuit_depth);

    cout << "Building the FHE Context" << endl;
    cout << "dcrtBits: "<< dcrtBits << " -- firstMod: " << firstMod << endl << "Ciphertexts depth: " 
         << circuit_depth << ", available multiplications: " << circuit_depth - 2 << endl;
   
    context = GenCryptoContext(parameters);
    context->Enable(PKE);
    context->Enable(KEYSWITCH);
    context->Enable(LEVELEDSHE);
    context->Enable(ADVANCEDSHE);
    context->Enable(FHE);
    
    cout << "Generate Keys ......" << endl;
    keyPair = context->KeyGen();
    context->EvalMultKeyGen(keyPair.secretKey);
    context->EvalSumKeyGen(keyPair.secretKey);

    ringDim = context->GetRingDimension();
    numSlots = ringDim / 2;
    usint halfnumSlots = numSlots/2;
    context->EvalBootstrapSetup(level_budget, bsgsDim, numSlots);
    context->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);

    /*** Initialize the ciphertext to be use for approximation */
    vector<double> initVec(numUsers, initVal);
    encnewtonInit = encrypt_inputData(initVec);

    auto sec_level = parameters.GetSecurityLevel();
    double logPQ = getlogPQ(keyPair.publicKey->GetPublicElements()[0]);
    cout << "Keys Generated." << endl;
    // cout << "Ciphertexts depth: " << circuit_depth << endl; 
    // cout << "Multiplication Depth: " << circuit_depth - 2 << endl;
    cout << "Cyclotomic Order: " << context->GetCyclotomicOrder() << endl;
    cout << "CKKS scheme is using ring dimension: " << ringDim  << endl;
    cout << "Avaliable numSlots: " << numSlots << "  - halfnumSlots: " << halfnumSlots << endl;
    cout << "Security Level: " << sec_level << endl;
    cout << "log PQ = " << logPQ << endl << endl;
    cout << "-----------------------------------------------------------" << endl;
    
    if(serialize){
        write_to_file(keys_folder + "/mult_depth.txt", to_string(mult_depth));
        write_to_file(keys_folder + "/num_slots.txt", to_string(num_slots));
        write_to_file(keys_folder + "/level_budget.txt", to_string(level_budget[0]) + "," + to_string(level_budget[1]));
        keys_serialization();
    }
    return;
}

void FHEController::keys_serialization(){
    cout << "------------------------------------------------------------" << endl;
    cout << "Now serializing keys ..." << endl;

    if (!fs::exists(keys_folder)) {
        if (!fs::create_directory(keys_folder)) {
            std::cerr << "Failed to create directory: " << keys_folder << std::endl;
            return;
        }
    }

    if (!Serial::SerializeToFile(keys_folder + "/crypto-context.bin", context, SerType::BINARY)) {
        cerr << "Error writing serialization of the crypto context to crypto-context.bin" << endl;
    } else {
        cout << "Crypto Context have been serialized" << std::endl;
    }

    ofstream multKeyFile(keys_folder + "/mult-keys.bin", ios::out | ios::binary);
    if (multKeyFile.is_open()) {
        if (!context->SerializeEvalMultKey(multKeyFile, SerType::BINARY)) {
            cerr << "Error writing eval mult keys" << std::endl;
            exit(1);
        }
        cout << "Relinearization Keys have been serialized" << std::endl;
        multKeyFile.close();
    }
    else {
        cerr << "Error serializing EvalMult keys in \"" << keys_folder + "/mult-keys.bin" << "\"" << endl;
        exit(1);
    }

    if (!Serial::SerializeToFile(keys_folder + "/public-key.bin", keyPair.publicKey, SerType::BINARY)) {
        cerr << "Error writing serialization of public key to public-key.bin" << endl;
    } else {
        cout << "Public Key has been serialized" << std::endl;
    }

    if (!Serial::SerializeToFile(keys_folder + "/secret-key.bin", keyPair.secretKey, SerType::BINARY)) {
        cerr << "Error writing serialization of public key to secret-key.bin" << endl;
    } else {
        cout << "Secret Key has been serialized" << std::endl;
    }
    return;
}

/********************************************************************
 * Author: Nges Brian,
 * Date: July 11, 2024
 * This function is used to load all the keys serialized and stored in files from the sskeys folder
 ********************************************************************/
void FHEController::load_context(bool verbose) {
    context->ClearEvalMultKeys();
    context->ClearEvalAutomorphismKeys();
    CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();

    cout << "------------------------------------------------------------" << endl;
    if (verbose) cout << "Reading serialized context..." << endl;

    if (!Serial::DeserializeFromFile(keys_folder + "/crypto-context.bin", context, SerType::BINARY)) {
        cerr << "I cannot read serialized data from: " << keys_folder + "/crypto-context.bin" << endl;
        exit(1);
    }

    PublicKey<DCRTPoly> clientPublicKey;
    if (!Serial::DeserializeFromFile(keys_folder + "/public-key.bin", clientPublicKey, SerType::BINARY)) {
        cerr << "I cannot read serialized data from public-key.bin" << endl;
        exit(1);
    }

    PrivateKey<DCRTPoly> serverSecretKey;
    if (!Serial::DeserializeFromFile(keys_folder + "/secret-key.bin", serverSecretKey, SerType::BINARY)) {
        cerr << "I cannot read serialized data from secret-key.bin" << endl;
        exit(1);
    }

    keyPair.publicKey = clientPublicKey;
    keyPair.secretKey = serverSecretKey;

    std::ifstream multKeyIStream(keys_folder + "/mult-keys.bin", ios::in | ios::binary);
    if (!multKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << "mult-keys.bin" << endl;
        exit(1);
    }
    if (!context->DeserializeEvalMultKey(multKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval multkey file" << endl;
        exit(1);
    }

    ifstream sumKeyIStream(keys_folder + "/sum-keys.bin", ios::in | ios::binary);
    if (!sumKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << "sum-keys.bin" << std::endl;
        exit(1);
    }
    if (!context->DeserializeEvalSumKey(sumKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval rot key file" << std::endl;
        exit(1);
    }

    mult_depth = stoi(read_from_file(keys_folder + "/mult_depth.txt"));
    level_budget[0] = read_from_file(keys_folder + "/level_budget.txt").at(0) - '0';
    level_budget[1] = read_from_file(keys_folder + "/level_budget.txt").at(2) - '0';

    uint32_t approxBootstrapDepth = 4 + 4;  
    uint32_t levelsUsedBeforeBootstrap = mult_depth;
    circuit_depth = levelsUsedBeforeBootstrap + FHECKKSRNS::GetBootstrapDepth(approxBootstrapDepth, level_budget, SPARSE_TERNARY);

    if (verbose) cout << "Circuit depth: " << circuit_depth << ", available multiplications: " << levelsUsedBeforeBootstrap - 2 << endl;

    cout << "Context Loaded" << endl;
    cout << "------------------------------------------------------------" << endl;
}

/***** Generate the Bootstrapping Keys */
void FHEController::generate_bootstrapping_keys(int bootstrap_slots, string filename, bool serialize) {
    
    context->EvalBootstrapSetup(level_budget, bsgsDim, 1<<bootstrap_slots);
    context->EvalBootstrapKeyGen(keyPair.secretKey, 1<<bootstrap_slots);
    
    context->EvalMultKeyGen(keyPair.secretKey);
    context->EvalSumKeyGen(keyPair.secretKey);

    if(serialize){
        ofstream multKeysFile(keys_folder + mult_prefix + filename, ios::out | ios::binary);
        if (multKeysFile.is_open()) {
            if (!context->SerializeEvalMultKey(multKeysFile, SerType::BINARY)) {
                cerr << "Error writing mult keys" << std::endl;
                exit(1);
            }
            cout << "mult keys \"" << filename << "\" have been serialized" << std::endl;
        } else {
            cerr << "Error serializing mult keys" << keys_folder + mult_prefix + filename << std::endl;
            exit(1);
        }

        ofstream sumKeysFile(keys_folder + sum_prefix + filename, ios::out | ios::binary);
        if (sumKeysFile.is_open()) {
            if (!context->SerializeEvalSumKey(sumKeysFile, SerType::BINARY)) {
                cerr << "Error writing sum keys" << std::endl;
                exit(1);
            }
            cout << "sum keys \"" << filename << "\" have been serialized" << std::endl;
        } else {
            cerr << "Error serializing sum keys" << keys_folder + sum_prefix + filename << std::endl;
            exit(1);
        }
    }
}

/** Genrate and serialize rotation keys */
void FHEController::generate_rotation_keys(const vector<int> rotations, std::string filename, bool serialize) {
    if (serialize && filename.size() == 0) {
        cout << "Filename cannot be empty when serializing rotation keys." << endl;
        return;
    }
    
    context->EvalRotateKeyGen(keyPair.secretKey, rotations);
    
    if (serialize) {
        ofstream rotationKeyFile(keys_folder + rotation_prefix + filename, ios::out | ios::binary);
        if (rotationKeyFile.is_open()) {
            if (!context->SerializeEvalAutomorphismKey(rotationKeyFile, SerType::BINARY)) {
                cerr << "Error writing rotation keys" << std::endl;
                exit(1);
            }
            cout << "Rotation keys \"" << filename << "\" have been serialized" << std::endl;
        } else {
            cerr << "Error serializing Rotation keys" << keys_folder + rotation_prefix + filename << std::endl;
            exit(1);
        }
    }
}

void FHEController::generate_bootstrapping_and_rotation_keys(vector<int> rotations, int bootstrap_slots, bool serialize, const string& filename) {
    if (serialize && filename.empty()) {
        cout << "Filename cannot be empty when serializing bootstrapping and rotation keys." << endl;
        return;
    }

    generate_bootstrapping_keys(bootstrap_slots, filename, serialize);
    generate_rotation_keys(rotations, filename, serialize);
}

void FHEController::load_bootstrapping_and_rotation_keys(const string& filename, int bootstrap_slots, bool verbose) {
    if (verbose) cout << endl << "Loading bootstrapping and rotations keys from " << filename << "..." << endl;

    context->EvalBootstrapSetup(level_budget, bsgsDim, 1<<bootstrap_slots);
    // context->EvalBootstrapKeyGen(keyPair.secretKey, 1<<bootstrap_slots);

    if (verbose)  cout << "(1/4) Bootstrapping precomputations completed!" << endl;
    
    ifstream multKeyIStream(keys_folder + mult_prefix + filename, ios::in | ios::binary);
    if (!multKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << keys_folder+ "/" << mult_prefix << filename << std::endl;
        exit(1);
    }
    if (!context->DeserializeEvalMultKey(multKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval rot key file" << std::endl;
        exit(1);
    }
    if (verbose)  cout << "(2/4) MultKey deserialized and loaded!" << endl;

    ifstream sumKeyIStream(keys_folder + sum_prefix + filename, ios::in | ios::binary);
    if (!sumKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << keys_folder+ "/" << sum_prefix << filename << std::endl;
        exit(1);
    }
    if (!context->DeserializeEvalSumKey(sumKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval rot key file" << std::endl;
        exit(1);
    }

    if (verbose)  cout << "(3/4) SumKey deserialized and loaded!" << endl;
    ifstream rotKeyIStream(keys_folder + rotation_prefix + filename, ios::in | ios::binary);
    if (!rotKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << keys_folder+ "/" << rotation_prefix << filename << std::endl;
        exit(1);
    }
    if (!context->DeserializeEvalAutomorphismKey(rotKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval rot key file" << std::endl;
        exit(1);
    }

    if (verbose) cout << "(4/4) Rotation keys deserialized and loaded!" << endl;
    if (verbose) cout << endl;
}


/***** Load rotation keys for a specific file */
void FHEController::load_rotation_keys(const string& filename, bool verbose) {

    if (verbose) cout << endl << "Loading rotations keys from " << filename << "..." << endl;
    
    ifstream rotKeyIStream(keys_folder + rotation_prefix + filename, ios::in | ios::binary);
    if (!rotKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " <<keys_folder + "/" << rotation_prefix << filename << std::endl;
        exit(1);
    }
    if (!context->DeserializeEvalAutomorphismKey(rotKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval rot key file" << std::endl;
        exit(1);
    }

    if (verbose) {
        cout << "(1/1) Rotation keys read!" << endl;
        cout << endl;
    }
}

/***Used to clear the rotation keys stored in context to generate others */
void FHEController::clear_rotation_keys() {
    // context->ClearEvalMultKeys();
    context->ClearEvalAutomorphismKeys();
}

void FHEController::clear_bootstrapping_and_rotation_keys(int bootstrap_num_slots) {
    //This lines would free more or less 1GB or precomputations, but requires access to the GetFHE function

    // FHECKKSRNS* derivedPtr = dynamic_cast<FHECKKSRNS*>(context->GetScheme()->GetFHE().get());
    // derivedPtr->m_bootPrecomMap.erase(bootstrap_num_slots);

    context->ClearEvalMultKeys();
    // context->ClearEvalSumKeys();
    context->ClearEvalAutomorphismKeys();
}

/*** clear the whole context multiplication keys as well.  */
void FHEController::clear_context(int bootstrapping_key_slots) {
    
    if (bootstrapping_key_slots != 0)
        clear_bootstrapping_and_rotation_keys(bootstrapping_key_slots);
    else
        clear_rotation_keys();
    // context->ClearEvalMultKeys();
}

/*** Bootstrap a ciphertext */
Ctext FHEController::bootstrap_function(Ctext& cipher, int level){
    // int circuit_depth = 18;
    Ctext boots_cipher = context->EvalBootstrap(cipher, level);
    return boots_cipher;
}


Ptext FHEController::encode_packedVector(vector<double> packedVector) {
    Ptext plaintext = context->MakeCKKSPackedPlaintext(packedVector, 1, 1);
    return plaintext;
}

Ptext FHEController::encode_packedVector(vector<double> packedVector, int encode_level, int noSlots) {
    Ptext plaintext = context->MakeCKKSPackedPlaintext(packedVector, 1, encode_level, nullptr, noSlots);
    return plaintext;
}

Ptext FHEController::encode_shortcut_packedVector(vector<double> packedVector, int colsSquare) {
    int dim1 = packedVector.size();
    vector<double> main_kernel;
    for(int t =0; t< dim1; t++){
        double cell_value = packedVector[t];
        vector<double> repeated(colsSquare, cell_value);
        main_kernel.insert(main_kernel.end(), repeated.begin(), repeated.end());
    }
    Ptext plaintext = context->MakeCKKSPackedPlaintext(main_kernel, 1, 1);
    return plaintext;
}

Ptext FHEController::encode_baisVector(vector<double> packedVector, int colsSquare, int level) {
    int dim1 = packedVector.size();
    vector<double> main_kernel;
    for(int t =0; t< dim1; t++){
        double cell_value = packedVector[t];
        vector<double> repeated(colsSquare, cell_value);
        main_kernel.insert(main_kernel.end(), repeated.begin(), repeated.end());
    }

    Ptext plaintext = context->MakeCKKSPackedPlaintext(main_kernel, 1, level);
    return plaintext;
}

Ctext FHEController::encrypt_packedVector(vector<double> inputData) {
    Ptext plaintext = context->MakeCKKSPackedPlaintext(inputData, 1, 1);
    plaintext->SetLength(inputData.size());
    auto encryptImage = context->Encrypt(keyPair.publicKey, plaintext);
    return encryptImage;
}

Ctext FHEController::reencrypt_data(Ptext plaintextVector) {
    
    auto encryptedData = context->Encrypt(keyPair.publicKey, plaintextVector);
    return encryptedData;
}

Ptext FHEController::decrypt_packedVector(Ctext encryptedpackedVector, int cols) {
    
    Ptext plaintextDec;
    context->Decrypt(keyPair.secretKey, encryptedpackedVector, &plaintextDec);
    plaintextDec->SetLength(cols);
    return plaintextDec;
}

vector<vector<Ctext>> FHEController::encrypt_kernel(vector<vector<vector<double>>> kernelData, int colsSquare){
    size_t dim1 = kernelData.size();
    if (dim1 == 0) return {};
    size_t dim2 = kernelData[0].size();
    if (dim2 == 0) return {};
    size_t dim3 = kernelData[0][0].size();
    if (dim3 == 0) return {}; 

    vector<vector<Ctext>> encrypt_kernel; 
    for (size_t k=0; k< dim1; k++){
        vector<Ctext> filters;
        for (size_t i=0; i< dim2; i++){
            for (size_t j=0; j< dim3 ; j++){
                double cell_value  = kernelData[k][i][j];
                vector<double> repeated(colsSquare, cell_value);
                Ctext encrypted_val = encrypt_packedVector(repeated);
                filters.push_back(encrypted_val);
            }
        }
        encrypt_kernel.push_back(filters);
    }
    return encrypt_kernel;
}

/****************
 * Encoding kernels for the fully connected layers
 * Author: Nges Brian ****/
vector<Ptext> FHEController::encode_kernel(vector<double> kernelData, int colsSquare){
    size_t dim1 = kernelData.size();
    if (dim1 == 0) return {};

    vector<Ptext> encrypt_kernel; 
     for (size_t j=0; j< dim1 ; j++){
        double cell_value  = kernelData[j];
        vector<double> repeated(colsSquare, cell_value);
        Ptext encrypted_val = encode_packedVector(repeated);
        encrypt_kernel.push_back(encrypted_val);
    }
    return encrypt_kernel;
}

/**** Select values of the corresponding kernel positions for all kernels and repeat by widthSq 
 * then put this values together as one long vector , encode the vector and
 *  return k^2 vector of repeated kernel values*/
vector<Ptext> FHEController::encode_kernel(vector<vector<vector<double>>> kernelData, int colsSquare){
    size_t dim1 = kernelData.size();
    if (dim1 == 0) return {};
    size_t dim2 = kernelData[0].size();
    if (dim2 == 0) return {};
    size_t dim3 = kernelData[0][0].size();
    if (dim3 == 0) return {}; 
    // cout <<"input kernel shape: " << dim1 << "*" << dim2 << "*" << dim3 <<endl;

    int kernelsize_sq = pow(dim2, 2);
    vector<vector<double>> main_kernel(kernelsize_sq, vector<double>());
    for (size_t k=0; k< dim1; k++){
        vector<vector<double>> filters;
        for (size_t i=0; i< dim2; i++){
            for (size_t j=0; j< dim3 ; j++){
                double cell_value  = kernelData[k][i][j];
                vector<double> repeated(colsSquare, cell_value);
                filters.push_back(repeated);
            }
        }
        for(int t =0; t< kernelsize_sq; t++){   
            main_kernel[t].insert(main_kernel[t].end(), filters[t].begin(), filters[t].end());
        }
    }
    vector<Ptext> encoded_kernel;
    for( int s =0; s< kernelsize_sq; s++){
        // cout << "Kernel size: " << main_kernel[s].size() << endl;
        Ptext encoded_val = encode_packedVector(main_kernel[s]);
        encoded_kernel.push_back(encoded_val);
    }
    return encoded_kernel;
}


/*** 
 * Author: Nges Brian
 * This function is used to set the number of slots in the ciphertext after downsampling. 
 * It is used to improve the performance by reducing the size of polynomial we are dealing with.*/
Ctext FHEController::change_numSlots(Ctext& encryptedInput, uint32_t numSlots){
    encryptedInput->SetSlots(1 << numSlots);
    return encryptedInput;
}

/**** Read the predicted Value */
int FHEController::read_inferencedValue(Ctext& inferencedData, int noElements, std::ofstream* outFile) {
    auto decryptedValue = decrypt_packedVector(inferencedData, noElements);
    auto decryptedVector = decryptedValue->GetRealPackedValue();

    auto maxElementIt = std::max_element(decryptedVector.begin(), decryptedVector.end());
    int maxIndex = std::distance(decryptedVector.begin(), maxElementIt);

    std::cout << "Predicted Value : " << maxIndex 
              << " Weight:  " << decryptedVector[maxIndex] << std::endl;

    if (outFile) {
        if (outFile->is_open()) {          // <-- dereference pointer
            (*outFile) << maxIndex << std::endl;
        } else {
            std::cout << "Unable to open file." << std::endl;
        }
    }

    return 0;
}

/**** Helper function to determine the min and max of data */
int FHEController::read_minmaxValue(Ctext inferencedData, int noElements){
    auto decryptedValue = decrypt_packedVector(inferencedData, noElements);
    auto decryptedVector = decryptedValue->GetRealPackedValue();

    // cout << "Decrypted Vector " << decryptedVector << endl;
    auto maxElementIt = max_element(decryptedVector.begin(), decryptedVector.end());
    int maxIndex = distance(decryptedVector.begin(), maxElementIt);
    auto minElementIt = min_element(decryptedVector.begin(), decryptedVector.end());
    int minIndex = distance(decryptedVector.begin(), minElementIt);
    cout << "------------------------------------------------------------------ " << endl;
    cout << "Range [ " << decryptedVector[minIndex] << " , " << decryptedVector[maxIndex] <<" ]" << endl;
    cout << "Index: " << maxIndex << endl;
    cout << "------------------------------------------------------------------ " << endl;
    return 0;
}

/**** this is temporal function to get the maximium value every convolution to use in relu.  */
int FHEController::read_scalingValue(Ctext inferencedData, int noElements){
    auto decryptedValue = decrypt_packedVector(inferencedData, noElements);
    auto decryptedVector = decryptedValue->GetRealPackedValue();

    double maxAbsValue = *std::max_element(decryptedVector.begin(), decryptedVector.end(), [](int a, int b) {
        return std::abs(a) < std::abs(b);
    });
    int roundedMaxAbsValue = static_cast<int>(std::ceil(std::abs(maxAbsValue)));
    return roundedMaxAbsValue;
}


/************************************************************************************************************************************************** 
 * 
 * SENTINALTOUCH - FINGERPRINT IDENTIFICATION AND AUTHENTICATION
 * AUTHOR: NGES BRIAN
 * 
************************************************************************************************************************************************/

/*** 
 * Action: Function is used to encrypted a packed
 * **/ 
Ctext FHEController::encrypt_inputData(vector<double> inputData) {
    Ptext plaintext = context->MakeCKKSPackedPlaintext(inputData, 1, 1);
    plaintext->SetLength(inputData.size());
    auto encryptImage = context->Encrypt(keyPair.publicKey, plaintext);
    return encryptImage;
}

void FHEController::decryptAndPrint(Ctext encryptedpackedVector, int cols) {
    
    Ptext plaintextDec;
    context->Decrypt(keyPair.secretKey, encryptedpackedVector, &plaintextDec);
    plaintextDec->SetLength(cols);
    vector<complex<double>> finalResult = plaintextDec->GetCKKSPackedValue();
    cout << finalResult << endl;
    cout << endl;
}

int FHEController::storeUsers(vector<Ctext>& registeredUsers, string modelName, int embedding_space){
    // cout << "Store User: " << "model name: " << modelName << endl;
    // decryptAndPrint(registeredUsers[0], num_users);
    // for(int i=0; i<embedding_space; i++){
    //     if (!lbcrypto::Serial::SerializeToFile(users_prefix+modelName+"_"+"ctxt_"+to_string(i)+".bin", registeredUsers[i], lbcrypto::SerType::BINARY)) {
    //         std::cerr << "Error Serializing ciphertext to file" << std::endl;
    //         return 1;
    //     }
    // }

    /*** work on the fly */
    users = registeredUsers;
    return 0;
}

vector<Ctext> FHEController::reeadRegisteredUsers(string modelName, int embedding_space){

    return users;
    // vector<Ctext> deserializedUsersCiphertext(embedding_space);
    
    // for(int i=0; i<embedding_space; i++){
    //     if (!Serial::DeserializeFromFile(users_prefix+modelName+"_"+"ctxt_"+to_string(i)+".bin", deserializedUsersCiphertext[i], SerType::BINARY)) {
    //         std::cerr << "Error deserializing ciphertext from file" << std::endl;
    //         // Handle error
    //     }
    // }
    // return deserializedUsersCiphertext;
}


Ctext FHEController::readEncryptedUserIndex(Ctext &encryptedInput, int numUsers){

    cout << "Start encrypted user identification" << endl;
    return encryptedInput;
}

int FHEController::resultsInterpreter(Ctext& inferencedData, int noElements, int label){

    auto decryptedValue = decrypt_packedVector(inferencedData, noElements);
    auto resultVec = decryptedValue->GetRealPackedValue();

    cout << "Results: " << resultVec << endl; 
    cout << "********************************************" << endl;

    // auto maxElementIt = std::max_element(resultVec.begin(), resultVec.end());
    // int maxIndex = std::distance(resultVec.begin(), maxElementIt);

    // std::cout << "User with Max Weight: " << maxIndex+1 
    //         << " Weight:  " << resultVec[maxIndex] << endl;

    // cout << "Decipher Values : " << resultVec  << endl;
    for(int i=0; i< noElements; i++){
        // double fractionalPart = resultVec[i] - floor(resultVec[i]); 
        // if(fractionalPart > 0.50 && fractionalPart < 1.5){
        if(resultVec[i] > 0.50){
            if(i == label){
                    
                cout << " *******CORRECT PREDICTION*************" << endl; 
                cout << " IDX:  " << label+1 << " -- Weight: " << resultVec[i] << endl;
            }
            else{
                // cout << "predicted client i: " << i << " -- True Label: " << label << endl; 
            }
        }
        else{
            //
        }
    }
    return 0;
}


int FHEController::secure_user_registration(Ctext &encryptedInput, int user_id, string model_name, int embeddings_space, int rotation_index){

    auto digits = context->EvalFastRotationPrecompute(encryptedInput);
    int num_users = numUsers;
    Ptext cleaning_mask = create_index_mask(0, num_users);
    Ctext tempIn = encryptedInput->Clone(); 
    vector<Ctext> userEmbSpace; 

    // cout << "Number of Users: " << num_users << " -- User ID: " << user_id << endl;
    /** create the user vertical representation */
    for(int i=0; i<embeddings_space; i++){
        if(i > 0){
            tempIn = context->EvalFastRotation(encryptedInput, i, context->GetCyclotomicOrder(), digits);
        } 
        if(user_id > 0){
            tempIn = context->EvalRotate(context->EvalMult(tempIn, cleaning_mask), -user_id);
        } else {
            tempIn = context->EvalMult(tempIn, cleaning_mask);
        }
        userEmbSpace.push_back(tempIn);
    }

    /**** Deserialize dataset and add user. */
    vector<Ctext> regUsers;
    if(user_id != 0){
        regUsers = reeadRegisteredUsers(model_name, embeddings_space);
        for(int i=0; i<embeddings_space; i++){
            regUsers[i] = context->EvalAdd(regUsers[i], userEmbSpace[i]);
        }
    }
    else{
        regUsers = userEmbSpace;
    }
    return storeUsers(regUsers, model_name, embeddings_space);
}


int FHEController::secure_user_deletion(int user_id, string model_name, int embeddings_space){
    int num_users = numUsers;
    Ptext cleaning_mask = create_index_mask(user_id, num_users);
    /**** Deserialize dataset and add user. */
    vector<Ctext> regUsers = reeadRegisteredUsers(model_name, embeddings_space);
    for(int i=0; i<embeddings_space; i++){
        regUsers[i] = context->EvalMult(regUsers[i], cleaning_mask);
    }

    return storeUsers(regUsers, model_name, embeddings_space);
}


Ctext FHEController::secure_cosine_similarity_hybrid(Ctext& encryptedInput, string model_name, int embedding_space, int rotation_index){
    // /***
    //  * Create the appropriate ciphertext format to compare.  */
    auto digits = context->EvalFastRotationPrecompute(encryptedInput);

    int num_users = numUsers;
    Ptext onesVec = create_index_mask(0, num_users);

    vector<Ctext> inputVec;
    Ctext tempIn = encryptedInput->Clone();
    for(int i=0; i<embedding_space; i++){
        if(i > 0){
            tempIn = context->EvalFastRotation(encryptedInput, i, context->GetCyclotomicOrder(), digits);
        }

        /** prepare the values to be equal to the number of users */
        tempIn = context->EvalMult(tempIn, onesVec);
        tempIn = context->EvalAdd(tempIn, context->EvalRotate(tempIn, -1));
        for(int j=1; j< log2(num_users); j++){
            int powVal = pow(2, j);
            tempIn = context->EvalAdd(tempIn, context->EvalRotate(tempIn, -powVal));
        }
        inputVec.push_back(tempIn);
    }
    
    /***
     * Calculate the dot product for all users in parral.*/
    vector<Ctext> dotVec;
    vector<Ctext> regUsers = reeadRegisteredUsers(model_name, embedding_space);
    for(int i=0; i<embedding_space; i++){
        tempIn = context->EvalMult(regUsers[i], inputVec[i]);
        dotVec.push_back(tempIn);
    }
    Ctext dotProduct = context->EvalAddMany(dotVec);
    return dotProduct;
}


Ctext FHEController::secure_cosine_similarity(Ctext& encryptedInput, string model_name, int num_users, int embedding_space, int rotation_index){
    
    /***
     * Create the appropriate ciphertext format to compare.  */
    auto digits = context->EvalFastRotationPrecompute(encryptedInput);
    num_users = nextPowerOf2(num_users);
    Ptext onesVec = create_index_mask(0, num_users);

    vector<Ctext> inputVec;
    Ctext tempIn = encryptedInput->Clone();
    for(int i=0; i<embedding_space; i++){
        if(i > 0){
            tempIn = context->EvalFastRotation(encryptedInput, i, context->GetCyclotomicOrder(), digits);
        }

        /** prepare the values to be equal to the number of users */
        tempIn = context->EvalMult(tempIn, onesVec);
        tempIn = context->EvalAdd(tempIn, context->EvalRotate(tempIn, -1));
        for(int j=1; j< log2(num_users); j++){
            int powVal = pow(2, j);
            tempIn = context->EvalAdd(tempIn, context->EvalRotate(tempIn, -powVal));
        }
        inputVec.push_back(tempIn);
    }

    /***
     * Calculate the dot product for all users in parral.*/
    vector<Ctext> dotVec;
    vector<Ctext> regUsers = reeadRegisteredUsers(model_name, embedding_space);
    for(int i=0; i<embedding_space; i++){
        tempIn = context->EvalMult(regUsers[i], inputVec[i]);
        dotVec.push_back(tempIn);
    }
    Ctext dotProduct = context->EvalAddMany(dotVec);

    /***
     *  Magnitudes of encrypted input */
    vector<Ctext> sqUserVec, sqRegVec;
    for(int i =0; i<embedding_space; i++){
        sqUserVec.push_back(context->EvalSquare(inputVec[i]));
        sqRegVec.push_back(context->EvalSquare(regUsers[i]));
    }
    Ctext squserInput = context->EvalAddMany(sqUserVec);
    Ctext sqReg = context->EvalAddMany(sqRegVec);


    /** multiply the sqs of the magnitude and approximate their 
     * inverse sqrt of the magnitude */
    Ctext magnitudeProduct = context->EvalMult(squserInput, sqReg);
    magnitudeProduct = bootstrap_function(magnitudeProduct);
    

    Ctext approxResults = approx_rsqrt(magnitudeProduct);
    // approxResults = bootstrap_function(approxResults);
    return approxResults;
    Ctext result = context->EvalMult(dotProduct, approxResults);
    
    return result; 
}


Ctext FHEController::secure_cosine_similarity(Ctext& encryptedInput, string model_name, int embedding_space, int rotation_index){
    
    /***
     * Create the appropriate ciphertext format to compare.  */
    auto digits = context->EvalFastRotationPrecompute(encryptedInput);
    int num_users = numUsers;
    Ptext onesVec = create_index_mask(0, num_users);

    vector<Ctext> inputVec;
    Ctext tempIn = encryptedInput->Clone();
    for(int i=0; i<embedding_space; i++){
        if(i > 0){
            tempIn = context->EvalFastRotation(encryptedInput, i, context->GetCyclotomicOrder(), digits);
        }

        /** prepare the values to be equal to the number of users */
        tempIn = context->EvalMult(tempIn, onesVec);
        tempIn = context->EvalAdd(tempIn, context->EvalRotate(tempIn, -1));
        for(int j=1; j< log2(num_users); j++){
            int powVal = pow(2, j);
            tempIn = context->EvalAdd(tempIn, context->EvalRotate(tempIn, -powVal));
        }
        inputVec.push_back(tempIn);
    }

    /***
     * Calculate the dot product for all users in parral.*/
    vector<Ctext> dotVec;
    vector<Ctext> regUsers = reeadRegisteredUsers(model_name, embedding_space);
    for(int i=0; i<embedding_space; i++){
        tempIn = context->EvalMult(regUsers[i], inputVec[i]);
        dotVec.push_back(tempIn);
    }
    Ctext dotProduct = context->EvalAddMany(dotVec);

    // /***
    // * Magnitudes of encrypted input */
    vector<Ctext> sqUserVec, sqRegVec;
    for(int i =0; i<embedding_space; i++){
        sqUserVec.push_back(context->EvalSquare(inputVec[i]));
        sqRegVec.push_back(context->EvalSquare(regUsers[i]));
    }
    Ctext squserInput = context->EvalAddMany(sqUserVec);
    Ctext sqReg = context->EvalAddMany(sqRegVec);

    /** multiply the sqs of the magnitude and approximate their 
     * inverse sqrt of the magnitude */
    Ctext magnitudeProduct = context->EvalMult(squserInput, sqReg);
    magnitudeProduct = bootstrap_function(magnitudeProduct);
    
    Ctext approxResults = approx_rsqrt(magnitudeProduct);
    Ctext result = context->EvalMult(dotProduct, approxResults);
    return result; 
}


/******* the square function */
Ctext FHEController::approx_sqrt(Ctext encryptedInput){
    auto ct_a = encryptedInput; 
    auto ct_b = context->EvalSub(ct_a, 1);

    int d = 5; 
    for (int n = 0; n < d; n++) {
        auto ct1 = context->EvalSub(1, context->EvalMult(ct_b, 0.5));
        ct_a = context->EvalMult(ct_a, ct1);
        auto ct2 = context->EvalMult(context->EvalSub(ct_b, 3), 0.25);
        ct_b = context->EvalMult(context->EvalSquare(ct_b), ct2);
    }
    return ct_a;

}

/*** Newton–Raphson Iteration (double-precision) */
Ctext FHEController::approx_rsqrt(Ctext encryptedInput) {
    Ctext y = encnewtonInit->Clone();
    int d = 2;  
    for (int n = 0; n < d; n++) {
        auto y2 = context->EvalSquare(y);
        auto g  = context->EvalMult(encryptedInput, y2);
        auto t = context->EvalSub(1.5, context->EvalMult(0.5, g));
        y = context->EvalMult(y, t);
        // y = bootstrap_function(y);
    }

    return y;
}


Ptext FHEController::create_index_mask(int index, int vectorLen) {
    vector<double> mask(vectorLen, 0.0);
    if (index >= 0 && index < vectorLen) {
        mask[index] = 1.0;
    }
    return context->MakeCKKSPackedPlaintext(mask, 1, 1);
}


Ctext FHEController::secure_batch_norm(Ctext &encryptedInput, int noElements){
    
    /** Compute the mean */
    Ctext sumVal = context->EvalSum(encryptedInput, noElements);
    double scaler = 1.0/noElements;
    Ctext meanVal = context->EvalMult(sumVal, scaler);

    /**** Repeat the mean for variance calculation */
    Ctext repeatedMean = context->EvalAdd(meanVal, context->EvalRotate(meanVal, -1));
    for(int j=1; j<log2(noElements); j++){
        int powVal = pow(2, j);
        repeatedMean = context->EvalAdd(repeatedMean, context->EvalRotate(repeatedMean, -powVal));
    }

    /**** Compute the variance */
    Ctext varVal = context->EvalMult(encryptedInput, repeatedMean);
    varVal = context->EvalSum(context->EvalSquare(varVal), noElements);
    varVal = context->EvalMult(varVal, scaler);

    Ctext topVec = context->EvalSub(encryptedInput, repeatedMean);

    Ctext approxResults = approx_rsqrt(varVal);
    Ctext result = context->EvalMult(topVec, approxResults);

    return result; 
}

