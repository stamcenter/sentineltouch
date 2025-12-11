
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
 * @brief FHE controller for defining and managing homomorphic ANN functions.
 *
 * This class provides different methods for convolution, pooling, fully connected
 *  relu, etc for neural network development on encrypted data using FHE.
 */


#include <fstream>
#include <filesystem>
#include <iostream>
#include <cmath>
#include <thread>
#include "ANNController.h"

namespace fs = std::filesystem;

void ANNController::setContext(CryptoContext<DCRTPoly>& in_context){
    context = in_context;
}

/****
 * Author: Nges Brian, Jul 19, 2024
 *  This function is to generate the rotation positons ******/


/*** 
 * Author: Nges Brian
 * Generate the rotation keys for the Convolution  layers.
 * Calculate the output shape and use the output width to determine the rotation positions for all layers **/
vector <int> ANNController::generate_conv_rotation_positions(int imgWidth, int inputChannels, int outputChannels,
                                                    int kernelSize, int paddingSize, int StrideLen){
        
    vector<int> keys_position;
    int imgWidth_sq = pow(imgWidth, 2);
    int padded_width = imgWidth+(2*paddingSize);
    int padding_width_sq = pow(padded_width, 2);
    int width_out = ((padded_width - (kernelSize - 1) - 1)/StrideLen)+1;
    int width_out_sq = pow(width_out,2);
    keys_position.push_back(imgWidth);
    keys_position.push_back(padded_width);
    keys_position.push_back(padding_width_sq);
    keys_position.push_back(imgWidth_sq);
    keys_position.push_back(width_out);
    keys_position.push_back(width_out_sq);
    int rot_val;

    /** Convolution rotations */
    for(int i=1; i < kernelSize;i++){
        keys_position.push_back(i);
    }

    for(int i=1; i<width_out; i++){
        rot_val = (i*width_out);
        keys_position.push_back(-rot_val);
    }
    for(int i=1; i<outputChannels; i++){
        rot_val = (i*width_out_sq);
        keys_position.push_back(-rot_val);
    }
    
    std::sort(keys_position.begin(), keys_position.end());
    auto new_end = std::remove(keys_position.begin(), keys_position.end(), 0);
    new_end = std::unique(keys_position.begin(), keys_position.end());
    unique(keys_position.begin(), keys_position.end());
    keys_position.erase(new_end, keys_position.end());
    std::sort(keys_position.begin(), keys_position.end());
    return keys_position;
}


/***
 * Author: Nges Brian
 * This is to generate the rotation positions for the optimized convolution. 
 * This is used alongiside optimized operations */
vector <int> ANNController::generate_optimized_convolution_rotation_positions(
    int imgWidth,  int inputChannels, int outputChannels, int StrideLen){
        
    vector<int> keys_position;
    int imgWidth_sq = pow(imgWidth, 2);
    int width_out = (imgWidth/StrideLen);
    int width_out_sq = pow(width_out,2);
    keys_position.push_back(-1);
    keys_position.push_back(1);
    keys_position.push_back(imgWidth_sq);
    keys_position.push_back(imgWidth);
    keys_position.push_back(-imgWidth);

    if(StrideLen > 1){
        for (int s=1; s<log2(width_out); s++) {
            keys_position.push_back( pow(2, s-1));
        }
        keys_position.push_back(pow(2, log2(width_out)-1));
        int rotAmount = (StrideLen * imgWidth - width_out);
        keys_position.push_back(rotAmount);

        for(int i=1; i<inputChannels; i++){
            int rot_val = (i*width_out_sq);
            keys_position.push_back(-rot_val);
        }
        int shift = (imgWidth_sq - width_out_sq)* ((outputChannels / StrideLen) - 1);
        keys_position.push_back(-shift);

        for(int i=1; i< outputChannels; i++){
            int rotateAmount = (i >= inputChannels) ? 
                            -inputChannels * width_out_sq - (i % inputChannels) * width_out_sq: 
                            -(i % inputChannels) * width_out_sq;
            keys_position.push_back(rotateAmount);
            // cout << "rotateAmount: " << rotateAmount << endl;
        }
    }
    else{
        for(int i=1; i<outputChannels; i++){
            int rot_val = (i*width_out_sq);
            keys_position.push_back(-rot_val);
        }
    }
    
    std::sort(keys_position.begin(), keys_position.end());
    auto new_end = std::remove(keys_position.begin(), keys_position.end(), 0);
    new_end = std::unique(keys_position.begin(), keys_position.end());
    unique(keys_position.begin(), keys_position.end());
    keys_position.erase(new_end, keys_position.end());
    std::sort(keys_position.begin(), keys_position.end());
    return keys_position;
}

vector <int> ANNController::generate_avgpool_optimized_rotation_positions(int imgWidth,  int inputChannels, 
                                        int kernelSize, int StrideLen, bool globalPooling){
    
    vector<int> keys_position;
    if(globalPooling){
        keys_position.push_back((imgWidth*imgWidth));
        keys_position.push_back(-inputChannels);
        return keys_position;
    }

    int width_avgpool_out = (imgWidth/StrideLen);
    int width_avgpool_sq = pow(width_avgpool_out, 2); 
    int width_sq = pow(imgWidth, 2);
    keys_position.push_back(width_sq);
    keys_position.push_back(imgWidth);
    keys_position.push_back(StrideLen);
    keys_position.push_back(width_avgpool_out);
    keys_position.push_back(width_avgpool_sq);
    keys_position.push_back((StrideLen*imgWidth));

    for(int i=1; i<inputChannels; i++){
        int rot_val = i*width_avgpool_sq;
        keys_position.push_back(-rot_val);
    }
    
    if(StrideLen>1){
        for (int s=1; s<log2(width_avgpool_out); s++) {
            keys_position.push_back( pow(2, s-1));
        }
        keys_position.push_back(pow(2, log2(width_avgpool_out)-1));
        int rotAmount = (StrideLen * imgWidth - width_avgpool_out);
        keys_position.push_back(rotAmount);
    }

    std::sort(keys_position.begin(), keys_position.end());
    auto new_end = std::remove(keys_position.begin(), keys_position.end(), 0);
    new_end = std::unique(keys_position.begin(), keys_position.end());
    unique(keys_position.begin(), keys_position.end());
    keys_position.erase(new_end, keys_position.end());
    std::sort(keys_position.begin(), keys_position.end());
    return keys_position;
}

/*** 
 * Author: Nges Brian
 * Generate the Fully connected layer rotation positions.
 * Here, we use one function to generate rotation keys for all fully connected layer. 
 * - Take the maximium output from the fc layers and the maximuim channelsOutput layer.
 * - Generate keys taking the maximum output channels divided by the maxium channels output 
 * since we already have 0...maxchanneloutput from the convolution layers */
vector <int> ANNController::generate_fullyconnected_rotation_positions(int maxFCLayeroutputs, int rotationPositions){
    vector<int> keys_position;
    for(int counter=0; counter<maxFCLayeroutputs; counter+=rotationPositions){
        // int rot_val =counter*rotationPositions;
        keys_position.push_back(-counter);
    }
    for(int i=1; i<=rotationPositions; i++){
        keys_position.push_back(i);
    }

    std::sort(keys_position.begin(), keys_position.end());
    auto new_end = std::remove(keys_position.begin(), keys_position.end(), 0);
    new_end = std::unique(keys_position.begin(), keys_position.end());
    unique(keys_position.begin(), keys_position.end());
    keys_position.erase(new_end, keys_position.end());
    std::sort(keys_position.begin(), keys_position.end());
    return keys_position;
}

vector <int> ANNController::generate_users_rotation_positions(int num_users, int embeddings_space){
    vector<int> keys_position;
    for(int i=1; i< log2(num_users); i++){
        int powVal = pow(2, i);
        keys_position.push_back(-powVal);
    }

    for(int i=1; i < embeddings_space; i++){
         keys_position.push_back(i);
    }

    return keys_position;
}


/*****
 * Author: Nges Brian
 * This function is defined to caryout con2d.
 * It takes in an encrypted vecor, imgWith, kernelSize, Kernel and strieLen*/
Ctext ANNController::secure_conv2D(Ctext& encryptedInput, vector<vector<Ptext>>& kernelData, Ptext& baisInput,
        int inputWidth, int inputChannels, int outputChannels,  int kernelWidth, int paddingLen, int strideLen) {

    int kernelSize = kernelWidth * kernelWidth;
    inputWidth += 2 * paddingLen;
    int inputSize = inputWidth * inputWidth;
    int outputWidth = ((inputWidth - kernelWidth) / strideLen) + 1;
    int outputSize = outputWidth * outputWidth;
    int encode_level = encryptedInput->GetLevel();

    // STEP 1 - Generate mixed mask for cleaning multi-channel inputs
    int zero_elements = inputSize * (inputChannels - 1);
    if(inputChannels  < 2){
        zero_elements = inputSize;
    }
    vector<double> mixed_mask = generate_mixed_mask(inputSize, zero_elements);
    Ptext cleaning_mask = context->MakeCKKSPackedPlaintext(mixed_mask, 1, encode_level);

    vector<double> mixed_mask_out = generate_mixed_mask(outputWidth, zero_elements);
    Ptext cleaning_mask_out = context->MakeCKKSPackedPlaintext(mixed_mask_out, 1, encode_level);

    // STEP 2 - ROTATE INPUT TO FORM k^2 slices
    vector<Ctext> rotated_ciphertexts;
    for (int i = 0; i < kernelWidth; i++) {
        if(i >0){
            encryptedInput = context->EvalRotate(encryptedInput, inputWidth);
        }
        rotated_ciphertexts.push_back(encryptedInput);
        for (int j = 1; j < kernelWidth; j++) {
            rotated_ciphertexts.push_back(context->EvalRotate(encryptedInput, j));
        }
    }

    // STEP 3-6 - Convolution over all output channels
    Ctext strided_cipher;
    vector<Ctext> final_vec;
    for (int out_ch = 0; out_ch < outputChannels; out_ch++) {
        vector<Ctext> mult_results;

        // Per-kernel value multiplies
        for (int k = 0; k < kernelSize; k++) {
            mult_results.push_back(context->EvalMult(rotated_ciphertexts[k], kernelData[out_ch][k]));
        }

        Ctext conv_sum = context->EvalAddMany(mult_results);

        // STEP 4 - Sum all input channels (rotating and adding)
        if (inputChannels > 1) {
            vector<Ctext> channel_sums = { conv_sum };
            for (int ch = 1; ch < inputChannels; ch++) {
                conv_sum = context->EvalRotate(conv_sum, inputSize);
                channel_sums.push_back(conv_sum);
            }
            conv_sum = context->EvalAddMany(channel_sums);
        }
        conv_sum = context->EvalMult(conv_sum, cleaning_mask);

        // STEP 5 - Striding
        if (strideLen > 1) {
            strided_cipher = generalized_downsample(conv_sum, inputWidth, strideLen);
        } 
        else {
            vector<Ctext> strided_vec;
            for (int l = 0; l < outputWidth; l++) {
                Ctext cleaned_cipher;
                if (l == 0) {
                    cleaned_cipher = context->EvalMult(conv_sum, cleaning_mask_out);
                } else {
                    conv_sum = context->EvalRotate(conv_sum, inputWidth); 
                    cleaned_cipher = context->EvalRotate(context->EvalMult(conv_sum, cleaning_mask_out), -(outputWidth * l));
                }
                strided_vec.push_back(cleaned_cipher);
            }
            strided_cipher = context->EvalAddMany(strided_vec);
        }

        // STEP 7 - Rotate for output layout reconstruction
        if (out_ch == 0) {
            final_vec.push_back(strided_cipher);
        } else {
            final_vec.push_back(context->EvalRotate(strided_cipher, -(out_ch * outputSize)));
        }
    }
    rotated_ciphertexts.clear();
    // STEP 8 - Add biases and return result
    return context->EvalAdd(context->EvalAddMany(final_vec), baisInput);;
}



/*****
 * This function is defined to caryout Optimized AvgPooling.
 * It takes in an encrypted vecor, imgWith, kernelSize, Kernel and strieLen */
Ctext ANNController::secure_optimzed_AvgPool(Ctext& encryptedInput,  int inputWidth, int inputChannels, int kernelWidth, int strideLen){

    int outputWidth = inputWidth/strideLen;
    int kernelSize = pow(kernelWidth, 2);
    int inputSize = pow(inputWidth, 2);
    int outputSize = pow(outputWidth, 2);
    int encode_level = encryptedInput->GetLevel();
    
    /*** STEP 1 - ROTATE THE CIPHERTEXT into by k^2-1 and create a k^2 rotated right positions ***/
    vector<Ctext> rotated_ciphertexts;
    auto digits = context->EvalFastRotationPrecompute(encryptedInput);
    rotated_ciphertexts.push_back(encryptedInput);
    rotated_ciphertexts.push_back(context->EvalFastRotation(encryptedInput, 1, context->GetCyclotomicOrder(), digits));
    rotated_ciphertexts.push_back(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits));
    rotated_ciphertexts.push_back(context->EvalRotate(context->EvalFastRotation(encryptedInput, inputWidth, context->GetCyclotomicOrder(), digits), 1));
    Ctext sum_cipher = context->EvalAddMany(rotated_ciphertexts);

    /*** STEP 3: Multiply the scale value with the sum cipher */
    int num_of_elements = inputChannels*inputSize;
    auto masked_data = generate_scale_mask(kernelSize, num_of_elements);
    auto masked_cipher =  context->MakeCKKSPackedPlaintext(masked_data, 1, encode_level);
    sum_cipher = context->EvalMult(sum_cipher, masked_cipher);

    /*** STEP 4: Extract the values needed in the ciphertext */
    // mainResult = generalized_downsample_with_channels(mainResult, inputWidth, strideLen, inputChannels);
    vector<Ctext> channel_ciphers;
    Ctext strided_cipher = generalized_downsample(sum_cipher, inputWidth,  strideLen);
    channel_ciphers.push_back(strided_cipher);
    for(int i = 1; i<inputChannels; i++){
        sum_cipher = context->EvalRotate(sum_cipher, inputSize);
        channel_ciphers.push_back(context->EvalRotate(generalized_downsample(sum_cipher, inputWidth, strideLen), -i*outputSize));
    }
    Ctext finalResult = context->EvalAddMany(channel_ciphers);
    channel_ciphers.clear();
    return finalResult;
}

/**
 * Author: Nges Brian
 * Description: This later is used for fully connected. It takes in an encypted vector which is results from 
 * conv2d and pooling etc, a weighted matrix and bais. It calculates the summation of w_i*x_i + b * **/ 
Ctext ANNController::secure_flinear(Ctext& encryptedInput, vector<Ptext>& weightMatrix, Ptext& baisInput, int inputSize, int outputSize, int rotatePositions){

    int output_size = weightMatrix.size();
    if(outputSize > output_size){
        /** need to handle error here because outputSize should never be grater than output_size */
        cout << "There is an error: ouputsize cannot be larger than weightedMatrix" << endl;
        return encryptedInput;
    }
    /* calculate the results of weights * encrypted vector + bais. 
    Shit the results by number of elements in inputsize*iteration value */
    vector<Ctext> result_matrix;
    vector<Ctext> inner_matrix(rotatePositions);
    int j = 0;
    int rotation_index = 0;
    for(int i = 0; i < outputSize; i++){
        inner_matrix[j] = context->EvalSum(context->EvalMult(encryptedInput, weightMatrix[i]), inputSize);
        j+=1;
        // std::cout << "Ciphertext 1 Scale: " << ", Level: " << inner_matrix[i]->GetLevel() << std::endl;
        
        /** check whether is equal to imgcols, merge them and rotate by imgCols. 
         * If i is equal to the outputSize, merge and rotate by imgCols */
        if(j == rotatePositions || i == (outputSize-1)){
            if(rotation_index > 0){
                result_matrix.push_back(context->EvalRotate(context->EvalMerge(inner_matrix), -rotation_index));
            }
            else{
                result_matrix.push_back(context->EvalMerge(inner_matrix));
            }
            // inner_matrix.clear();
            rotation_index +=rotatePositions;
            j=0;
        }
    }

    /**** convert everything to one vector. and add the baisInput  ***/
    Ctext fResults = context->EvalAddMany(result_matrix);
    inner_matrix.clear();
    result_matrix.clear();
    return context->EvalAdd(fResults, baisInput);
}

/**** 
 * Nges Brian
 * This function is to calculate the relu value using the EvalChebyfunction
 * To do this, we scale the values to a range between -1 and 1 so as to ensure that function works well. 
 * Reludegree tells us the degree pf the data */
Ctext ANNController::secure_relu(Ctext& encryptedInput, double scaleValue,  int vectorSize, int polyDegree) {
    double lowerBound = -1;
    double upperBound = 1;
    
    auto innVector = encryptedInput;
    if(scaleValue > 1){
        auto mask_data = context->MakeCKKSPackedPlaintext(generate_scale_mask(scaleValue, vectorSize), 1, 1);
        innVector = context->EvalMult(encryptedInput, mask_data);
    }
    else{
        scaleValue = 1;
    }
    
    Ctext relu_result = context->EvalChebyshevFunction(
        [scaleValue](double x) -> double { if (x < 0) return 0; else return scaleValue*x; }, 
                                            innVector,
                                            lowerBound,
                                            upperBound, 
                                            polyDegree);
    return relu_result;
}


/*** 
 * Nges Brian
 * This is the function to carryout striding. 
 * Shall use it for all functions using striding */
Ctext ANNController::generalized_downsample(const Ctext& input, int inputWidth, int stride) {
    
    Ctext result = input->Clone();
    const int outputWidth = inputWidth / stride;
    const int inputSize = inputWidth * inputWidth;
    
    // Step 1: Binary decomposition for row juxtaposition
    result = context->EvalMult(result, first_mask(inputWidth, inputSize, stride, input->GetLevel()));
    for (int s = 1; s < log2(outputWidth); s++) {
        result = context->EvalMult(
            context->EvalAdd(result, context->EvalRotate(result, pow(2, s-1))),
            gen_binary_mask(pow(2,s), inputSize, stride, input->GetLevel())
        );
    }
    result = context->EvalAdd(result, context->EvalRotate(result, pow(2, (log2(outputWidth)-1))));
    // Step 2: Row processing with optimized rotations
    Ctext downsampledrows = context->EvalMult(input, gen_zero_mask(inputSize, input->GetLevel()));
    
    for (int row = 0; row < outputWidth; ++row) {
        Ctext masked = context->EvalMult(result, gen_row_mask(row, outputWidth, inputSize, stride, input->GetLevel()));
        downsampledrows = context->EvalAdd(downsampledrows, masked);
        if(row < outputWidth-1){
            result = context->EvalRotate(result, (stride*inputWidth - outputWidth));
        }
    }
    return downsampledrows;
}

Ptext ANNController::first_mask(int width, int inputSize, int stride, int level) {
    vector<double> mask(inputSize, 0);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            if (j % stride == 0 && i % stride == 0) {
                int index = i * width + j;
                mask[index] = 1.0;
            }
        }
    }
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

// Mask generation helpers
Ptext ANNController::gen_binary_mask(int pattern, int inputSize, int stride, int level) {
    vector<double> mask;
    int copy_interval = pattern;
    for (int i = 0; i < inputSize; i++) {
        if (copy_interval > 0) {
            mask.push_back(1);
        } else {
            mask.push_back(0);
        }

        copy_interval--;

        if (copy_interval <= -pattern) {
            copy_interval = pattern;
        }
    }
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

Ptext ANNController::gen_row_mask(int row, int width, int inputSize, int stride, int level) {
    vector<double> mask;

    for (int j = 0; j < (row * width); j++) {
        mask.push_back(0);
    }
    for (int j = 0; j < width; j++) {
        mask.push_back(1);
    }
    for (int j = 0; j < (inputSize - width - (row * width)); j++) {
        mask.push_back(0);
    }
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}

Ptext ANNController::gen_zero_mask(int size, int level) {
    vector<double> mask(size, 0.0);
    return context->MakeCKKSPackedPlaintext(mask, 1.0, level);
}
