
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


#ifndef SECURESPIKENET_ANNController_H
#define SECURESPIKENET_ANNController_H

#include <openfhe.h>
#include <thread>
#include "FHEController.h"

#include "Utils.h"
#include "UtilsData.h"

using namespace lbcrypto;
using namespace std;

/** securespikenet defined utils */
using namespace utils;
using namespace utilsdata;

class ANNController{

private:
    CryptoContext<DCRTPoly> context;

public:
    string public_data = "sskeys";
    int num_slots = 1 << 12;
    
    ANNController(CryptoContext<DCRTPoly>& ctx) : context(ctx) {}

    void setContext(CryptoContext<DCRTPoly>& in_context);
    
    vector<int> generate_conv_rotation_positions(int imgWidth, int inputChannels, int outputChannels,
                                                    int kernelSize, int paddingSize, int StrideLen);
    vector<int> generate_fullyconnected_rotation_positions(int maxFCLayeroutputs, int rotationPosition);
    vector<int> generate_optimized_convolution_rotation_positions(int imgWidth,  int inputChannels, 
                                            int outputChannels, int StrideLen = 1);
    vector<int> generate_avgpool_optimized_rotation_positions(int imgWidth,  int inputChannels, 
                                            int kernelSize, int StrideLen, bool globalPooling=false);
    
    Ctext secure_conv2D(Ctext& encryptedInput, vector<vector<Ptext>>& kernelData, Ptext& baisInput,
                        int imgWidth, int inputChannels, int outputChannels, int kernelWidth, int paddingSize=0, int stridingLen=1);
    Ctext secure_optimzed_AvgPool(Ctext& encryptedInput,  int imgWidth, int outputChannels, int kernelSize, int StrideLen);
    Ctext secure_flinear(Ctext& encryptedInput, vector<Ptext>& weightMatrix, Ptext& baisInput, int inputSize, int outputSize, int rotatePositions);
    Ctext secure_relu(Ctext& encryptedInput, double scale, int vectorSize, int polyDegree = 59);
    
    Ptext first_mask(int width, int inputSize, int stride, int level);
    Ptext gen_binary_mask(int pattern, int inputSize, int stride, int level);
    Ptext gen_row_mask(int row, int width, int inputSize, int stride, int level);
    Ptext gen_zero_mask(int size, int level);

    /*******************************************    SENTINAL TOUCH *************************************/
    vector <int> generate_users_rotation_positions(int num_users, int embeddings_space=1);
    

private:
    vector<uint32_t> private_data = {8, 8};
    Ctext generalized_downsample(const Ctext& input, int inputWidth, int stride);
};

#endif // FHEON_ANNController_H