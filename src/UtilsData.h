
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


#ifndef FHEON_DATAUTILS_H
#define FHEON_DATAUTILS_H

#include <iostream>
#include <cmath>
#include <openfhe.h>
#include <filesystem>
#include <vector>
#include <string>

using namespace std;
using namespace std::chrono;
using namespace lbcrypto;
namespace fs = std::filesystem;

namespace utilsdata {

    /***
     * Author: Nges Brian, July 11, 2024
     * Action: This function is used to print the 3D matrix, 2Dmatrix and Vector. 
     * Inputs: 
     */
    static inline void printVector(vector<double> &vecData) {
        cout << vecData <<endl;
        cout << endl;
    }

    static inline void print2DMatrix(vector<vector<double>> matrix2D){
        int rows = matrix2D.size();
        int cols = matrix2D[0].size();
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                cout << matrix2D[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    static inline void print3DMatrix(vector<vector<vector<double>>> matrix3D){
        int depth = matrix3D.size();
        int rows = matrix3D[0].size();
        int cols = matrix3D[0][0].size();
        for (int d = 0; d < depth; ++d) {
            cout << "Depth " << d << ":\n";
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    cout << matrix3D[d][i][j] << " ";
                }
               cout << endl;
            }
            cout << endl;
        }
    }

    /*** 
     * Name: Nges Brian
     * Action: Functions are used to generate 3D Vector, 2D and Vectors
     * Inputs: very explanatry 
     * **/ 
    static inline vector<double> createVector(int cols, int minValue, int maxValue) {
        // Initialize random seed
        static bool seedInitialized = false;
        if (!seedInitialized) {
            srand(static_cast<unsigned int>(time(nullptr)));
            seedInitialized = true;
        }

        vector<double> vectorData(cols);
        for (int i = 0; i < cols; i++) {
            vectorData[i] = minValue + static_cast<double>(rand()) / RAND_MAX * (maxValue - minValue);
        }
        return vectorData;
    }

    static inline vector<vector<double>> create2DMatrix(int rows, int cols, int minValue, int maxValue) {
        vector<vector<double>> matrix2D;
        matrix2D.reserve(rows); // Reserve space to avoid multiple allocations

        for (int i = 0; i < rows; i++) {
            matrix2D.push_back(createVector(cols, minValue, maxValue));
        }
        return matrix2D;
    }
    
    static inline vector<vector<vector<double>>> create3DMatrix(int depth, int rows, int cols, int minValue, int maxValue) {
        vector<vector<vector<double>>> matrix3D;
        matrix3D.reserve(depth); // Reserve space to avoid multiple allocations

        for (int d = 0; d < depth; d++) {
            matrix3D.push_back(create2DMatrix(rows, cols, minValue, maxValue));
        }
        return matrix3D;
    }

    /****
     * Author: Nges Brian, July 11, 2024
     * This function is used to Flatten a 3D Matrix to a vector to be able to pack it as a ciphertext
     */
    static inline vector<double> flatten3DMatrix(const vector<vector<vector<double>>> matrix3D) {
        vector<double> flatVec;
        for (const auto& matrix : matrix3D) {
            for (const auto& row : matrix) {
                for (const auto& elem : row) {
                    flatVec.push_back(elem);
                }
            }
        }
        return flatVec;
    }
    
    /*** This function is used to print decrypted Data after encryption */
    static inline void printPtextVector(Plaintext packedVec) {
        vector<complex<double>> finalResult = packedVec->GetCKKSPackedValue();
        cout << finalResult << endl;
        cout << endl;
    }


    /**** Generate the ones and zeros mask */
    static inline vector<double> generate_mixed_mask(int ones_width, int vector_size){
        vector<double> ones_vector(ones_width, 1);
        vector<double> zeros_vector((vector_size - ones_width), 0.0);
        ones_vector.insert(ones_vector.end(), zeros_vector.begin(), zeros_vector.end());
        return ones_vector;
    }

    /**** Generate a scaled mask */
    static inline vector<double> generate_scale_mask(int scale_value, int vector_size){
        double scale_val = (1.0/scale_value);

        vector<double> scaled_vector(vector_size, scale_val);
        return scaled_vector;
    }

    /**** Generate a value mask */
    static inline vector<double> generate_value_mask(double scale_value, int vector_size){

        vector<double> scaled_vector(vector_size, scale_value);
        return scaled_vector;
    }


    /**
     * @brief Find the next power of 2.
     * 
     * @param n Input value.
     * @return Next power of 2.
     */
    static inline int nextPowerOf2(unsigned int n) {
            if (n == 0) return 1;

            n--;                     // make sure exact powers of 2 stay unchanged
            n |= n >> 1;
            n |= n >> 2;
            n |= n >> 4;
            n |= n >> 8;
            n |= n >> 16;
            n++;                     // result is next power of 2
            return n;
     }


     /***
     * Action: used in the spiking function to evaluate the approximation of the greater than using TFHE. 
     * Input: 
     *   x => The evaluation value
     *   0.5 => This is the threshold value
     */
    static inline int greaterFunction(double x) {
        double threshold_value = 0;
        double spike_value = 0;
        int scale_value = 10;
        if(x > threshold_value){
            spike_value = x*scale_value;
            return spike_value; 

        }
        else{
            return 0;
        }
    }

    // Function to approximate the greater-than step function
    static inline double approximateGreaterFunction(double x){
        double threshold_value = 0.05;
        double steepness = 100.0;
        double spike_value = 0.5 * (1 + tanh(steepness * (x - threshold_value)));
        return spike_value; 
    }

    static inline double innerRelu(double x, double scale){
        if (x < 0) return 0; else return (1 / scale) * x;
    }

    static inline vector<double> avgPoolFilter(int kernel_width){
        int numVals = pow(kernel_width, 2);
        double scaled_value = (1.0/numVals);
        vector<double> avgPoolFilter(numVals, scaled_value);

        return avgPoolFilter;
    }
    
    static inline vector<vector<double>> loadCSV(const string& fileName) {
        std::vector<std::vector<double>> data;
        std::ifstream file(fileName);
        
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << fileName << std::endl;
            return data;
        }

        std::string line;
        while (std::getline(file, line)) {
            std::vector<double> row;
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ',')) {
                try {
                    row.push_back(std::stod(cell));
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid number: " << cell << std::endl;
                    row.push_back(0.0);
                }
            }
            data.push_back(row);
        }
        file.close();
        return data;
    }

    static inline vector<double> load_bias(string fileName){
        vector<vector<double>> data = loadCSV(fileName);
        vector<double> bias; 
        for (size_t i = 0; i< data.size(); i++) {
            bias = data[0];
            // cout << " bias data: "<< bias << endl;
        }
        return bias;
    }

    static inline vector<vector<vector<vector<double>>>> load_weights(string fileName, int outputChannels, int inputChannels, 
                int rowsWidth, int imgCols) {
        vector<vector<double>> data = loadCSV(fileName);
        vector<double> raw_weights;
        vector<vector<vector<vector<double>>>> reshapedData(outputChannels, 
                        vector<vector<vector<double>>>(inputChannels, 
                        vector<vector<double>>(rowsWidth, vector<double>(imgCols))));    
        int indexVal = 0; 

        for (size_t i = 0; i< data.size(); i++) {
            raw_weights = data[0];
        }
        for(int i = 0; i< outputChannels; i++){
            for(int j=0; j<inputChannels; j++){
                for(int k=0; k<rowsWidth; k++){
                    for(int l=0; l<imgCols; l++){
                        reshapedData[i][j][k][l] = raw_weights[indexVal];
                        indexVal+=1;
                    }
                }
            }
        }
        data.clear();
        raw_weights.clear();
        return reshapedData;
    }

    static inline vector<vector<double>> load_fc_weights(string fileName, int outputChannels, int inputChannels){
        vector<vector<double>> data = loadCSV(fileName);
        vector<double> raw_weights;
        for (size_t i = 0; i< data.size(); i++) {
            raw_weights = data[0];
        }

        vector<vector<double>> reshapedData(outputChannels, vector<double>(inputChannels));    
        int indexVal = 0; 
        for(int i = 0; i< outputChannels; i++){
            for(int j=0; j< inputChannels; j++){
                reshapedData[i][j] = raw_weights[indexVal];
                indexVal+=1;
            }
        }
        return reshapedData;
    }

    /***
     * Read the CSV files with the pca for users
     */
    static inline vector<double> load_user_input(string dirName, int indx){
        
        vector<string> files;
        for (const auto& entry : fs::directory_iterator(dirName)) {
            files.push_back(entry.path().filename().string()); 
        }
        
        string fileName = files[indx];
        // cout << "File Name: " << dirName+"/"+fileName << endl;
        vector<vector<double>> data = loadCSV(dirName+"/"+fileName);
        std::vector<double> flattened;
        for (size_t i = 0; i < data.size(); i++) {
            for (size_t j = 0; j < data[i].size(); j++) {
                flattened.push_back(data[i][j]);
            }
        }
        return flattened;
    }

    /***
     * Use to write data into files especially especially at the context level */
    static inline void write_to_file(string filename, string content) {
        ofstream file;
        file.open (filename);
        file << content.c_str();
        file.close();
    }

    /***
     * Use to write read data from files especially especially at the context level */
    static inline string read_from_file(string filename) {
        //It reads only the first line!!
        string line;
        ifstream myfile (filename);
        if (myfile.is_open()) {
            if (getline(myfile, line)) {
                myfile.close();
                return line;
            } else {
                cerr << "Could not open " << filename << "." <<endl;
                exit(1);
            }
        } else {
            cerr << "Could not open " << filename << "." <<endl;
            exit(1);
        }
    }

    static inline vector<int> serialize_rotation_keys( vector<vector<int>> rotation_keys){

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

        return rotation_positions;
    }


}

#endif //FHEON_DATAUTILS_H