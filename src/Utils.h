
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


#ifndef FHEON_UTILS_H
#define FHEON_UTILS_H

#include <iostream>
#include <openfhe.h>

using namespace std;
using namespace std::chrono;
using namespace lbcrypto;
using Ctext = Ciphertext<DCRTPoly>;

namespace utils {

    static duration<long long, ratio<1, 1000>> total_time;

    /***
     * Author: Nges Brian, July 11, 2024
     * Function is used to print the welcome message to the project. Just formating and data
     */
    static inline void printWelcomeMessage(){
        cout<< "----------------------------------------------------------------------------------- " << endl;
        cout<< "-------------------------------- WELCOME TO FHEON --------------------------------- " << endl;
        cout<< "---------------------------AUTHORS: NGES BRIAN, ERIC JAHNS ------------------------ " << endl;
        cout<< "----------------------------- STAM CENTER, ASU ------------------------------------ " << endl;
        cout<< "----------------------------------------------------------------------------------- " << endl;
    }

    /***
     * Author: Nges Brian, July 11, 2024
     * Functions, start_time and print_duration are used to calculate and print the duration of piece of code. 
     * start_time record and pass the value to print_duration with a string to print alongside
     */
    static inline chrono::time_point<steady_clock, nanoseconds> startTime() {
        return steady_clock::now();
    }

    /*******
     * Arguments
     * -> start: Provides the starting time to measure. 
     * -> caption: Provides the caption of print statements
     * ->  global_time: used to determine whether the entire time of project should be printed or just a small piece
     */
    static inline void printDuration(chrono::time_point<steady_clock, nanoseconds> start, const string &caption="Time Taken is: ", bool global_time=false) {
        auto ms = duration_cast<milliseconds>(steady_clock::now() - start);

        static duration<long long, ratio<1, 1000>> total_duration; 
        if(global_time){
            total_duration  = total_time + ms;
            total_time = total_duration;
        }
        else{
            total_duration = ms;
        }

        auto secs = duration_cast<seconds>(ms);
        ms -= duration_cast<milliseconds>(secs);
        auto mins = duration_cast<minutes>(secs);
        secs -= duration_cast<seconds>(mins);

        cout<< endl;
        if (mins.count() < 1) {
            cout << "------- " << caption << ": " << secs.count() << ":" << ms.count() << "s" << " (Total: " << duration_cast<seconds>(total_duration).count() << "s)" << " -------- " << endl;
        } else {
            cout << "-------- " << caption << ": " << mins.count() << "." << secs.count() << ":" << ms.count() << " (Total: " << duration_cast<minutes>(total_duration).count() << "mins)" << " -------- " << endl;
        }
        cout<< endl;
    }

    static inline void printBootsrappingData(Ctext ciphertextIn, int depth){
        std::cout << "Number of levels remaining: "
              << depth - ciphertextIn->GetLevel() - (ciphertextIn->GetNoiseScaleDeg() - 1) 
              << " ***Level: " << ciphertextIn->GetLevel() << " ***noiseScaleDeg: " 
              << ciphertextIn->GetNoiseScaleDeg() << std::endl;
    }

    // Calculate the duration in seconds and return the count
    static inline int measure_time(const time_point<high_resolution_clock>& start, const time_point<high_resolution_clock>& end) {
        auto duration = duration_cast<milliseconds>(end - start);
        return duration.count();
    }

    static inline time_point<high_resolution_clock> get_current_time() {
        return high_resolution_clock::now();
    }

    static inline int totalTime(vector<int> measuring, string message){
        int total = accumulate(measuring.begin(), measuring.end(), 0);
        cout << "------- " << message << " Time: " << total << "ms" << "(" << total/1000 <<"s)" << endl;
        return total;
    }

 
    static inline vector<double> gen_mask(int n, int num_slots, int level) {
        vector<double> mask;

        int copy_interval = n;

        for (int i = 0; i < num_slots; i++) {
            if (copy_interval > 0) {
                mask.push_back(1);
            } else {
                mask.push_back(0);
            }

            copy_interval--;

            if (copy_interval <= -n) {
                copy_interval = n;
            }
        }
        return mask;
    }

    static inline vector<double> mask_first_n(int n, int num_slots, int level) {
        vector<double> mask;
    
        for (int i = 0; i < num_slots; i++) {
            if (i < n) {
                mask.push_back(1);
            } else {
                mask.push_back(0);
            }
        }
    
        return mask;
    }

    static inline vector<double> mask_second_n(int n, int num_slots, int level) {
        vector<double> mask;
    
        for (int i = 0; i < num_slots; i++) {
            if (i >= n) {
                mask.push_back(1);
            } else {
                mask.push_back(0);
            }
        }
    
        return mask;
    }

    static inline vector<double> mask_first_n_mod(int n, int padding, int pos, int level) {
        vector<double> mask;
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < (pos * n); j++) {
                mask.push_back(0);
            }
            for (int j = 0; j < n; j++) {
                mask.push_back(1);
            }
            for (int j = 0; j < (padding - n - (pos * n)); j++) {
                mask.push_back(0);
            }
        }

        return mask;
    }

    static inline vector<double> mask_first_n_mod2(int n, int padding, int pos, int level) {
        vector<double> mask;
        for (int i = 0; i < 64; i++) {
            for (int j = 0; j < (pos * n); j++) {
                mask.push_back(0);
            }
            for (int j = 0; j < n; j++) {
                mask.push_back(1);
            }
            for (int j = 0; j < (padding - n - (pos * n)); j++) {
                mask.push_back(0);
            }
        }
        return mask;
    }

    static inline vector<double> mask_channel(int n, int level) {
        vector<double> mask;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 1024; j++) {
                mask.push_back(0);
            }
        }

        for (int i = 0; i < 256; i++) {
            mask.push_back(1);
        }

        for (int i = 0; i < 1024 - 256; i++) {
            mask.push_back(0);
        }

        for (int i = 0; i < 31 - n; i++) {
            for (int j = 0; j < 1024; j++) {
                mask.push_back(0);
            }
        }

        return mask;
    }

    // static inline vector<double> mask_channel_full(int n, int in_elements, int out_elements, int num_channels, int level) {
    //     vector<double> mask;

    //     for (int i = 0; i < n; i++) {
    //         for (int j = 0; j < in_elements; j++) {
    //             mask.push_back(0);
    //         }
    //     }

    //     for (int i = 0; i < out_elements; i++) {
    //         mask.push_back(1);
    //     }

    //     for (int i = 0; i < in_elements - out_elements; i++) {
    //         mask.push_back(0);
    //     }

    //     for (int i = 0; i < (num_channels-1) - n; i++) {
    //         for (int j = 0; j < in_elements; j++) {
    //             mask.push_back(0);
    //         }
    //     }

    //     return mask;
    // }

    static inline vector<double> mask_channel_2(int n, int level) {
        vector<double> mask;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 256; j++) {
                mask.push_back(0);
            }
        }

        for (int i = 0; i < 64; i++) {
            mask.push_back(1);
        }

        for (int i = 0; i < 256 - 64; i++) {
            mask.push_back(0);
        }

        for (int i = 0; i < 63 - n; i++) {
            for (int j = 0; j < 256; j++) {
                mask.push_back(0);
            }
        }

        return mask;
    }

    static inline vector<double> mask_mod(int n, int num_slots, int level, double custom_val) {
        vector<double> vec;

        for (int i = 0; i < num_slots; i++) {
            if (i % n == 0) {
                vec.push_back(custom_val);
            } else {
                vec.push_back(0);
            }
        }

        return vec;
    }
    

}

#endif //FHEON_UTILS_H