#include <bits/stdc++.h>
#include <chrono>
#include <papi.h>
using namespace std;
using namespace std::chrono;

// Input dimensions
int input_width  = 1<<6;
int input_length = 1<<6;
int input_height = 1<<6;

int K = 3;

// Output dimensions
int out_w = input_width - K + 1;
int out_l = input_length - K + 1;
int out_h = input_height - K + 1;


// Naive 3D convolution
void conv_naive(const vector<int>& input, const vector<vector<vector<int>>>& filter,
                vector<int>& result) {
    for (int x = 0; x < out_w; x++) {
        for (int y = 0; y < out_l; y++) {
            for (int z = 0; z < out_h; z++) {
                int sum = 0;
                for (int i = 0; i < K; i++) {
                    for (int j = 0; j < K; j++) {
                        for (int k = 0; k < K; k++) {
                            int idx = (x + i) * input_length * input_height +
                                      (y + j) * input_height +
                                      (z + k);
                            sum += input[idx] * filter[i][j][k];
                        }
                    }
                }
                result[x * out_l * out_h + y * out_h + z] = sum;
            }
        }
    }
}

// Blocked 3D convolution
void conv_blocked(const vector<int>& input, const vector<vector<vector<int>>>& filter,
                  vector<int>& result, int block) {
    for (int x = 0; x < out_w; x += block) {
        for (int y = 0; y < out_l; y += block) {
            for(int z=0; z < out_h; z+=block){
                for (int xx = x; xx < x + block && xx < out_w; xx++) {
                    for (int yy = y; yy < y + block && yy < out_l; yy++) {
                        for (int zz = z; zz < z+block && zz<out_h ; zz++) {
                            int sum = 0;
                            for (int i = 0; i < K; i++) {
                                for (int j = 0; j < K; j++) {
                                    for (int k = 0; k < K; k++) {
                                        int idx = (xx + i) * input_length * input_height +
                                                  (yy + j) * input_height +
                                                  (zz + k);
                                        sum += input[idx] * filter[i][j][k];
                                    }
                                }
                            }
                            result[xx * out_l * out_h + yy * out_h + zz] = sum;
                        }
                    }
                }
            }
        }
    }
}

// Run a test with PAPI counters
template <typename Func>
void run_with_papi(const string& name, Func f) {
    // Init arrays
    vector<vector<vector<int>>> filter = {
        {{2, 1, 3}, {2, 1, 3}, {2, 1, 3}},
        {{2, 1, 3}, {2, 1, 3}, {2, 1, 3}},
        {{2, 1, 3}, {2, 1, 3}, {2, 1, 3}}
    };
    vector<int> input(input_width * input_length * input_height, 1);
    vector<int> result(out_w * out_l * out_h, 0);

    int EventSet = PAPI_NULL;
    int retval = PAPI_create_eventset(&EventSet);
    if (retval != PAPI_OK) { cerr << "PAPI create event set error\n"; exit(1); }

    // Add events: L1, L2, LLC (accesses + misses)
    PAPI_add_event(EventSet, PAPI_L1_DCA);
    PAPI_add_event(EventSet, PAPI_L1_DCM);
    PAPI_add_event(EventSet, PAPI_L2_DCA);
    PAPI_add_event(EventSet, PAPI_L2_DCM);
    PAPI_add_event(EventSet, PAPI_L3_DCA);
    PAPI_add_event(EventSet, PAPI_L3_DCM);

    long long values[6] = {0};

    auto start = high_resolution_clock::now();
    retval = PAPI_start(EventSet);
    if (retval != PAPI_OK) cerr << "Error starting PAPI\n";

    // Run function
    f(input, filter, result);

    retval = PAPI_stop(EventSet, values);
    if (retval != PAPI_OK) cerr << "Error stopping PAPI\n";
    auto stop = high_resolution_clock::now();

    auto time = duration_cast<microseconds>(stop - start);

    cout << "=== " << name << " ===\n";
    cout << "Execution time (us): " << time.count() << endl;
    cout <<"L1 Data Accesses: "<<values[0]<< "  L1 Misses: " << values[1] << endl;
    cout <<"L2 Data Accesses: "<< values[2]<< "  L2 Misses: " << values[3] << endl;
    cout <<"LLC Data Accesses: "<<values[4]<< "  LLC Misses: " << values[5] << endl;
    cout << "-------------------------------------\n";

    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
}

int main() {
    // Init PAPI
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        cerr << "PAPI library init error!" << endl;
        return 1;
    }
    run_with_papi("Naive convolution", [](auto& input, auto& filter, auto& result) {
            conv_naive(input, filter, result);
        });
        

    for (int block : {32}) {
        cout << "\n===== Block size = " << block << " =====\n";


        run_with_papi("Blocked convolution", [block](auto& input, auto& filter, auto& result) {
            conv_blocked(input, filter, result, block);
        });
    }

    return 0;
}

