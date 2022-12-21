#include <dali/c_api.h>
#include <vector>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "dali/operators.h"

using namespace std;


string read_bin(const std::string &filename) {
  std::ifstream fin(filename, std::ios::binary);
  if (!fin)
    throw std::runtime_error(std::string("Failed to open model file: ") + filename);
  std::stringstream ss;
  ss << fin.rdbuf();
  return ss.str();
}


bool do_sleep = true;


string dali_backend_repo_path = "/home/mszolucha/clion_deploy/Triton/dali_backend/";


void RunDali(int batch_size) {
  string serialized_model = dali_backend_repo_path +
                            "/docs/examples/perf_analyzer/model_repository/dali/1/model.dali";
  string test_sample = "/home/mszolucha/Pictures/608832.jpg";

  cout << "Starting test, note GPU memory\n";
  if (do_sleep) sleep(10);
  daliPipelineHandle h;
  auto ser_bin = read_bin(serialized_model);

  cout << "Creating Pipeline... ";
  daliDeserializeDefault(&h, ser_bin.c_str(), ser_bin.length());
  cout << "DONE\n";
  if (do_sleep) sleep(10);

  cout << "Setting batch size... ";
  daliSetExternalInputBatchSize(&h, "DALI_INPUT_0", batch_size);
  cout << "DONE\n";
  if (do_sleep) sleep(10);

  cout << "Load test sample... ";
  auto test = read_bin(test_sample);
  vector<char> data(batch_size * test.length());
  vector<int64_t> shapes(batch_size);
  size_t k = 0;
  for (int i = 0; i < batch_size; i++) {
    for (char j: test) {
      data[k++] = j;
    }
    shapes[i] = test.length();
  }
  cout << "DONE\n";
  if (do_sleep) sleep(10);

  cout << "Feeding input... ";
  cudaStream_t stream = 0;
  daliSetExternalInput(&h, "DALI_INPUT_0", device_type_t::CPU, data.data(),
                       dali_data_type_t::DALI_UINT8, shapes.data(), 1, NULL, DALI_ext_default);
  cout << "DONE\n";
  if (do_sleep) sleep(10);

  cout << "Running pipeline and waiting for outputs... ";
  daliRun(&h);
  daliOutput(&h);
  cout << "DONE\n";
  if (do_sleep) sleep(10);

  cout << "Deleting DALI pipeline... ";
  daliDeletePipeline(&h);
  cout << "DONE\n";
  if (do_sleep) sleep(10);
}


int main() {
  daliInitialize();
  daliInitOperators();
  cout << "BATCH SIZE 64\n";
  RunDali(64);
  cout << "TEST FINISHED\n";

  cout << "BATCH SIZE 256\n";
  RunDali(256);
  cout << "TEST FINISHED\n";
}