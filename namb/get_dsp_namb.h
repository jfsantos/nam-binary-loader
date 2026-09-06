#pragma once
// Binary .namb loader for NAM models
// No dependency on nlohmann/json - suitable for embedded targets

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <NAM/compiler.h>
#include <NAM/dsp.h>

namespace nam
{

namespace namb
{

struct NambLayerArrayInfo
{
  int input_size = 0;
  int condition_size = 0;
  int head_size = 0;
  int channels = 0;
  int bottleneck = 0;
  int head_kernel_size = 1;
  int head_dilation = 1;
  bool head_bias = false;
  int groups_input = 1;
  int groups_input_mixin = 1;
  int num_dilations = 0;
  std::vector<int> dilations;
  std::vector<int> kernel_sizes;
  bool layer1x1_active = false;
  int layer1x1_groups = 1;
  bool head1x1_active = false;
  int head1x1_out_channels = 0;
  int head1x1_groups = 1;
};

struct NambModelInfo
{
  struct LinearInfo
  {
    int receptive_field = 0;
    bool bias = false;
    int in_channels = 0;
    int out_channels = 0;
  };

  struct ConvNetInfo
  {
    int channels = 0;
    bool batchnorm = false;
    int groups = 1;
    int in_channels = 0;
    int out_channels = 0;
    int num_dilations = 0;
    std::vector<int> dilations;
  };

  struct LstmInfo
  {
    int num_layers = 0;
    int input_size = 0;
    int hidden_size = 0;
    int in_channels = 0;
    int out_channels = 0;
  };

  struct WaveNetInfo
  {
    int in_channels = 0;
    bool with_head = false;
    int num_layer_arrays = 0;
    bool has_condition_dsp = false;
    std::vector<NambLayerArrayInfo> layer_arrays;
  };

  bool ok = false;
  std::string error;
  uint8_t architecture = 0;
  std::string architecture_name;
  uint32_t total_weight_count = 0;
  ModelMetadata metadata;

  std::optional<LinearInfo> linear;
  std::optional<ConvNetInfo> convnet;
  std::optional<LstmInfo> lstm;
  std::optional<WaveNetInfo> wavenet;
};

bool inspect_namb(const uint8_t* data, size_t size, NambModelInfo& out);

} // namespace namb

/// \brief Load a NAM model from a memory buffer containing .namb data
/// \param data Pointer to the binary data
/// \param size Size of the data in bytes
/// \return Unique pointer to a DSP object, or nullptr if the data is not a valid model
std::unique_ptr<DSP> get_dsp_namb(const uint8_t* data, size_t size);

} // namespace nam
