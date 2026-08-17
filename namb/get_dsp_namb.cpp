// Binary .namb loader for NAM models
// Uses the unified create_dsp() path shared with the JSON loader.
// Loading never throws: every failure returns nullptr.

#include <algorithm>
#include <cmath>
#include <cstring>

#include "get_dsp_namb.h"

#include <NAM/activations.h>
#include <NAM/convnet.h>
#include <NAM/dsp.h>
#include <NAM/get_dsp.h>
#include <NAM/lstm.h>
#include <NAM/model_config.h>
#if defined(NAM_ENABLE_A2_FAST)
  #include <NAM/wavenet/a2_fast.h>
#endif
#include <NAM/wavenet/model.h>
#include "binary_parser_registry.h"
#include "namb_format.h"

using namespace nam::namb;

namespace
{

// =============================================================================
// Activation config reading
// =============================================================================

nam::activations::ActivationConfig read_activation_config(BinaryReader& r)
{
  nam::activations::ActivationConfig config;
  config.type = static_cast<nam::activations::ActivationType>(r.read_u8());
  uint8_t param_count = r.read_u8();

  switch (config.type)
  {
    case nam::activations::ActivationType::LeakyReLU:
      if (param_count >= 1)
      {
        config.negative_slope = r.read_f32();
        for (uint8_t i = 1; i < param_count; i++)
          r.read_f32(); // skip extra
      }
      break;

    case nam::activations::ActivationType::PReLU:
      if (param_count == 1)
      {
        config.negative_slope = r.read_f32();
      }
      else if (param_count > 1)
      {
        std::vector<float> slopes;
        slopes.reserve(param_count);
        for (uint8_t i = 0; i < param_count; i++)
          slopes.push_back(r.read_f32());
        config.negative_slopes = std::move(slopes);
      }
      break;

    case nam::activations::ActivationType::LeakyHardtanh:
      if (param_count >= 4)
      {
        config.min_val = r.read_f32();
        config.max_val = r.read_f32();
        config.min_slope = r.read_f32();
        config.max_slope = r.read_f32();
        for (uint8_t i = 4; i < param_count; i++)
          r.read_f32(); // skip extra
      }
      else
      {
        for (uint8_t i = 0; i < param_count; i++)
          r.read_f32(); // skip
      }
      break;

    default:
      // Simple activation - skip any params
      for (uint8_t i = 0; i < param_count; i++)
        r.read_f32();
      break;
  }

  return config;
}

// =============================================================================
// FiLM params reading (4 bytes)
// =============================================================================

nam::wavenet::_FiLMParams read_film_params(BinaryReader& r)
{
  uint8_t flags = r.read_u8();
  r.read_u8(); // reserved
  uint16_t groups = r.read_u16();

  bool active = (flags & 0x01) != 0;
  bool shift = (flags & 0x02) != 0;

  return nam::wavenet::_FiLMParams(active, shift, groups);
}

// =============================================================================
// Metadata parsing
// =============================================================================

struct ParsedMetadata
{
  uint8_t version_major = 0;
  uint8_t version_minor = 0;
  uint8_t version_patch = 0;
  uint8_t meta_flags = 0;
  double sample_rate = -1.0;
  double loudness = 0.0;
  double input_level = 0.0;
  double output_level = 0.0;
};

struct NambBinaryHeader
{
  uint32_t total_file_size = 0;
  uint32_t weights_offset = 0;
  uint32_t total_weight_count = 0;
  uint32_t stored_checksum = 0;
};

ParsedMetadata read_metadata_block(BinaryReader& r)
{
  ParsedMetadata m;
  m.version_major = r.read_u8();
  m.version_minor = r.read_u8();
  m.version_patch = r.read_u8();
  m.meta_flags = r.read_u8();
  m.sample_rate = r.read_f64();
  m.loudness = r.read_f64();
  m.input_level = r.read_f64();
  m.output_level = r.read_f64();
  r.skip(12); // reserved
  return m;
}

nam::ModelMetadata to_model_metadata(const ParsedMetadata& pm)
{
  nam::ModelMetadata meta;
  meta.version = std::to_string(pm.version_major) + "." + std::to_string(pm.version_minor) + "."
                 + std::to_string(pm.version_patch);
  meta.sample_rate = pm.sample_rate;
  if (pm.meta_flags & META_HAS_LOUDNESS)
    meta.loudness = pm.loudness;
  if (pm.meta_flags & META_HAS_INPUT_LEVEL)
    meta.input_level = pm.input_level;
  if (pm.meta_flags & META_HAS_OUTPUT_LEVEL)
    meta.output_level = pm.output_level;
  return meta;
}

const char* architecture_name(const uint8_t arch)
{
  switch (arch)
  {
    case ARCH_LINEAR: return "Linear";
    case ARCH_CONVNET: return "ConvNet";
    case ARCH_LSTM: return "LSTM";
    case ARCH_WAVENET: return "WaveNet";
    default: return "Unknown";
  }
}

bool parse_namb_header(const uint8_t* data, size_t size, NambBinaryHeader& out)
{
  if (data == nullptr || size < FILE_HEADER_SIZE + METADATA_BLOCK_SIZE)
    return false;

  BinaryReader header_reader(data, FILE_HEADER_SIZE);

  if (header_reader.read_u32() != MAGIC)
    return false;

  if (header_reader.read_u16() != FORMAT_VERSION)
    return false;

  header_reader.read_u16(); // flags
  out.total_file_size = header_reader.read_u32();
  out.weights_offset = header_reader.read_u32();
  out.total_weight_count = header_reader.read_u32();
  header_reader.read_u32(); // model_block_size
  out.stored_checksum = header_reader.read_u32();

  if (!header_reader.ok())
    return false;

  if (size < out.total_file_size || out.weights_offset < MODEL_BLOCK_OFFSET)
    return false;

  if (compute_file_crc32(data, out.total_file_size) != out.stored_checksum)
    return false;

  const size_t expected_weights_end = out.weights_offset + (size_t)out.total_weight_count * sizeof(float);
  if (expected_weights_end > out.total_file_size)
    return false;

  return true;
}

void skip_activation_config(BinaryReader& r)
{
  r.read_u8(); // activation type
  const uint8_t param_count = r.read_u8();
  for (uint8_t i = 0; i < param_count; i++)
    r.read_f32();
}

bool parse_wavenet_inspection(BinaryReader& r, nam::namb::NambModelInfo::WaveNetInfo& out)
{
  out.in_channels = r.read_u8();
  const uint8_t has_head = r.read_u8();
  const uint8_t num_layer_arrays = r.read_u8();
  const uint8_t has_condition_dsp = r.read_u8();

  out.with_head = (has_head != 0);
  out.num_layer_arrays = static_cast<int>(num_layer_arrays);
  out.has_condition_dsp = (has_condition_dsp != 0);

  if (out.has_condition_dsp)
  {
    r.read_u32(); // condition DSP weight count
    read_metadata_block(r);
    r.read_u8(); // condition DSP architecture
    r.read_u8(); // reserved
    const uint16_t cdsp_config_size = r.read_u16();
    r.skip(cdsp_config_size);
  }

  out.layer_arrays.reserve(num_layer_arrays);
  for (uint8_t la = 0; la < num_layer_arrays; la++)
  {
    nam::namb::NambLayerArrayInfo s;
    s.input_size = r.read_u16();
    s.condition_size = r.read_u16();
    s.head_size = r.read_u16();
    s.channels = r.read_u16();
    s.bottleneck = r.read_u16();
    const uint16_t head_kernel_size_raw = r.read_u16();
    s.head_kernel_size = (head_kernel_size_raw == 0) ? 1 : static_cast<int>(head_kernel_size_raw);
    s.head_dilation = r.read_u16();

    s.head_bias = r.read_u8() != 0;
    const uint8_t num_dilations = r.read_u8();
    s.num_dilations = num_dilations;
    s.groups_input = r.read_u16();
    s.groups_input_mixin = r.read_u16();

    s.layer1x1_active = r.read_u8() != 0;
    s.layer1x1_groups = r.read_u16();
    r.read_u8(); // reserved

    s.head1x1_active = r.read_u8() != 0;
    s.head1x1_out_channels = r.read_u16();
    s.head1x1_groups = r.read_u16();
    r.read_u8(); // reserved

    for (int i = 0; i < 8; i++)
      read_film_params(r);

    s.dilations.reserve(num_dilations);
    for (uint8_t i = 0; i < num_dilations; i++)
      s.dilations.push_back(r.read_i32());

    s.kernel_sizes.reserve(num_dilations);
    for (uint8_t i = 0; i < num_dilations; i++)
      s.kernel_sizes.push_back(r.read_u16());

    for (uint8_t i = 0; i < num_dilations; i++)
      skip_activation_config(r);

    for (uint8_t i = 0; i < num_dilations; i++)
      r.read_u8();

    for (uint8_t i = 0; i < num_dilations; i++)
      skip_activation_config(r);

    out.layer_arrays.push_back(std::move(s));
  }

  return r.ok();
}

#if defined(NAM_ENABLE_A2_FAST)
bool close_to(const float value, const float target)
{
  return std::fabs(value - target) <= 1e-7f;
}

bool film_inactive(const nam::wavenet::_FiLMParams& params)
{
  return !params.active;
}

bool is_a2_channel_count(const int channels)
{
  return channels == 3 || channels == 8;
}

bool matches_a2_channeling(const nam::wavenet::LayerArrayParams& layer)
{
  return layer.input_size == 1 && layer.condition_size == 1 && layer.head_size == 1
         && layer.bottleneck == layer.channels && is_a2_channel_count(layer.channels);
}

bool matches_a2_head(const nam::wavenet::LayerArrayParams& layer)
{
  return layer.head_kernel_size == nam::wavenet::a2_fast::kHeadKernelSize && layer.head_dilation == 1
         && layer.head_bias;
}

bool matches_a2_grouping(const nam::wavenet::LayerArrayParams& layer)
{
  if (layer.groups_input != 1 || layer.groups_input_mixin != 1)
    return false;
  if (!layer.layer1x1_params.active || layer.layer1x1_params.groups != 1)
    return false;
  return !layer.head1x1_params.active;
}

bool matches_a2_film(const nam::wavenet::LayerArrayParams& layer)
{
  return film_inactive(layer.conv_pre_film_params) && film_inactive(layer.conv_post_film_params)
         && film_inactive(layer.input_mixin_pre_film_params) && film_inactive(layer.input_mixin_post_film_params)
         && film_inactive(layer.activation_pre_film_params) && film_inactive(layer.activation_post_film_params)
         && film_inactive(layer._layer1x1_post_film_params) && film_inactive(layer.head1x1_post_film_params);
}

bool matches_a2_topology(const nam::wavenet::LayerArrayParams& layer)
{
  if (layer.dilations.size() != nam::wavenet::a2_fast::kNumLayers)
    return false;
  if (layer.kernel_sizes.size() != nam::wavenet::a2_fast::kNumLayers)
    return false;
  if (layer.activation_configs.size() != nam::wavenet::a2_fast::kNumLayers)
    return false;
  if (layer.gating_modes.size() != nam::wavenet::a2_fast::kNumLayers)
    return false;

  for (int i = 0; i < nam::wavenet::a2_fast::kNumLayers; i++)
  {
    if (layer.dilations[i] != nam::wavenet::a2_fast::kDilations[i])
      return false;
    if (layer.kernel_sizes[i] != nam::wavenet::a2_fast::kKernelSizes[i])
      return false;
    if (layer.gating_modes[i] != nam::wavenet::GatingMode::NONE)
      return false;

    const auto& activation = layer.activation_configs[i];
    if (activation.type != nam::activations::ActivationType::LeakyReLU)
      return false;
    if (!activation.negative_slope.has_value())
      return false;
    if (!close_to(activation.negative_slope.value(), nam::wavenet::a2_fast::kLeakySlope))
      return false;
  }

  return true;
}

bool is_valid_a2(const nam::wavenet::WaveNetConfig& config)
{
  if (config.in_channels != 1)
    return false;
  if (config.with_head || config.head_params.has_value())
    return false;
  if (config.condition_dsp != nullptr)
    return false;
  if (config.layer_array_params.size() != 1)
    return false;

  const auto& layer = config.layer_array_params[0];

  if (!matches_a2_channeling(layer))
    return false;
  if (!matches_a2_head(layer))
    return false;
  if (!matches_a2_grouping(layer))
    return false;
  if (!matches_a2_film(layer))
    return false;
  if (!matches_a2_topology(layer))
    return false;

  return true;
}

int get_a2_channels(const nam::wavenet::WaveNetConfig& config)
{
  return config.layer_array_params[0].channels;
}
#endif

// =============================================================================
// Binary parsing into typed configs
// =============================================================================

// Forward declaration
std::unique_ptr<nam::ModelConfig> load_model(BinaryReader& r, const float*& weights, size_t& weight_count,
                                             const nam::ModelMetadata& meta);

// --- Linear ---

std::unique_ptr<nam::ModelConfig> load_linear(BinaryReader& r, const float*& /*weights*/, size_t& /*weight_count*/,
                                              const nam::ModelMetadata&)
{
  auto cfg = std::make_unique<nam::linear::LinearConfig>();
  cfg->receptive_field = r.read_i32();
  cfg->bias = r.read_u8() != 0;
  cfg->in_channels = r.read_u8();
  cfg->out_channels = r.read_u8();
  r.read_u8(); // reserved
  return cfg;
}

// --- LSTM ---

std::unique_ptr<nam::ModelConfig> load_lstm(BinaryReader& r, const float*& /*weights*/, size_t& /*weight_count*/,
                                            const nam::ModelMetadata&)
{
  auto cfg = std::make_unique<nam::lstm::LSTMConfig>();
  cfg->num_layers = r.read_u16();
  cfg->input_size = r.read_u16();
  cfg->hidden_size = r.read_u16();
  cfg->in_channels = r.read_u8();
  cfg->out_channels = r.read_u8();
  r.skip(2); // reserved
  return cfg;
}

// --- ConvNet ---
std::unique_ptr<nam::ModelConfig> load_convnet(BinaryReader& r, const float*& /*weights*/, size_t& /*weight_count*/,
                                               const nam::ModelMetadata&)
{
  auto cfg = std::make_unique<nam::convnet::ConvNetConfig>();
  cfg->channels = r.read_u16();
  cfg->batchnorm = r.read_u8() != 0;
  uint8_t num_dilations = r.read_u8();
  cfg->groups = r.read_u16();
  cfg->in_channels = r.read_u8();
  cfg->out_channels = r.read_u8();

  cfg->activation = read_activation_config(r);

  cfg->dilations.reserve(num_dilations);
  for (int i = 0; i < num_dilations; i++)
    cfg->dilations.push_back(r.read_i32());

  return cfg;
}

// --- WaveNet ---

std::unique_ptr<nam::ModelConfig> load_wavenet(BinaryReader& r, const float*& weights, size_t& weight_count,
                                               const nam::ModelMetadata& meta)
{
  auto wc = std::make_unique<nam::wavenet::WaveNetConfig>();
  wc->in_channels = r.read_u8();
  uint8_t has_head = r.read_u8();
  uint8_t num_layer_arrays = r.read_u8();
  uint8_t has_condition_dsp = r.read_u8();

  wc->with_head = (has_head != 0);

  // Condition DSP
  if (has_condition_dsp)
  {
    uint32_t cdsp_weight_count = r.read_u32();

    // Read condition DSP metadata (48 bytes)
    ParsedMetadata cdsp_pm = read_metadata_block(r);
    nam::ModelMetadata cdsp_meta = to_model_metadata(cdsp_pm);

    // Load condition DSP model recursively via create_dsp
    // Use local copies so load_model doesn't advance the outer pointers
    const float* cdsp_weights = weights;
    size_t cdsp_wc = cdsp_weight_count;
    auto cdsp_config = load_model(r, cdsp_weights, cdsp_wc, cdsp_meta);
    std::vector<float> cdsp_weight_vec(weights, weights + cdsp_weight_count);
    wc->condition_dsp = nam::create_dsp(std::move(cdsp_config), std::move(cdsp_weight_vec), cdsp_meta);

    // Advance past condition DSP weights
    weights += cdsp_weight_count;
    weight_count -= cdsp_weight_count;
  }

  // Parse layer array params
  for (int la = 0; la < num_layer_arrays; la++)
  {
    uint16_t input_size = r.read_u16();
    uint16_t condition_size = r.read_u16();
    uint16_t head_size = r.read_u16();
    uint16_t la_channels = r.read_u16();
    uint16_t bottleneck = r.read_u16();
    uint16_t head_kernel_size_raw = r.read_u16();
    int head_kernel_size = (head_kernel_size_raw == 0) ? 1 : static_cast<int>(head_kernel_size_raw);
    uint16_t head_dilation = r.read_u16();

    bool head_bias = r.read_u8() != 0;
    uint8_t num_dilations = r.read_u8();
    uint16_t groups_input = r.read_u16();
    uint16_t groups_input_mixin = r.read_u16();

    // layer1x1 (4 bytes)
    bool layer1x1_active = r.read_u8() != 0;
    uint16_t layer1x1_groups = r.read_u16();
    r.read_u8(); // reserved

    // head1x1 (6 bytes)
    bool head1x1_active = r.read_u8() != 0;
    uint16_t head1x1_out_channels = r.read_u16();
    uint16_t head1x1_groups = r.read_u16();
    r.read_u8(); // reserved

    // 8 FiLM params (32 bytes)
    nam::wavenet::_FiLMParams conv_pre_film = read_film_params(r);
    nam::wavenet::_FiLMParams conv_post_film = read_film_params(r);
    nam::wavenet::_FiLMParams input_mixin_pre_film = read_film_params(r);
    nam::wavenet::_FiLMParams input_mixin_post_film = read_film_params(r);
    nam::wavenet::_FiLMParams activation_pre_film = read_film_params(r);
    nam::wavenet::_FiLMParams activation_post_film = read_film_params(r);
    nam::wavenet::_FiLMParams layer1x1_post_film = read_film_params(r);
    nam::wavenet::_FiLMParams head1x1_post_film = read_film_params(r);

    // Dilations [N * int32]
    std::vector<int> dilations;
    dilations.reserve(num_dilations);
    for (int i = 0; i < num_dilations; i++)
      dilations.push_back(r.read_i32());

    // Kernel sizes [N * uint16]
    std::vector<int> kernel_sizes;
    kernel_sizes.reserve(num_dilations);
    for (int i = 0; i < num_dilations; i++)
      kernel_sizes.push_back(r.read_u16());

    // Activation configs [N * variable]
    std::vector<nam::activations::ActivationConfig> activation_configs;
    activation_configs.reserve(num_dilations);
    for (int i = 0; i < num_dilations; i++)
      activation_configs.push_back(read_activation_config(r));

    // Gating modes [N * uint8]
    std::vector<nam::wavenet::GatingMode> gating_modes;
    gating_modes.reserve(num_dilations);
    for (int i = 0; i < num_dilations; i++)
    {
      uint8_t gm = r.read_u8();
      switch (gm)
      {
        case GATING_GATED: gating_modes.push_back(nam::wavenet::GatingMode::GATED); break;
        case GATING_BLENDED: gating_modes.push_back(nam::wavenet::GatingMode::BLENDED); break;
        default: gating_modes.push_back(nam::wavenet::GatingMode::NONE); break;
      }
    }

    // Secondary activation configs [N * variable]
    std::vector<nam::activations::ActivationConfig> secondary_activation_configs;
    secondary_activation_configs.reserve(num_dilations);
    for (int i = 0; i < num_dilations; i++)
      secondary_activation_configs.push_back(read_activation_config(r));

    nam::wavenet::Layer1x1Params layer1x1_params(layer1x1_active, layer1x1_groups);
    nam::wavenet::Head1x1Params head1x1_params(head1x1_active, head1x1_out_channels, head1x1_groups);

    wc->layer_array_params.emplace_back(
      static_cast<int>(input_size), static_cast<int>(condition_size), static_cast<int>(head_size),
      static_cast<int>(head_dilation), head_kernel_size, static_cast<int>(la_channels), static_cast<int>(bottleneck),
      std::move(kernel_sizes), std::move(dilations), std::move(activation_configs),
      std::move(gating_modes),
      head_bias, static_cast<int>(groups_input), static_cast<int>(groups_input_mixin),
      layer1x1_params, head1x1_params,
      std::move(secondary_activation_configs), conv_pre_film, conv_post_film,
      input_mixin_pre_film, input_mixin_post_film, activation_pre_film,
      activation_post_film, layer1x1_post_film, head1x1_post_film);
  }

  // head_scale is the last weight value, but set_weights_ will overwrite it.
  // Pass 0.0f; set_weights_ will set the correct value from weights.
  wc->head_scale = 0.0f;

#if defined(NAM_ENABLE_A2_FAST)
  if (is_valid_a2(*wc))
    return nam::wavenet::a2_fast::create_a2_fast_config(get_a2_channels(*wc));
#endif

  return wc;
}

// =============================================================================
// Static registration of binary parsers
// =============================================================================

static nam::namb::BinaryConfigParserHelper _register_linear(ARCH_LINEAR, load_linear);
static nam::namb::BinaryConfigParserHelper _register_lstm(ARCH_LSTM, load_lstm);
static nam::namb::BinaryConfigParserHelper _register_convnet(ARCH_CONVNET, load_convnet);
static nam::namb::BinaryConfigParserHelper _register_wavenet(ARCH_WAVENET, load_wavenet);

// =============================================================================
// Dispatch to architecture-specific loader via registry
// =============================================================================

std::unique_ptr<nam::ModelConfig> load_model(BinaryReader& r, const float*& weights, size_t& weight_count,
                                             const nam::ModelMetadata& meta)
{
  uint8_t arch = r.read_u8();
  r.read_u8(); // reserved
  r.read_u16(); // config_size

  return BinaryConfigParserRegistry::instance().parse(arch, r, weights, weight_count, meta);
}

} // anonymous namespace

// =============================================================================
// Public API
// =============================================================================

std::unique_ptr<nam::DSP> nam::get_dsp_namb(const uint8_t* data, size_t size)
{
  NambBinaryHeader header;
  if (!parse_namb_header(data, size, header))
    return nullptr;

  // Metadata block (at offset 32)
  BinaryReader meta_reader(data + FILE_HEADER_SIZE, METADATA_BLOCK_SIZE);
  ParsedMetadata pm = read_metadata_block(meta_reader);
  ModelMetadata meta = to_model_metadata(pm);
  if (!meta_reader.ok() || nam::is_version_supported(meta.version) == nam::Supported::NO)
    return nullptr;

  const float* weights = reinterpret_cast<const float*>(data + header.weights_offset);
  size_t weight_count = header.total_weight_count;

  // Model block (at offset 80)
  BinaryReader model_reader(data + MODEL_BLOCK_OFFSET, header.weights_offset - MODEL_BLOCK_OFFSET);

  auto config = load_model(model_reader, weights, weight_count, meta);
  if (config == nullptr || !model_reader.ok())
    return nullptr;

  std::vector<float> weight_vec(weights, weights + weight_count);
  return create_dsp(std::move(config), std::move(weight_vec), meta);
}

bool nam::namb::inspect_namb(const uint8_t* data, size_t size, NambModelInfo& out)
{
  out = NambModelInfo{};

  NambBinaryHeader header;
  if (!parse_namb_header(data, size, header))
  {
    out.error = "Invalid namb header or checksum";
    return false;
  }

  BinaryReader meta_reader(data + FILE_HEADER_SIZE, METADATA_BLOCK_SIZE);
  ParsedMetadata pm = read_metadata_block(meta_reader);
  out.metadata = to_model_metadata(pm);
  if (!meta_reader.ok())
  {
    out.error = "Failed to parse metadata";
    return false;
  }

  out.total_weight_count = header.total_weight_count;

  BinaryReader model_reader(data + MODEL_BLOCK_OFFSET, header.weights_offset - MODEL_BLOCK_OFFSET);
  out.architecture = model_reader.read_u8();
  model_reader.read_u8(); // reserved
  model_reader.read_u16(); // config_size
  out.architecture_name = architecture_name(out.architecture);

  if (!model_reader.ok())
  {
    out.error = "Failed to parse model header";
    return false;
  }

  switch (out.architecture)
  {
    case ARCH_LINEAR:
    {
      NambModelInfo::LinearInfo info;
      info.receptive_field = model_reader.read_i32();
      info.bias = model_reader.read_u8() != 0;
      info.in_channels = model_reader.read_u8();
      info.out_channels = model_reader.read_u8();
      model_reader.read_u8(); // reserved
      out.linear = std::move(info);
      break;
    }
    case ARCH_CONVNET:
    {
      NambModelInfo::ConvNetInfo info;
      info.channels = model_reader.read_u16();
      info.batchnorm = model_reader.read_u8() != 0;
      const uint8_t num_dilations = model_reader.read_u8();
      info.groups = model_reader.read_u16();
      info.in_channels = model_reader.read_u8();
      info.out_channels = model_reader.read_u8();
      info.num_dilations = num_dilations;
      skip_activation_config(model_reader);
      info.dilations.reserve(num_dilations);
      for (uint8_t i = 0; i < num_dilations; i++)
        info.dilations.push_back(model_reader.read_i32());
      out.convnet = std::move(info);
      break;
    }
    case ARCH_LSTM:
    {
      NambModelInfo::LstmInfo info;
      info.num_layers = model_reader.read_u16();
      info.input_size = model_reader.read_u16();
      info.hidden_size = model_reader.read_u16();
      info.in_channels = model_reader.read_u8();
      info.out_channels = model_reader.read_u8();
      model_reader.skip(2);
      out.lstm = std::move(info);
      break;
    }
    case ARCH_WAVENET:
    {
      NambModelInfo::WaveNetInfo info;
      if (!parse_wavenet_inspection(model_reader, info))
      {
        out.error = "Failed to parse WaveNet config";
        return false;
      }
      out.wavenet = std::move(info);
      break;
    }
    default:
      out.error = "Unsupported architecture id";
      return false;
  }

  if (!model_reader.ok())
  {
    out.error = "Failed while parsing model body";
    return false;
  }

  out.ok = true;
  return true;
}

