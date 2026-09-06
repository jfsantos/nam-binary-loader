#include <stdlib.h>

#include <algorithm>
#include <limits>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <NAM/dsp.h>
#include <NAM/get_dsp.h>
#include <namb/get_dsp_namb.h>

namespace
{

static void print_usage()
{
  fprintf(stderr, "Usage: loadmodel [--verbose|-v] [--buffer-size N|-b N] <model_path>\n");
}

static bool read_file_bytes(const std::filesystem::path& path, std::vector<uint8_t>& out)
{
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open())
    return false;

  const std::streamsize file_size = file.tellg();
  if (file_size <= 0)
    return false;
  file.seekg(0, std::ios::beg);

  out.resize((size_t)file_size);
  return file.read(reinterpret_cast<char*>(out.data()), file_size).good();
}

static std::string human_bytes(const size_t bytes)
{
  const char* units[] = {"B", "KiB", "MiB", "GiB"};
  double value = static_cast<double>(bytes);
  size_t unit = 0;
  while (value >= 1024.0 && unit < 3)
  {
    value /= 1024.0;
    unit++;
  }

  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.2f %s", value, units[unit]);
  return std::string(buf);
}

struct RuntimeEstimate
{
  size_t total_bytes = 0;
  size_t exact_ring_bytes = 0;
  size_t lower_bound_workspace_bytes = 0;

  // Optional detailed breakdown (populated when available).
  size_t condition_buffers_bytes = 0;
  size_t rechannel_ring_bytes = 0;
  size_t rechannel_output_bytes = 0;
  size_t head_rechannel_ring_bytes = 0;
  size_t head_rechannel_output_bytes = 0;
  size_t layer_array_workspace_bytes = 0;
  size_t layer_conv_ring_bytes = 0;
  size_t layer_conv_output_bytes = 0;
  size_t layer_z_bytes = 0;
  size_t layer_output_next_bytes = 0;
  size_t layer_output_head_bytes = 0;

  bool has_detailed_breakdown = false;
};

static void add_bytes(size_t& total, const size_t bytes)
{
  const size_t max_size = std::numeric_limits<size_t>::max();
  if (bytes > max_size - total)
    total = max_size;
  else
    total += bytes;
}

static size_t matrix_bytes(const int rows, const int cols)
{
  if (rows <= 0 || cols <= 0)
    return 0;
  return static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float);
}

static RuntimeEstimate estimate_runtime_bytes(const nam::namb::NambModelInfo& info, const int max_buffer_size)
{
  RuntimeEstimate result;
  const int frames = std::max(1, max_buffer_size);

  if (info.linear.has_value())
  {
    const auto& l = info.linear.value();
    result.lower_bound_workspace_bytes = matrix_bytes(std::max(1, l.out_channels), std::max(1, frames));
    result.total_bytes = result.lower_bound_workspace_bytes;
    return result;
  }

  if (info.convnet.has_value())
  {
    const auto& c = info.convnet.value();
    const int channels = std::max(1, c.channels);
    const int layers = std::max(1, c.num_dilations);
    // Lower bound: one output matrix plus one temp matrix per layer.
    result.lower_bound_workspace_bytes =
      matrix_bytes(channels, frames) + static_cast<size_t>(layers) * matrix_bytes(channels, frames);
    result.total_bytes = result.lower_bound_workspace_bytes;
    return result;
  }

  if (info.lstm.has_value())
  {
    const auto& l = info.lstm.value();
    const int hidden = std::max(1, l.hidden_size);
    const int layers = std::max(1, l.num_layers);
    // Lower bound: hidden and cell states plus one output buffer.
    result.lower_bound_workspace_bytes = static_cast<size_t>(layers) * matrix_bytes(hidden, 2) + matrix_bytes(hidden, frames);
    result.total_bytes = result.lower_bound_workspace_bytes;
    return result;
  }

  if (info.wavenet.has_value())
  {
    const auto& w = info.wavenet.value();

    result.has_detailed_breakdown = true;

    // WaveNet top-level buffers in SetMaxBufferSize.
    const size_t condition_input_bytes = matrix_bytes(std::max(1, w.in_channels), frames);
    const size_t condition_output_bytes = matrix_bytes(std::max(1, w.in_channels), frames);
    result.condition_buffers_bytes = condition_input_bytes + condition_output_bytes;
    add_bytes(result.lower_bound_workspace_bytes, result.condition_buffers_bytes);

    for (const auto& la : w.layer_arrays)
    {
      const int channels = std::max(1, la.channels);
      const int bottleneck = std::max(1, la.bottleneck);
      const int head_in = std::max(1, la.head1x1_active ? la.head1x1_out_channels : la.bottleneck);
      const int head_out = std::max(1, la.head_size);

      // _rechannel Conv1D: lookback 0 (kernel=1, dilation=1)
      const size_t rechannel_ring = matrix_bytes(std::max(1, la.input_size), frames);
      add_bytes(result.rechannel_ring_bytes, rechannel_ring);
      add_bytes(result.exact_ring_bytes, rechannel_ring);

      const size_t rechannel_output = matrix_bytes(channels, frames);
      add_bytes(result.rechannel_output_bytes, rechannel_output);
      add_bytes(result.lower_bound_workspace_bytes, rechannel_output); // _rechannel output

      // _head_rechannel Conv1D
      const int head_lookback = std::max(0, la.head_dilation) * std::max(0, la.head_kernel_size - 1);
      const int head_cols = 2 * head_lookback + frames;
      const size_t head_rechannel_ring = matrix_bytes(head_in, head_cols);
      add_bytes(result.head_rechannel_ring_bytes, head_rechannel_ring);
      add_bytes(result.exact_ring_bytes, head_rechannel_ring);

      const size_t head_rechannel_output = matrix_bytes(head_out, frames);
      add_bytes(result.head_rechannel_output_bytes, head_rechannel_output);
      add_bytes(result.lower_bound_workspace_bytes, head_rechannel_output); // _head_rechannel output

      // LayerArray work buffers
      const size_t layer_outputs_bytes = matrix_bytes(channels, frames);
      const size_t head_inputs_bytes = matrix_bytes(head_in, frames);
      add_bytes(result.layer_array_workspace_bytes, layer_outputs_bytes);
      add_bytes(result.layer_array_workspace_bytes, head_inputs_bytes);
      add_bytes(result.lower_bound_workspace_bytes, layer_outputs_bytes); // _layer_outputs
      add_bytes(result.lower_bound_workspace_bytes, head_inputs_bytes); // _head_inputs

      const int num_layers = std::min(la.num_dilations,
                                      static_cast<int>(std::min(la.dilations.size(), la.kernel_sizes.size())));
      for (int i = 0; i < num_layers; ++i)
      {
        const int dilation = std::max(0, la.dilations[i]);
        const int kernel = std::max(1, la.kernel_sizes[i]);
        const int lookback = dilation * (kernel - 1);
        const int ring_cols = 2 * lookback + frames;

        // Layer conv ring buffer is exact from dilation/kernel/channels.
        const size_t layer_conv_ring = matrix_bytes(channels, ring_cols);
        add_bytes(result.layer_conv_ring_bytes, layer_conv_ring);
        add_bytes(result.exact_ring_bytes, layer_conv_ring);

        // Lower bound workspace for each layer:
        // _conv output (ungated minimum is bottleneck), _z, _output_next_layer, _output_head(min bottleneck).
        const size_t layer_conv_output = matrix_bytes(bottleneck, frames);
        const size_t layer_z = matrix_bytes(bottleneck, frames);
        const size_t layer_output_next = matrix_bytes(channels, frames);
        const size_t layer_output_head = matrix_bytes(bottleneck, frames);

        add_bytes(result.layer_conv_output_bytes, layer_conv_output);
        add_bytes(result.layer_z_bytes, layer_z);
        add_bytes(result.layer_output_next_bytes, layer_output_next);
        add_bytes(result.layer_output_head_bytes, layer_output_head);

        add_bytes(result.lower_bound_workspace_bytes, layer_conv_output);
        add_bytes(result.lower_bound_workspace_bytes, layer_z);
        add_bytes(result.lower_bound_workspace_bytes, layer_output_next);
        add_bytes(result.lower_bound_workspace_bytes, layer_output_head);
      }
    }

    result.total_bytes = result.exact_ring_bytes;
    add_bytes(result.total_bytes, result.lower_bound_workspace_bytes);
    return result;
  }

  return result;
}

static void print_metadata(const nam::ModelMetadata& m)
{
  fprintf(stderr, "Metadata:\n");
  fprintf(stderr, "  Version: %s\n", m.version.c_str());
  fprintf(stderr, "  Sample rate: %.2f Hz\n", m.sample_rate);
  if (m.loudness.has_value())
    fprintf(stderr, "  Loudness: %.4f dB\n", m.loudness.value());
  if (m.input_level.has_value())
    fprintf(stderr, "  Input level: %.4f dBu\n", m.input_level.value());
  if (m.output_level.has_value())
    fprintf(stderr, "  Output level: %.4f dBu\n", m.output_level.value());
}

static void print_namb_info(const nam::namb::NambModelInfo& info, const bool verbose, const int max_buffer_size)
{
  fprintf(stderr, "Model structure:\n");
  fprintf(stderr, "  Architecture: %s (id=%u)\n", info.architecture_name.c_str(), info.architecture);

  if (info.linear.has_value())
  {
    const auto& l = info.linear.value();
    fprintf(stderr, "  Linear: receptive_field=%d, in_channels=%d, out_channels=%d, bias=%s\n", l.receptive_field,
            l.in_channels, l.out_channels, l.bias ? "true" : "false");
  }

  if (info.convnet.has_value())
  {
    const auto& c = info.convnet.value();
    fprintf(stderr,
            "  ConvNet: channels=%d, layers=%d, in_channels=%d, out_channels=%d, groups=%d, batchnorm=%s\n",
            c.channels, c.num_dilations, c.in_channels, c.out_channels, c.groups, c.batchnorm ? "true" : "false");
    if (verbose)
    {
      fprintf(stderr, "  ConvNet dilations:");
      for (const int d : c.dilations)
        fprintf(stderr, " %d", d);
      fprintf(stderr, "\n");
    }
  }

  if (info.lstm.has_value())
  {
    const auto& l = info.lstm.value();
    fprintf(stderr, "  LSTM: num_layers=%d, input_size=%d, hidden_size=%d, in_channels=%d, out_channels=%d\n",
            l.num_layers, l.input_size, l.hidden_size, l.in_channels, l.out_channels);
  }

  if (info.wavenet.has_value())
  {
    const auto& w = info.wavenet.value();
    fprintf(stderr, "  WaveNet: in_channels=%d, layer_arrays=%d, has_head=%s, has_condition_dsp=%s\n", w.in_channels,
            w.num_layer_arrays, w.with_head ? "true" : "false", w.has_condition_dsp ? "true" : "false");

    if (verbose)
    {
      for (size_t i = 0; i < w.layer_arrays.size(); i++)
      {
        const auto& la = w.layer_arrays[i];
        fprintf(stderr,
                "    LayerArray[%zu]: channels=%d, bottleneck=%d, input=%d, condition=%d, head=%d, "
                "head_kernel=%d, head_dilation=%d, head_bias=%s, groups_input=%d, groups_input_mixin=%d, "
                "dilations=%d\n",
                i, la.channels, la.bottleneck, la.input_size, la.condition_size, la.head_size, la.head_kernel_size,
                la.head_dilation, la.head_bias ? "true" : "false", la.groups_input, la.groups_input_mixin,
                la.num_dilations);

        fprintf(stderr, "      dilations:");
        for (const int d : la.dilations)
          fprintf(stderr, " %d", d);
        fprintf(stderr, "\n");

        fprintf(stderr, "      kernel_sizes:");
        for (const int k : la.kernel_sizes)
          fprintf(stderr, " %d", k);
        fprintf(stderr, "\n");
      }
    }
  }

      const size_t weight_bytes = static_cast<size_t>(info.total_weight_count) * sizeof(float);
      const RuntimeEstimate runtime = estimate_runtime_bytes(info, max_buffer_size);
  fprintf(stderr, "Memory:\n");
  fprintf(stderr, "  Weights: %u float32 = %zu bytes (%s)\n", info.total_weight_count, weight_bytes,
          human_bytes(weight_bytes).c_str());
      fprintf(stderr, "  Runtime buffers (max_buffer_size=%d):\n", max_buffer_size);
      if (runtime.has_detailed_breakdown)
      {
        fprintf(stderr, "    Condition buffers: %zu bytes (%s)\n", runtime.condition_buffers_bytes,
          human_bytes(runtime.condition_buffers_bytes).c_str());
        fprintf(stderr, "    Rechannel ring history (exact): %zu bytes (%s)\n", runtime.rechannel_ring_bytes,
          human_bytes(runtime.rechannel_ring_bytes).c_str());
        fprintf(stderr, "    Rechannel outputs: %zu bytes (%s)\n", runtime.rechannel_output_bytes,
          human_bytes(runtime.rechannel_output_bytes).c_str());
        fprintf(stderr, "    Head rechannel ring history (exact): %zu bytes (%s)\n", runtime.head_rechannel_ring_bytes,
          human_bytes(runtime.head_rechannel_ring_bytes).c_str());
        fprintf(stderr, "    Head rechannel outputs: %zu bytes (%s)\n", runtime.head_rechannel_output_bytes,
          human_bytes(runtime.head_rechannel_output_bytes).c_str());
        fprintf(stderr, "    LayerArray workspace: %zu bytes (%s)\n", runtime.layer_array_workspace_bytes,
          human_bytes(runtime.layer_array_workspace_bytes).c_str());
        fprintf(stderr, "    Layer conv ring history (exact): %zu bytes (%s)\n", runtime.layer_conv_ring_bytes,
          human_bytes(runtime.layer_conv_ring_bytes).c_str());
        fprintf(stderr, "    Layer conv outputs: %zu bytes (%s)\n", runtime.layer_conv_output_bytes,
          human_bytes(runtime.layer_conv_output_bytes).c_str());
        fprintf(stderr, "    Layer z buffers: %zu bytes (%s)\n", runtime.layer_z_bytes,
          human_bytes(runtime.layer_z_bytes).c_str());
        fprintf(stderr, "    Layer residual outputs: %zu bytes (%s)\n", runtime.layer_output_next_bytes,
          human_bytes(runtime.layer_output_next_bytes).c_str());
        fprintf(stderr, "    Layer head outputs: %zu bytes (%s)\n", runtime.layer_output_head_bytes,
          human_bytes(runtime.layer_output_head_bytes).c_str());

        fprintf(stderr, "    Conv/Ring history total (exact): %zu bytes (%s)\n", runtime.exact_ring_bytes,
          human_bytes(runtime.exact_ring_bytes).c_str());
        fprintf(stderr, "    Workspace total (lower bound): %zu bytes (%s)\n", runtime.lower_bound_workspace_bytes,
          human_bytes(runtime.lower_bound_workspace_bytes).c_str());
        fprintf(stderr, "    Total runtime (lower-bound estimate): %zu bytes (%s)\n", runtime.total_bytes,
          human_bytes(runtime.total_bytes).c_str());
      }
      else if (runtime.exact_ring_bytes > 0)
      {
        fprintf(stderr, "    Conv/Ring history (exact): %zu bytes (%s)\n", runtime.exact_ring_bytes,
          human_bytes(runtime.exact_ring_bytes).c_str());
        fprintf(stderr, "    Workspace (lower bound): %zu bytes (%s)\n", runtime.lower_bound_workspace_bytes,
          human_bytes(runtime.lower_bound_workspace_bytes).c_str());
        fprintf(stderr, "    Total (lower-bound estimate): %zu bytes (%s)\n", runtime.total_bytes,
          human_bytes(runtime.total_bytes).c_str());
      }
      else
      {
        fprintf(stderr, "    Estimate: %zu bytes (%s)\n", runtime.total_bytes, human_bytes(runtime.total_bytes).c_str());
      }
}

} // namespace

int main(int argc, char* argv[])
{
  bool verbose = false;
  const char* model_path = nullptr;
  int max_buffer_size = 128;

  for (int i = 1; i < argc; i++)
  {
    const std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h")
    {
      print_usage();
      return 0;
    }
    if (arg == "--verbose" || arg == "-v")
    {
      verbose = true;
      continue;
    }
    if ((arg == "--buffer-size" || arg == "--buffer-size" || arg == "-b") && (i + 1 < argc))
    {
      max_buffer_size = std::max(1, atoi(argv[++i]));
      continue;
    }
    if (model_path == nullptr)
    {
      model_path = argv[i];
      continue;
    }
    fprintf(stderr, "Unexpected argument: %s\n", argv[i]);
    print_usage();
    return 1;
  }

  if (model_path == nullptr)
  {
    print_usage();
    return 1;
  }

  fprintf(stderr, "Loading model [%s]\n", model_path);

  std::filesystem::path path(model_path);
  std::unique_ptr<nam::DSP> model;
  const bool is_namb = (path.extension() == ".namb");

  if (is_namb)
  {
    std::vector<uint8_t> data;
    if (!read_file_bytes(path, data))
    {
      fprintf(stderr, "Warning: failed to read namb file for introspection\n");
      return 1;
    }
    else
    {
      nam::namb::NambModelInfo info;
      if (!nam::namb::inspect_namb(data.data(), data.size(), info))
      {
        fprintf(stderr, "Warning: failed to inspect namb details: %s\n", info.error.c_str());
      }
      else
      {
        print_metadata(info.metadata);
        print_namb_info(info, verbose, max_buffer_size);
      }

      model = nam::get_dsp_namb(data.data(), data.size());
    }
  }
  else
  {
    fprintf(stderr, "Info: detailed layer/memory introspection is available for .namb models.\n");
    model = nam::get_dsp(path);
  }

  if (model != nullptr)
  {
    fprintf(stderr, "Runtime DSP:\n");
    fprintf(stderr, "  Input channels: %d\n", model->NumInputChannels());
    fprintf(stderr, "  Output channels: %d\n", model->NumOutputChannels());
    fprintf(stderr, "  Expected sample rate: %.2f Hz\n", model->GetExpectedSampleRate());
    fprintf(stderr, "  Has loudness: %s\n", model->HasLoudness() ? "true" : "false");
    fprintf(stderr, "  Has input level: %s\n", model->HasInputLevel() ? "true" : "false");
    fprintf(stderr, "  Has output level: %s\n", model->HasOutputLevel() ? "true" : "false");
    fprintf(stderr, "Model loaded successfully\n");
    return 0;
  }

  fprintf(stderr, "Failed to load model\n");
  return 1;
}
