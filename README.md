# nam-binary-loader

A compact binary model format (`.namb`) for [Neural Amp Modeler](https://github.com/sdatkinson/NeuralAmpModelerCore) models.

## Motivation

NAM models are distributed as `.nam` files, which are JSON documents containing model configuration and weights encoded as arrays of floating-point numbers in text form. While JSON is convenient and human-readable, it has significant drawbacks in resource-constrained environments:

- **Parsing overhead**: The nlohmann/json library adds substantial code size and requires dynamic memory allocation during parsing. On embedded targets (e.g. STM32H7 with 512 KB RAM), this can be prohibitive.
- **File size**: Text-encoded floats are roughly 10x larger than their binary representation. A typical WaveNet model shrinks by ~80% when converted to `.namb`.
- **Load time**: Binary deserialization is essentially `memcpy` for weights, whereas JSON parsing must tokenize, validate, and convert each number from text.

The `.namb` format addresses these issues by storing model configuration as fixed-layout binary structures and weights as raw IEEE 754 floats. The loader (`get_dsp_namb`) has no dependency on nlohmann/json, making it suitable for embedded targets where only the binary loader needs to be linked.

We wrote more about what led us to develop this and other optimizations for embedded devices [here](https://www.tone3000.com/blog/running-nam-on-embedded-hardware).

## File format overview

All multi-byte values are little-endian. Format version: 2.

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | Magic number (`0x4E414D42` = "NAMB") |
| 4 | 2 | Format version |
| 6 | 2 | Flags (reserved) |
| 8 | 4 | Total file size |
| 12 | 4 | Weights offset |
| 16 | 4 | Total weight count |
| 20 | 4 | Model block size |
| 24 | 4 | CRC32 checksum (IEEE 802.3, excludes this field) |
| 28 | 4 | Reserved |
| 32 | 48 | Metadata block (version, sample rate, loudness, levels) |
| 80 | var | Model block (architecture ID + config) |
| ... | ... | Padding (4-byte aligned) |
| weights_offset | weight_count * 4 | Weight data (raw float32) |

Supported architectures: Linear, ConvNet, LSTM, WaveNet (including recursive condition DSP).

## Repository structure

```
namb/                        # Library: format spec, reader/writer, loader
  namb_format.h              #   Binary format constants, CRC32, BinaryReader/BinaryWriter
  get_dsp_namb.h             #   Public API: get_dsp_namb()
  get_dsp_namb.cpp           #   Binary loader implementation
  binary_parser_registry.h   #   Architecture parser registry
tools/
  nam2namb.cpp               # CLI converter: .nam (JSON) -> .namb (binary)
  loadmodel.cpp              # Model loader supporting both .nam and .namb
  render_namb.cpp            # Offline renderer: process WAV through a .namb model
test/
  test_namb.cpp              # Tests: CRC32, format validation, round-trip, size reduction
```

## Dependencies

- [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore)
- Eigen (transitive, via NeuralAmpModelerCore)
- nlohmann/json (only for `nam2namb` converter tool, **not** for the loader)

## Building

### Windows (MSVC)

From the repository root:

```powershell
mkdir build && cd build
cmake .. -DNAM_CORE_PATH="/path/to/NeuralAmpModelerCore"
cmake --build . --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

### Linux and macOS

```bash
mkdir build && cd build
cmake .. -DNAM_CORE_PATH=/path/to/NeuralAmpModelerCore
make
```

`NAM_CORE_PATH` defaults to `../NeuralAmpModelerCore` (sibling directory).

## Usage

### Converting models

```bash
./nam2namb model.nam model.namb
```

If the output path is omitted, the `.nam` extension is replaced with `.namb`.

### Loading in C++

```cpp
#include <namb/get_dsp_namb.h>

// From file
auto model = nam::get_dsp_namb("model.namb");

// From memory buffer (useful for embedded/QSPI flash)
auto model = nam::get_dsp_namb(data_ptr, data_size);
```

### Inspecting memory with loadmodel

`loadmodel` prints architecture details plus a runtime memory estimate. For
buffered models (especially WaveNet), runtime memory depends strongly on the
audio block size. Use `--buffer-size` to match your
target:

```bash
./loadmodel --buffer-size 128 model.namb   # e.g. 128-samples per block
./loadmodel --buffer-size 48 model.namb    # e.g. 48-samples per block
```

For WaveNet, the report now includes a per-component breakdown:

- `Condition buffers`: top-level condition input/output matrices.
- `Rechannel ring history (exact)` and `Head rechannel ring history (exact)`.
- `Rechannel outputs` and `Head rechannel outputs`.
- `LayerArray workspace`: aggregate of per-array work matrices.
- `Layer conv ring history (exact)`: sum across all dilated conv layer rings.
- `Layer conv outputs`, `Layer z buffers`, `Layer residual outputs`, and
  `Layer head outputs`.
- `Conv/Ring history total (exact)`, `Workspace total (lower bound)`, and
  `Total runtime (lower-bound estimate)`.

### Embedded integration (Daisy / STM32)

On embedded targets, you typically store the `.namb` file in external flash (e.g. QSPI) and load it directly from memory. The memory-buffer overload of `get_dsp_namb` avoids any filesystem dependency:

```cpp
#include <namb/get_dsp_namb.h>

// Model stored in QSPI flash at a known address
extern const uint8_t namb_data[];
extern const size_t  namb_size;

auto model = nam::get_dsp_namb(namb_data, namb_size);
model->prewarm();
```

The loader (`get_dsp_namb.cpp`) and its headers are self-contained — they do **not** depend on nlohmann/json. The only external dependencies are NeuralAmpModelerCore (for `create_dsp` and the DSP model classes) and Eigen (transitive, via NeuralAmpModelerCore).

To add the loader to a Daisy project, include these source/header files in your build:

```
namb/get_dsp_namb.h              # Public API
namb/get_dsp_namb.cpp            # Loader implementation
namb/namb_format.h               # Format constants, CRC32, BinaryReader
namb/binary_parser_registry.h    # Architecture parser registry
```

Plus the NeuralAmpModelerCore sources your target architecture requires (e.g. `NAM/lstm.cpp`, `NAM/wavenet.cpp`).

### Rendering audio offline

```bash
./render_namb model.namb input.wav [output.wav]
```

Processes `input.wav` through the model and writes the result to `output.wav` (defaults to `output.wav` if omitted). Input sample rate must match the model's expected rate.

### Running tests

```bash
./test_namb
```

Tests require example models from the NeuralAmpModelerCore `example_models/` directory to be accessible from the working directory.

When built with CMake, run the registered test from any directory with:

```bash
ctest --test-dir build -C Debug --output-on-failure
```
