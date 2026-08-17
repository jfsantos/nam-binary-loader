#pragma once
// Binary config parser registry for .namb format
// Mirrors ConfigParserRegistry from model_config.h but maps uint8_t architecture IDs
// to binary parser functions instead of string names to JSON parsers.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <NAM/model_config.h>
#include "namb_format.h"

namespace nam
{
namespace namb
{

using BinaryConfigParserFunction = std::function<std::unique_ptr<ModelConfig>(
    BinaryReader& reader, const float*& weights, size_t& weight_count, const ModelMetadata& meta)>;

class BinaryConfigParserRegistry
{
public:
  static BinaryConfigParserRegistry& instance()
  {
    static BinaryConfigParserRegistry registry;
    return registry;
  }

  void registerParser(uint8_t arch_id, BinaryConfigParserFunction func)
  {
    parsers_[arch_id] = std::move(func);
  }

  bool has(uint8_t arch_id) const { return parsers_.find(arch_id) != parsers_.end(); }

  /// \brief Parse a config for an architecture ID, or nullptr if it is not registered
  std::unique_ptr<ModelConfig> parse(uint8_t arch_id, BinaryReader& reader, const float*& weights,
                                      size_t& weight_count, const ModelMetadata& meta) const
  {
    auto it = parsers_.find(arch_id);
    if (it == parsers_.end())
      return nullptr;
    return it->second(reader, weights, weight_count, meta);
  }

private:
  BinaryConfigParserRegistry() = default;
  std::unordered_map<uint8_t, BinaryConfigParserFunction> parsers_;
};

struct BinaryConfigParserHelper
{
  BinaryConfigParserHelper(uint8_t arch_id, BinaryConfigParserFunction func)
  {
    BinaryConfigParserRegistry::instance().registerParser(arch_id, std::move(func));
  }
};

} // namespace namb
} // namespace nam
