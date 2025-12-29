#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct Options {
  std::string input;
  std::string output;
  std::size_t desired_count{100};
  bool has_header{true};
};

bool starts_with(const std::string &arg, const std::string &prefix) {
  return arg.rfind(prefix, 0) == 0;
}

Options parse_args(int argc, char **argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
      opts.input = argv[++i];
    } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
      opts.output = argv[++i];
    } else if ((arg == "-n" || arg == "--count") && i + 1 < argc) {
      opts.desired_count = static_cast<std::size_t>(std::stoul(argv[++i]));
    } else if (arg == "--no-header") {
      opts.has_header = false;
    } else if (starts_with(arg, "-h")) {
      throw std::runtime_error("Usage: csv_downsampler -i INPUT -o OUTPUT [-n COUNT] [--no-header]");
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (opts.input.empty()) {
    throw std::runtime_error("Missing required --input path");
  }
  if (opts.output.empty()) {
    throw std::runtime_error("Missing required --output path");
  }
  if (opts.desired_count == 0) {
    throw std::runtime_error("Target count must be > 0");
  }
  return opts;
}

std::vector<std::string> split_csv_line(const std::string &line) {
  std::vector<std::string> tokens;
  std::stringstream ss(line);
  std::string token;
  while (std::getline(ss, token, ',')) {
    tokens.emplace_back(std::move(token));
  }
  return tokens;
}

struct CsvData {
  std::vector<std::string> header;
  std::vector<std::vector<std::string>> rows;
};

CsvData load_csv(const std::string &path, bool has_header) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    throw std::runtime_error("Failed to open input CSV: " + path);
  }

  CsvData data;
  std::string line;

  if (has_header && std::getline(ifs, line)) {
    data.header = split_csv_line(line);
  }

  while (std::getline(ifs, line)) {
    if (line.empty()) {
      continue;
    }
    data.rows.emplace_back(split_csv_line(line));
  }

  if (data.rows.empty()) {
    throw std::runtime_error("Input CSV contains no data rows: " + path);
  }

  return data;
}

std::vector<std::size_t> pick_indices(std::size_t total, std::size_t desired) {
  if (desired >= total) {
    std::vector<std::size_t> all(total);
    std::iota(all.begin(), all.end(), 0);
    return all;
  }

  const std::size_t last = total - 1;
  const double denom = static_cast<double>(std::max<std::size_t>(desired - 1, 1));

  std::vector<std::size_t> idxs;
  idxs.reserve(desired);
  std::unordered_set<std::size_t> seen;

  for (std::size_t i = 0; i < desired; ++i) {
    const double ratio = static_cast<double>(i) / denom;
    const auto idx = static_cast<std::size_t>(std::llround(ratio * static_cast<double>(last)));
    if (seen.insert(idx).second) {
      idxs.push_back(idx);
    }
  }

  if (!seen.count(0)) {
    idxs.push_back(0);
    seen.insert(0);
  }
  if (!seen.count(last)) {
    idxs.push_back(last);
    seen.insert(last);
  }

  for (std::size_t candidate = 0; idxs.size() < desired && candidate < total; ++candidate) {
    if (seen.insert(candidate).second) {
      idxs.push_back(candidate);
    }
  }

  std::sort(idxs.begin(), idxs.end());
  return idxs;
}

void write_csv(const std::string &path,
               const std::vector<std::string> &header,
               const std::vector<std::vector<std::string>> &rows) {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    throw std::runtime_error("Failed to open output CSV: " + path);
  }

  auto write_row = [&ofs](const std::vector<std::string> &row) {
    for (std::size_t i = 0; i < row.size(); ++i) {
      ofs << row[i];
      if (i + 1 < row.size()) {
        ofs << ",";
      }
    }
    ofs << "\n";
  };

  if (!header.empty()) {
    write_row(header);
  }
  for (const auto &row : rows) {
    write_row(row);
  }
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options opts = parse_args(argc, argv);
    CsvData data = load_csv(opts.input, opts.has_header);
    const auto indices = pick_indices(data.rows.size(), opts.desired_count);

    std::vector<std::vector<std::string>> sampled;
    sampled.reserve(indices.size());
    for (const auto idx : indices) {
      sampled.push_back(data.rows.at(idx));
    }

    write_csv(opts.output, opts.has_header ? data.header : std::vector<std::string>{}, sampled);

    std::cout << "Downsampled " << data.rows.size() << " -> " << sampled.size()
              << " rows into " << opts.output << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
