/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/* \file
   \brief Class performing output during profiling
*/

#pragma once

#include <vector>
#include <fstream>

// CUTLASS Profiler includes
#include "options.h"
#include "enumerated_types.h"
#include "performance_result.h"

// CUTLASS Library includes
#include "cutlass/library/library.h"

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

class PerformanceReport {
private:

  /// Reference to options
  Options const &options_;

  /// Operation kind
  library::OperationKind op_kind_;

  /// Operation file name containing performance report of op_kind
  std::string op_file_name_;

  /// Output file containing results
  std::ofstream output_file_;

  /// Flag indicating the performance report is valid
  bool good_;

  /// Vector of argument names
  std::vector<std::string> argument_names_;

  /// Counter uniquely identifying problem within the report
  size_t problem_index_;

  /// Collection of all results
  PerformanceResultVector concatenated_results_;

public:

  PerformanceReport(Options const &options, std::vector<std::string> const &argument_names, library::OperationKind const &op_kind);

  bool good() const { return good_; }

  void next_problem();
  void append_result(PerformanceResult result);
  void append_results(PerformanceResultVector const &results);

  void close();

public:

  /// Prints the CSV header
  std::ostream & print_csv_header_(std::ostream &out);

  /// Prints the CSV
  std::ostream & print_result_csv_(std::ostream &out, PerformanceResult const &result);

  /// Prints the result in human readable form
  std::ostream & print_result_pretty_(
    std::ostream &out, 
    PerformanceResult const &result);
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

