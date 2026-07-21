/*
 * (C) Copyright 2026 Colorado State University
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

/// \file EvolvingGaussianShiftEquivalence.cc
///
/// An evolving-Gaussian run with a Gaussian density at mode +m must produce
/// the same analysis as a stock Gaussian run with a fixed obs bias of -m,
/// and a different analysis than a bias of +m. This pins the departure sign
/// convention (the density is evaluated at H(x) - y) and exercises the
/// mode-subtracting Jo value end to end. l95 adds the bias inside
/// simulateObs, so the saved oman columns of identical analyses differ by
/// exactly the bias; the comparison removes it.

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "eckit/config/LocalConfiguration.h"
#include "eckit/testing/Test.h"
#include "lorenz95/ObsTable.h"
#include "oops/runs/Run.h"
#include "oops/runs/Test.h"
#include "oops/util/Expect.h"
#include "oops/util/TimeWindow.h"
#include "test/TestEnvironment.h"

namespace test {

struct RunOut {
  std::vector<double> oman;
  double bias = 0.0;
};

// Read a run's config, then its output obs file, and return oman + the bias.
RunOut loadRun(const std::string & cfgfile) {
  const eckit::PathName confPath = cfgfile;
  const eckit::YAMLConfiguration conf(confPath);
  const eckit::LocalConfiguration costConf(conf, "cost function");
  const std::vector<eckit::LocalConfiguration> obsConfs =
      costConf.getSubConfigurations("observations.observers");
  const eckit::LocalConfiguration timeWindowConf(costConf, "time window");

  RunOut out;
  out.bias = obsConfs[0].getDouble("obs bias.bias", 0.0);

  eckit::LocalConfiguration obsSpaceConf;
  obsSpaceConf.set("obsdatain.obsfile",
                   obsConfs[0].getString("obs space.obsdataout.obsfile"));
  lorenz95::ObsTable obsSpace(obsSpaceConf, oops::mpi::world(),
                              util::TimeWindow(timeWindowConf),
                              oops::mpi::myself());
  obsSpace.getdb("oman", out.oman);
  return out;
}

CASE("l95_evolvinggaussian_shift_equivalence") {
  const eckit::Configuration & conf = TestEnvironment::config();
  const RunOut s = loadRun(conf.getString("shifted config"));
  const RunOut m = loadRun(conf.getString("matching config"));
  const RunOut p = loadRun(conf.getString("differing config"));
  EXPECT_EQUAL(s.oman.size(), m.oman.size());
  EXPECT_EQUAL(s.oman.size(), p.oman.size());

  // oman = y - (H(x_a) + bias): identical analyses leave residual ~0 after
  // removing the bias; a different analysis leaves an O(bias) residual.
  double dm = 0.0;
  double dp = 0.0;
  for (size_t jj = 0; jj < s.oman.size(); ++jj) {
    dm = std::max(dm, std::abs(s.oman[jj] - m.oman[jj] - m.bias));
    dp = std::max(dp, std::abs(s.oman[jj] - p.oman[jj] - p.bias));
  }
  oops::Log::test() << "shifted vs matching bias: max residual = " << dm << std::endl;
  oops::Log::test() << "shifted vs differing bias: max residual = " << dp << std::endl;
  EXPECT(dm < conf.getDouble("match tolerance", 1.0e-4));
  EXPECT(dp > conf.getDouble("differ threshold", 5.0e-2));
}

class EvolvingGaussianShiftEquivalence : public oops::Test {
 public:
  EvolvingGaussianShiftEquivalence() {}

 private:
  std::string testid() const override {return "test::EvolvingGaussianShiftEquivalence";}

  void register_tests() const override {}

  void clear() const override {}
};

}  // namespace test

int main(int argc, char **argv) {
  oops::Run run(argc, argv);
  test::EvolvingGaussianShiftEquivalence tests;
  return run.execute(tests);
}
