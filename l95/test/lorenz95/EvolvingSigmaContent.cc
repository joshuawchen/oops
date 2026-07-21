/*
 * (C) Copyright 2026 Colorado State University
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

/// \file EvolvingSigmaContent.cc
///
/// The Gaussian-equivalence run configures the evolving-Gaussian term with a
/// density that IS Gaussian with sigma = 0.4, so every value the diagnostic
/// writes must be exactly 0.4. Anything else is a bug in the save flag, the
/// group naming, the outer-loop index, or the missing-value handling.

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

CASE("l95_evolvingsigma_content") {
  const eckit::Configuration & conf = TestEnvironment::config();
  const double expected = conf.getDouble("expected sigma");
  const double tol = conf.getDouble("tolerance", 1.0e-6);

  const eckit::PathName confPath = conf.getString("run config");
  const eckit::YAMLConfiguration runConf(confPath);
  const eckit::LocalConfiguration costConf(runConf, "cost function");
  const std::vector<eckit::LocalConfiguration> obsConfs =
      costConf.getSubConfigurations("observations.observers");
  const eckit::LocalConfiguration timeWindowConf(costConf, "time window");

  eckit::LocalConfiguration obsSpaceConf;
  obsSpaceConf.set("obsdatain.obsfile",
                   obsConfs[0].getString("obs space.obsdataout.obsfile"));
  lorenz95::ObsTable obsSpace(obsSpaceConf, oops::mpi::world(),
                              util::TimeWindow(timeWindowConf),
                              oops::mpi::myself());

  for (const std::string & group : conf.getStringVector("groups")) {
    std::vector<double> sig;
    obsSpace.getdb(group, sig);
    EXPECT(!sig.empty());
    double dev = 0.0;
    for (double v : sig) dev = std::max(dev, std::abs(v - expected));
    oops::Log::test() << group << ": " << sig.size() << " values, max deviation from "
                      << expected << " = " << dev << std::endl;
    EXPECT(dev <= tol);
  }
}

class EvolvingSigmaContent : public oops::Test {
 public:
  EvolvingSigmaContent() {}

 private:
  std::string testid() const override {return "test::EvolvingSigmaContent";}

  void register_tests() const override {}

  void clear() const override {}
};

}  // namespace test

int main(int argc, char **argv) {
  oops::Run run(argc, argv);
  test::EvolvingSigmaContent tests;
  return run.execute(tests);
}
