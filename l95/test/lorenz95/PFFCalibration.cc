/*
 * (C) Copyright 2026 Colorado State University
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

/// \file PFFCalibration.cc
///
/// Calibration gate for the conformant PFF (Hu & van Leeuwen 2021): with
/// unperturbed observations all particles share ONE likelihood, so in the
/// linear-Gaussian case the flow targets the posterior of N(mean background,
/// B) x likelihood -- whose mean is exactly the 3D-Var analysis started from
/// the mean background (pff_calibration_exact). Checks the ensemble MEAN
/// against that analysis and the ensemble SPREAD against a configured band.
/// This is the correctness gate the PFF norm diagnostic cannot be: the
/// repulsion term is pairwise-antisymmetric and cancels exactly in the
/// summed-update norm, so the norm is blind to spread pathologies.

#include <cmath>
#include <string>
#include <vector>

#include "eckit/config/LocalConfiguration.h"
#include "eckit/testing/Test.h"
#include "lorenz95/Resolution.h"
#include "lorenz95/StateL95.h"
#include "oops/runs/Run.h"
#include "oops/runs/Test.h"
#include "oops/util/Expect.h"
#include "oops/util/Logger.h"
#include "test/TestEnvironment.h"

namespace test {

CASE("test_pff_calibration") {
  const eckit::Configuration & conf = TestEnvironment::config();
  lorenz95::Resolution resol(eckit::LocalConfiguration(conf, "geometry"),
                             oops::mpi::myself());

  const auto memberConfs = conf.getSubConfigurations("member analyses");
  EXPECT(memberConfs.size() >= 2);
  std::vector<std::vector<double>> members;
  for (const auto & mc : memberConfs) {
    lorenz95::StateL95 xx(resol, mc);
    members.push_back(xx.getField().asVector());
  }
  lorenz95::StateL95 xexact(resol,
                            eckit::LocalConfiguration(conf, "exact analysis"));
  const std::vector<double> & exact = xexact.getField().asVector();

  const size_t nx = exact.size();
  const size_t nm = members.size();
  std::vector<double> mean(nx, 0.0);
  for (const auto & m : members) {
    EXPECT_EQUAL(m.size(), nx);
    for (size_t i = 0; i < nx; ++i) mean[i] += m[i];
  }
  for (size_t i = 0; i < nx; ++i) mean[i] /= static_cast<double>(nm);

  double meandev2 = 0.0;
  for (size_t i = 0; i < nx; ++i) {
    const double d = mean[i] - exact[i];
    meandev2 += d * d;
  }
  const double meanDev = std::sqrt(meandev2 / static_cast<double>(nx));

  double spread2 = 0.0;
  for (const auto & m : members)
    for (size_t i = 0; i < nx; ++i) {
      const double d = m[i] - mean[i];
      spread2 += d * d;
    }
  const double spread =
      std::sqrt(spread2 / static_cast<double>(nx * (nm - 1)));

  oops::Log::test() << "PFF calibration: ensemble mean rms deviation from the "
                    << "exact posterior mean: " << meanDev << std::endl;
  oops::Log::test() << "PFF calibration: ensemble spread: " << spread
                    << std::endl;

  EXPECT(meanDev <= conf.getDouble("mean tolerance"));
  EXPECT(spread >= conf.getDouble("spread min"));
  EXPECT(spread <= conf.getDouble("spread max"));
}

class PFFCalibration : public oops::Test {
 public:
  PFFCalibration() {}
 private:
  std::string testid() const override {return "test::PFFCalibration";}
  void register_tests() const override {}
  void clear() const override {}
};

}  // namespace test

int main(int argc, char **argv) {
  oops::Run run(argc, argv);
  test::PFFCalibration tests;
  return run.execute(tests);
}
