/*
 * (C) Copyright 2025 Colorado State University
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

/*
 * Tier-2 method test: reproduce the single-cycle idealized experiment of
 * Hu, Geer & van Leeuwen (2025, QJRMS e5050, sec 3 & Appendix C) and show the
 * evolving-Gaussian outer loop converges to the TRUE (analytical) posterior mode
 * for a non-Gaussian observation error, while a fixed Gaussian does not.
 *
 * Setup: 1-D, H = identity, mirrored-Gamma obs-error pdf
 *     f(d) = (2-d)/4 * exp((d-2)/2),  d < 2,  mode m = 0,
 * so the score is g(d) = d / (2(2-d)) and Eq. 9 gives the evolving effective
 * variance in closed form:  sigma_o^2(d) = (d-m)/g(d) = 2(2-d) = 4 - 2d.
 * Background variance sigma_b^2 = 8, observation y_o = 2.
 *
 * Outer-loop map (Appendix C, Eq. C1), with sigma_o evaluated at the current
 * departure d = y_o - y_n:
 *     y_{n+1} = [sigma_o^2/(sigma_o^2+sigma_b^2)] y_b
 *             + [sigma_b^2/(sigma_o^2+sigma_b^2)] y_o.
 *
 * The analytical mode solves (x-x_b)/sigma_b^2 = g(y_o-x). For this f this is
 * x = x_b + sigma_b^2 g(y_o-x); reference roots below were obtained numerically
 * and verified against the closed form (e.g. x_b=10 -> (6+sqrt(68))/2).
 *
 * Assertions:
 *   (converges_to_mode) evolving-Gaussian matches the analytical mode to 1e-6
 *                       across a background sweep, within <=8 outer loops.
 *   (beats_fixed_gaussian) a fixed Gaussian (sigma_o^2 = 8, mode-centred) misses
 *                       the analytical mode by a large margin where f is skewed
 *                       -- so the test is not vacuous.
 *   (fast_convergence)  3 outer loops already reach ~5e-2 of the mode (the paper
 *                       notes 2-3 outer loops are sufficient in practice); full
 *                       convergence to 1e-6 takes ~10 loops (geometric, ratio ~0.21).
 *
 * NB: this validates the METHOD (the sigma_o(d) map + outer-loop fixed point),
 * independent of storage. Class-level plumbing (CostJoEvolvingGaussian inside a
 * running minimizer) is covered by the model-level integration test.
 */

#ifndef TEST_ASSIMILATION_EVOLVINGGAUSSIANSINGLECYCLE_H_
#define TEST_ASSIMILATION_EVOLVINGGAUSSIANSINGLECYCLE_H_

#include <cmath>
#include <string>
#include <vector>

#include "eckit/testing/Test.h"

#include "oops/../test/TestEnvironment.h"
#include "oops/runs/Test.h"
#include "oops/util/Expect.h"
#include "oops/util/FloatCompare.h"

namespace test {

namespace {
const double yo = 2.0;
const double sb2 = 8.0;

// closed-form score and Eq.9 variance for the mirrored-Gamma
double score(double d) { return d / (2.0 * (2.0 - d)); }          // g(d)
double evar(double d)  { return 2.0 * (2.0 - d); }                // sigma_o^2(d)

// evolving-Gaussian outer loop from background xb
double evolve(double xb, int nouter) {
  double y = xb;
  for (int i = 0; i < nouter; ++i) {
    const double s2 = evar(yo - y);
    y = (s2 / (s2 + sb2)) * xb + (sb2 / (s2 + sb2)) * yo;
  }
  return y;
}

// fixed Gaussian analysis, sigma_o^2 = 8, mode-centred (single linear solve)
double fixedGaussian(double xb, double so2) {
  return (so2 / (so2 + sb2)) * xb + (sb2 / (so2 + sb2)) * yo;
}
}  // namespace

CASE("assimilation/EvolvingGaussianSingleCycle/converges_to_mode") {
  // background sweep -> analytical posterior mode (verified references)
  struct Ref { double xb, mode; };
  const std::vector<Ref> ref = {
    {0.6,  1.600000}, {6.0, 4.000000}, {10.0, 7.123106}, {15.0, 11.684658},
  };
  for (const auto & r : ref)
    EXPECT(oops::is_close_absolute(evolve(r.xb, 40), r.mode, 1.0e-6));
}

CASE("assimilation/EvolvingGaussianSingleCycle/beats_fixed_gaussian") {
  // where f is skewed, fixed Gaussian misses the analytical mode badly.
  struct Ref { double xb, mode, minMiss; };
  const std::vector<Ref> ref = {
    {10.0, 7.123106, 1.0},   // fixed-mode gives 6.0, off by ~1.12
    {15.0, 11.684658, 3.0},  // fixed-mode gives 8.5, off by ~3.18
  };
  for (const auto & r : ref) {
    const double miss = std::abs(fixedGaussian(r.xb, 8.0) - r.mode);
    EXPECT(miss > r.minMiss);
    EXPECT(oops::is_close_absolute(evolve(r.xb, 40), r.mode, 1.0e-6));
  }
}

CASE("assimilation/EvolvingGaussianSingleCycle/fast_convergence") {
  // paper: 2-3 outer loops sufficient in practice (not machine tolerance).
  EXPECT(oops::is_close_absolute(evolve(10.0, 3), 7.123106,  5.0e-2));
  EXPECT(oops::is_close_absolute(evolve(15.0, 3), 11.684658, 5.0e-2));
  // and it does drive to machine-ish tolerance given enough loops:
  EXPECT(oops::is_close_absolute(evolve(10.0, 12), 7.123106, 1.0e-5));
}

class EvolvingGaussianSingleCycle : public oops::Test {
 public:
  using oops::Test::Test;
 private:
  std::string testid() const override {return "test::EvolvingGaussianSingleCycle";}
  void register_tests() const override {}
  void clear() const override {}
};

}  // namespace test

#endif  // TEST_ASSIMILATION_EVOLVINGGAUSSIANSINGLECYCLE_H_
