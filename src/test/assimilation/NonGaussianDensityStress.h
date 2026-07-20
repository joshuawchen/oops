/*
 * (C) Copyright 2025 Colorado State University
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

/*
 * Robustness of the non-Gaussian observation-error density under adversarial
 * and degenerate configurations. A real DOEE estimate is a noisy histogram, so
 * the failure modes here are not hypothetical.
 *
 * Motivation. Before validation was added, three failures were SILENT:
 *   - a "log slopes" array of the wrong length clamped the bin index and
 *     returned values belonging to a different bin;
 *   - a density that is not unimodal makes the score disagree in sign with
 *     (d - m), so Eq. 9 returns a NEGATIVE variance, which was then clamped to
 *     the floor and gave those observations a weight of order 1e6;
 *   - a positive tail curvature describes a density growing without bound and
 *     produced the same catastrophic clamping in the tails.
 * Each is now rejected at construction, and the evaluator falls back to the
 * fitted sigma at the mode rather than to the floor if a bad value survives.
 */

#ifndef TEST_ASSIMILATION_NONGAUSSIANDENSITYSTRESS_H_
#define TEST_ASSIMILATION_NONGAUSSIANDENSITYSTRESS_H_

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "eckit/config/LocalConfiguration.h"
#include "eckit/exception/Exceptions.h"
#include "eckit/testing/Test.h"

#include "oops/../test/TestEnvironment.h"
#include "oops/assimilation/NonGaussianDensity.h"
#include "oops/runs/Test.h"
#include "oops/util/Expect.h"
#include "oops/util/FloatCompare.h"

namespace test {

/// A valid, unimodal, skewed baseline. Individual fields are then corrupted.
inline eckit::LocalConfiguration stressBase() {
  eckit::LocalConfiguration c;
  c.set("mode", 0.0);
  c.set("grid spacing", 1.0);
  c.set("stable min", -3.0);
  c.set("stable max", 3.0);
  c.set("log slopes", std::vector<double>{0.9, 0.6, 0.2, -0.4, -0.8, -1.5});
  c.set("left log slope", 0.9);
  c.set("left curvature", -0.10);
  c.set("right log slope", -1.5);
  c.set("right curvature", -0.20);
  c.set("sigma at mode", 1.0);
  c.set("sigma floor", 1.0e-3);
  c.set("mode window", 0.0);
  return c;
}

inline oops::NonGaussianDensity build(const eckit::LocalConfiguration & c) {
  oops::NonGaussianDensityParameters p;
  p.deserialize(c);
  return oops::NonGaussianDensity(p);
}

CASE("assimilation/NonGaussianDensityStress/baseline_is_valid") {
  EXPECT_NO_THROW(build(stressBase()));
}

CASE("assimilation/NonGaussianDensityStress/rejects_bad_configurations") {
  // slopes array inconsistent with the grid -- previously clamped silently
  {
    eckit::LocalConfiguration c = stressBase();
    c.set("log slopes", std::vector<double>{0.9, 0.6, 0.2});
    EXPECT_THROWS_AS(build(c), eckit::BadParameter);
  }
  // not unimodal: log density still rising above the mode
  {
    eckit::LocalConfiguration c = stressBase();
    c.set("log slopes", std::vector<double>{0.9, 0.6, 0.2, +0.4, -0.8, -1.5});
    EXPECT_THROWS_AS(build(c), eckit::BadParameter);
  }
  // not unimodal: log density already falling below the mode
  {
    eckit::LocalConfiguration c = stressBase();
    c.set("log slopes", std::vector<double>{-0.9, 0.6, 0.2, -0.4, -0.8, -1.5});
    EXPECT_THROWS_AS(build(c), eckit::BadParameter);
  }
  // improper density: tail grows without bound
  {
    eckit::LocalConfiguration c = stressBase();
    c.set("right curvature", 0.5);
    EXPECT_THROWS_AS(build(c), eckit::BadParameter);
  }
  {
    eckit::LocalConfiguration c = stressBase();
    c.set("left curvature", 0.5);
    EXPECT_THROWS_AS(build(c), eckit::BadParameter);
  }
  // degenerate grid
  {
    eckit::LocalConfiguration c = stressBase();
    c.set("grid spacing", 0.0);
    EXPECT_THROWS_AS(build(c), eckit::BadParameter);
  }
  {
    eckit::LocalConfiguration c = stressBase();
    c.set("stable max", -4.0);
    EXPECT_THROWS_AS(build(c), eckit::BadParameter);
  }
  // sigma at the mode must be usable: it is the fallback everywhere
  {
    eckit::LocalConfiguration c = stressBase();
    c.set("sigma at mode", 0.0);
    EXPECT_THROWS_AS(build(c), eckit::BadParameter);
  }
  {
    eckit::LocalConfiguration c = stressBase();
    c.set("sigma at mode", -1.0);
    EXPECT_THROWS_AS(build(c), eckit::BadParameter);
  }
  // mode outside the represented support
  {
    eckit::LocalConfiguration c = stressBase();
    c.set("mode", 7.0);
    EXPECT_THROWS_AS(build(c), eckit::BadParameter);
  }
  // empty density
  {
    eckit::LocalConfiguration c = stressBase();
    c.set("log slopes", std::vector<double>{});
    EXPECT_THROWS_AS(build(c), eckit::BadParameter);
  }
}

CASE("assimilation/NonGaussianDensityStress/finite_and_positive_everywhere") {
  // Sweep far beyond the represented support, including the bin edges and the
  // mode, and demand a usable Jo Hessian at every point. A single non-finite or
  // non-positive value would break the Krylov solve.
  const oops::NonGaussianDensity f = build(stressBase());
  std::vector<double> pts{0.0, 1.0e-12, -1.0e-12, 1.0e-6, -1.0e-6,
                          1.0e2, -1.0e2, 1.0e4, -1.0e4, 1.0e300, -1.0e300};
  for (double d = -6.0; d <= 6.0; d += 0.05) pts.push_back(d);
  for (double e : {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0}) {   // exact bin edges
    pts.push_back(e);
    pts.push_back(std::nextafter(e, -10.0));
    pts.push_back(std::nextafter(e, 10.0));
  }

  for (double d : pts) {
    const double v = f.evolvingVariance(d);
    EXPECT(std::isfinite(v));
    EXPECT(v > 0.0);
    EXPECT(std::isfinite(f.score(d)));
    EXPECT(f.scoreSlope(d) > 0.0);     // curvature proxy must stay positive
  }
}

CASE("assimilation/NonGaussianDensityStress/heavy_tail_downweights") {
  // A flatter tail must mean a LARGER effective error, i.e. less weight for
  // outliers. This is the qualitative behaviour the method exists to provide,
  // so assert the monotone trend rather than particular values.
  const oops::NonGaussianDensity f = build(stressBase());
  double prev = f.evolvingVariance(3.5);
  for (double d = 4.0; d <= 40.0; d += 0.5) {
    const double v = f.evolvingVariance(d);
    EXPECT(v >= prev * (1.0 - 1.0e-12));   // non-decreasing into the tail
    prev = v;
  }
}

CASE("assimilation/NonGaussianDensityStress/mode_window_bounds_the_weight") {
  // Without a mode neighbourhood the effective variance collapses towards zero
  // as the departure approaches the mode inside an offset bin, so the weight
  // 1/sigma_o^2 grows without bound. With the window the weight is bounded by
  // the fitted sigma at the mode.
  eckit::LocalConfiguration c = stressBase();
  c.set("mode window", 0.0);
  const oops::NonGaussianDensity nowin = build(c);
  c.set("mode window", 1.0);
  const oops::NonGaussianDensity win = build(c);

  const double sig2 = 1.0;   // sigma at mode squared
  double smallest = std::numeric_limits<double>::max();
  for (double d = 1.0e-6; d < 0.5; d *= 2.0)
    smallest = std::min(smallest, nowin.evolvingVariance(d));
  EXPECT(smallest < 0.01 * sig2);            // unbounded weight without the window

  for (double d = 1.0e-6; d < 0.5; d *= 2.0)
    EXPECT(oops::is_close_absolute(win.evolvingVariance(d), sig2, 1.0e-12));
}

class NonGaussianDensityStress : public oops::Test {
 public:
  using oops::Test::Test;
 private:
  std::string testid() const override {return "test::NonGaussianDensityStress";}
  void register_tests() const override {}
  void clear() const override {}
};

}  // namespace test

#endif  // TEST_ASSIMILATION_NONGAUSSIANDENSITYSTRESS_H_
