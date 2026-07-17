/*
 * (C) Copyright 2025 Colorado State University
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

/*
 * Tier-1 unit test for the non-Gaussian Jo machinery, exact to machine
 * precision. Tests NonGaussianDensity -- the piece that carries ALL the science:
 * score g(d)=d/dd[-log f], score-slope g'(d), and the evolving-Gaussian variance
 * sigma_o^2(d)=(d-m)/g(d)  [Eq. 9 of Hu, Geer & van Leeuwen 2025, QJRMS e5050].
 * Reference values were produced by the stable_DOEE python oracle and verified
 * independently.
 *
 *   (branches)            score/scoreSlope/evolvingVariance match closed form at
 *                         points hitting left tail, interior, exact mode, right tail.
 *   (gaussian_reduction)  a density with exactly-quadratic log yields
 *                         evolvingVariance == sigma^2 at EVERY d (machine precision):
 *                         the non-Gaussian path is a strict generalization that
 *                         does not perturb the Gaussian case.
 *   (spd_positive)        evolvingVariance>0 everywhere -> Jo Hessian 1/sigma_o^2
 *                         is SPD, keeping the Krylov inner loop valid.
 *   (gradient_consistency) score at a bin centre equals -slope (the gradient of
 *                         the reconstructed -log f).
 */

#ifndef TEST_ASSIMILATION_NONGAUSSIANDENSITY_H_
#define TEST_ASSIMILATION_NONGAUSSIANDENSITY_H_

#include <cmath>
#include <string>
#include <vector>

#include "eckit/config/LocalConfiguration.h"
#include "eckit/testing/Test.h"

#include "oops/../test/TestEnvironment.h"
#include "oops/assimilation/NonGaussianDensity.h"
#include "oops/runs/Test.h"
#include "oops/util/Expect.h"
#include "oops/util/FloatCompare.h"

namespace test {

const double tol = 1.0e-12;

inline oops::NonGaussianDensity fixtureDensity() {
  eckit::LocalConfiguration c;
  c.set("mode", 0.0);
  c.set("grid spacing", 0.5);
  c.set("stable min", -2.0);
  c.set("stable max", 2.0);
  c.set("log slopes", std::vector<double>{1.0, 0.6, 0.3, -0.3, -0.6, -1.0, -1.3, -1.5});
  c.set("left log slope", 1.0);
  c.set("left curvature", -0.4);
  c.set("right log slope", -1.5);
  c.set("right curvature", -0.5);
  c.set("sigma at mode", 1.2);
  c.set("sigma floor", 0.3);
  oops::NonGaussianDensityParameters p;
  p.deserialize(c);
  return oops::NonGaussianDensity(p);
}

// log f exactly quadratic: slope_log(d) = -(bin centre)/s^2 -> score=d/s^2,
// evolvingVariance = s^2 exactly.
inline oops::NonGaussianDensity gaussianDensity(double s2) {
  const double dx = 0.25, lo = -6.0, hi = 6.0;
  std::vector<double> slopes;
  for (double d = lo; d < hi - 1.0e-9; d += dx) slopes.push_back(-(d + 0.5*dx) / s2);
  eckit::LocalConfiguration c;
  c.set("mode", 0.0);
  c.set("grid spacing", dx);
  c.set("stable min", lo);
  c.set("stable max", hi);
  c.set("log slopes", slopes);
  c.set("left log slope", -lo / s2);
  c.set("left curvature", -1.0 / s2);
  c.set("right log slope", -hi / s2);
  c.set("right curvature", -1.0 / s2);
  c.set("sigma at mode", std::sqrt(s2));
  c.set("sigma floor", 1.0e-6);
  oops::NonGaussianDensityParameters p;
  p.deserialize(c);
  return oops::NonGaussianDensity(p);
}

CASE("assimilation/NonGaussianDensity/branches") {
  const oops::NonGaussianDensity f = fixtureDensity();
  struct Ref { double d, score, slope, evar; };
  const std::vector<Ref> ref = {
    {-3.00, -1.400000000000e+00, 4.000000000000e-01, 2.142857142857e+00},
    {-2.00, -1.000000000000e+00, 4.000000000000e-01, 2.000000000000e+00},
    {-1.10, -6.000000000000e-01, 6.944444444444e-01, 1.833333333333e+00},
    {-0.50,  3.000000000000e-01, 6.944444444444e-01, 9.000000000000e-02},
    { 0.00,  6.000000000000e-01, 6.944444444444e-01, 1.440000000000e+00},
    { 0.40,  6.000000000000e-01, 6.944444444444e-01, 6.666666666667e-01},
    { 0.90,  1.000000000000e+00, 6.944444444444e-01, 9.000000000000e-01},
    { 2.00,  1.500000000000e+00, 5.000000000000e-01, 1.333333333333e+00},
    { 3.00,  2.000000000000e+00, 5.000000000000e-01, 1.500000000000e+00},
  };
  for (const auto & r : ref) {
    EXPECT(oops::is_close_absolute(f.score(r.d),            r.score, tol));
    EXPECT(oops::is_close_absolute(f.scoreSlope(r.d),       r.slope, tol));
    EXPECT(oops::is_close_absolute(f.evolvingVariance(r.d), r.evar,  tol));
  }
}

CASE("assimilation/NonGaussianDensity/gaussian_reduction") {
  for (double s2 : {0.5, 2.0, 9.0}) {
    const oops::NonGaussianDensity f = gaussianDensity(s2);
    for (double d : {-3.1, -1.3, -0.7, 0.6, 1.9, 3.3})
      EXPECT(oops::is_close_absolute(f.evolvingVariance(d), s2, 1.0e-10));
  }
}

CASE("assimilation/NonGaussianDensity/spd_positive") {
  const oops::NonGaussianDensity f = fixtureDensity();
  for (double d = -5.0; d <= 5.0; d += 0.1) EXPECT(f.evolvingVariance(d) > 0.0);
}

CASE("assimilation/NonGaussianDensity/gradient_consistency") {
  const oops::NonGaussianDensity f = fixtureDensity();
  const std::vector<double> slopes{1.0, 0.6, 0.3, -0.3, -0.6, -1.0, -1.3, -1.5};
  const double dx = 0.5, sMin = -2.0;
  for (size_t j = 0; j < slopes.size(); ++j) {
    const double dc = sMin + (static_cast<double>(j) + 0.5) * dx;
    EXPECT(oops::is_close_absolute(f.score(dc), -slopes[j], tol));
  }
}

class NonGaussianDensity : public oops::Test {
 public:
  using oops::Test::Test;
 private:
  std::string testid() const override {return "test::NonGaussianDensity";}
  void register_tests() const override {}
  void clear() const override {}
};

}  // namespace test

#endif  // TEST_ASSIMILATION_NONGAUSSIANDENSITY_H_
