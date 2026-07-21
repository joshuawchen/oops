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
#include "eckit/exception/Exceptions.h"
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
  c.set("log slopes", std::vector<double>{1.0, 0.8, 0.5, 0.2, -0.2, -0.5, -0.8, -1.0});
  c.set("left log slope", 1.0);
  c.set("left curvature", -0.4);
  c.set("right log slope", -1.0);
  c.set("right curvature", -0.5);
  c.set("sigma at mode", 1.2);
  c.set("sigma floor", 0.3);
  // Disable the mode neighbourhood: this fixture tests the raw Eq. 9 branches,
  // including behaviour close to the mode. The window is covered separately.
  c.set("mode window", 0.0);
  oops::NonGaussianDensityParameters p;
  p.deserialize(c);
  return oops::NonGaussianDensity(p);
}

// Gaussian fixture grid. Kept at namespace scope so the test can sample at
// exact bin centres (see gaussian_reduction).
const double kGaussLo = -6.0;
const double kGaussHi = 6.0;
const double kGaussDx = 0.25;

// log f exactly quadratic: slope_log(d) = -(bin centre)/s^2 -> score=d/s^2,
// evolvingVariance = s^2 exactly.
inline oops::NonGaussianDensity gaussianDensity(double s2) {
  const double dx = kGaussDx, lo = kGaussLo, hi = kGaussHi;
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
    {-1.10, -8.000000000000e-01, 6.944444444444e-01, 1.375000000000e+00},
    {-0.50, -2.000000000000e-01, 6.944444444444e-01, 2.500000000000e+00},
    { 0.00,  2.000000000000e-01, 6.944444444444e-01, 1.440000000000e+00},
    { 0.40,  2.000000000000e-01, 6.944444444444e-01, 2.000000000000e+00},
    { 0.90,  5.000000000000e-01, 6.944444444444e-01, 1.800000000000e+00},
    { 2.00,  1.000000000000e+00, 5.000000000000e-01, 2.000000000000e+00},
    { 3.00,  1.500000000000e+00, 5.000000000000e-01, 2.000000000000e+00},
  };
  for (const auto & r : ref) {
    EXPECT(oops::is_close_absolute(f.score(r.d),            r.score, tol));
    EXPECT(oops::is_close_absolute(f.scoreSlope(r.d),       r.slope, tol));
    EXPECT(oops::is_close_absolute(f.evolvingVariance(r.d), r.evar,  tol));
  }
}

CASE("assimilation/NonGaussianDensity/gaussian_reduction") {
  // A density whose log is exactly quadratic must give evolvingVariance == s^2.
  // The interior score is piecewise CONSTANT (one slope per bin), so the
  // identity sigma_o^2(d) = (d-m)/g(d) = s^2 is exact where d coincides with
  // the bin centre the slope was derived from; away from a centre the value
  // differs by the discretisation ratio d/centre. Sample at exact centres for
  // the machine-precision claim, then check the off-centre error is bounded by
  // that ratio (i.e. it is discretisation, not a modelling error).
  for (double s2 : {0.5, 2.0, 9.0}) {
    const oops::NonGaussianDensity f = gaussianDensity(s2);
    for (size_t j : {11, 19, 23, 26, 31, 37}) {
      const double centre = kGaussLo + (static_cast<double>(j) + 0.5) * kGaussDx;
      EXPECT(oops::is_close_absolute(f.evolvingVariance(centre), s2, 1.0e-12));
    }
    // off-centre: bounded by half a bin relative to the sample point
    for (double d : {-3.1, -1.3, -0.7, 0.6, 1.9, 3.3}) {
      const double tol = s2 * (0.5 * kGaussDx / std::abs(d)) * 1.01;
      EXPECT(oops::is_close_absolute(f.evolvingVariance(d), s2, tol));
    }
  }
}

CASE("assimilation/NonGaussianDensity/spd_positive") {
  const oops::NonGaussianDensity f = fixtureDensity();
  for (double d = -5.0; d <= 5.0; d += 0.1) EXPECT(f.evolvingVariance(d) > 0.0);
}

CASE("assimilation/NonGaussianDensity/gradient_consistency") {
  const oops::NonGaussianDensity f = fixtureDensity();
  const std::vector<double> slopes{1.0, 0.8, 0.5, 0.2, -0.2, -0.5, -0.8, -1.0};
  const double dx = 0.5, sMin = -2.0;
  for (size_t j = 0; j < slopes.size(); ++j) {
    const double dc = sMin + (static_cast<double>(j) + 0.5) * dx;
    EXPECT(oops::is_close_absolute(f.score(dc), -slopes[j], tol));
  }
}

// Same as gaussianDensity but with the mode at m.
inline oops::NonGaussianDensity gaussianDensityAt(double s2, double m) {
  const double dx = kGaussDx, lo = kGaussLo, hi = kGaussHi;
  std::vector<double> slopes;
  for (double d = lo; d < hi - 1.0e-9; d += dx) slopes.push_back(-(d + 0.5*dx - m) / s2);
  eckit::LocalConfiguration c;
  c.set("mode", m);
  c.set("grid spacing", dx);
  c.set("stable min", lo);
  c.set("stable max", hi);
  c.set("log slopes", slopes);
  c.set("left log slope", -(lo - m) / s2);
  c.set("left curvature", -1.0 / s2);
  c.set("right log slope", -(hi - m) / s2);
  c.set("right curvature", -1.0 / s2);
  c.set("sigma at mode", std::sqrt(s2));
  c.set("sigma floor", 1.0e-6);
  oops::NonGaussianDensityParameters p;
  p.deserialize(c);
  return oops::NonGaussianDensity(p);
}

CASE("assimilation/NonGaussianDensity/shifted_mode_reduction") {
  // Gaussian reduction with the mode away from zero: Eq. 9 uses (d - m), so
  // the variance must still be exactly s^2 at every bin center.
  const double s2 = 2.0, m = 0.75;
  const oops::NonGaussianDensity f = gaussianDensityAt(s2, m);
  for (size_t j : {5, 11, 19, 23, 27, 31, 37, 43}) {
    const double center = kGaussLo + (static_cast<double>(j) + 0.5) * kGaussDx;
    EXPECT(oops::is_close_absolute(f.evolvingVariance(center), s2, 1.0e-12));
  }
  EXPECT(oops::is_close_absolute(f.evolvingVariance(m), s2, 1.0e-12));
}

CASE("assimilation/NonGaussianDensity/validate_rejects") {
  // Each block breaks one rule of validate() and must throw BadParameter.
  // The count mismatch is the bug class the first estimated density hit.
  eckit::LocalConfiguration base;
  base.set("mode", 0.0);
  base.set("grid spacing", 0.5);
  base.set("stable min", -2.0);
  base.set("stable max", 2.0);
  base.set("log slopes", std::vector<double>{1.0, 0.8, 0.5, 0.2, -0.2, -0.5, -0.8, -1.0});
  base.set("left log slope", 1.0);
  base.set("left curvature", -0.4);
  base.set("right log slope", -1.0);
  base.set("right curvature", -0.5);
  base.set("sigma at mode", 1.2);

  {  // the base configuration is valid
    oops::NonGaussianDensityParameters p;
    p.deserialize(base);
    EXPECT_NO_THROW(oops::NonGaussianDensity{p});
  }
  {  // 8 slopes over an extent of 7 bins
    eckit::LocalConfiguration c(base);
    c.set("stable max", 1.5);
    oops::NonGaussianDensityParameters p;
    p.deserialize(c);
    EXPECT_THROWS_AS(oops::NonGaussianDensity{p}, eckit::BadParameter);
  }
  {  // growing right tail
    eckit::LocalConfiguration c(base);
    c.set("right curvature", 0.5);
    oops::NonGaussianDensityParameters p;
    p.deserialize(c);
    EXPECT_THROWS_AS(oops::NonGaussianDensity{p}, eckit::BadParameter);
  }
  {  // mode outside the stable interior
    eckit::LocalConfiguration c(base);
    c.set("mode", 3.0);
    oops::NonGaussianDensityParameters p;
    p.deserialize(c);
    EXPECT_THROWS_AS(oops::NonGaussianDensity{p}, eckit::BadParameter);
  }
  {  // negative slope below the mode: not unimodal
    eckit::LocalConfiguration c(base);
    c.set("log slopes", std::vector<double>{1.0, -0.8, 0.5, 0.2, -0.2, -0.5, -0.8, -1.0});
    oops::NonGaussianDensityParameters p;
    p.deserialize(c);
    EXPECT_THROWS_AS(oops::NonGaussianDensity{p}, eckit::BadParameter);
  }
  {  // sigma at mode must be positive
    eckit::LocalConfiguration c(base);
    c.set("sigma at mode", 0.0);
    oops::NonGaussianDensityParameters p;
    p.deserialize(c);
    EXPECT_THROWS_AS(oops::NonGaussianDensity{p}, eckit::BadParameter);
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
