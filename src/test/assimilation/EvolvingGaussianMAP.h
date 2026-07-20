/*
 * (C) Copyright 2025 Colorado State University
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

/*
 * Convergence of the evolving-Gaussian method to an ANALYTICALLY KNOWN
 * posterior mode (MAP), driven by a non-trivial observation-error density read
 * from YAML through NonGaussianDensityParameters -- i.e. the same
 * configuration path a real experiment uses.
 *
 * Toy problem (scalar, H = identity):
 *     prior      x ~ N(x_b, sigma_b^2)
 *     obs        y_o with non-Gaussian error density f
 *     cost       J(x) = (x-x_b)^2/(2 sigma_b^2) - log f(d),   d = y_o - x
 *
 * Stationarity of J gives the MAP:
 *     (x - x_b)/sigma_b^2 = g(d),   g(d) = d/dd[-log f](d),   d = y_o - x
 *     =>  x* = x_b + sigma_b^2 g(d*)
 *
 * Because the density is piecewise-linear in log f, the score g is piecewise
 * CONSTANT on the interior (g = -log_slope[j]), so on the bin containing d*
 *     x* = x_b + sigma_b^2 * (-log_slope[j])
 * is EXACT ARITHMETIC -- no numerical root finding. In the tails log f is
 * quadratic, so g is linear in d and x* is the exact solution of a linear
 * equation. Both are used below, so the interior and both tail branches of the
 * density are covered.
 *
 * The evolving-Gaussian outer loop (paper Eqs. 8-9 and C1) is
 *     sigma_o^2(d) = (d - m)/g(d)
 *     x_{k+1} = [sigma_o^2/(sigma_o^2+sigma_b^2)] x_b
 *             + [sigma_b^2/(sigma_o^2+sigma_b^2)] y_o
 * Its fixed point satisfies x - x_b = sigma_b^2 d/sigma_o^2 = sigma_b^2 g(d),
 * which is exactly the MAP condition above. So this test asserts the method
 * CONVERGES TO THE TRUE POSTERIOR MODE, not merely that it converges.
 *
 * sigma_o^2(d) is obtained from oops::NonGaussianDensity built from the YAML
 * block, so the density parsing, the interior/tail branch selection and Eq. 9
 * are all exercised through the production code path.
 *
 * Reference: Hu, Geer & van Leeuwen (2025), QJRMS 151:e5050.
 */

#ifndef TEST_ASSIMILATION_EVOLVINGGAUSSIANMAP_H_
#define TEST_ASSIMILATION_EVOLVINGGAUSSIANMAP_H_

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

/// A deliberately NON-TRIVIAL density: skewed, 12 interior bins, quadratic-log
/// tails. Heavy left tail (gentle log-slopes -> large sigma_o -> outliers below
/// the mode are down-weighted) and a light right tail (steep slopes -> small
/// sigma_o). This is the shape a real DOEE estimate has, not a symmetric toy.
inline eckit::LocalConfiguration mapDensityConfig() {
  eckit::LocalConfiguration c;
  c.set("mode", 0.0);
  c.set("grid spacing", 1.0);
  c.set("stable min", -6.0);
  c.set("stable max", 6.0);
  c.set("log slopes", std::vector<double>{
        0.30, 0.35, 0.40, 0.50, 0.70, 1.00,      // d < 0, rising toward mode
       -1.60, -1.90, -2.10, -2.30, -2.40, -2.50});  // d > 0, steep decay
  c.set("left log slope", 0.30);
  c.set("left curvature", -0.05);
  c.set("right log slope", -2.50);
  c.set("right curvature", -0.05);
  c.set("sigma at mode", 1.10);
  c.set("sigma floor", 1.0e-3);
  return c;
}

inline oops::NonGaussianDensity mapDensity() {
  oops::NonGaussianDensityParameters p;
  p.deserialize(mapDensityConfig());
  return oops::NonGaussianDensity(p);
}

/// One evolving-Gaussian outer loop, using sigma_o^2 from the density class.
inline double outerLoop(const oops::NonGaussianDensity & f,
                        double x, double xb, double sb2, double yo) {
  const double s2 = f.evolvingVariance(yo - x);
  return (s2 / (s2 + sb2)) * xb + (sb2 / (s2 + sb2)) * yo;
}

/// Reference cases. MAP values are EXACT closed-form solutions of the
/// stationarity condition (see header comment), not numerically fitted.
///   xb, sigma_b^2, yo, exact MAP, branch of the density the solution lies in
struct MapCase { double xb, sb2, yo, map; const char * where; };
const std::vector<MapCase> mapCases = {
  {  12.0, 1.0, 0.0,  11.428571428571429, "left tail"  },
  {  -8.0, 0.5, 0.0,  -6.731707317073171, "right tail" },
  {  -5.0, 1.0, 0.0,  -2.900000000000000, "interior, d>0" },
  {  -2.0, 1.0, 0.0,  -0.400000000000000, "interior, d>0, near mode" },
  {   5.0, 4.0, 0.0,   3.400000000000000, "interior, d<0" },
  {   3.0, 0.5, 1.0,   2.650000000000000, "interior, d<0" },
};

CASE("assimilation/EvolvingGaussianMAP/converges_to_analytic_map") {
  const oops::NonGaussianDensity f = mapDensity();
  for (const MapCase & c : mapCases) {
    double x = c.xb;
    for (int k = 0; k < 200; ++k) x = outerLoop(f, x, c.xb, c.sb2, c.yo);
    EXPECT(oops::is_close_absolute(x, c.map, 1.0e-10));
  }
}

CASE("assimilation/EvolvingGaussianMAP/is_a_fixed_point") {
  // Starting AT the MAP must leave it unchanged: the analytic mode really is
  // the fixed point of the iteration, independent of the starting guess.
  const oops::NonGaussianDensity f = mapDensity();
  for (const MapCase & c : mapCases) {
    const double x = outerLoop(f, c.map, c.xb, c.sb2, c.yo);
    EXPECT(oops::is_close_absolute(x, c.map, 1.0e-12));
  }
}

CASE("assimilation/EvolvingGaussianMAP/beats_fixed_gaussian") {
  // A fixed Gaussian cannot reproduce these analyses: with constant sigma_o the
  // analysis is linear in x_b, whereas the true MAP is not. Use the density's
  // own sigma at the mode as the fairest fixed choice.
  const oops::NonGaussianDensity f = mapDensity();
  const double so2 = 1.10 * 1.10;
  double worst = 0.0;
  for (const MapCase & c : mapCases) {
    const double fixed = (so2/(so2 + c.sb2))*c.xb + (c.sb2/(so2 + c.sb2))*c.yo;
    worst = std::max(worst, std::abs(fixed - c.map));
  }
  EXPECT(worst > 0.5);   // the fixed Gaussian is far off for at least one case
}

CASE("assimilation/EvolvingGaussianMAP/stationarity_residual") {
  // Independent check that the converged point satisfies the ORIGINAL
  // stationarity condition x - x_b = sigma_b^2 g(d), using the score directly
  // rather than the variance -- so an error in Eq. 9 could not cancel out.
  const oops::NonGaussianDensity f = mapDensity();
  for (const MapCase & c : mapCases) {
    const double resid = (c.map - c.xb) - c.sb2 * f.score(c.yo - c.map);
    EXPECT(oops::is_close_absolute(resid, 0.0, 1.0e-12));
  }
}

class EvolvingGaussianMAP : public oops::Test {
 public:
  using oops::Test::Test;
 private:
  std::string testid() const override {return "test::EvolvingGaussianMAP";}
  void register_tests() const override {}
  void clear() const override {}
};

}  // namespace test

#endif  // TEST_ASSIMILATION_EVOLVINGGAUSSIANMAP_H_
