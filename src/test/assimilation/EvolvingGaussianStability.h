/*
 * (C) Copyright 2025 Colorado State University
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

/*
 * Characterisation of the KNOWN convergence limitation of the evolving-Gaussian
 * method, and of the stability that holds regardless.
 *
 * The reference is explicit that the method is not guaranteed to reach the
 * posterior mode when it is applied only in the outer loop (Hu, Geer & van
 * Leeuwen 2025, section 5 and Appendix C). Appendix C gives the condition: with
 * y_o < y_b, if sigma_o DECREASES between them, successive iterates alternate
 * about the mode, and convergence additionally requires the double map
 * F = f o f to have no fixed point other than the analysis. Their Figure C3
 * shows a case where F has three fixed points and the sequence oscillates
 * indefinitely.
 *
 * This test documents that boundary rather than pretending it does not exist,
 * and pins down what IS guaranteed: the iterates stay bounded and finite, and
 * the effective variance stays positive, so a non-converging configuration
 * degrades into a bounded oscillation rather than diverging or producing
 * non-finite weights inside a minimisation.
 *
 * The density used is deliberately LIGHT-tailed (super-Gaussian): the log
 * slopes grow quickly with |d|, so sigma_o^2(d) = (d-m)/g(d) DECREASES with
 * |d|, which is exactly the regime Appendix C identifies. It is nevertheless a
 * perfectly valid unimodal density and passes construction validation.
 *
 * Note the contrast case: the SAME density converges for a different
 * background and background error. Non-convergence is a property of the
 * configuration, not of the density alone.
 */

#ifndef TEST_ASSIMILATION_EVOLVINGGAUSSIANSTABILITY_H_
#define TEST_ASSIMILATION_EVOLVINGGAUSSIANSTABILITY_H_

#include <algorithm>
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

/// Light-tailed unimodal density: sigma_o decreases with |d|.
inline oops::NonGaussianDensity lightTailedDensity() {
  eckit::LocalConfiguration c;
  c.set("mode", 0.0);
  c.set("grid spacing", 1.0);
  c.set("stable min", -6.0);
  c.set("stable max", 6.0);
  c.set("log slopes", std::vector<double>{
        9.0, 6.0, 3.5, 1.8, 0.8, 0.2, -0.2, -0.8, -1.8, -3.5, -6.0, -9.0});
  c.set("left log slope", 9.0);
  c.set("left curvature", -3.0);
  c.set("right log slope", -9.0);
  c.set("right curvature", -3.0);
  c.set("sigma at mode", 0.5);
  c.set("sigma floor", 1.0e-3);
  c.set("mode window", 0.0);
  oops::NonGaussianDensityParameters p;
  p.deserialize(c);
  return oops::NonGaussianDensity(p);
}

inline double step(const oops::NonGaussianDensity & f,
                   double x, double xb, double sb2, double yo) {
  const double s2 = f.evolvingVariance(yo - x);
  return (s2 / (s2 + sb2)) * xb + (sb2 / (s2 + sb2)) * yo;
}

CASE("assimilation/EvolvingGaussianStability/sigma_decreases_with_departure") {
  // The precondition for the Appendix C oscillation regime.
  const oops::NonGaussianDensity f = lightTailedDensity();
  double prev = f.evolvingVariance(0.5);
  for (double d = 1.5; d <= 5.5; d += 1.0) {
    const double v = f.evolvingVariance(d);
    EXPECT(v < prev);
    prev = v;
  }
}

CASE("assimilation/EvolvingGaussianStability/known_non_convergence") {
  // Documented limitation: the outer loop need not converge. Here it settles
  // into an exact period-two cycle, so successive iterates differ by more than
  // one state unit no matter how many outer loops are taken.
  const oops::NonGaussianDensity f = lightTailedDensity();
  const double xb = -4.0, sb2 = 4.0, yo = 0.0;
  double x = xb;
  for (int k = 0; k < 200; ++k) x = step(f, x, xb, sb2, yo);
  const double a = x;
  const double b = step(f, a, xb, sb2, yo);
  const double c = step(f, b, xb, sb2, yo);

  EXPECT(std::abs(b - a) > 1.0);                        // genuinely not converged
  EXPECT(oops::is_close_absolute(c, a, 1.0e-12));       // exact period-two cycle
  EXPECT(oops::is_close_absolute(a, -2.133333333333333, 1.0e-12));
  EXPECT(oops::is_close_absolute(b, -0.914285714285714, 1.0e-12));
}

CASE("assimilation/EvolvingGaussianStability/oscillation_stays_bounded") {
  // What IS guaranteed: no divergence, no non-finite values. A configuration
  // that fails to converge degrades into a bounded oscillation, which is the
  // property that matters for not destroying a cycling system.
  const oops::NonGaussianDensity f = lightTailedDensity();
  const double xb = -4.0, sb2 = 4.0, yo = 0.0;
  double x = xb, lo = xb, hi = xb;
  for (int k = 0; k < 5000; ++k) {
    x = step(f, x, xb, sb2, yo);
    EXPECT(std::isfinite(x));
    lo = std::min(lo, x);
    hi = std::max(hi, x);
  }
  // the iterates never leave the interval spanned by background and observation
  EXPECT(lo >= std::min(xb, yo) - 1.0e-12);
  EXPECT(hi <= std::max(xb, yo) + 1.0e-12);
}

CASE("assimilation/EvolvingGaussianStability/same_density_can_converge") {
  // Non-convergence is a property of the configuration, not of the density.
  const oops::NonGaussianDensity f = lightTailedDensity();
  const double xb = -1.0, sb2 = 0.5, yo = 0.0;
  double x = xb;
  for (int k = 0; k < 500; ++k) x = step(f, x, xb, sb2, yo);
  const double next = step(f, x, xb, sb2, yo);
  EXPECT(oops::is_close_absolute(next, x, 1.0e-12));    // a genuine fixed point
  EXPECT(oops::is_close_absolute(x, -0.9, 1.0e-12));
}

class EvolvingGaussianStability : public oops::Test {
 public:
  using oops::Test::Test;
 private:
  std::string testid() const override {return "test::EvolvingGaussianStability";}
  void register_tests() const override {}
  void clear() const override {}
};

}  // namespace test

#endif  // TEST_ASSIMILATION_EVOLVINGGAUSSIANSTABILITY_H_
