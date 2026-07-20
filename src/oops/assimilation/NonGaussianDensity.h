/*
 * NonGaussianDensity: 1-D observation-error density, embedded in YAML, that
 * supplies everything a non-Gaussian Jo term needs at DA time:
 *   - the mode m (static),
 *   - the score g(d)      = d/dd[-log f](d)     (Jo GRADIENT, shared),
 *   - the score slope g'(d) = d^2/dd^2[-log f](d) (true-GN Jo HESSIAN option),
 *   - optionally the true cost -log f(d) (for honest Jo-table reporting).
 *
 * Representation = the piecewise-linear-log-density "cache" from stable_DOEE:
 * piecewise-linear log f on a stable interior (slopes only are needed for g),
 * quadratic-log tails with explicit edge curvatures. This is "Format A":
 * carrying log-density slopes lets us reconstruct BOTH g and g', so both the
 * evolving-Gaussian (secant) and true-GN (tangent) Hessians are expressible.
 *
 * NOTE: g and g' do NOT depend on the normalization; only the log-density
 * SLOPES are required. Intercepts / logZ are needed only if reportTrueCost.
 */
#ifndef OOPS_ASSIMILATION_NONGAUSSIANDENSITY_H_
#define OOPS_ASSIMILATION_NONGAUSSIANDENSITY_H_

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "eckit/config/Configuration.h"
#include "oops/util/parameters/Parameter.h"
#include "oops/util/parameters/Parameters.h"
#include "oops/util/parameters/RequiredParameter.h"

namespace oops {

/// YAML spec for one 1-D non-Gaussian observation-error density (per obs space
/// / channel). Embedded directly under the observer config -- no external file.
class NonGaussianDensityParameters : public Parameters {
  OOPS_CONCRETE_PARAMETERS(NonGaussianDensityParameters, Parameters)
 public:
  /// static mode m of f (Gaussian surrogate mean). Usually 0.
  RequiredParameter<double> mode{"mode", this};
  /// interior grid: even spacing dx and left edge stable_min; log-density
  /// slopes on each interior bin (piecewise-const score on interior).
  RequiredParameter<double> dx{"grid spacing", this};
  RequiredParameter<double> stableMin{"stable min", this};
  RequiredParameter<double> stableMax{"stable max", this};
  RequiredParameter<std::vector<double>> logSlopes{"log slopes", this};
  /// quadratic-log tails: score coeffs beyond the interior edges.
  RequiredParameter<double> leftLogSlope{"left log slope", this};
  RequiredParameter<double> leftCurvature{"left curvature", this};   // left_dd
  RequiredParameter<double> rightLogSlope{"right log slope", this};
  RequiredParameter<double> rightCurvature{"right curvature", this}; // right_dd
  /// sigma_o at the mode, resolving the Eq.(9) 0/0 there (from the paper's
  /// +-window quadratic fit at export time). Also serves as the interior
  /// curvature scale for the true-GN option.
  RequiredParameter<double> sigmaAtMode{"sigma at mode", this};
  /// Half-width of the neighbourhood of the mode over which sigmaAtMode is used
  /// instead of Eq. 9. Necessary because the interior score is piecewise
  /// CONSTANT: as d -> m inside a bin whose centre is offset from m,
  /// sigma_o^2 = (d-m)/g(d) -> 0 and the observation weight 1/sigma_o^2 grows
  /// without bound. The paper handles this by fitting a quadratic in a window
  /// about the mode and using a constant sigma there (their Appendix A).
  /// Default is one grid spacing; set to 0 to disable (point guard only).
  Parameter<double> modeWindow{"mode window", -1.0, this};
  /// SPD floor on any Jo-Hessian (guards true-GN indefiniteness & mode).
  Parameter<double> sigmaFloor{"sigma floor", 1.0e-3, this};
  /// Diagnostics: write the evolving sigma_o(d) to the ObsSpace each outer loop
  /// so it appears in the obs diagnostic files. Off by default (costs one extra
  /// ObsVector save per outer loop when enabled).
  Parameter<bool> saveSigma{"save evolving sigma", false, this};
  /// Group name prefix for the saved sigma; the outer-loop index is appended,
  /// e.g. EvolvingSigma0, EvolvingSigma1, ...
  Parameter<std::string> sigmaGroup{"evolving sigma group", "EvolvingSigma", this};
};

/// Evaluator. All O(1); index j = clamp((d - stableMin)/dx). Mirrors the
/// stable_DOEE python logpdf/dlogpdf so the ctest oracle matches to ~1e-6.
class NonGaussianDensity {
 public:
  explicit NonGaussianDensity(const NonGaussianDensityParameters & p)
    : m_(p.mode), dx_(p.dx), sMin_(p.stableMin), sMax_(p.stableMax),
      slopes_(p.logSlopes), lSlope_(p.leftLogSlope), lDD_(p.leftCurvature),
      rSlope_(p.rightLogSlope), rDD_(p.rightCurvature),
      sigMode_(p.sigmaAtMode), sigFloor_(p.sigmaFloor),
      saveSigma_(p.saveSigma), sigmaGroup_(p.sigmaGroup),
      modeWindow_(p.modeWindow >= 0.0 ? p.modeWindow.value() : p.dx.value()) {}

  double mode() const {return m_;}
  bool saveSigma() const {return saveSigma_;}
  const std::string & sigmaGroup() const {return sigmaGroup_;}

  /// score g(d) = -(d/dd) log f(d) = d/dd[-log f]. This is the Jo GRADIENT
  /// per obs (shared by every non-Gaussian implementation).
  double score(double d) const {
    if (d <= sMin_) return -(lSlope_ + lDD_ * (d - sMin_));
    if (d >= sMax_) return -(rSlope_ + rDD_ * (d - sMax_));
    return -slopes_[bin(d)];
  }

  /// score slope g'(d) = d^2/dd^2[-log f](d). Interior of a piecewise-linear
  /// log f is 0 (degenerate!) -> we return the mode-curvature scale there so
  /// the true-GN option is usable; tails carry -left_dd/-right_dd.
  double scoreSlope(double d) const {
    if (d <= sMin_) return -lDD_;
    if (d >= sMax_) return -rDD_;
    return 1.0 / (sigMode_ * sigMode_);   // interior curvature proxy
  }

  /// Evolving-Gaussian effective variance sigma_o^2(d) = (d - m)/g(d), Eq.(9),
  /// with the mode neighbourhood and SPD floor handled.
  double evolvingVariance(double d) const {
    const double num = d - m_;
    // Near the mode Eq. 9 is 0/0 and, for a piecewise-constant score, tends to
    // zero rather than to the curvature limit. Use the fitted sigma there.
    if (std::abs(num) <= modeWindow_) return sigMode_ * sigMode_;
    const double g = score(d);
    double var;
    if (std::abs(g) < 1.0e-12)
      var = sigMode_ * sigMode_;
    else
      var = num / g;                      // sign-safe for unimodal f
    return std::max(var, sigFloor_ * sigFloor_);
  }

 private:
  int bin(double d) const {
    int j = static_cast<int>((d - sMin_) / dx_);
    return std::max(0, std::min(j, static_cast<int>(slopes_.size()) - 1));
  }
  double m_, dx_, sMin_, sMax_;
  std::vector<double> slopes_;
  double lSlope_, lDD_, rSlope_, rDD_, sigMode_, sigFloor_;
  bool saveSigma_;
  std::string sigmaGroup_;
  double modeWindow_;
};

}  // namespace oops
#endif  // OOPS_ASSIMILATION_NONGAUSSIANDENSITY_H_
