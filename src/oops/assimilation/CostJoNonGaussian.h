/*
 * CostJoNonGaussian<MODEL, OBS>  (abstract)
 * -----------------------------------------
 * A Jo cost term whose observation error is NON-Gaussian. Reuses ALL of
 * CostJo's observer / H(x) / TL-AD plumbing (by subclassing CostJo) and changes
 * only two things:
 *
 *  (1) Jo GRADIENT  = the true score g(d) = d/dd[-log f](d), not R^{-1} d.
 *      SHARED by every subclass; this is what pins the fixed point to the true
 *      posterior mode regardless of which Hessian a subclass uses. Implemented
 *      here by overriding CostJo's setGradientFG hook.
 *
 *  (2) Jo HESSIAN action = PURE VIRTUAL applyJoHessian(). This is the ONLY
 *      thing a concrete non-Gaussian Jo must supply. The optimizer reaches it
 *      through the inherited, UNCHANGED CostTermBase method multiplyCoInv()
 *      (which this base routes to applyJoHessian). No optimizer / no other JEDI
 *      class is modified or renamed; applyJoHessian is purely additive.
 *
 * Terminology: (2) is the "Jo Hessian", not a precision. In the Gaussian case
 * it equals R^{-1}; in general it is the surrogate curvature. Its inverse
 * (applyJoHessianInverse) plays R's role for preconditioning / dual space.
 *
 * Depends on the behavior-preserving CostJo extension hooks (patch
 * 0001-CostJo-add-extension-hooks.patch): setGradientFG virtual, multiply*
 * virtual, Rmat_/gradFG_ protected. Stock CostJo behavior is byte-identical.
 */
#ifndef OOPS_ASSIMILATION_COSTJONONGAUSSIAN_H_
#define OOPS_ASSIMILATION_COSTJONONGAUSSIAN_H_

#include <memory>
#include <vector>

#include "oops/assimilation/CostJo.h"
#include "oops/assimilation/NonGaussianDensity.h"
#include "oops/base/Departures.h"
#include "oops/util/missingValues.h"

namespace oops {

template <typename MODEL, typename OBS>
class CostJoNonGaussian : public CostJo<MODEL, OBS> {
  typedef Departures<OBS> Departures_;

 public:
  CostJoNonGaussian(const eckit::Configuration & conf, const eckit::mpi::Comm & comm,
                    const util::TimeWindow & tw,
                    const eckit::mpi::Comm & ctime = oops::mpi::myself())
    : CostJo<MODEL, OBS>(conf, comm, tw, ctime) {
    for (const auto & oc :
         eckit::LocalConfiguration(conf, "observers").getSubConfigurations()) {
      NonGaussianDensityParameters p;
      p.deserialize(eckit::LocalConfiguration(oc, "non gaussian cost"));
      densities_.emplace_back(p);
    }
  }
  virtual ~CostJoNonGaussian() = default;

  /// THE JO HESSIAN (pure virtual): dy_i <- A_i dy_i. Called by the inner loop
  /// via the inherited multiplyCoInv. Subclasses choose the curvature model.
  virtual void applyJoHessian(Departures_ & dy) const = 0;
  /// Its inverse: dy_i <- A_i^{-1} dy_i (R's role in preconditioning).
  virtual void applyJoHessianInverse(Departures_ & dy) const = 0;

 protected:
  /// (1) Jo gradient = true score g(d). ALSO caches the outer-loop departure
  ///     d_i so applyJoHessian can evaluate A at the re-linearization point.
  void setGradientFG(Departures_ & dep) override {
    depCache_.reset(new Departures_(dep));      // d_i, before overwrite
    for (size_t jj = 0; jj < dep.size(); ++jj) {
      auto & v = dep[jj];
      for (size_t k = 0; k < v.size(); ++k)
        if (v[k] != util::missingValue<double>())
          v[k] = densities_[jj].score(v[k]);    // dep <- g(d_i)
    }
  }

  /// The cached outer-loop departure d_i (nullptr before first computeCost).
  const Departures_ & departure() const { return *depCache_; }
  const std::vector<NonGaussianDensity> & densities() const { return densities_; }

  /// route the inherited (unchanged-name) covariance methods to the Jo Hessian
  std::unique_ptr<GeneralizedDepartures>
  multiplyCoInv(const GeneralizedDepartures & v1) const override {
    auto y1 = std::make_unique<Departures_>(dynamic_cast<const Departures_ &>(v1));
    applyJoHessian(*y1);
    return std::move(y1);
  }
  std::unique_ptr<GeneralizedDepartures>
  multiplyCovar(const GeneralizedDepartures & v1) const override {
    auto y1 = std::make_unique<Departures_>(dynamic_cast<const Departures_ &>(v1));
    applyJoHessianInverse(*y1);
    return std::move(y1);
  }

 private:
  std::vector<NonGaussianDensity> densities_;
  std::unique_ptr<Departures_> depCache_;   // d_i for the current outer loop
};

}  // namespace oops
#endif  // OOPS_ASSIMILATION_COSTJONONGAUSSIAN_H_
