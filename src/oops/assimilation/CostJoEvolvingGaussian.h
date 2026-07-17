/*
 * CostJoEvolvingGaussian<MODEL, OBS>
 * ----------------------------------
 * Concrete CostJoNonGaussian: Hu, Geer & van Leeuwen (2025), QJRMS 151 e5050.
 * Jo Hessian = evolving-Gaussian SECANT curvature
 *      A_i = 1 / sigma_o^2(d_i),   sigma_o^2(d) = (d - m)/g(d)      [Eq. 9]
 * evaluated at the CACHED outer-loop departure d_i (not the multiplied vector).
 * SPD-by-construction for unimodal f. Static mode m from YAML. Gradient (score)
 * inherited unchanged -> fixed point is the true posterior mode.
 *
 * Registered via  jo type: "evolving gaussian".
 */
#ifndef OOPS_ASSIMILATION_COSTJOEVOLVINGGAUSSIAN_H_
#define OOPS_ASSIMILATION_COSTJOEVOLVINGGAUSSIAN_H_

#include "oops/assimilation/CostJoNonGaussian.h"
#include "oops/util/missingValues.h"

namespace oops {

template <typename MODEL, typename OBS>
class CostJoEvolvingGaussian : public CostJoNonGaussian<MODEL, OBS> {
  typedef Departures<OBS> Departures_;
 public:
  using CostJoNonGaussian<MODEL, OBS>::CostJoNonGaussian;

  void applyJoHessian(Departures_ & dy) const override        { apply(dy, false); }
  void applyJoHessianInverse(Departures_ & dy) const override { apply(dy, true);  }

 private:
  void apply(Departures_ & dy, bool inverse) const {
    const Departures_ & di = this->departure();      // cached d_i
    const auto & dens = this->densities();
    for (size_t jj = 0; jj < dy.size(); ++jj) {
      auto & v = dy[jj];
      const auto & d = di[jj];
      for (size_t k = 0; k < v.size(); ++k) {
        if (v[k] == util::missingValue<double>()) continue;
        const double var = dens[jj].evolvingVariance(d[k]);   // sigma_o^2(d_i)
        v[k] = inverse ? v[k] * var : v[k] / var;
      }
    }
  }
};

}  // namespace oops
#endif  // OOPS_ASSIMILATION_COSTJOEVOLVINGGAUSSIAN_H_
