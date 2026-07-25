/*
 * (C) Copyright 2020 UCAR.
 * (C) Crown Copyright 2024, the Met Office.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include <sstream>
#include <string>

#include "eckit/exception/Exceptions.h"

#include "oops/assimilation/MinimizerUtils.h"

#include "oops/util/formats.h"
#include "oops/util/Logger.h"

namespace oops {

void checkQuadraticCostFunction(const double costJb, const double costJoJc) {
  auto checkFinite = [](const std::string & costname, const double J) {
    if (std::isinf(J)) {
      throw eckit::BadValue(costname + " is infinite", Here());
    } else if (std::isnan(J)) {
      throw eckit::BadValue(costname + " is NaN", Here());
    }
  };
  checkFinite("Jb", costJb);
  checkFinite("JoJc", costJoJc);
  if (costJb < 0) {
    std::stringstream sserr;
    sserr << "Jb is negative (" << costJb << ")";
    throw eckit::BadValue(sserr.str(), Here());
  }
  if (costJoJc < 0) {
    // A Gaussian quadratic Jo is a sum of squares, so a negative value
    // signals corruption. A non-Gaussian (evolving-variance) Jo is not:
    // the secant Hessian makes the quadratic model's Jo minimum exactly
    // zero outside the mode window, but inside the window and at the
    // sigma floor the variance is deliberately not (d-m)/g(d), and small
    // negative excursions of the quadratic model are legitimate there.
    Log::warning() << "Quadratic cost function: JoJc is negative ("
                   << costJoJc << "); expected only for non-Gaussian Jo"
                   << std::endl;
  }
}

void printNormReduction(int iteration, const double & grad, const double & norm) {
  Log::info() << "  Residual norm (" << std::setw(2) << iteration << ") = "
              << util::full_precision(grad) << std::endl
              << "  Norm reduction (" << std::setw(2) << iteration << ") = "
              << util::full_precision(norm) << std::endl << std::endl;
}

void printQuadraticCostFunction(int iteration, const double & costJ,
                                const double & costJb, const double & costJoJc) {
  Log::info() << "  Quadratic cost function: J   (" << std::setw(2) << iteration << ") = "
              << util::full_precision(costJ)        << std::endl
              << "  Quadratic cost function: Jb  (" << std::setw(2) << iteration << ") = "
              << util::full_precision(costJb)       << std::endl
              << "  Quadratic cost function: JoJc(" << std::setw(2) << iteration << ") = "
              << util::full_precision(costJoJc)     << std::endl << std::endl;
}

// -----------------------------------------------------------------------------

}  // namespace oops
