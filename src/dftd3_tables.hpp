#pragma once

namespace gpu4dft::dftd3::detail {

inline constexpr double kAngstromToBohr = 1.0 / 0.529177210903;
inline constexpr int kMaxAtomicNumber = 103;
inline constexpr int kTotalRefs = 338;
inline constexpr int kC6UpperTriSize = 57291;

extern const double kRcovBohr[104];
extern const double kR4R2[104];
extern const double kCnRef[104][7];
extern const int kRefCount[104];
extern const int kRefOffset[104];
extern const double kC6UpperTri[57291];

}  // namespace gpu4dft::dftd3::detail
