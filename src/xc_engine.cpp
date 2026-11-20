#include "gpu4dft/xc.hpp"

#include <xc.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu4dft/generated/xc_kernels.hpp"

namespace gpu4dft {

namespace detail {
#if defined(GPU4DFT_HAVE_CUDA)
int xc_cuda_device_count();
void eval_xc_points_cuda(int func_id,
                         std::size_t n,
                         const double* rho,
                         const double* sigma,
                         const double* tau,
                         int deriv,
                         double rho_cutoff,
                         XcPointwiseResult& out);

void eval_ao_grid_cuda(const FlatBasisData& flat_basis,
                       const double* coords_xyz,
                       std::size_t npts,
                       int deriv,
                       double* out_ao);
#endif
}  // namespace detail

namespace detail {
struct LibxcState {
  struct Term {
    double coeff = 1.0;
    int func_id = 0;
    int family = 0;  // 0=LDA, 1=GGA, 2=MGGA
    xc_func_type func_unpol{};
    xc_func_type func_pol{};
    bool has_unpol = false;
    bool has_pol = false;
  };

  std::vector<Term> terms;
  int family = 0;
  bool is_hf = false;
  RshCoefficients rsh{};

  LibxcState() = default;
  ~LibxcState() { cleanup(); }

  LibxcState(const LibxcState&) = delete;
  LibxcState& operator=(const LibxcState&) = delete;

  void cleanup() noexcept {
    for (Term& t : terms) {
      if (t.has_unpol) {
        xc_func_end(&t.func_unpol);
        t.has_unpol = false;
      }
      if (t.has_pol) {
        xc_func_end(&t.func_pol);
        t.has_pol = false;
      }
    }
    terms.clear();
  }
};
}  // namespace detail

struct XcEngine::HostFxcState {
  bool ready = false;
  std::vector<double> v2rho2;
  std::vector<double> g0x;
  std::vector<double> g0y;
  std::vector<double> g0z;
  std::vector<double> vsigma;
  std::vector<double> v2rhosigma;
  std::vector<double> v2sigma2;
  std::vector<double> v2rhotau;
  std::vector<double> v2sigmatau;
  std::vector<double> v2tau2;
  bool ao_cached = false;
  std::vector<std::size_t> batch_phi_offset;
  std::vector<double> phi_cache;
};

#if defined(GPU4DFT_HAVE_CUDA)
struct XcEngine::CudaState {
  void* impl = nullptr;
};

namespace detail {
void* xc_cuda_engine_create(const FlatBasisData& flat_basis,
                            const MolecularGrid& grid,
                            int func_id,
                            int family,
                            bool is_hf,
                            const void* libxc_state,
                            const XcOptions& opts);
void xc_cuda_engine_destroy(void* impl) noexcept;
void xc_cuda_set_grid_response(void* impl, bool enabled) noexcept;
double xc_cuda_eval_vxc_device(void* impl,
                               const double* d_dm,
                               double* d_vxc,
                               double* nelec_out,
                               void* cuda_stream);
double xc_cuda_eval_vxc_host(void* impl,
                             const double* h_dm,
                             double* h_vxc,
                             double* nelec_out);
double xc_cuda_eval_vxc_uks_device(void* impl,
                                   const double* d_dm_alpha,
                                   const double* d_dm_beta,
                                   double* d_vxc_alpha,
                                   double* d_vxc_beta,
                                   double* nelec_alpha_out,
                                   double* nelec_beta_out,
                                   void* cuda_stream);
double xc_cuda_eval_vxc_uks_host(void* impl,
                                 const double* h_dm_alpha,
                                 const double* h_dm_beta,
                                 double* h_vxc_alpha,
                                 double* h_vxc_beta,
                                 double* nelec_alpha_out,
                                 double* nelec_beta_out);
void xc_cuda_set_reference_density_device(void* impl,
                                          const double* d_dm,
                                          void* cuda_stream);
void xc_cuda_set_reference_density_host(void* impl,
                                        const double* h_dm);
bool xc_cuda_has_reference_density(const void* impl) noexcept;
void xc_cuda_eval_fxc_device(void* impl,
                             const double* d_delta_dm,
                             double* d_fxc_out,
                             void* cuda_stream);
void xc_cuda_eval_fxc_host(void* impl,
                           const double* h_delta_dm,
                           double* h_fxc_out);
std::vector<double> xc_cuda_nuclear_grad(void* impl,
                                         const MolecularGrid& grid,
                                         std::span<const XcAtom> atoms,
                                         const int* ao_to_atom,
                                         const double* h_dm);
std::vector<double> xc_cuda_nuclear_grad_uks(void* impl,
                                             const MolecularGrid& grid,
                                             std::span<const XcAtom> atoms,
                                             const int* ao_to_atom,
                                             const double* h_dm_alpha,
                                             const double* h_dm_beta);
}  // namespace detail
#else
struct XcEngine::CudaState {};
#endif

namespace {

std::string trim_copy(std::string_view s) {
  std::size_t beg = 0;
  std::size_t end = s.size();
  while (beg < end && std::isspace(static_cast<unsigned char>(s[beg])) != 0) ++beg;
  while (end > beg && std::isspace(static_cast<unsigned char>(s[end - 1])) != 0) --end;
  return std::string(s.substr(beg, end - beg));
}

std::string normalize_alias_key(std::string_view s) {
  std::string t = trim_copy(s);
  std::string out;
  out.reserve(t.size());
  for (char ch : t) {
    if (std::isspace(static_cast<unsigned char>(ch)) != 0) continue;
    char lc = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (lc == '-') lc = '_';
    out.push_back(lc);
  }
  return out;
}

int lookup_libxc_id_by_name(const std::string& name) {
  int fid = xc_functional_get_number(name.c_str());
  if (fid > 0) return fid;
  std::string underscored = name;
  for (char& ch : underscored) {
    if (ch == '-') ch = '_';
  }
  if (underscored != name) {
    fid = xc_functional_get_number(underscored.c_str());
    if (fid > 0) return fid;
  }
  return -1;
}

int resolve_single_token_id(const std::string& raw_tok, bool& is_hf_token) {
  is_hf_token = false;
  const std::string k = normalize_alias_key(raw_tok);
  if (k == "hf" || k == "exx" || k == "rhf" || k == "uhf") {
    is_hf_token = true;
    return 0;
  }
  // Exact single-component aliases from codegen/xc_codegen/functionals.py
  if (k == "lda_x" || k == "slater" || k == "lda") return 1;
  if (k == "lda_c_vwn5" || k == "lda_c_vwn" || k == "vwn5" || k == "vwn") return 7;
  if (k == "lda_c_vwn_rpa" || k == "vwn_rpa") return 8;
  if (k == "lda_c_pz" || k == "pz81" || k == "pz") return 9;
  if (k == "gga_x_pbe" || k == "pbe_x" || k == "pbe") return 101;
  if (k == "gga_c_pbe" || k == "pbe_c") return 130;
  if (k == "gga_x_b88" || k == "b88") return 106;
  if (k == "gga_c_lyp" || k == "lyp") return 131;
  if (k == "mgga_x_tpss" || k == "tpss_x" || k == "tpss") return 202;
  if (k == "mgga_c_tpss" || k == "tpss_c") return 231;

  // Integer ID?
  if (!k.empty() && std::all_of(k.begin(), k.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) {
    const int fid = std::stoi(k);
    char* nm = xc_functional_get_name(fid);
    if (nm != nullptr) {
      std::free(nm);
      return fid;
    }
    throw std::invalid_argument("Unknown Libxc functional ID: " + k);
  }

  const int fid = lookup_libxc_id_by_name(k);
  if (fid > 0) return fid;
  throw std::invalid_argument("Unknown XC functional token: '" + trim_copy(raw_tok) + "'");
}

struct ParsedTermSpec {
  double coeff = 1.0;
  int func_id = 0;
  bool is_hf = false;
};

void parse_sum_terms(std::string_view expr, std::vector<ParsedTermSpec>& out, bool is_corr_side) {
  std::size_t pos = 0;
  while (pos < expr.size()) {
    const std::size_t plus = expr.find('+', pos);
    const std::size_t len = (plus == std::string_view::npos) ? (expr.size() - pos) : (plus - pos);
    const std::string part = trim_copy(expr.substr(pos, len));
    pos = (plus == std::string_view::npos) ? expr.size() : (plus + 1);
    if (part.empty()) continue;

    double coeff = 1.0;
    std::string tok = part;
    const std::size_t star = part.find('*');
    if (star != std::string::npos) {
      const std::string c_str = trim_copy(part.substr(0, star));
      tok = trim_copy(part.substr(star + 1));
      try {
        coeff = std::stod(c_str);
      } catch (...) {
        throw std::invalid_argument("Invalid XC term coefficient: '" + c_str + "'");
      }
    }

    const std::string k = normalize_alias_key(tok);
    bool is_hf = false;
    int fid = 0;
    if (is_corr_side && k == "pbe") {
      fid = 130;  // GGA_C_PBE
    } else if (is_corr_side && k == "tpss") {
      fid = 231;  // MGGA_C_TPSS
    } else if (is_corr_side && k == "lda") {
      fid = 7;    // LDA_C_VWN
    } else {
      fid = resolve_single_token_id(tok, is_hf);
    }
    out.push_back(ParsedTermSpec{coeff, fid, is_hf});
  }
}

std::unique_ptr<detail::LibxcState> build_libxc_state(std::string_view xc_name, double rho_cutoff) {
  auto st = std::make_unique<detail::LibxcState>();
  const std::string key = normalize_alias_key(xc_name);

  if (key.empty() || key == "hf" || key == "rhf" || key == "uhf") {
    st->is_hf = true;
    st->family = 0;
    st->rsh = RshCoefficients{0.0, 1.0, 0.0};
    return st;
  }

  std::vector<ParsedTermSpec> specs;
  double explicit_hf_coeff = 0.0;

  auto resolve_named_or_libxc = [&](const std::string& libxc_sym, int fallback_id) -> int {
    const int fid = xc_functional_get_number(libxc_sym.c_str());
    return (fid > 0) ? fid : fallback_id;
  };

  if (key == "lda" || key == "svwn" || key == "svwn5" || key == "lda,vwn" || key == "lda,vwn5") {
    specs = {{1.0, 1, false}, {1.0, 7, false}};
  } else if (key == "svwn_rpa" || key == "lda,vwn_rpa") {
    specs = {{1.0, 1, false}, {1.0, 8, false}};
  } else if (key == "spz81" || key == "lda,pz" || key == "lda,pz81") {
    specs = {{1.0, 1, false}, {1.0, 9, false}};
  } else if (key == "pbe" || key == "pbe,pbe") {
    specs = {{1.0, 101, false}, {1.0, 130, false}};
  } else if (key == "blyp" || key == "b88,lyp") {
    specs = {{1.0, 106, false}, {1.0, 131, false}};
  } else if (key == "b3lyp" || key == "b3lypg") {
    specs = {{1.0, resolve_named_or_libxc("hyb_gga_xc_b3lyp", 402), false}};
  } else if (key == "b3lyp5") {
    specs = {{1.0, resolve_named_or_libxc("hyb_gga_xc_b3lyp5", 475), false}};
  } else if (key == "pbe0" || key == "pbeh") {
    specs = {{1.0, resolve_named_or_libxc("hyb_gga_xc_pbeh", 406), false}};
  } else if (key == "tpss" || key == "tpss,tpss") {
    specs = {{1.0, 202, false}, {1.0, 231, false}};
  } else if (key == "cam_b3lyp" || key == "camb3lyp") {
    specs = {{1.0, resolve_named_or_libxc("hyb_gga_xc_cam_b3lyp", 433), false}};
  } else if (key == "wb97x") {
    specs = {{1.0, resolve_named_or_libxc("hyb_gga_xc_wb97x", 464), false}};
  } else if (key == "wb97x_d" || key == "wb97xd") {
    specs = {{1.0, resolve_named_or_libxc("hyb_gga_xc_wb97x_d", 471), false}};
  } else {
    const std::string trimmed = trim_copy(xc_name);
    const std::size_t comma = trimmed.find(',');
    if (comma != std::string::npos) {
      if (trimmed.find(',', comma + 1) != std::string::npos) {
        throw std::invalid_argument("Unsupported XC functional syntax (multiple commas): '" +
                                    trimmed + "'");
      }
      parse_sum_terms(std::string_view(trimmed).substr(0, comma), specs, false);
      parse_sum_terms(std::string_view(trimmed).substr(comma + 1), specs, true);
    } else if (trimmed.find('+') != std::string::npos || trimmed.find('*') != std::string::npos) {
      parse_sum_terms(trimmed, specs, false);
    } else {
      bool is_hf_tok = false;
      const int fid = resolve_single_token_id(trimmed, is_hf_tok);
      specs.push_back(ParsedTermSpec{1.0, fid, is_hf_tok});
    }
  }

  for (const ParsedTermSpec& sp : specs) {
    if (sp.is_hf) {
      explicit_hf_coeff += sp.coeff;
      continue;
    }
    detail::LibxcState::Term term;
    term.coeff = sp.coeff;
    term.func_id = sp.func_id;
    if (xc_func_init(&term.func_unpol, sp.func_id, XC_UNPOLARIZED) != 0) {
      throw std::runtime_error("xc_func_init failed (XC_UNPOLARIZED) for functional ID " +
                               std::to_string(sp.func_id));
    }
    term.has_unpol = true;
    if (xc_func_init(&term.func_pol, sp.func_id, XC_POLARIZED) != 0) {
      xc_func_end(&term.func_unpol);
      term.has_unpol = false;
      throw std::runtime_error("xc_func_init failed (XC_POLARIZED) for functional ID " +
                               std::to_string(sp.func_id));
    }
    term.has_pol = true;

    xc_func_set_dens_threshold(&term.func_unpol, rho_cutoff);
    xc_func_set_dens_threshold(&term.func_pol, rho_cutoff);

    const int flags = term.func_unpol.info->flags;
    if ((flags & XC_FLAGS_NEEDS_LAPLACIAN) != 0) {
      st->terms.push_back(std::move(term));
      throw NotImplemented("Laplacian-dependent meta-GGA functionals are not supported yet");
    }
#ifdef XC_FLAGS_HYB_CAMY
    if ((flags & XC_FLAGS_HYB_CAMY) != 0) {
      st->terms.push_back(std::move(term));
      throw NotImplemented("Yukawa range-separated hybrids (CAMY) are not supported yet");
    }
#endif
#ifdef XC_FLAGS_HYB_LCY
    if ((flags & XC_FLAGS_HYB_LCY) != 0) {
      st->terms.push_back(std::move(term));
      throw NotImplemented("Yukawa range-separated hybrids (LCY) are not supported yet");
    }
#endif

    const int fam_raw = term.func_unpol.info->family;
    bool is_hybrid = false;
    switch (fam_raw) {
      case XC_FAMILY_LDA:
        term.family = 0;
        break;
#ifdef XC_FAMILY_HYB_LDA
      case XC_FAMILY_HYB_LDA:
        term.family = 0;
        is_hybrid = true;
        break;
#endif
      case XC_FAMILY_GGA:
        term.family = 1;
        break;
      case XC_FAMILY_HYB_GGA:
        term.family = 1;
        is_hybrid = true;
        break;
      case XC_FAMILY_MGGA:
        term.family = 2;
        break;
      case XC_FAMILY_HYB_MGGA:
        term.family = 2;
        is_hybrid = true;
        break;
      default:
        st->terms.push_back(std::move(term));
        throw NotImplemented("Unsupported Libxc functional family " + std::to_string(fam_raw));
    }

    st->family = std::max(st->family, term.family);

    if (is_hybrid) {
      if ((flags & XC_FLAGS_HYB_CAM) != 0) {
        double om = 0.0;
        double al = 0.0;
        double be = 0.0;
        xc_hyb_cam_coef(&term.func_unpol, &om, &al, &be);
        if (st->rsh.omega > 0.0 && std::abs(st->rsh.omega - om) > 1e-12) {
          st->terms.push_back(std::move(term));
          throw std::invalid_argument("Multiple RSH components with different omega are unsupported");
        }
        st->rsh.omega = om;
        st->rsh.alpha += sp.coeff * al;
        st->rsh.beta += sp.coeff * be;
      } else {
        const double exx = xc_hyb_exx_coef(&term.func_unpol);
        st->rsh.alpha += sp.coeff * exx;
      }
    }

    st->terms.push_back(std::move(term));
  }

  st->rsh.alpha += explicit_hf_coeff;
  if (st->terms.empty() && explicit_hf_coeff != 0.0) {
    st->is_hf = true;
  }
  return st;
}

void eval_libxc_rks_chunk(const detail::LibxcState& st,
                          std::size_t n,
                          const double* weights,
                          const double* rho,
                          const double* gx,
                          const double* gy,
                          const double* gz,
                          const double* tau,
                          bool for_grad,
                          double* wv0,
                          double* wvx,
                          double* wvy,
                          double* wvz,
                          double* wvt,
                          double* e_xc_point_out,
                          double& exc_acc,
                          double& nelec_acc,
                          std::vector<double>& rho_buf,
                          std::vector<double>& sigma_buf,
                          std::vector<double>& tau_buf,
                          std::vector<double>& zk_acc,
                          std::vector<double>& vrho_acc,
                          std::vector<double>& vsigma_acc,
                          std::vector<double>& vtau_acc,
                          std::vector<double>& zk_tmp,
                          std::vector<double>& vrho_tmp,
                          std::vector<double>& vsigma_tmp,
                          std::vector<double>& vtau_tmp) {
  if (n == 0) return;

  if (st.is_hf) {
    for (std::size_t g = 0; g < n; ++g) {
      if (rho[g] > 0.0) nelec_acc += weights[g] * rho[g];
      wv0[g] = 0.0;
      if (wvx) wvx[g] = 0.0;
      if (wvy) wvy[g] = 0.0;
      if (wvz) wvz[g] = 0.0;
      if (wvt) wvt[g] = 0.0;
      if (e_xc_point_out) e_xc_point_out[g] = 0.0;
    }
    return;
  }

  rho_buf.resize(n);
  for (std::size_t g = 0; g < n; ++g) {
    rho_buf[g] = (rho[g] > 0.0) ? rho[g] : 0.0;
  }

  if (st.family >= 1) {
    sigma_buf.resize(n);
    for (std::size_t g = 0; g < n; ++g) {
      if (rho[g] > 0.0) {
        sigma_buf[g] = gx[g] * gx[g] + gy[g] * gy[g] + gz[g] * gz[g];
      } else {
        sigma_buf[g] = 0.0;
      }
    }
  }

  if (st.family == 2) {
    tau_buf.resize(n);
    for (std::size_t g = 0; g < n; ++g) {
      const double r = rho_buf[g];
      if (r > 0.0) {
        const double denom = 8.0 * r;
        const double tau_w = (denom > 0.0) ? (sigma_buf[g] / denom) : 0.0;
        tau_buf[g] = std::isfinite(tau_w) ? std::max(tau[g], tau_w) : std::max(tau[g], 0.0);
      } else {
        tau_buf[g] = 0.0;
      }
    }
  }

  zk_acc.assign(n, 0.0);
  vrho_acc.assign(n, 0.0);
  if (st.family >= 1) vsigma_acc.assign(n, 0.0);
  if (st.family == 2) vtau_acc.assign(n, 0.0);

  zk_tmp.resize(n);
  vrho_tmp.resize(n);
  if (st.family >= 1) vsigma_tmp.resize(n);
  if (st.family == 2) vtau_tmp.resize(n);

  const int i_n = static_cast<int>(n);
  for (const auto& t : st.terms) {
    if (t.family == 0) {
      xc_lda_exc_vxc(&t.func_unpol, i_n, rho_buf.data(), zk_tmp.data(), vrho_tmp.data());
    } else if (t.family == 1) {
      xc_gga_exc_vxc(&t.func_unpol, i_n, rho_buf.data(), sigma_buf.data(), zk_tmp.data(),
                     vrho_tmp.data(), vsigma_tmp.data());
    } else {
      xc_mgga_exc_vxc(&t.func_unpol, i_n, rho_buf.data(), sigma_buf.data(), nullptr, tau_buf.data(),
                      zk_tmp.data(), vrho_tmp.data(), vsigma_tmp.data(), nullptr, vtau_tmp.data());
    }

    const double c = t.coeff;
    for (std::size_t g = 0; g < n; ++g) {
      zk_acc[g] += c * zk_tmp[g];
      vrho_acc[g] += c * vrho_tmp[g];
    }
    if (t.family >= 1) {
      for (std::size_t g = 0; g < n; ++g) {
        vsigma_acc[g] += c * vsigma_tmp[g];
      }
    }
    if (t.family == 2) {
      for (std::size_t g = 0; g < n; ++g) {
        vtau_acc[g] += c * vtau_tmp[g];
      }
    }
  }

  const double rho_scale = for_grad ? 1.0 : 0.5;
  const double tau_scale = for_grad ? 1.0 : 0.25;

  for (std::size_t g = 0; g < n; ++g) {
    const double wg = weights[g];
    const double r = rho_buf[g];
    if (r > 0.0) {
      nelec_acc += wg * r;
    }
    const double e_pt = r * zk_acc[g];
    exc_acc += wg * e_pt;
    if (e_xc_point_out) {
      e_xc_point_out[g] = e_pt;
    }

    wv0[g] = rho_scale * wg * vrho_acc[g];
    if (st.family >= 1) {
      const double two_w_vs = 2.0 * wg * vsigma_acc[g];
      const double gx_s = (r > 0.0) ? gx[g] : 0.0;
      const double gy_s = (r > 0.0) ? gy[g] : 0.0;
      const double gz_s = (r > 0.0) ? gz[g] : 0.0;
      wvx[g] = two_w_vs * gx_s;
      wvy[g] = two_w_vs * gy_s;
      wvz[g] = two_w_vs * gz_s;
    }
    if (st.family == 2) {
      wvt[g] = tau_scale * wg * vtau_acc[g];
    }
  }
}

void eval_libxc_uks_chunk(const detail::LibxcState& st,
                          std::size_t n,
                          const double* weights,
                          const double* rho_a,
                          const double* rho_b,
                          const double* gxa,
                          const double* gya,
                          const double* gza,
                          const double* gxb,
                          const double* gyb,
                          const double* gzb,
                          const double* tau_a,
                          const double* tau_b,
                          bool for_grad,
                          double* wv0_a,
                          double* wv0_b,
                          double* wvx_a,
                          double* wvy_a,
                          double* wvz_a,
                          double* wvx_b,
                          double* wvy_b,
                          double* wvz_b,
                          double* wvt_a,
                          double* wvt_b,
                          double* e_xc_point_out,
                          double& exc_acc,
                          double& nelec_a_acc,
                          double& nelec_b_acc,
                          std::vector<double>& rho2_buf,
                          std::vector<double>& sigma3_buf,
                          std::vector<double>& tau2_buf,
                          std::vector<double>& zk_acc,
                          std::vector<double>& vrho2_acc,
                          std::vector<double>& vsigma3_acc,
                          std::vector<double>& vtau2_acc,
                          std::vector<double>& zk_tmp,
                          std::vector<double>& vrho2_tmp,
                          std::vector<double>& vsigma3_tmp,
                          std::vector<double>& vtau2_tmp) {
  if (n == 0) return;

  if (st.is_hf) {
    for (std::size_t g = 0; g < n; ++g) {
      if (rho_a[g] > 0.0) nelec_a_acc += weights[g] * rho_a[g];
      if (rho_b[g] > 0.0) nelec_b_acc += weights[g] * rho_b[g];
      wv0_a[g] = 0.0;
      wv0_b[g] = 0.0;
      if (wvx_a) {
        wvx_a[g] = 0.0;
        wvy_a[g] = 0.0;
        wvz_a[g] = 0.0;
        wvx_b[g] = 0.0;
        wvy_b[g] = 0.0;
        wvz_b[g] = 0.0;
      }
      if (wvt_a) {
        wvt_a[g] = 0.0;
        wvt_b[g] = 0.0;
      }
      if (e_xc_point_out) e_xc_point_out[g] = 0.0;
    }
    return;
  }

  rho2_buf.resize(2 * n);
  for (std::size_t g = 0; g < n; ++g) {
    rho2_buf[2 * g + 0] = (rho_a[g] > 0.0) ? rho_a[g] : 0.0;
    rho2_buf[2 * g + 1] = (rho_b[g] > 0.0) ? rho_b[g] : 0.0;
  }

  if (st.family >= 1) {
    sigma3_buf.resize(3 * n);
    for (std::size_t g = 0; g < n; ++g) {
      const bool pos_a = (rho_a[g] > 0.0);
      const bool pos_b = (rho_b[g] > 0.0);
      const double gxa_s = pos_a ? gxa[g] : 0.0;
      const double gya_s = pos_a ? gya[g] : 0.0;
      const double gza_s = pos_a ? gza[g] : 0.0;
      const double gxb_s = pos_b ? gxb[g] : 0.0;
      const double gyb_s = pos_b ? gyb[g] : 0.0;
      const double gzb_s = pos_b ? gzb[g] : 0.0;
      sigma3_buf[3 * g + 0] = gxa_s * gxa_s + gya_s * gya_s + gza_s * gza_s;
      // NEVER clamp sigma_ab < 0: grad_rho_a . grad_rho_b can be negative!
      sigma3_buf[3 * g + 1] = gxa_s * gxb_s + gya_s * gyb_s + gza_s * gzb_s;
      sigma3_buf[3 * g + 2] = gxb_s * gxb_s + gyb_s * gyb_s + gzb_s * gzb_s;
    }
  }

  if (st.family == 2) {
    tau2_buf.resize(2 * n);
    for (std::size_t g = 0; g < n; ++g) {
      const double ra = rho2_buf[2 * g + 0];
      const double rb = rho2_buf[2 * g + 1];
      if (ra > 0.0) {
        const double denom_a = 8.0 * ra;
        const double tau_wa = (denom_a > 0.0) ? (sigma3_buf[3 * g + 0] / denom_a) : 0.0;
        tau2_buf[2 * g + 0] =
            std::isfinite(tau_wa) ? std::max(tau_a[g], tau_wa) : std::max(tau_a[g], 0.0);
      } else {
        tau2_buf[2 * g + 0] = 0.0;
      }
      if (rb > 0.0) {
        const double denom_b = 8.0 * rb;
        const double tau_wb = (denom_b > 0.0) ? (sigma3_buf[3 * g + 2] / denom_b) : 0.0;
        tau2_buf[2 * g + 1] =
            std::isfinite(tau_wb) ? std::max(tau_b[g], tau_wb) : std::max(tau_b[g], 0.0);
      } else {
        tau2_buf[2 * g + 1] = 0.0;
      }
    }
  }

  zk_acc.assign(n, 0.0);
  vrho2_acc.assign(2 * n, 0.0);
  if (st.family >= 1) vsigma3_acc.assign(3 * n, 0.0);
  if (st.family == 2) vtau2_acc.assign(2 * n, 0.0);

  zk_tmp.resize(n);
  vrho2_tmp.resize(2 * n);
  if (st.family >= 1) vsigma3_tmp.resize(3 * n);
  if (st.family == 2) vtau2_tmp.resize(2 * n);

  const int i_n = static_cast<int>(n);
  for (const auto& t : st.terms) {
    if (t.family == 0) {
      xc_lda_exc_vxc(&t.func_pol, i_n, rho2_buf.data(), zk_tmp.data(), vrho2_tmp.data());
    } else if (t.family == 1) {
      xc_gga_exc_vxc(&t.func_pol, i_n, rho2_buf.data(), sigma3_buf.data(), zk_tmp.data(),
                     vrho2_tmp.data(), vsigma3_tmp.data());
    } else {
      xc_mgga_exc_vxc(&t.func_pol, i_n, rho2_buf.data(), sigma3_buf.data(), nullptr,
                      tau2_buf.data(), zk_tmp.data(), vrho2_tmp.data(), vsigma3_tmp.data(),
                      nullptr, vtau2_tmp.data());
    }

    const double c = t.coeff;
    for (std::size_t g = 0; g < n; ++g) {
      zk_acc[g] += c * zk_tmp[g];
    }
    for (std::size_t k = 0; k < 2 * n; ++k) {
      vrho2_acc[k] += c * vrho2_tmp[k];
    }
    if (t.family >= 1) {
      for (std::size_t k = 0; k < 3 * n; ++k) {
        vsigma3_acc[k] += c * vsigma3_tmp[k];
      }
    }
    if (t.family == 2) {
      for (std::size_t k = 0; k < 2 * n; ++k) {
        vtau2_acc[k] += c * vtau2_tmp[k];
      }
    }
  }

  const double rho_scale = for_grad ? 1.0 : 0.5;
  const double tau_scale = for_grad ? 1.0 : 0.25;

  for (std::size_t g = 0; g < n; ++g) {
    const double wg = weights[g];
    const double ra = (rho_a[g] > 0.0) ? rho_a[g] : 0.0;
    const double rb = (rho_b[g] > 0.0) ? rho_b[g] : 0.0;
    if (ra > 0.0) nelec_a_acc += wg * ra;
    if (rb > 0.0) nelec_b_acc += wg * rb;

    const double e_pt = (ra + rb) * zk_acc[g];
    exc_acc += wg * e_pt;
    if (e_xc_point_out) {
      e_xc_point_out[g] = e_pt;
    }

    wv0_a[g] = rho_scale * wg * vrho2_acc[2 * g + 0];
    wv0_b[g] = rho_scale * wg * vrho2_acc[2 * g + 1];

    if (st.family >= 1) {
      const double gxa_s = (ra > 0.0) ? gxa[g] : 0.0;
      const double gya_s = (ra > 0.0) ? gya[g] : 0.0;
      const double gza_s = (ra > 0.0) ? gza[g] : 0.0;
      const double gxb_s = (rb > 0.0) ? gxb[g] : 0.0;
      const double gyb_s = (rb > 0.0) ? gyb[g] : 0.0;
      const double gzb_s = (rb > 0.0) ? gzb[g] : 0.0;

      const double two_vs_aa = 2.0 * wg * vsigma3_acc[3 * g + 0];
      const double vs_ab = wg * vsigma3_acc[3 * g + 1];
      const double two_vs_bb = 2.0 * wg * vsigma3_acc[3 * g + 2];

      wvx_a[g] = two_vs_aa * gxa_s + vs_ab * gxb_s;
      wvy_a[g] = two_vs_aa * gya_s + vs_ab * gyb_s;
      wvz_a[g] = two_vs_aa * gza_s + vs_ab * gzb_s;

      wvx_b[g] = two_vs_bb * gxb_s + vs_ab * gxa_s;
      wvy_b[g] = two_vs_bb * gyb_s + vs_ab * gya_s;
      wvz_b[g] = two_vs_bb * gzb_s + vs_ab * gza_s;
    }

    if (st.family == 2) {
      wvt_a[g] = tau_scale * wg * vtau2_acc[2 * g + 0];
      wvt_b[g] = tau_scale * wg * vtau2_acc[2 * g + 1];
    }
  }
}

// Cache-friendly Host blocked GEMM C[M x N] = A[M x K] * B[K x N]
void host_gemm_nn(int M, int N, int K, const double* A, const double* B, double* C,
                  bool accum = false) {
  if (!accum) {
    std::fill_n(C, static_cast<std::size_t>(M) * N, 0.0);
  }
  for (int i = 0; i < M; ++i) {
    double* ci = C + static_cast<std::size_t>(i) * N;
    const double* ai = A + static_cast<std::size_t>(i) * K;
    for (int k = 0; k < K; ++k) {
      const double aik = ai[k];
      if (aik == 0.0) continue;
      const double* bk = B + static_cast<std::size_t>(k) * N;
      for (int j = 0; j < N; ++j) {
        ci[j] += aik * bk[j];
      }
    }
  }
}

void host_gemm_nt(int M, int N, int K, const double* A, const double* B, double* C,
                  bool accum = false) {
  if (!accum) {
    std::fill_n(C, static_cast<std::size_t>(M) * N, 0.0);
  }
  for (int i = 0; i < M; ++i) {
    const double* ai = A + static_cast<std::size_t>(i) * K;
    double* ci = C + static_cast<std::size_t>(i) * N;
    for (int j = 0; j < N; ++j) {
      const double* bj = B + static_cast<std::size_t>(j) * K;
      double sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += ai[k] * bj[k];
      }
      ci[j] += sum;
    }
  }
}

std::vector<int> build_ao_to_atom(const Basis& basis, std::span<const XcAtom> atoms) {
  std::vector<int> ao_to_atom(static_cast<std::size_t>(basis.nao()), -1);
  for (std::size_t is = 0; is < basis.nshell(); ++is) {
    const Shell& sh = basis.shell(is);
    int matched = -1;
    for (std::size_t ia = 0; ia < atoms.size(); ++ia) {
      const double dx = sh.center[0] - atoms[ia].center[0];
      const double dy = sh.center[1] - atoms[ia].center[1];
      const double dz = sh.center[2] - atoms[ia].center[2];
      if (dx * dx + dy * dy + dz * dz < 1e-10) {
        matched = static_cast<int>(ia);
        break;
      }
    }
    const int ao0 = basis.ao_offset(is);
    const int nm = 2 * sh.l + 1;
    for (int m = 0; m < nm; ++m) {
      ao_to_atom[static_cast<std::size_t>(ao0 + m)] = matched;
    }
  }
  return ao_to_atom;
}

}  // namespace

namespace detail {

void eval_libxc_rks_grid(const void* libxc_state,
                         int family,
                         double /*rho_cutoff*/,
                         std::size_t npts,
                         const double* weights,
                         const double* dens_in,
                         bool for_grad,
                         double* wv_out,
                         double* e_xc_point_out,
                         double* exc_out,
                         double* nelec_out) {
  if (npts == 0) {
    if (exc_out) *exc_out = 0.0;
    if (nelec_out) *nelec_out = 0.0;
    return;
  }
  const auto& st = *static_cast<const detail::LibxcState*>(libxc_state);
  constexpr std::size_t kChunk = 2048;
  const std::size_t n_chunks = (npts + kChunk - 1) / kChunk;

  double total_exc = 0.0;
  double total_nelec = 0.0;

#if defined(_OPENMP)
#pragma omp parallel reduction(+ : total_exc, total_nelec)
#endif
  {
    std::vector<double> rho_buf, sigma_buf, tau_buf, zk_acc, vrho_acc, vsigma_acc, vtau_acc;
    std::vector<double> zk_tmp, vrho_tmp, vsigma_tmp, vtau_tmp;

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
    for (std::ptrdiff_t ic = 0; ic < static_cast<std::ptrdiff_t>(n_chunks); ++ic) {
      const std::size_t off = static_cast<std::size_t>(ic) * kChunk;
      const std::size_t len = std::min(kChunk, npts - off);

      const double* rho = dens_in + 0 * npts + off;
      const double* gx = (family >= 1) ? (dens_in + 1 * npts + off) : nullptr;
      const double* gy = (family >= 1) ? (dens_in + 2 * npts + off) : nullptr;
      const double* gz = (family >= 1) ? (dens_in + 3 * npts + off) : nullptr;
      const double* tau = (family == 2) ? (dens_in + 4 * npts + off) : nullptr;

      double* wv0 = wv_out + 0 * npts + off;
      double* wvx = (family >= 1) ? (wv_out + 1 * npts + off) : nullptr;
      double* wvy = (family >= 1) ? (wv_out + 2 * npts + off) : nullptr;
      double* wvz = (family >= 1) ? (wv_out + 3 * npts + off) : nullptr;
      double* wvt = (family == 2) ? (wv_out + 4 * npts + off) : nullptr;
      double* exc_pt = e_xc_point_out ? (e_xc_point_out + off) : nullptr;

      eval_libxc_rks_chunk(st, len, weights + off, rho, gx, gy, gz, tau, for_grad,
                           wv0, wvx, wvy, wvz, wvt, exc_pt, total_exc, total_nelec,
                           rho_buf, sigma_buf, tau_buf, zk_acc, vrho_acc, vsigma_acc, vtau_acc,
                           zk_tmp, vrho_tmp, vsigma_tmp, vtau_tmp);
    }
  }

  if (exc_out) *exc_out = total_exc;
  if (nelec_out) *nelec_out = total_nelec;
}

void eval_libxc_uks_grid(const void* libxc_state,
                         int family,
                         double /*rho_cutoff*/,
                         std::size_t npts,
                         const double* weights,
                         const double* dens_in,
                         bool for_grad,
                         double* wv_out,
                         double* e_xc_point_out,
                         double* exc_out,
                         double* nelec_a_out,
                         double* nelec_b_out) {
  if (npts == 0) {
    if (exc_out) *exc_out = 0.0;
    if (nelec_a_out) *nelec_a_out = 0.0;
    if (nelec_b_out) *nelec_b_out = 0.0;
    return;
  }
  const auto& st = *static_cast<const detail::LibxcState*>(libxc_state);
  constexpr std::size_t kChunk = 2048;
  const std::size_t n_chunks = (npts + kChunk - 1) / kChunk;

  double total_exc = 0.0;
  double total_nelec_a = 0.0;
  double total_nelec_b = 0.0;

#if defined(_OPENMP)
#pragma omp parallel reduction(+ : total_exc, total_nelec_a, total_nelec_b)
#endif
  {
    std::vector<double> rho2_buf, sigma3_buf, tau2_buf;
    std::vector<double> zk_acc, vrho2_acc, vsigma3_acc, vtau2_acc;
    std::vector<double> zk_tmp, vrho2_tmp, vsigma3_tmp, vtau2_tmp;

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
    for (std::ptrdiff_t ic = 0; ic < static_cast<std::ptrdiff_t>(n_chunks); ++ic) {
      const std::size_t off = static_cast<std::size_t>(ic) * kChunk;
      const std::size_t len = std::min(kChunk, npts - off);

      const double* rho_a = dens_in + 0 * npts + off;
      const double* rho_b = dens_in + 1 * npts + off;
      const double* gxa = (family >= 1) ? (dens_in + 2 * npts + off) : nullptr;
      const double* gya = (family >= 1) ? (dens_in + 3 * npts + off) : nullptr;
      const double* gza = (family >= 1) ? (dens_in + 4 * npts + off) : nullptr;
      const double* gxb = (family >= 1) ? (dens_in + 5 * npts + off) : nullptr;
      const double* gyb = (family >= 1) ? (dens_in + 6 * npts + off) : nullptr;
      const double* gzb = (family >= 1) ? (dens_in + 7 * npts + off) : nullptr;
      const double* tau_a = (family == 2) ? (dens_in + 8 * npts + off) : nullptr;
      const double* tau_b = (family == 2) ? (dens_in + 9 * npts + off) : nullptr;

      const std::size_t n_wv_comp = (family == 0) ? 1 : ((family == 1) ? 4 : 5);
      double* wv_a_base = wv_out;
      double* wv_b_base = wv_out + n_wv_comp * npts;
      double* wv0_a = wv_a_base + 0 * npts + off;
      double* wv0_b = wv_b_base + 0 * npts + off;
      double* wvx_a = (family >= 1) ? (wv_a_base + 1 * npts + off) : nullptr;
      double* wvy_a = (family >= 1) ? (wv_a_base + 2 * npts + off) : nullptr;
      double* wvz_a = (family >= 1) ? (wv_a_base + 3 * npts + off) : nullptr;
      double* wvx_b = (family >= 1) ? (wv_b_base + 1 * npts + off) : nullptr;
      double* wvy_b = (family >= 1) ? (wv_b_base + 2 * npts + off) : nullptr;
      double* wvz_b = (family >= 1) ? (wv_b_base + 3 * npts + off) : nullptr;
      double* wvt_a = (family == 2) ? (wv_a_base + 4 * npts + off) : nullptr;
      double* wvt_b = (family == 2) ? (wv_b_base + 4 * npts + off) : nullptr;
      double* exc_pt = e_xc_point_out ? (e_xc_point_out + off) : nullptr;

      eval_libxc_uks_chunk(st, len, weights + off,
                           rho_a, rho_b, gxa, gya, gza, gxb, gyb, gzb, tau_a, tau_b,
                           for_grad,
                           wv0_a, wv0_b, wvx_a, wvy_a, wvz_a, wvx_b, wvy_b, wvz_b, wvt_a, wvt_b,
                           exc_pt, total_exc, total_nelec_a, total_nelec_b,
                           rho2_buf, sigma3_buf, tau2_buf,
                           zk_acc, vrho2_acc, vsigma3_acc, vtau2_acc,
                           zk_tmp, vrho2_tmp, vsigma3_tmp, vtau2_tmp);
    }
  }

  if (exc_out) *exc_out = total_exc;
  if (nelec_a_out) *nelec_a_out = total_nelec_a;
  if (nelec_b_out) *nelec_b_out = total_nelec_b;
}

}  // namespace detail

double xc_exact_exchange_fraction(std::string_view xc_name) {
  return xc_rsh_coefficients(xc_name).c_sr();
}

RshCoefficients xc_rsh_coefficients(std::string_view xc_name) {
  auto st = build_libxc_state(xc_name, 1e-15);
  return st->rsh;
}

XcPointwiseResult eval_xc_points(std::string_view xc_name,
                                 std::span<const double> rho,
                                 std::span<const double> sigma,
                                 std::span<const double> tau,
                                 int deriv,
                                 Backend backend,
                                 double rho_cutoff) {
  if (deriv < 0 || deriv > 2) {
    throw std::invalid_argument("eval_xc_points: deriv must be in {0, 1, 2}");
  }
  const std::size_t n = rho.size();
  if (!sigma.empty() && sigma.size() != n) {
    throw std::invalid_argument("eval_xc_points: sigma size mismatch");
  }
  if (!tau.empty() && tau.size() != n) {
    throw std::invalid_argument("eval_xc_points: tau size mismatch");
  }

  const auto id = xc_gen::parse_xc_functional(xc_name);
  XcPointwiseResult res;
  res.zk.assign(n, 0.0);
  if (deriv >= 1) {
    res.vrho.assign(n, 0.0);
    res.vsigma.assign(n, 0.0);
    res.vtau.assign(n, 0.0);
  }
  if (deriv >= 2) {
    res.v2rho2.assign(n, 0.0);
    res.v2rhosigma.assign(n, 0.0);
    res.v2sigma2.assign(n, 0.0);
    res.v2rhotau.assign(n, 0.0);
    res.v2sigmatau.assign(n, 0.0);
    res.v2tau2.assign(n, 0.0);
  }
  if (n == 0) return res;

  if (backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    if (detail::xc_cuda_device_count() == 0) {
      throw std::runtime_error("eval_xc_points: no CUDA device available");
    }
    std::vector<double> sig_buf;
    std::vector<double> tau_buf;
    const double* sig_ptr = sigma.data();
    const double* tau_ptr = tau.data();
    if (sigma.empty()) {
      sig_buf.assign(n, 0.0);
      sig_ptr = sig_buf.data();
    }
    if (tau.empty()) {
      tau_buf.assign(n, 0.0);
      tau_ptr = tau_buf.data();
    }
    detail::eval_xc_points_cuda(static_cast<int>(id), n, rho.data(), sig_ptr, tau_ptr,
                                deriv, rho_cutoff, res);
    return res;
#else
    throw std::runtime_error("eval_xc_points: gpu4dft was built without CUDA");
#endif
  }
  if (backend != Backend::Host) {
    throw std::runtime_error("eval_xc_points: unsupported backend");
  }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    const double r = rho[idx];
    const double s = sigma.empty() ? 0.0 : sigma[idx];
    const double t = tau.empty() ? 0.0 : tau[idx];
    xc_gen::XcPointOut pt;
    xc_gen::eval_xc_point(id, r, s, t, deriv, pt, rho_cutoff);
    res.zk[idx] = pt.zk;
    if (deriv >= 1) {
      res.vrho[idx] = pt.vrho;
      res.vsigma[idx] = pt.vsigma;
      res.vtau[idx] = pt.vtau;
    }
    if (deriv >= 2) {
      res.v2rho2[idx] = pt.v2rho2;
      res.v2rhosigma[idx] = pt.v2rhosigma;
      res.v2sigma2[idx] = pt.v2sigma2;
      res.v2rhotau[idx] = pt.v2rhotau;
      res.v2sigmatau[idx] = pt.v2sigmatau;
      res.v2tau2[idx] = pt.v2tau2;
    }
  }
  return res;
}

std::vector<double> eval_ao_grid(const Basis& basis,
                                 std::span<const double> coords_xyz,
                                 int deriv,
                                 Backend backend) {
  if (deriv < 0 || deriv > 2) {
    throw std::invalid_argument("eval_ao_grid: deriv must be in {0, 1, 2}");
  }
  if (coords_xyz.size() % 3 != 0) {
    throw std::invalid_argument("eval_ao_grid: coords_xyz length must be divisible by 3");
  }
  const std::size_t npts = coords_xyz.size() / 3;
  const std::size_t nao = static_cast<std::size_t>(basis.nao());
  const std::size_t ncomp = (deriv == 0) ? 1 : ((deriv == 1) ? 4 : 10);
  std::vector<double> out(ncomp * npts * nao, 0.0);
  if (npts == 0 || nao == 0) return out;

  const detail::FlatBasisData flat(basis);
  if (backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    if (detail::xc_cuda_device_count() == 0) {
      throw std::runtime_error("eval_ao_grid: no CUDA device available");
    }
    detail::eval_ao_grid_cuda(flat, coords_xyz.data(), npts, deriv, out.data());
    return out;
#else
    throw std::runtime_error("eval_ao_grid: gpu4dft was built without CUDA");
#endif
  }
  if (backend != Backend::Host) {
    throw std::runtime_error("eval_ao_grid: unsupported backend");
  }

  const detail::FlatBasisView bview = flat.view();
  const std::size_t comp_stride = npts * nao;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t ig = 0; ig < static_cast<std::ptrdiff_t>(npts); ++ig) {
    const std::size_t g = static_cast<std::size_t>(ig);
    const double gx = coords_xyz[3 * g + 0];
    const double gy = coords_xyz[3 * g + 1];
    const double gz = coords_xyz[3 * g + 2];
    for (int is = 0; is < bview.nshell; ++is) {
      const std::size_t ao0 = static_cast<std::size_t>(bview.shell_ao_offset[is]);
      detail::eval_shell_at_point(bview, is, gx, gy, gz, deriv,
                                  out.data() + g * nao + ao0, comp_stride, 1);
    }
  }
  return out;
}

XcEngine::XcEngine(const Basis& basis, const MolecularGrid& grid,
                   std::string_view xc_name, const XcOptions& opts)
    : basis_(basis), grid_(grid), xc_name_(xc_name), opts_(opts) {
  if (grid_.nao() != basis_.nao() && grid_.npoints() > 0) {
    throw std::invalid_argument("XcEngine: grid.nao() does not match basis.nao()");
  }

  libxc_ = build_libxc_state(xc_name_, opts_.rho_cutoff);
  xc_family_ = libxc_->family;
  rsh_ = libxc_->rsh;
  exx_fraction_ = rsh_.c_sr();

  try {
    const auto fid = xc_gen::parse_xc_functional(xc_name_);
    xc_func_id_ = static_cast<int>(fid);
    xc_codegen_supported_ = true;
  } catch (const std::exception&) {
    xc_func_id_ = 0;
    xc_codegen_supported_ = false;
  }

  if (!opts_.use_libxc && !xc_codegen_supported_) {
    throw std::invalid_argument(
        "XcEngine: use_libxc=false is only supported for built-in xc_codegen functionals; got '" +
        xc_name_ + "'");
  }

  if (opts_.backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    if (detail::xc_cuda_device_count() > 0) {
      const detail::FlatBasisData flat(basis_);
      cuda_ = std::make_unique<CudaState>();
      cuda_->impl = detail::xc_cuda_engine_create(
          flat, grid_, xc_func_id_, xc_family_, libxc_->is_hf, libxc_.get(), opts_);
      active_backend_ = Backend::Cuda;
    } else {
      throw std::runtime_error("XcEngine: Backend::Cuda requested, but no CUDA device is available");
    }
#else
    throw std::runtime_error("XcEngine: gpu4dft was built without CUDA");
#endif
  } else if (opts_.backend == Backend::Host) {
    active_backend_ = Backend::Host;
  } else {
    throw std::runtime_error("XcEngine: unsupported backend");
  }
}

XcEngine::~XcEngine() {
#if defined(GPU4DFT_HAVE_CUDA)
  if (cuda_ && cuda_->impl) {
    detail::xc_cuda_engine_destroy(cuda_->impl);
    cuda_->impl = nullptr;
  }
#endif
}

XcEngine::XcEngine(XcEngine&&) noexcept = default;
XcEngine& XcEngine::operator=(XcEngine&&) noexcept = default;

void XcEngine::set_grid_response(bool enabled) noexcept {
  opts_.grid_response = enabled;
#if defined(GPU4DFT_HAVE_CUDA)
  if (active_backend_ == Backend::Cuda && cuda_ && cuda_->impl) {
    detail::xc_cuda_set_grid_response(cuda_->impl, enabled);
  }
#endif
}

XcResult XcEngine::eval_vxc(std::span<const double> dm) const {
  const std::size_t n = static_cast<std::size_t>(basis_.nao());
  if (dm.size() != n * n) {
    throw std::invalid_argument("XcEngine::eval_vxc: dm size must equal nao * nao");
  }
  XcResult res;
  res.vxc.assign(n * n, 0.0);
  res.exc = eval_vxc(dm.data(), res.vxc.data(), &res.nelec);
  return res;
}

double XcEngine::eval_vxc(const double* dm, double* vxc_out, double* nelec_out) const {
  const std::size_t nao = static_cast<std::size_t>(basis_.nao());
  std::fill_n(vxc_out, nao * nao, 0.0);
  if (nelec_out) *nelec_out = 0.0;
  if (nao == 0 || grid_.npoints() == 0) return 0.0;

  if (active_backend_ == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    return detail::xc_cuda_eval_vxc_host(cuda_->impl, dm, vxc_out, nelec_out);
#endif
  }

  const detail::FlatBasisData flat(basis_);
  const detail::FlatBasisView bview = flat.view();
  const auto fid = static_cast<xc_gen::XcFunctionalId>(xc_func_id_);
  const bool use_libxc_path = opts_.use_libxc || !xc_codegen_supported_;
  const int deriv = (xc_family_ == 0) ? 0 : 1;
  const std::size_t ncomp = (deriv == 0) ? 1 : 4;
  const auto& batches = grid_.batches();
  const auto& coords = grid_.coords();
  const auto& weights = grid_.weights();

  double total_exc = 0.0;
  double total_nelec = 0.0;

#if defined(_OPENMP)
#pragma omp parallel reduction(+ : total_exc, total_nelec)
#endif
  {
    std::vector<double> local_vxc(nao * nao, 0.0);
    std::vector<double> phi;
    std::vector<double> d_act;
    std::vector<double> t_mat;
    std::vector<double> w_mat;
    std::vector<double> v_half;
    std::vector<double> rho_g, gx_g, gy_g, gz_g, tau_g;
    std::vector<double> wv0, wvx, wvy, wvz, wvt;
    std::vector<double> rho_buf, sigma_buf, tau_buf, zk_acc, vrho_acc, vsigma_acc, vtau_acc;
    std::vector<double> zk_tmp, vrho_tmp, vsigma_tmp, vtau_tmp;

#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
    for (std::ptrdiff_t ib = 0; ib < static_cast<std::ptrdiff_t>(batches.size()); ++ib) {
      const SpatialBatch& batch = batches[static_cast<std::size_t>(ib)];
      const std::size_t ng = batch.npoints;
      const std::size_t na = batch.active_aos.size();
      if (ng == 0 || na == 0) continue;

      const std::size_t comp_stride = na * ng;
      phi.assign(ncomp * comp_stride, 0.0);

      std::size_t local_ao_cursor = 0;
      for (int is : batch.active_shells) {
        const int nm = 2 * bview.shell_l[is] + 1;
        for (std::size_t g = 0; g < ng; ++g) {
          const std::size_t gp = batch.point_offset + g;
          detail::eval_shell_at_point(
              bview, is,
              coords[3 * gp + 0], coords[3 * gp + 1], coords[3 * gp + 2],
              deriv,
              phi.data() + local_ao_cursor * ng + g,
              comp_stride, ng);
        }
        local_ao_cursor += static_cast<std::size_t>(nm);
      }

      d_act.resize(na * na);
      for (std::size_t u = 0; u < na; ++u) {
        const std::size_t gu = static_cast<std::size_t>(batch.active_aos[u]);
        for (std::size_t v = 0; v < na; ++v) {
          const std::size_t gv = static_cast<std::size_t>(batch.active_aos[v]);
          d_act[u * na + v] = 0.5 * (dm[gu * nao + gv] + dm[gv * nao + gu]);
        }
      }

      const int i_na = static_cast<int>(na);
      const int i_ng = static_cast<int>(ng);
      const std::size_t n_tcomp = (xc_family_ == 2) ? 4 : 1;
      t_mat.resize(n_tcomp * comp_stride);

      host_gemm_nn(i_na, i_ng, i_na, d_act.data(), phi.data(), t_mat.data(), false);
      if (xc_family_ == 2) {
        for (std::size_t b = 1; b <= 3; ++b) {
          host_gemm_nn(i_na, i_ng, i_na, d_act.data(),
                       phi.data() + b * comp_stride,
                       t_mat.data() + b * comp_stride, false);
        }
      }

      rho_g.assign(ng, 0.0);
      if (xc_family_ >= 1) {
        gx_g.assign(ng, 0.0);
        gy_g.assign(ng, 0.0);
        gz_g.assign(ng, 0.0);
      }
      if (xc_family_ == 2) {
        tau_g.assign(ng, 0.0);
      }

      const double* phi0 = phi.data();
      const double* phix = (xc_family_ >= 1) ? (phi.data() + 1 * comp_stride) : nullptr;
      const double* phiy = (xc_family_ >= 1) ? (phi.data() + 2 * comp_stride) : nullptr;
      const double* phiz = (xc_family_ >= 1) ? (phi.data() + 3 * comp_stride) : nullptr;
      const double* t0 = t_mat.data();

      for (std::size_t u = 0; u < na; ++u) {
        for (std::size_t g = 0; g < ng; ++g) {
          const double t0_ug = t0[u * ng + g];
          rho_g[g] += phi0[u * ng + g] * t0_ug;
          if (xc_family_ >= 1) {
            gx_g[g] += 2.0 * phix[u * ng + g] * t0_ug;
            gy_g[g] += 2.0 * phiy[u * ng + g] * t0_ug;
            gz_g[g] += 2.0 * phiz[u * ng + g] * t0_ug;
          }
          if (xc_family_ == 2) {
            const double tx = t_mat[1 * comp_stride + u * ng + g];
            const double ty = t_mat[2 * comp_stride + u * ng + g];
            const double tz = t_mat[3 * comp_stride + u * ng + g];
            tau_g[g] += 0.5 * (phix[u * ng + g] * tx + phiy[u * ng + g] * ty +
                               phiz[u * ng + g] * tz);
          }
        }
      }

      wv0.resize(ng);
      if (xc_family_ >= 1) {
        wvx.resize(ng);
        wvy.resize(ng);
        wvz.resize(ng);
      }
      if (xc_family_ == 2) {
        wvt.resize(ng);
      }

      if (use_libxc_path) {
        eval_libxc_rks_chunk(
            *libxc_, ng, weights.data() + batch.point_offset,
            rho_g.data(),
            (xc_family_ >= 1) ? gx_g.data() : nullptr,
            (xc_family_ >= 1) ? gy_g.data() : nullptr,
            (xc_family_ >= 1) ? gz_g.data() : nullptr,
            (xc_family_ == 2) ? tau_g.data() : nullptr,
            /*for_grad=*/false,
            wv0.data(),
            (xc_family_ >= 1) ? wvx.data() : nullptr,
            (xc_family_ >= 1) ? wvy.data() : nullptr,
            (xc_family_ >= 1) ? wvz.data() : nullptr,
            (xc_family_ == 2) ? wvt.data() : nullptr,
            /*e_xc_point_out=*/nullptr,
            total_exc, total_nelec,
            rho_buf, sigma_buf, tau_buf, zk_acc, vrho_acc, vsigma_acc, vtau_acc,
            zk_tmp, vrho_tmp, vsigma_tmp, vtau_tmp);
      } else {
        for (std::size_t g = 0; g < ng; ++g) {
          const double wg = weights[batch.point_offset + g];
          const double r = rho_g[g];
          const double s = (xc_family_ >= 1)
                               ? (gx_g[g] * gx_g[g] + gy_g[g] * gy_g[g] + gz_g[g] * gz_g[g])
                               : 0.0;
          const double t = (xc_family_ == 2) ? tau_g[g] : 0.0;
          if (r > 0.0) total_nelec += wg * r;
          xc_gen::XcPointOut out_pt;
          xc_gen::eval_xc_point(fid, r, s, t, 1, out_pt, opts_.rho_cutoff);
          total_exc += wg * r * out_pt.zk;
          wv0[g] = 0.5 * wg * out_pt.vrho;
          if (xc_family_ >= 1) {
            wvx[g] = 2.0 * wg * out_pt.vsigma * gx_g[g];
            wvy[g] = 2.0 * wg * out_pt.vsigma * gy_g[g];
            wvz[g] = 2.0 * wg * out_pt.vsigma * gz_g[g];
          }
          if (xc_family_ == 2) {
            wvt[g] = 0.25 * wg * out_pt.vtau;
          }
        }
      }

      w_mat.assign(comp_stride, 0.0);
      for (std::size_t u = 0; u < na; ++u) {
        for (std::size_t g = 0; g < ng; ++g) {
          double w_ug = wv0[g] * phi0[u * ng + g];
          if (xc_family_ >= 1) {
            w_ug += wvx[g] * phix[u * ng + g] + wvy[g] * phiy[u * ng + g] +
                    wvz[g] * phiz[u * ng + g];
          }
          w_mat[u * ng + g] = w_ug;
        }
      }

      v_half.resize(na * na);
      host_gemm_nt(i_na, i_na, i_ng, phi0, w_mat.data(), v_half.data(), false);

      if (xc_family_ == 2) {
        for (std::size_t b = 1; b <= 3; ++b) {
          const double* phib = phi.data() + b * comp_stride;
          for (std::size_t u = 0; u < na; ++u) {
            for (std::size_t g = 0; g < ng; ++g) {
              w_mat[u * ng + g] = wvt[g] * phib[u * ng + g];
            }
          }
          host_gemm_nt(i_na, i_na, i_ng, phib, w_mat.data(), v_half.data(), true);
        }
      }

      for (std::size_t u = 0; u < na; ++u) {
        const std::size_t gu = static_cast<std::size_t>(batch.active_aos[u]);
        for (std::size_t v = 0; v < na; ++v) {
          const std::size_t gv = static_cast<std::size_t>(batch.active_aos[v]);
          local_vxc[gu * nao + gv] += v_half[u * na + v] + v_half[v * na + u];
        }
      }
    }

#if defined(_OPENMP)
#pragma omp critical
#endif
    {
      for (std::size_t idx = 0; idx < nao * nao; ++idx) {
        vxc_out[idx] += local_vxc[idx];
      }
    }
  }

  if (nelec_out) *nelec_out = total_nelec;
  return total_exc;
}

XcUksResult XcEngine::eval_vxc_uks(std::span<const double> dm_alpha,
                                   std::span<const double> dm_beta) const {
  const std::size_t n = static_cast<std::size_t>(basis_.nao());
  if (dm_alpha.size() != n * n || dm_beta.size() != n * n) {
    throw std::invalid_argument(
        "XcEngine::eval_vxc_uks: dm_alpha and dm_beta sizes must equal nao * nao");
  }
  XcUksResult res;
  res.vxc_alpha.assign(n * n, 0.0);
  res.vxc_beta.assign(n * n, 0.0);
  res.exc = eval_vxc_uks(dm_alpha.data(), dm_beta.data(), res.vxc_alpha.data(),
                         res.vxc_beta.data(), &res.nelec_alpha, &res.nelec_beta);
  return res;
}

double XcEngine::eval_vxc_uks(const double* dm_alpha,
                              const double* dm_beta,
                              double* vxc_alpha_out,
                              double* vxc_beta_out,
                              double* nelec_alpha_out,
                              double* nelec_beta_out) const {
  const std::size_t nao = static_cast<std::size_t>(basis_.nao());
  std::fill_n(vxc_alpha_out, nao * nao, 0.0);
  std::fill_n(vxc_beta_out, nao * nao, 0.0);
  if (nelec_alpha_out) *nelec_alpha_out = 0.0;
  if (nelec_beta_out) *nelec_beta_out = 0.0;
  if (nao == 0 || grid_.npoints() == 0) return 0.0;

  if (active_backend_ == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    return detail::xc_cuda_eval_vxc_uks_host(
        cuda_->impl, dm_alpha, dm_beta, vxc_alpha_out, vxc_beta_out,
        nelec_alpha_out, nelec_beta_out);
#endif
  }

  const detail::FlatBasisData flat(basis_);
  const detail::FlatBasisView bview = flat.view();
  const int deriv = (xc_family_ == 0) ? 0 : 1;
  const std::size_t ncomp = (deriv == 0) ? 1 : 4;
  const auto& batches = grid_.batches();
  const auto& coords = grid_.coords();
  const auto& weights = grid_.weights();

  double total_exc = 0.0;
  double total_nelec_a = 0.0;
  double total_nelec_b = 0.0;

#if defined(_OPENMP)
#pragma omp parallel reduction(+ : total_exc, total_nelec_a, total_nelec_b)
#endif
  {
    std::vector<double> local_vxc_a(nao * nao, 0.0);
    std::vector<double> local_vxc_b(nao * nao, 0.0);
    std::vector<double> phi;
    std::vector<double> d_act_a, d_act_b;
    std::vector<double> t_mat_a, t_mat_b;
    std::vector<double> w_mat, v_half;
    std::vector<double> rho_a, rho_b, gxa, gya, gza, gxb, gyb, gzb, tau_a, tau_b;
    std::vector<double> wv0_a, wv0_b, wvx_a, wvy_a, wvz_a, wvx_b, wvy_b, wvz_b, wvt_a, wvt_b;
    std::vector<double> rho2_buf, sigma3_buf, tau2_buf;
    std::vector<double> zk_acc, vrho2_acc, vsigma3_acc, vtau2_acc;
    std::vector<double> zk_tmp, vrho2_tmp, vsigma3_tmp, vtau2_tmp;

#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
    for (std::ptrdiff_t ib = 0; ib < static_cast<std::ptrdiff_t>(batches.size()); ++ib) {
      const SpatialBatch& batch = batches[static_cast<std::size_t>(ib)];
      const std::size_t ng = batch.npoints;
      const std::size_t na = batch.active_aos.size();
      if (ng == 0 || na == 0) continue;

      const std::size_t comp_stride = na * ng;
      phi.assign(ncomp * comp_stride, 0.0);

      std::size_t local_ao_cursor = 0;
      for (int is : batch.active_shells) {
        const int nm = 2 * bview.shell_l[is] + 1;
        for (std::size_t g = 0; g < ng; ++g) {
          const std::size_t gp = batch.point_offset + g;
          detail::eval_shell_at_point(
              bview, is,
              coords[3 * gp + 0], coords[3 * gp + 1], coords[3 * gp + 2],
              deriv, phi.data() + local_ao_cursor * ng + g, comp_stride, ng);
        }
        local_ao_cursor += static_cast<std::size_t>(nm);
      }

      d_act_a.resize(na * na);
      d_act_b.resize(na * na);
      for (std::size_t u = 0; u < na; ++u) {
        const std::size_t gu = static_cast<std::size_t>(batch.active_aos[u]);
        for (std::size_t v = 0; v < na; ++v) {
          const std::size_t gv = static_cast<std::size_t>(batch.active_aos[v]);
          d_act_a[u * na + v] = 0.5 * (dm_alpha[gu * nao + gv] + dm_alpha[gv * nao + gu]);
          d_act_b[u * na + v] = 0.5 * (dm_beta[gu * nao + gv] + dm_beta[gv * nao + gu]);
        }
      }

      const int i_na = static_cast<int>(na);
      const int i_ng = static_cast<int>(ng);
      const std::size_t n_tcomp = (xc_family_ == 2) ? 4 : 1;
      t_mat_a.resize(n_tcomp * comp_stride);
      t_mat_b.resize(n_tcomp * comp_stride);

      host_gemm_nn(i_na, i_ng, i_na, d_act_a.data(), phi.data(), t_mat_a.data(), false);
      host_gemm_nn(i_na, i_ng, i_na, d_act_b.data(), phi.data(), t_mat_b.data(), false);
      if (xc_family_ == 2) {
        for (std::size_t b = 1; b <= 3; ++b) {
          host_gemm_nn(i_na, i_ng, i_na, d_act_a.data(),
                       phi.data() + b * comp_stride,
                       t_mat_a.data() + b * comp_stride, false);
          host_gemm_nn(i_na, i_ng, i_na, d_act_b.data(),
                       phi.data() + b * comp_stride,
                       t_mat_b.data() + b * comp_stride, false);
        }
      }

      rho_a.assign(ng, 0.0);
      rho_b.assign(ng, 0.0);
      if (xc_family_ >= 1) {
        gxa.assign(ng, 0.0);
        gya.assign(ng, 0.0);
        gza.assign(ng, 0.0);
        gxb.assign(ng, 0.0);
        gyb.assign(ng, 0.0);
        gzb.assign(ng, 0.0);
      }
      if (xc_family_ == 2) {
        tau_a.assign(ng, 0.0);
        tau_b.assign(ng, 0.0);
      }

      const double* phi0 = phi.data();
      const double* phix = (xc_family_ >= 1) ? (phi.data() + 1 * comp_stride) : nullptr;
      const double* phiy = (xc_family_ >= 1) ? (phi.data() + 2 * comp_stride) : nullptr;
      const double* phiz = (xc_family_ >= 1) ? (phi.data() + 3 * comp_stride) : nullptr;
      const double* t0a = t_mat_a.data();
      const double* t0b = t_mat_b.data();

      for (std::size_t u = 0; u < na; ++u) {
        for (std::size_t g = 0; g < ng; ++g) {
          const std::size_t off = u * ng + g;
          const double p0 = phi0[off];
          const double t0a_ug = t0a[off];
          const double t0b_ug = t0b[off];
          rho_a[g] += p0 * t0a_ug;
          rho_b[g] += p0 * t0b_ug;
          if (xc_family_ >= 1) {
            const double px = phix[off];
            const double py = phiy[off];
            const double pz = phiz[off];
            gxa[g] += 2.0 * px * t0a_ug;
            gya[g] += 2.0 * py * t0a_ug;
            gza[g] += 2.0 * pz * t0a_ug;
            gxb[g] += 2.0 * px * t0b_ug;
            gyb[g] += 2.0 * py * t0b_ug;
            gzb[g] += 2.0 * pz * t0b_ug;
            if (xc_family_ == 2) {
              tau_a[g] += 0.5 * (px * t_mat_a[1 * comp_stride + off] +
                                 py * t_mat_a[2 * comp_stride + off] +
                                 pz * t_mat_a[3 * comp_stride + off]);
              tau_b[g] += 0.5 * (px * t_mat_b[1 * comp_stride + off] +
                                 py * t_mat_b[2 * comp_stride + off] +
                                 pz * t_mat_b[3 * comp_stride + off]);
            }
          }
        }
      }

      wv0_a.resize(ng);
      wv0_b.resize(ng);
      if (xc_family_ >= 1) {
        wvx_a.resize(ng);
        wvy_a.resize(ng);
        wvz_a.resize(ng);
        wvx_b.resize(ng);
        wvy_b.resize(ng);
        wvz_b.resize(ng);
      }
      if (xc_family_ == 2) {
        wvt_a.resize(ng);
        wvt_b.resize(ng);
      }

      eval_libxc_uks_chunk(
          *libxc_, ng, weights.data() + batch.point_offset,
          rho_a.data(), rho_b.data(),
          (xc_family_ >= 1) ? gxa.data() : nullptr,
          (xc_family_ >= 1) ? gya.data() : nullptr,
          (xc_family_ >= 1) ? gza.data() : nullptr,
          (xc_family_ >= 1) ? gxb.data() : nullptr,
          (xc_family_ >= 1) ? gyb.data() : nullptr,
          (xc_family_ >= 1) ? gzb.data() : nullptr,
          (xc_family_ == 2) ? tau_a.data() : nullptr,
          (xc_family_ == 2) ? tau_b.data() : nullptr,
          /*for_grad=*/false,
          wv0_a.data(), wv0_b.data(),
          (xc_family_ >= 1) ? wvx_a.data() : nullptr,
          (xc_family_ >= 1) ? wvy_a.data() : nullptr,
          (xc_family_ >= 1) ? wvz_a.data() : nullptr,
          (xc_family_ >= 1) ? wvx_b.data() : nullptr,
          (xc_family_ >= 1) ? wvy_b.data() : nullptr,
          (xc_family_ >= 1) ? wvz_b.data() : nullptr,
          (xc_family_ == 2) ? wvt_a.data() : nullptr,
          (xc_family_ == 2) ? wvt_b.data() : nullptr,
          /*e_xc_point_out=*/nullptr,
          total_exc, total_nelec_a, total_nelec_b,
          rho2_buf, sigma3_buf, tau2_buf,
          zk_acc, vrho2_acc, vsigma3_acc, vtau2_acc,
          zk_tmp, vrho2_tmp, vsigma3_tmp, vtau2_tmp);

      auto assemble_spin_vxc = [&](const std::vector<double>& wv0_s,
                                   const std::vector<double>& wvx_s,
                                   const std::vector<double>& wvy_s,
                                   const std::vector<double>& wvz_s,
                                   const std::vector<double>& wvt_s,
                                   std::vector<double>& local_vxc) {
        w_mat.assign(comp_stride, 0.0);
        for (std::size_t u = 0; u < na; ++u) {
          for (std::size_t g = 0; g < ng; ++g) {
            const std::size_t off = u * ng + g;
            double w_ug = wv0_s[g] * phi0[off];
            if (xc_family_ >= 1) {
              w_ug += wvx_s[g] * phix[off] + wvy_s[g] * phiy[off] + wvz_s[g] * phiz[off];
            }
            w_mat[off] = w_ug;
          }
        }
        v_half.resize(na * na);
        host_gemm_nt(i_na, i_na, i_ng, phi0, w_mat.data(), v_half.data(), false);
        if (xc_family_ == 2) {
          for (std::size_t b = 1; b <= 3; ++b) {
            const double* phib = phi.data() + b * comp_stride;
            for (std::size_t u = 0; u < na; ++u) {
              for (std::size_t g = 0; g < ng; ++g) {
                w_mat[u * ng + g] = wvt_s[g] * phib[u * ng + g];
              }
            }
            host_gemm_nt(i_na, i_na, i_ng, phib, w_mat.data(), v_half.data(), true);
          }
        }
        for (std::size_t u = 0; u < na; ++u) {
          const std::size_t gu = static_cast<std::size_t>(batch.active_aos[u]);
          for (std::size_t v = 0; v < na; ++v) {
            const std::size_t gv = static_cast<std::size_t>(batch.active_aos[v]);
            local_vxc[gu * nao + gv] += v_half[u * na + v] + v_half[v * na + u];
          }
        }
      };

      assemble_spin_vxc(wv0_a, wvx_a, wvy_a, wvz_a, wvt_a, local_vxc_a);
      assemble_spin_vxc(wv0_b, wvx_b, wvy_b, wvz_b, wvt_b, local_vxc_b);
    }

#if defined(_OPENMP)
#pragma omp critical
#endif
    {
      for (std::size_t idx = 0; idx < nao * nao; ++idx) {
        vxc_alpha_out[idx] += local_vxc_a[idx];
        vxc_beta_out[idx] += local_vxc_b[idx];
      }
    }
  }

  if (nelec_alpha_out) *nelec_alpha_out = total_nelec_a;
  if (nelec_beta_out) *nelec_beta_out = total_nelec_b;
  return total_exc;
}

double XcEngine::eval_vxc_device(const double* d_dm, double* d_vxc, void* cuda_stream) const {
  if (active_backend_ != Backend::Cuda) {
    throw std::runtime_error("XcEngine::eval_vxc_device requires Backend::Cuda");
  }
#if defined(GPU4DFT_HAVE_CUDA)
  return detail::xc_cuda_eval_vxc_device(cuda_->impl, d_dm, d_vxc, nullptr, cuda_stream);
#else
  (void)d_dm;
  (void)d_vxc;
  (void)cuda_stream;
  throw std::runtime_error("gpu4dft was built without CUDA");
#endif
}

double XcEngine::eval_vxc_uks_device(const double* d_dm_alpha,
                                     const double* d_dm_beta,
                                     double* d_vxc_alpha,
                                     double* d_vxc_beta,
                                     void* cuda_stream) const {
  if (active_backend_ != Backend::Cuda) {
    throw std::runtime_error("XcEngine::eval_vxc_uks_device requires Backend::Cuda");
  }
#if defined(GPU4DFT_HAVE_CUDA)
  return detail::xc_cuda_eval_vxc_uks_device(
      cuda_->impl, d_dm_alpha, d_dm_beta, d_vxc_alpha, d_vxc_beta, nullptr, nullptr, cuda_stream);
#else
  (void)d_dm_alpha;
  (void)d_dm_beta;
  (void)d_vxc_alpha;
  (void)d_vxc_beta;
  (void)cuda_stream;
  throw std::runtime_error("gpu4dft was built without CUDA");
#endif
}

void XcEngine::set_reference_density(std::span<const double> h_dm) {
  const std::size_t n = static_cast<std::size_t>(basis_.nao());
  if (h_dm.size() != n * n) {
    throw std::invalid_argument("XcEngine::set_reference_density: dm size must equal nao * nao");
  }
  set_reference_density(h_dm.data());
}

void XcEngine::set_reference_density(const double* h_dm) {
  if (!xc_codegen_supported_) {
    throw NotImplemented("XcEngine::set_reference_density/eval_fxc is not implemented for '" +
                         xc_name_ + "'");
  }
  const std::size_t nao = static_cast<std::size_t>(basis_.nao());
  if (nao > 0 && h_dm == nullptr) {
    throw std::invalid_argument("XcEngine::set_reference_density: h_dm must not be null");
  }

  if (active_backend_ == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    detail::xc_cuda_set_reference_density_host(cuda_->impl, h_dm);
    return;
#endif
  }

  if (!host_fxc_) {
    host_fxc_ = std::make_unique<HostFxcState>();
  }
  HostFxcState& st = *host_fxc_;
  const std::size_t npts = grid_.npoints();
  st.v2rho2.assign(npts, 0.0);
  if (xc_family_ >= 1) {
    st.g0x.assign(npts, 0.0);
    st.g0y.assign(npts, 0.0);
    st.g0z.assign(npts, 0.0);
    st.vsigma.assign(npts, 0.0);
    st.v2rhosigma.assign(npts, 0.0);
    st.v2sigma2.assign(npts, 0.0);
  }
  if (xc_family_ == 2) {
    st.v2rhotau.assign(npts, 0.0);
    st.v2sigmatau.assign(npts, 0.0);
    st.v2tau2.assign(npts, 0.0);
  }
  if (nao == 0 || npts == 0) {
    st.ready = true;
    return;
  }

  const detail::FlatBasisData flat(basis_);
  const detail::FlatBasisView bview = flat.view();
  const auto fid = static_cast<xc_gen::XcFunctionalId>(xc_func_id_);
  const int deriv = (xc_family_ == 0) ? 0 : 1;
  const std::size_t ncomp = (deriv == 0) ? 1 : 4;
  const auto& batches = grid_.batches();
  const auto& coords = grid_.coords();

  if (!st.ao_cached) {
    std::size_t total_elems = 0;
    st.batch_phi_offset.resize(batches.size(), 0);
    for (std::size_t ib = 0; ib < batches.size(); ++ib) {
      st.batch_phi_offset[ib] = total_elems;
      total_elems += ncomp * batches[ib].active_aos.size() * batches[ib].npoints;
    }
    if (total_elems > 0 && total_elems * sizeof(double) <= (512ULL << 20)) {
      st.phi_cache.assign(total_elems, 0.0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
      for (std::ptrdiff_t ib = 0; ib < static_cast<std::ptrdiff_t>(batches.size()); ++ib) {
        const SpatialBatch& batch = batches[static_cast<std::size_t>(ib)];
        const std::size_t ng = batch.npoints;
        const std::size_t na = batch.active_aos.size();
        if (ng == 0 || na == 0) continue;
        const std::size_t comp_stride = na * ng;
        double* dst = st.phi_cache.data() + st.batch_phi_offset[static_cast<std::size_t>(ib)];
        std::size_t local_ao_cursor = 0;
        for (int is : batch.active_shells) {
          const int nm = 2 * bview.shell_l[is] + 1;
          for (std::size_t g = 0; g < ng; ++g) {
            const std::size_t gp = batch.point_offset + g;
            detail::eval_shell_at_point(
                bview, is,
                coords[3 * gp + 0], coords[3 * gp + 1], coords[3 * gp + 2],
                deriv, dst + local_ao_cursor * ng + g, comp_stride, ng);
          }
          local_ao_cursor += static_cast<std::size_t>(nm);
        }
      }
      st.ao_cached = true;
    } else {
      st.batch_phi_offset.clear();
    }
  }

#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
    std::vector<double> phi_local;
    std::vector<double> d_act;
    std::vector<double> t_mat;
    std::vector<double> rho_g, gx_g, gy_g, gz_g, tau_g;

#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
    for (std::ptrdiff_t ib = 0; ib < static_cast<std::ptrdiff_t>(batches.size()); ++ib) {
      const std::size_t uib = static_cast<std::size_t>(ib);
      const SpatialBatch& batch = batches[uib];
      const std::size_t ng = batch.npoints;
      const std::size_t na = batch.active_aos.size();
      if (ng == 0 || na == 0) continue;

      const std::size_t comp_stride = na * ng;
      const double* phi_ptr = nullptr;
      if (st.ao_cached) {
        phi_ptr = st.phi_cache.data() + st.batch_phi_offset[uib];
      } else {
        phi_local.assign(ncomp * comp_stride, 0.0);
        std::size_t local_ao_cursor = 0;
        for (int is : batch.active_shells) {
          const int nm = 2 * bview.shell_l[is] + 1;
          for (std::size_t g = 0; g < ng; ++g) {
            const std::size_t gp = batch.point_offset + g;
            detail::eval_shell_at_point(
                bview, is,
                coords[3 * gp + 0], coords[3 * gp + 1], coords[3 * gp + 2],
                deriv, phi_local.data() + local_ao_cursor * ng + g, comp_stride, ng);
          }
          local_ao_cursor += static_cast<std::size_t>(nm);
        }
        phi_ptr = phi_local.data();
      }

      d_act.resize(na * na);
      for (std::size_t u = 0; u < na; ++u) {
        const std::size_t gu = static_cast<std::size_t>(batch.active_aos[u]);
        for (std::size_t v = 0; v < na; ++v) {
          const std::size_t gv = static_cast<std::size_t>(batch.active_aos[v]);
          d_act[u * na + v] = 0.5 * (h_dm[gu * nao + gv] + h_dm[gv * nao + gu]);
        }
      }

      const int i_na = static_cast<int>(na);
      const int i_ng = static_cast<int>(ng);
      const std::size_t n_tcomp = (xc_family_ == 2) ? 4 : 1;
      t_mat.resize(n_tcomp * comp_stride);
      host_gemm_nn(i_na, i_ng, i_na, d_act.data(), phi_ptr, t_mat.data(), false);
      if (xc_family_ == 2) {
        for (std::size_t b = 1; b <= 3; ++b) {
          host_gemm_nn(i_na, i_ng, i_na, d_act.data(),
                       phi_ptr + b * comp_stride,
                       t_mat.data() + b * comp_stride, false);
        }
      }

      rho_g.assign(ng, 0.0);
      if (xc_family_ >= 1) {
        gx_g.assign(ng, 0.0);
        gy_g.assign(ng, 0.0);
        gz_g.assign(ng, 0.0);
      }
      if (xc_family_ == 2) {
        tau_g.assign(ng, 0.0);
      }

      const double* phi0 = phi_ptr;
      const double* phix = (xc_family_ >= 1) ? (phi_ptr + 1 * comp_stride) : nullptr;
      const double* phiy = (xc_family_ >= 1) ? (phi_ptr + 2 * comp_stride) : nullptr;
      const double* phiz = (xc_family_ >= 1) ? (phi_ptr + 3 * comp_stride) : nullptr;
      const double* t0 = t_mat.data();

      for (std::size_t u = 0; u < na; ++u) {
        for (std::size_t g = 0; g < ng; ++g) {
          const std::size_t off = u * ng + g;
          const double t0_ug = t0[off];
          rho_g[g] += phi0[off] * t0_ug;
          if (xc_family_ >= 1) {
            gx_g[g] += 2.0 * phix[off] * t0_ug;
            gy_g[g] += 2.0 * phiy[off] * t0_ug;
            gz_g[g] += 2.0 * phiz[off] * t0_ug;
          }
          if (xc_family_ == 2) {
            tau_g[g] += 0.5 * (phix[off] * t_mat[1 * comp_stride + off] +
                               phiy[off] * t_mat[2 * comp_stride + off] +
                               phiz[off] * t_mat[3 * comp_stride + off]);
          }
        }
      }

      for (std::size_t g = 0; g < ng; ++g) {
        const std::size_t gp = batch.point_offset + g;
        const double r = rho_g[g];
        const double s = (xc_family_ >= 1)
                             ? (gx_g[g] * gx_g[g] + gy_g[g] * gy_g[g] + gz_g[g] * gz_g[g])
                             : 0.0;
        const double t = (xc_family_ == 2) ? tau_g[g] : 0.0;

        xc_gen::XcPointOut pt;
        xc_gen::eval_xc_point(fid, r, s, t, 2, pt, opts_.rho_cutoff);
        st.v2rho2[gp] = pt.v2rho2;
        if (xc_family_ >= 1) {
          st.g0x[gp] = gx_g[g];
          st.g0y[gp] = gy_g[g];
          st.g0z[gp] = gz_g[g];
          st.vsigma[gp] = pt.vsigma;
          st.v2rhosigma[gp] = pt.v2rhosigma;
          st.v2sigma2[gp] = pt.v2sigma2;
        }
        if (xc_family_ == 2) {
          st.v2rhotau[gp] = pt.v2rhotau;
          st.v2sigmatau[gp] = pt.v2sigmatau;
          st.v2tau2[gp] = pt.v2tau2;
        }
      }
    }
  }
  st.ready = true;
}

void XcEngine::set_reference_density_device(const double* d_dm, void* cuda_stream) {
  if (!xc_codegen_supported_) {
    throw NotImplemented("XcEngine::set_reference_density_device is not implemented for '" +
                         xc_name_ + "'");
  }
  if (active_backend_ != Backend::Cuda) {
    throw std::runtime_error("XcEngine::set_reference_density_device requires Backend::Cuda");
  }
  if (basis_.nao() > 0 && d_dm == nullptr) {
    throw std::invalid_argument("XcEngine::set_reference_density_device: d_dm must not be null");
  }
#if defined(GPU4DFT_HAVE_CUDA)
  detail::xc_cuda_set_reference_density_device(cuda_->impl, d_dm, cuda_stream);
#else
  (void)d_dm;
  (void)cuda_stream;
  throw std::runtime_error("gpu4dft was built without CUDA");
#endif
}

bool XcEngine::has_reference_density() const noexcept {
  if (active_backend_ == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    return cuda_ && cuda_->impl && detail::xc_cuda_has_reference_density(cuda_->impl);
#else
    return false;
#endif
  }
  return host_fxc_ && host_fxc_->ready;
}

std::vector<double> XcEngine::eval_fxc(std::span<const double> h_delta_dm) const {
  const std::size_t n = static_cast<std::size_t>(basis_.nao());
  if (h_delta_dm.size() != n * n) {
    throw std::invalid_argument("XcEngine::eval_fxc: delta_dm size must equal nao * nao");
  }
  return eval_fxc(h_delta_dm.data());
}

std::vector<double> XcEngine::eval_fxc(const double* h_delta_dm) const {
  const std::size_t n = static_cast<std::size_t>(basis_.nao());
  std::vector<double> fxc(n * n, 0.0);
  eval_fxc(h_delta_dm, fxc.data());
  return fxc;
}

void XcEngine::eval_fxc(const double* h_delta_dm, double* h_fxc_out) const {
  const std::size_t nao = static_cast<std::size_t>(basis_.nao());
  if (nao > 0 && h_fxc_out == nullptr) {
    throw std::invalid_argument("XcEngine::eval_fxc: h_fxc_out must not be null");
  }
  if (nao > 0 && h_delta_dm == nullptr) {
    throw std::invalid_argument("XcEngine::eval_fxc: h_delta_dm must not be null");
  }
  if (!has_reference_density()) {
    throw std::logic_error(
        "XcEngine::eval_fxc: reference density not set; call set_reference_density first");
  }
  std::fill_n(h_fxc_out, nao * nao, 0.0);
  if (nao == 0 || grid_.npoints() == 0) return;

  if (active_backend_ == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    detail::xc_cuda_eval_fxc_host(cuda_->impl, h_delta_dm, h_fxc_out);
    return;
#endif
  }

  const HostFxcState& st = *host_fxc_;
  const detail::FlatBasisData flat(basis_);
  const detail::FlatBasisView bview = flat.view();
  const int deriv = (xc_family_ == 0) ? 0 : 1;
  const std::size_t ncomp = (deriv == 0) ? 1 : 4;
  const auto& batches = grid_.batches();
  const auto& coords = grid_.coords();
  const auto& weights = grid_.weights();

#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
    std::vector<double> local_fxc(nao * nao, 0.0);
    std::vector<double> phi_local;
    std::vector<double> d_act;
    std::vector<double> t_mat;
    std::vector<double> w_mat;
    std::vector<double> v_half;
    std::vector<double> drho, dgx, dgy, dgz, dtau, vtau_w;

#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
    for (std::ptrdiff_t ib = 0; ib < static_cast<std::ptrdiff_t>(batches.size()); ++ib) {
      const std::size_t uib = static_cast<std::size_t>(ib);
      const SpatialBatch& batch = batches[uib];
      const std::size_t ng = batch.npoints;
      const std::size_t na = batch.active_aos.size();
      if (ng == 0 || na == 0) continue;

      const std::size_t comp_stride = na * ng;
      const double* phi_ptr = nullptr;
      if (st.ao_cached) {
        phi_ptr = st.phi_cache.data() + st.batch_phi_offset[uib];
      } else {
        phi_local.assign(ncomp * comp_stride, 0.0);
        std::size_t local_ao_cursor = 0;
        for (int is : batch.active_shells) {
          const int nm = 2 * bview.shell_l[is] + 1;
          for (std::size_t g = 0; g < ng; ++g) {
            const std::size_t gp = batch.point_offset + g;
            detail::eval_shell_at_point(
                bview, is,
                coords[3 * gp + 0], coords[3 * gp + 1], coords[3 * gp + 2],
                deriv, phi_local.data() + local_ao_cursor * ng + g, comp_stride, ng);
          }
          local_ao_cursor += static_cast<std::size_t>(nm);
        }
        phi_ptr = phi_local.data();
      }

      d_act.resize(na * na);
      for (std::size_t u = 0; u < na; ++u) {
        const std::size_t gu = static_cast<std::size_t>(batch.active_aos[u]);
        for (std::size_t v = 0; v < na; ++v) {
          const std::size_t gv = static_cast<std::size_t>(batch.active_aos[v]);
          d_act[u * na + v] = 0.5 * (h_delta_dm[gu * nao + gv] + h_delta_dm[gv * nao + gu]);
        }
      }

      const int i_na = static_cast<int>(na);
      const int i_ng = static_cast<int>(ng);
      const std::size_t n_tcomp = (xc_family_ == 2) ? 4 : 1;
      t_mat.resize(n_tcomp * comp_stride);
      host_gemm_nn(i_na, i_ng, i_na, d_act.data(), phi_ptr, t_mat.data(), false);
      if (xc_family_ == 2) {
        for (std::size_t b = 1; b <= 3; ++b) {
          host_gemm_nn(i_na, i_ng, i_na, d_act.data(),
                       phi_ptr + b * comp_stride,
                       t_mat.data() + b * comp_stride, false);
        }
      }

      drho.assign(ng, 0.0);
      if (xc_family_ >= 1) {
        dgx.assign(ng, 0.0);
        dgy.assign(ng, 0.0);
        dgz.assign(ng, 0.0);
      }
      if (xc_family_ == 2) {
        dtau.assign(ng, 0.0);
        vtau_w.assign(ng, 0.0);
      }

      const double* phi0 = phi_ptr;
      const double* phix = (xc_family_ >= 1) ? (phi_ptr + 1 * comp_stride) : nullptr;
      const double* phiy = (xc_family_ >= 1) ? (phi_ptr + 2 * comp_stride) : nullptr;
      const double* phiz = (xc_family_ >= 1) ? (phi_ptr + 3 * comp_stride) : nullptr;
      const double* t0 = t_mat.data();

      for (std::size_t u = 0; u < na; ++u) {
        for (std::size_t g = 0; g < ng; ++g) {
          const std::size_t off = u * ng + g;
          const double t0_ug = t0[off];
          drho[g] += phi0[off] * t0_ug;
          if (xc_family_ >= 1) {
            dgx[g] += 2.0 * phix[off] * t0_ug;
            dgy[g] += 2.0 * phiy[off] * t0_ug;
            dgz[g] += 2.0 * phiz[off] * t0_ug;
          }
          if (xc_family_ == 2) {
            dtau[g] += 0.5 * (phix[off] * t_mat[1 * comp_stride + off] +
                              phiy[off] * t_mat[2 * comp_stride + off] +
                              phiz[off] * t_mat[3 * comp_stride + off]);
          }
        }
      }

      w_mat.assign(comp_stride, 0.0);
      for (std::size_t g = 0; g < ng; ++g) {
        const std::size_t gp = batch.point_offset + g;
        const double wg = weights[gp];
        const double dr = drho[g];

        double d_vrho = st.v2rho2[gp] * dr;
        double wvx = 0.0, wvy = 0.0, wvz = 0.0;
        if (xc_family_ >= 1) {
          const double gx0 = st.g0x[gp];
          const double gy0 = st.g0y[gp];
          const double gz0 = st.g0z[gp];
          const double dsig = 2.0 * (gx0 * dgx[g] + gy0 * dgy[g] + gz0 * dgz[g]);
          d_vrho += st.v2rhosigma[gp] * dsig;
          double d_vsigma = st.v2rhosigma[gp] * dr + st.v2sigma2[gp] * dsig;
          if (xc_family_ == 2) {
            const double dt = dtau[g];
            d_vrho += st.v2rhotau[gp] * dt;
            d_vsigma += st.v2sigmatau[gp] * dt;
            const double d_vtau =
                st.v2rhotau[gp] * dr + st.v2sigmatau[gp] * dsig + st.v2tau2[gp] * dt;
            vtau_w[g] = 0.5 * wg * d_vtau;
          }
          const double vs0 = st.vsigma[gp];
          wvx = 2.0 * wg * (d_vsigma * gx0 + vs0 * dgx[g]);
          wvy = 2.0 * wg * (d_vsigma * gy0 + vs0 * dgy[g]);
          wvz = 2.0 * wg * (d_vsigma * gz0 + vs0 * dgz[g]);
        }
        const double wv0 = 0.5 * wg * d_vrho;

        for (std::size_t u = 0; u < na; ++u) {
          const std::size_t off = u * ng + g;
          double w_ug = wv0 * phi0[off];
          if (xc_family_ >= 1) {
            w_ug += wvx * phix[off] + wvy * phiy[off] + wvz * phiz[off];
          }
          w_mat[off] = w_ug;
        }
      }

      v_half.resize(na * na);
      host_gemm_nt(i_na, i_na, i_ng, phi0, w_mat.data(), v_half.data(), false);

      if (xc_family_ == 2) {
        for (std::size_t b = 1; b <= 3; ++b) {
          const double* phib = phi_ptr + b * comp_stride;
          for (std::size_t u = 0; u < na; ++u) {
            for (std::size_t g = 0; g < ng; ++g) {
              w_mat[u * ng + g] = 0.5 * vtau_w[g] * phib[u * ng + g];
            }
          }
          host_gemm_nt(i_na, i_na, i_ng, phib, w_mat.data(), v_half.data(), true);
        }
      }

      for (std::size_t u = 0; u < na; ++u) {
        const std::size_t gu = static_cast<std::size_t>(batch.active_aos[u]);
        for (std::size_t v = 0; v < na; ++v) {
          const std::size_t gv = static_cast<std::size_t>(batch.active_aos[v]);
          local_fxc[gu * nao + gv] += v_half[u * na + v] + v_half[v * na + u];
        }
      }
    }

#if defined(_OPENMP)
#pragma omp critical
#endif
    {
      for (std::size_t idx = 0; idx < nao * nao; ++idx) {
        h_fxc_out[idx] += local_fxc[idx];
      }
    }
  }
}

void XcEngine::eval_fxc_device(const double* d_delta_dm, double* d_fxc_out,
                               void* cuda_stream) const {
  if (active_backend_ != Backend::Cuda) {
    throw std::runtime_error("XcEngine::eval_fxc_device requires Backend::Cuda");
  }
  if (!has_reference_density()) {
    throw std::logic_error(
        "XcEngine::eval_fxc_device: reference density not set; call set_reference_density_device first");
  }
  if (basis_.nao() > 0 && (d_delta_dm == nullptr || d_fxc_out == nullptr)) {
    throw std::invalid_argument("XcEngine::eval_fxc_device: device pointers must not be null");
  }
#if defined(GPU4DFT_HAVE_CUDA)
  detail::xc_cuda_eval_fxc_device(cuda_->impl, d_delta_dm, d_fxc_out, cuda_stream);
#else
  (void)d_delta_dm;
  (void)d_fxc_out;
  (void)cuda_stream;
  throw std::runtime_error("gpu4dft was built without CUDA");
#endif
}

std::vector<double> XcEngine::nuclear_grad(std::span<const XcAtom> atoms,
                                           std::span<const double> dm) const {
  const std::size_t nao = static_cast<std::size_t>(basis_.nao());
  if (dm.size() != nao * nao) {
    throw std::invalid_argument("XcEngine::nuclear_grad: dm size must equal nao * nao");
  }
  const int natom = static_cast<int>(atoms.size());
  std::vector<double> grad(static_cast<std::size_t>(natom) * 3, 0.0);
  if (natom == 0 || nao == 0 || grid_.npoints() == 0) return grad;

  const std::vector<int> ao_to_atom = build_ao_to_atom(basis_, atoms);

  if (active_backend_ == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    return detail::xc_cuda_nuclear_grad(cuda_->impl, grid_, atoms, ao_to_atom.data(), dm.data());
#endif
  }

  const detail::FlatBasisData flat(basis_);
  const detail::FlatBasisView bview = flat.view();
  const auto fid = static_cast<xc_gen::XcFunctionalId>(xc_func_id_);
  const bool use_libxc_path = opts_.use_libxc || !xc_codegen_supported_;
  const bool do_grid_response = opts_.grid_response && grid_.has_grid_response_data();
  const int deriv = (xc_family_ == 0) ? 1 : 2;
  const std::size_t ncomp = (deriv == 1) ? 4 : 10;
  const auto& batches = grid_.batches();
  const auto& coords = grid_.coords();
  const auto& weights = grid_.weights();
  const auto& parent_atom = grid_.parent_atoms();

  std::vector<double> e_xc_point(do_grid_response ? grid_.npoints() : 0, 0.0);

  static constexpr int kIdx2[3][3] = {
      {4, 5, 6},
      {5, 7, 8},
      {6, 8, 9},
  };

#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
    std::vector<double> local_grad(static_cast<std::size_t>(natom) * 3, 0.0);
    std::vector<double> phi;
    std::vector<double> d_act;
    std::vector<double> f_mat;
    std::vector<double> rho_g, gx_g, gy_g, gz_g, tau_g;
    std::vector<double> wv0, wvx, wvy, wvz, wvt;
    std::vector<double> rho_buf, sigma_buf, tau_buf, zk_acc, vrho_acc, vsigma_acc, vtau_acc;
    std::vector<double> zk_tmp, vrho_tmp, vsigma_tmp, vtau_tmp;

#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
    for (std::ptrdiff_t ib = 0; ib < static_cast<std::ptrdiff_t>(batches.size()); ++ib) {
      const SpatialBatch& batch = batches[static_cast<std::size_t>(ib)];
      const std::size_t ng = batch.npoints;
      const std::size_t na = batch.active_aos.size();
      if (ng == 0 || na == 0) continue;

      const std::size_t comp_stride = na * ng;
      phi.assign(ncomp * comp_stride, 0.0);

      std::size_t local_ao_cursor = 0;
      for (int is : batch.active_shells) {
        const int nm = 2 * bview.shell_l[is] + 1;
        for (std::size_t g = 0; g < ng; ++g) {
          const std::size_t gp = batch.point_offset + g;
          detail::eval_shell_at_point(
              bview, is,
              coords[3 * gp + 0], coords[3 * gp + 1], coords[3 * gp + 2],
              deriv,
              phi.data() + local_ao_cursor * ng + g,
              comp_stride, ng);
        }
        local_ao_cursor += static_cast<std::size_t>(nm);
      }

      d_act.resize(na * na);
      for (std::size_t u = 0; u < na; ++u) {
        const std::size_t gu = static_cast<std::size_t>(batch.active_aos[u]);
        for (std::size_t v = 0; v < na; ++v) {
          const std::size_t gv = static_cast<std::size_t>(batch.active_aos[v]);
          d_act[u * na + v] = 0.5 * (dm[gu * nao + gv] + dm[gv * nao + gu]);
        }
      }

      const int i_na = static_cast<int>(na);
      const int i_ng = static_cast<int>(ng);
      const std::size_t n_fcomp = (xc_family_ >= 1) ? 4 : 1;
      f_mat.resize(n_fcomp * comp_stride);
      for (std::size_t b = 0; b < n_fcomp; ++b) {
        host_gemm_nn(i_na, i_ng, i_na, d_act.data(),
                     phi.data() + b * comp_stride,
                     f_mat.data() + b * comp_stride, false);
      }

      rho_g.assign(ng, 0.0);
      gx_g.assign(ng, 0.0);
      gy_g.assign(ng, 0.0);
      gz_g.assign(ng, 0.0);
      tau_g.assign(ng, 0.0);

      const double* phi0 = phi.data();
      const double* phix = phi.data() + 1 * comp_stride;
      const double* phiy = phi.data() + 2 * comp_stride;
      const double* phiz = phi.data() + 3 * comp_stride;
      const double* f0 = f_mat.data();

      for (std::size_t u = 0; u < na; ++u) {
        for (std::size_t g = 0; g < ng; ++g) {
          const double f0_ug = f0[u * ng + g];
          rho_g[g] += phi0[u * ng + g] * f0_ug;
          if (xc_family_ >= 1) {
            gx_g[g] += 2.0 * phix[u * ng + g] * f0_ug;
            gy_g[g] += 2.0 * phiy[u * ng + g] * f0_ug;
            gz_g[g] += 2.0 * phiz[u * ng + g] * f0_ug;
          }
          if (xc_family_ == 2) {
            tau_g[g] += 0.5 * (phix[u * ng + g] * f_mat[1 * comp_stride + u * ng + g] +
                               phiy[u * ng + g] * f_mat[2 * comp_stride + u * ng + g] +
                               phiz[u * ng + g] * f_mat[3 * comp_stride + u * ng + g]);
          }
        }
      }

      wv0.resize(ng);
      wvx.assign(ng, 0.0);
      wvy.assign(ng, 0.0);
      wvz.assign(ng, 0.0);
      wvt.assign(ng, 0.0);

      if (use_libxc_path) {
        double dummy_exc = 0.0, dummy_nelec = 0.0;
        eval_libxc_rks_chunk(
            *libxc_, ng, weights.data() + batch.point_offset,
            rho_g.data(),
            (xc_family_ >= 1) ? gx_g.data() : nullptr,
            (xc_family_ >= 1) ? gy_g.data() : nullptr,
            (xc_family_ >= 1) ? gz_g.data() : nullptr,
            (xc_family_ == 2) ? tau_g.data() : nullptr,
            /*for_grad=*/true,
            wv0.data(),
            (xc_family_ >= 1) ? wvx.data() : nullptr,
            (xc_family_ >= 1) ? wvy.data() : nullptr,
            (xc_family_ >= 1) ? wvz.data() : nullptr,
            (xc_family_ == 2) ? wvt.data() : nullptr,
            do_grid_response ? (e_xc_point.data() + batch.point_offset) : nullptr,
            dummy_exc, dummy_nelec,
            rho_buf, sigma_buf, tau_buf, zk_acc, vrho_acc, vsigma_acc, vtau_acc,
            zk_tmp, vrho_tmp, vsigma_tmp, vtau_tmp);
      } else {
        for (std::size_t g = 0; g < ng; ++g) {
          const double wg = weights[batch.point_offset + g];
          const double r = rho_g[g];
          const double s = (xc_family_ >= 1)
                               ? (gx_g[g] * gx_g[g] + gy_g[g] * gy_g[g] + gz_g[g] * gz_g[g])
                               : 0.0;
          const double t = (xc_family_ == 2) ? tau_g[g] : 0.0;
          xc_gen::XcPointOut pt;
          xc_gen::eval_xc_point(fid, r, s, t, 1, pt, opts_.rho_cutoff);
          if (do_grid_response) {
            e_xc_point[batch.point_offset + g] = r * pt.zk;
          }
          wv0[g] = wg * pt.vrho;
          if (xc_family_ >= 1) {
            wvx[g] = 2.0 * wg * pt.vsigma * gx_g[g];
            wvy[g] = 2.0 * wg * pt.vsigma * gy_g[g];
            wvz[g] = 2.0 * wg * pt.vsigma * gz_g[g];
          }
          if (xc_family_ == 2) {
            wvt[g] = wg * pt.vtau;
          }
        }
      }

      for (std::size_t u = 0; u < na; ++u) {
        const int ia = ao_to_atom[static_cast<std::size_t>(batch.active_aos[u])];
        if (ia < 0 && !do_grid_response) continue;

        double gx_acc = 0.0, gy_acc = 0.0, gz_acc = 0.0;
        for (std::size_t g = 0; g < ng; ++g) {
          const double f0_ug = f0[u * ng + g];
          double w0_ug = wv0[g] * f0_ug;
          double sx_ug = 0.0, sy_ug = 0.0, sz_ug = 0.0;
          if (xc_family_ >= 1) {
            const double fx_ug = f_mat[1 * comp_stride + u * ng + g];
            const double fy_ug = f_mat[2 * comp_stride + u * ng + g];
            const double fz_ug = f_mat[3 * comp_stride + u * ng + g];
            w0_ug += wvx[g] * fx_ug + wvy[g] * fy_ug + wvz[g] * fz_ug;

            const double wb_x = 2.0 * wvx[g] * f0_ug + ((xc_family_ == 2) ? wvt[g] * fx_ug : 0.0);
            const double wb_y = 2.0 * wvy[g] * f0_ug + ((xc_family_ == 2) ? wvt[g] * fy_ug : 0.0);
            const double wb_z = 2.0 * wvz[g] * f0_ug + ((xc_family_ == 2) ? wvt[g] * fz_ug : 0.0);
            const double wb[3] = {wb_x, wb_y, wb_z};

            double sec[3] = {0.0, 0.0, 0.0};
            for (int c = 0; c < 3; ++c) {
              for (int b = 0; b < 3; ++b) {
                sec[c] += phi[static_cast<std::size_t>(kIdx2[b][c]) * comp_stride + u * ng + g] * wb[b];
              }
            }
            sx_ug = sec[0];
            sy_ug = sec[1];
            sz_ug = sec[2];
          }
          const double fx_tot = 2.0 * phix[u * ng + g] * w0_ug + sx_ug;
          const double fy_tot = 2.0 * phiy[u * ng + g] * w0_ug + sy_ug;
          const double fz_tot = 2.0 * phiz[u * ng + g] * w0_ug + sz_ug;

          gx_acc -= fx_tot;
          gy_acc -= fy_tot;
          gz_acc -= fz_tot;

          if (do_grid_response) {
            const int ib_atom = parent_atom[batch.point_offset + g];
            if (ib_atom >= 0 && ib_atom < natom) {
              local_grad[static_cast<std::size_t>(ib_atom) * 3 + 0] += fx_tot;
              local_grad[static_cast<std::size_t>(ib_atom) * 3 + 1] += fy_tot;
              local_grad[static_cast<std::size_t>(ib_atom) * 3 + 2] += fz_tot;
            }
          }
        }
        if (ia >= 0) {
          local_grad[static_cast<std::size_t>(ia) * 3 + 0] += gx_acc;
          local_grad[static_cast<std::size_t>(ia) * 3 + 1] += gy_acc;
          local_grad[static_cast<std::size_t>(ia) * 3 + 2] += gz_acc;
        }
      }
    }

#if defined(_OPENMP)
#pragma omp critical
#endif
    {
      for (std::size_t k = 0; k < local_grad.size(); ++k) {
        grad[k] += local_grad[k];
      }
    }
  }

  if (do_grid_response) {
    detail::compute_partition_weight_grad_host(grid_, atoms, e_xc_point, grad);
  }
  return grad;
}

std::vector<double> XcEngine::nuclear_grad_uks(std::span<const XcAtom> atoms,
                                               std::span<const double> dm_alpha,
                                               std::span<const double> dm_beta) const {
  const std::size_t nao = static_cast<std::size_t>(basis_.nao());
  if (dm_alpha.size() != nao * nao || dm_beta.size() != nao * nao) {
    throw std::invalid_argument(
        "XcEngine::nuclear_grad_uks: dm_alpha and dm_beta sizes must equal nao * nao");
  }
  const int natom = static_cast<int>(atoms.size());
  std::vector<double> grad(static_cast<std::size_t>(natom) * 3, 0.0);
  if (natom == 0 || nao == 0 || grid_.npoints() == 0) return grad;

  const std::vector<int> ao_to_atom = build_ao_to_atom(basis_, atoms);

  if (active_backend_ == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    return detail::xc_cuda_nuclear_grad_uks(
        cuda_->impl, grid_, atoms, ao_to_atom.data(), dm_alpha.data(), dm_beta.data());
#endif
  }

  const detail::FlatBasisData flat(basis_);
  const detail::FlatBasisView bview = flat.view();
  const bool do_grid_response = opts_.grid_response && grid_.has_grid_response_data();
  const int deriv = (xc_family_ == 0) ? 1 : 2;
  const std::size_t ncomp = (deriv == 1) ? 4 : 10;
  const auto& batches = grid_.batches();
  const auto& coords = grid_.coords();
  const auto& weights = grid_.weights();
  const auto& parent_atom = grid_.parent_atoms();

  std::vector<double> e_xc_point(do_grid_response ? grid_.npoints() : 0, 0.0);

  static constexpr int kIdx2[3][3] = {
      {4, 5, 6},
      {5, 7, 8},
      {6, 8, 9},
  };

#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
    std::vector<double> local_grad(static_cast<std::size_t>(natom) * 3, 0.0);
    std::vector<double> phi;
    std::vector<double> d_act_a, d_act_b;
    std::vector<double> f_mat_a, f_mat_b;
    std::vector<double> rho_a, rho_b, gxa, gya, gza, gxb, gyb, gzb, tau_a, tau_b;
    std::vector<double> wv0_a, wv0_b, wvx_a, wvy_a, wvz_a, wvx_b, wvy_b, wvz_b, wvt_a, wvt_b;
    std::vector<double> rho2_buf, sigma3_buf, tau2_buf;
    std::vector<double> zk_acc, vrho2_acc, vsigma3_acc, vtau2_acc;
    std::vector<double> zk_tmp, vrho2_tmp, vsigma3_tmp, vtau2_tmp;

#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
    for (std::ptrdiff_t ib = 0; ib < static_cast<std::ptrdiff_t>(batches.size()); ++ib) {
      const SpatialBatch& batch = batches[static_cast<std::size_t>(ib)];
      const std::size_t ng = batch.npoints;
      const std::size_t na = batch.active_aos.size();
      if (ng == 0 || na == 0) continue;

      const std::size_t comp_stride = na * ng;
      phi.assign(ncomp * comp_stride, 0.0);

      std::size_t local_ao_cursor = 0;
      for (int is : batch.active_shells) {
        const int nm = 2 * bview.shell_l[is] + 1;
        for (std::size_t g = 0; g < ng; ++g) {
          const std::size_t gp = batch.point_offset + g;
          detail::eval_shell_at_point(
              bview, is,
              coords[3 * gp + 0], coords[3 * gp + 1], coords[3 * gp + 2],
              deriv, phi.data() + local_ao_cursor * ng + g, comp_stride, ng);
        }
        local_ao_cursor += static_cast<std::size_t>(nm);
      }

      d_act_a.resize(na * na);
      d_act_b.resize(na * na);
      for (std::size_t u = 0; u < na; ++u) {
        const std::size_t gu = static_cast<std::size_t>(batch.active_aos[u]);
        for (std::size_t v = 0; v < na; ++v) {
          const std::size_t gv = static_cast<std::size_t>(batch.active_aos[v]);
          d_act_a[u * na + v] = 0.5 * (dm_alpha[gu * nao + gv] + dm_alpha[gv * nao + gu]);
          d_act_b[u * na + v] = 0.5 * (dm_beta[gu * nao + gv] + dm_beta[gv * nao + gu]);
        }
      }

      const int i_na = static_cast<int>(na);
      const int i_ng = static_cast<int>(ng);
      const std::size_t n_fcomp = (xc_family_ >= 1) ? 4 : 1;
      f_mat_a.resize(n_fcomp * comp_stride);
      f_mat_b.resize(n_fcomp * comp_stride);
      for (std::size_t b = 0; b < n_fcomp; ++b) {
        host_gemm_nn(i_na, i_ng, i_na, d_act_a.data(),
                     phi.data() + b * comp_stride,
                     f_mat_a.data() + b * comp_stride, false);
        host_gemm_nn(i_na, i_ng, i_na, d_act_b.data(),
                     phi.data() + b * comp_stride,
                     f_mat_b.data() + b * comp_stride, false);
      }

      rho_a.assign(ng, 0.0);
      rho_b.assign(ng, 0.0);
      gxa.assign(ng, 0.0);
      gya.assign(ng, 0.0);
      gza.assign(ng, 0.0);
      gxb.assign(ng, 0.0);
      gyb.assign(ng, 0.0);
      gzb.assign(ng, 0.0);
      tau_a.assign(ng, 0.0);
      tau_b.assign(ng, 0.0);

      const double* phi0 = phi.data();
      const double* phix = phi.data() + 1 * comp_stride;
      const double* phiy = phi.data() + 2 * comp_stride;
      const double* phiz = phi.data() + 3 * comp_stride;
      const double* f0a = f_mat_a.data();
      const double* f0b = f_mat_b.data();

      for (std::size_t u = 0; u < na; ++u) {
        for (std::size_t g = 0; g < ng; ++g) {
          const std::size_t off = u * ng + g;
          const double p0 = phi0[off];
          const double f0a_ug = f0a[off];
          const double f0b_ug = f0b[off];
          rho_a[g] += p0 * f0a_ug;
          rho_b[g] += p0 * f0b_ug;
          if (xc_family_ >= 1) {
            const double px = phix[off];
            const double py = phiy[off];
            const double pz = phiz[off];
            gxa[g] += 2.0 * px * f0a_ug;
            gya[g] += 2.0 * py * f0a_ug;
            gza[g] += 2.0 * pz * f0a_ug;
            gxb[g] += 2.0 * px * f0b_ug;
            gyb[g] += 2.0 * py * f0b_ug;
            gzb[g] += 2.0 * pz * f0b_ug;
            if (xc_family_ == 2) {
              tau_a[g] += 0.5 * (px * f_mat_a[1 * comp_stride + off] +
                                 py * f_mat_a[2 * comp_stride + off] +
                                 pz * f_mat_a[3 * comp_stride + off]);
              tau_b[g] += 0.5 * (px * f_mat_b[1 * comp_stride + off] +
                                 py * f_mat_b[2 * comp_stride + off] +
                                 pz * f_mat_b[3 * comp_stride + off]);
            }
          }
        }
      }

      wv0_a.resize(ng);
      wv0_b.resize(ng);
      wvx_a.assign(ng, 0.0);
      wvy_a.assign(ng, 0.0);
      wvz_a.assign(ng, 0.0);
      wvx_b.assign(ng, 0.0);
      wvy_b.assign(ng, 0.0);
      wvz_b.assign(ng, 0.0);
      wvt_a.assign(ng, 0.0);
      wvt_b.assign(ng, 0.0);

      double dummy_exc = 0.0, dummy_na = 0.0, dummy_nb = 0.0;
      eval_libxc_uks_chunk(
          *libxc_, ng, weights.data() + batch.point_offset,
          rho_a.data(), rho_b.data(),
          (xc_family_ >= 1) ? gxa.data() : nullptr,
          (xc_family_ >= 1) ? gya.data() : nullptr,
          (xc_family_ >= 1) ? gza.data() : nullptr,
          (xc_family_ >= 1) ? gxb.data() : nullptr,
          (xc_family_ >= 1) ? gyb.data() : nullptr,
          (xc_family_ >= 1) ? gzb.data() : nullptr,
          (xc_family_ == 2) ? tau_a.data() : nullptr,
          (xc_family_ == 2) ? tau_b.data() : nullptr,
          /*for_grad=*/true,
          wv0_a.data(), wv0_b.data(),
          (xc_family_ >= 1) ? wvx_a.data() : nullptr,
          (xc_family_ >= 1) ? wvy_a.data() : nullptr,
          (xc_family_ >= 1) ? wvz_a.data() : nullptr,
          (xc_family_ >= 1) ? wvx_b.data() : nullptr,
          (xc_family_ >= 1) ? wvy_b.data() : nullptr,
          (xc_family_ >= 1) ? wvz_b.data() : nullptr,
          (xc_family_ == 2) ? wvt_a.data() : nullptr,
          (xc_family_ == 2) ? wvt_b.data() : nullptr,
          do_grid_response ? (e_xc_point.data() + batch.point_offset) : nullptr,
          dummy_exc, dummy_na, dummy_nb,
          rho2_buf, sigma3_buf, tau2_buf,
          zk_acc, vrho2_acc, vsigma3_acc, vtau2_acc,
          zk_tmp, vrho2_tmp, vsigma3_tmp, vtau2_tmp);

      for (std::size_t u = 0; u < na; ++u) {
        const int ia = ao_to_atom[static_cast<std::size_t>(batch.active_aos[u])];
        if (ia < 0 && !do_grid_response) continue;

        double gx_acc = 0.0, gy_acc = 0.0, gz_acc = 0.0;
        for (std::size_t g = 0; g < ng; ++g) {
          const std::size_t off = u * ng + g;
          const double f0a_ug = f0a[off];
          const double f0b_ug = f0b[off];
          double w0_ug = wv0_a[g] * f0a_ug + wv0_b[g] * f0b_ug;
          double sx_ug = 0.0, sy_ug = 0.0, sz_ug = 0.0;
          if (xc_family_ >= 1) {
            const double fxa = f_mat_a[1 * comp_stride + off];
            const double fya = f_mat_a[2 * comp_stride + off];
            const double fza = f_mat_a[3 * comp_stride + off];
            const double fxb = f_mat_b[1 * comp_stride + off];
            const double fyb = f_mat_b[2 * comp_stride + off];
            const double fzb = f_mat_b[3 * comp_stride + off];

            w0_ug += wvx_a[g] * fxa + wvy_a[g] * fya + wvz_a[g] * fza +
                     wvx_b[g] * fxb + wvy_b[g] * fyb + wvz_b[g] * fzb;

            const double wb_x = 2.0 * (wvx_a[g] * f0a_ug + wvx_b[g] * f0b_ug) +
                                ((xc_family_ == 2) ? (wvt_a[g] * fxa + wvt_b[g] * fxb) : 0.0);
            const double wb_y = 2.0 * (wvy_a[g] * f0a_ug + wvy_b[g] * f0b_ug) +
                                ((xc_family_ == 2) ? (wvt_a[g] * fya + wvt_b[g] * fyb) : 0.0);
            const double wb_z = 2.0 * (wvz_a[g] * f0a_ug + wvz_b[g] * f0b_ug) +
                                ((xc_family_ == 2) ? (wvt_a[g] * fza + wvt_b[g] * fzb) : 0.0);
            const double wb[3] = {wb_x, wb_y, wb_z};

            double sec[3] = {0.0, 0.0, 0.0};
            for (int c = 0; c < 3; ++c) {
              for (int b = 0; b < 3; ++b) {
                sec[c] += phi[static_cast<std::size_t>(kIdx2[b][c]) * comp_stride + off] * wb[b];
              }
            }
            sx_ug = sec[0];
            sy_ug = sec[1];
            sz_ug = sec[2];
          }
          const double fx_tot = 2.0 * phix[off] * w0_ug + sx_ug;
          const double fy_tot = 2.0 * phiy[off] * w0_ug + sy_ug;
          const double fz_tot = 2.0 * phiz[off] * w0_ug + sz_ug;

          gx_acc -= fx_tot;
          gy_acc -= fy_tot;
          gz_acc -= fz_tot;

          if (do_grid_response) {
            const int ib_atom = parent_atom[batch.point_offset + g];
            if (ib_atom >= 0 && ib_atom < natom) {
              local_grad[static_cast<std::size_t>(ib_atom) * 3 + 0] += fx_tot;
              local_grad[static_cast<std::size_t>(ib_atom) * 3 + 1] += fy_tot;
              local_grad[static_cast<std::size_t>(ib_atom) * 3 + 2] += fz_tot;
            }
          }
        }
        if (ia >= 0) {
          local_grad[static_cast<std::size_t>(ia) * 3 + 0] += gx_acc;
          local_grad[static_cast<std::size_t>(ia) * 3 + 1] += gy_acc;
          local_grad[static_cast<std::size_t>(ia) * 3 + 2] += gz_acc;
        }
      }
    }

#if defined(_OPENMP)
#pragma omp critical
#endif
    {
      for (std::size_t k = 0; k < local_grad.size(); ++k) {
        grad[k] += local_grad[k];
      }
    }
  }

  if (do_grid_response) {
    detail::compute_partition_weight_grad_host(grid_, atoms, e_xc_point, grad);
  }
  return grad;
}

std::vector<double> xc_nuclear_grad(const Basis& basis,
                                    const MolecularGrid& grid,
                                    std::span<const XcAtom> atoms,
                                    std::string_view xc_name,
                                    const double* dm,
                                    Backend backend,
                                    bool grid_response) {
  XcOptions opts;
  opts.backend = backend;
  opts.grid_response = grid_response;
  XcEngine engine(basis, grid, xc_name, opts);
  const std::size_t nao = static_cast<std::size_t>(basis.nao());
  return engine.nuclear_grad(atoms, std::span<const double>(dm, nao * nao));
}

}  // namespace gpu4dft
