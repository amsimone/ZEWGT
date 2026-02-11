#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "EWPOZ.h"
#include "EWPOZ2.h"
#include "SMval.h"
#include "SMvalG.h"
#include "classes.h"
#include "xscnnlo.h"

using namespace std;
using namespace griffin;

namespace {

struct Vec4 {
  Vec4() : e(0.0), px(0.0), py(0.0), pz(0.0) {}
  Vec4(double e_in, double px_in, double py_in, double pz_in)
      : e(e_in), px(px_in), py(py_in), pz(pz_in) {}

  double e = 0.0;
  double px = 0.0;
  double py = 0.0;
  double pz = 0.0;

  Vec4 operator+(const Vec4& o) const {
    return Vec4(e + o.e, px + o.px, py + o.py, pz + o.pz);
  }

  Vec4 operator-(const Vec4& o) const {
    return Vec4(e - o.e, px - o.px, py - o.py, pz - o.pz);
  }

  double m2() const { return e * e - (px * px + py * py + pz * pz); }
};

struct Particle {
  int id = 0;
  int status = 0;
  int m1 = 0;
  int m2 = 0;
  double px = 0.0;
  double py = 0.0;
  double pz = 0.0;
  double e = 0.0;
};

struct Card {
  map<string, string> str;
  map<string, double> num;
};

struct UubData {
  bool has_p1 = false;
  bool has_p2 = false;
  bool has_p3 = false;
  bool has_p4 = false;
  bool has_flav = false;
  Vec4 p1;
  Vec4 p2;
  Vec4 p3;
  Vec4 p4;
  array<int, 4> flav{{0, 0, 0, 0}};
};

struct QuarkAssignment {
  bool ok = false;
  int abs_flav = 0;
  bool quark_from_beam1 = true;
};

struct LeptonPair {
  bool ok = false;
  Particle lminus;
  Particle lplus;
  int outtype = 0;
};

struct Projection {
  bool ok = false;
  int intype = 0;
  int outtype = 0;
  double shat = numeric_limits<double>::quiet_NaN();
  double costh = numeric_limits<double>::quiet_NaN();
  string source;
  string detail;
};

struct EventInfo {
  vector<Particle> particles;
  int rwgt_type = -1;
  int rwgt_index = -1;
  UubData uub;
};

struct Stats {
  long long nevents = 0;
  long long n_uub = 0;
  long long n_rwgt_flavreg = 0;
  long long n_incoming = 0;
  long long n_unresolved = 0;
};

static string ltrim(string s) {
  s.erase(s.begin(), find_if(s.begin(), s.end(), [](unsigned char c) {
    return !isspace(c);
  }));
  return s;
}

static string rtrim(string s) {
  s.erase(find_if(s.rbegin(), s.rend(), [](unsigned char c) {
    return !isspace(c);
  }).base(), s.end());
  return s;
}

static string trim(string s) { return rtrim(ltrim(s)); }

static string tolower_str(string s) {
  transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(tolower(c));
  });
  return s;
}

static vector<string> split_ws(const string& s) {
  istringstream iss(s);
  vector<string> out;
  string tok;
  while (iss >> tok) out.push_back(tok);
  return out;
}

static bool parse_double(const string& s, double& out) {
  char* endptr = nullptr;
  out = strtod(s.c_str(), &endptr);
  if (endptr == s.c_str()) return false;
  while (*endptr != '\0') {
    if (!isspace(static_cast<unsigned char>(*endptr))) return false;
    ++endptr;
  }
  return true;
}

static Card load_card(const string& path) {
  ifstream in(path.c_str());
  if (!in) {
    cerr << "Failed to open card: " << path << "\n";
    exit(1);
  }
  Card card;
  string line;
  while (getline(in, line)) {
    size_t cpos = line.find('#');
    size_t ep = line.find('!');
    if (ep != string::npos && (cpos == string::npos || ep < cpos)) cpos = ep;
    if (cpos != string::npos) line = line.substr(0, cpos);
    line = trim(line);
    if (line.empty()) continue;

    string key, val;
    size_t eq = line.find('=');
    if (eq != string::npos) {
      key = trim(line.substr(0, eq));
      val = trim(line.substr(eq + 1));
    } else {
      istringstream iss(line);
      iss >> key;
      if (!(iss >> val)) continue;
    }

    key = tolower_str(key);
    card.str[key] = val;

    double dval = 0.0;
    if (parse_double(val, dval)) card.num[key] = dval;
  }
  return card;
}

static double get_num(const Card& card, const string& key, bool required) {
  auto it = card.num.find(key);
  if (it != card.num.end()) return it->second;
  if (required) {
    cerr << "Missing numeric parameter in card: " << key << "\n";
    exit(1);
  }
  return numeric_limits<double>::quiet_NaN();
}

static string get_str(const Card& card, const string& key) {
  auto it = card.str.find(key);
  if (it != card.str.end()) return tolower_str(it->second);
  return "";
}

static int parse_fermion(const string& s) {
  string v = tolower_str(s);
  if (v == "e" || v == "ele" || v == "electron" || v == "11") return ELE;
  if (v == "mu" || v == "muo" || v == "muon" || v == "13") return MUO;
  if (v == "tau" || v == "15") return TAU;
  if (v == "u" || v == "2") return UQU;
  if (v == "d" || v == "1") return DQU;
  if (v == "s" || v == "3") return SQU;
  if (v == "c" || v == "4") return CQU;
  if (v == "b" || v == "5") return BQU;
  return 0;
}

static int pdg_quark_to_griffin(int abs_pdg) {
  switch (abs_pdg) {
    case 1:
      return DQU;
    case 2:
      return UQU;
    case 3:
      return SQU;
    case 4:
      return CQU;
    case 5:
      return BQU;
    default:
      return 0;
  }
}

template <class MatT>
static double unpolarized_xsec(MatT& m, double cost) {
  Cplx resvv, resav, resva, resaa;
  m.setform(VEC, VEC);
  resvv = m.result();
  m.setform(AXV, VEC);
  resav = m.result();
  m.setform(VEC, AXV);
  resva = m.result();
  m.setform(AXV, AXV);
  resaa = m.result();

  Cplx sum = (1 + cost * cost) * (resvv * conj(resvv) + resav * conj(resav)
                                  + resva * conj(resva) + resaa * conj(resaa))
             + 4 * cost * (resvv * conj(resaa) + resva * conj(resav));
  return real(sum);
}

static void set_param(inval& input, const Card& card, const string& key,
                      int idx, bool required) {
  double v = get_num(card, key, required);
  if (isfinite(v)) input.set(idx, v);
}

static void apply_common_inputs(inval& input, const Card& card, bool require_mw) {
  set_param(input, card, "al", al, true);
  set_param(input, card, "alpha", al, false);
  set_param(input, card, "alphaem", al, false);
  set_param(input, card, "als", als, true);
  set_param(input, card, "delal", Delal, true);

  set_param(input, card, "mz", MZ, true);
  if (require_mw) set_param(input, card, "mw", MW, true);
  set_param(input, card, "gamz", GamZ, true);
  if (require_mw) set_param(input, card, "gamw", GamW, true);
  set_param(input, card, "mh", MH, true);
  set_param(input, card, "mt", MT, true);
  set_param(input, card, "mb", MB, true);
  set_param(input, card, "mc", MC, true);
  set_param(input, card, "ms", MS, true);
  set_param(input, card, "mu", MU, true);
  set_param(input, card, "md", MD, true);
  set_param(input, card, "ml", ML, true);
  set_param(input, card, "mm", MM, true);
  set_param(input, card, "me", ME, true);
}

class GriffinWeightComputer {
 public:
  GriffinWeightComputer(const Card& card, const string& mode_override,
                        const string& scheme_override)
      : mode_(mode_override), scheme_(scheme_override) {
    if (mode_.empty()) mode_ = get_str(card, "mode");
    if (scheme_.empty()) scheme_ = get_str(card, "ewscheme");
    if (mode_.empty()) mode_ = "nlo";
    if (scheme_.empty()) scheme_ = "alpha";

    if (scheme_ == "gmu") {
      unique_ptr<SMvalGmu> inp(new SMvalGmu());
      apply_common_inputs(*inp, card, false);
      set_param(*inp, card, "gmu", Gmu, true);
      input_ = move(inp);
    } else if (scheme_ == "alpha") {
      unique_ptr<SMval> inp(new SMval());
      apply_common_inputs(*inp, card, true);
      input_ = move(inp);
    } else {
      cerr << "Unsupported ewscheme: " << scheme_ << "\n";
      exit(1);
    }

    if (mode_ != "nlo" && mode_ != "nnlo") {
      cerr << "Unsupported mode: " << mode_ << " (use nlo or nnlo)\n";
      exit(1);
    }
  }

  double compute_weight(int intype, int outtype, double shat, double costh) const {
    if (!isfinite(shat) || !isfinite(costh) || shat <= 0.0 || fabs(costh) > 1.0) {
      return numeric_limits<double>::quiet_NaN();
    }

    FA_SMLO fai_lo(intype, *input_), faf_lo(outtype, *input_);
    SW_SMLO swi_lo(intype, *input_), swf_lo(outtype, *input_);
    matel mlo(intype, outtype, VEC, VEC, fai_lo, faf_lo, swi_lo, swf_lo, shat, costh,
              *input_);
    double xsec_lo = unpolarized_xsec(mlo, costh);
    if (xsec_lo == 0.0) return numeric_limits<double>::quiet_NaN();

    double xsec_corr = 0.0;
    if (mode_ == "nlo") {
      FA_SMNLO fai(intype, *input_), faf(outtype, *input_);
      SW_SMNLO swi(intype, *input_), swf(outtype, *input_);
      matel mnlo(intype, outtype, VEC, VEC, fai, faf, swi, swf, shat, costh, *input_);
      xsec_corr = unpolarized_xsec(mnlo, costh);
    } else {
      FA_SMNNLO fai(intype, *input_), faf(outtype, *input_);
      SW_SMNNLO swi(intype, *input_), swf(outtype, *input_);
      mat_SMNNLO mnnlo(intype, outtype, VEC, VEC, fai, faf, swi, swf, shat, costh,
                       *input_);
      xsec_corr = unpolarized_xsec(mnnlo, costh);
    }

    return xsec_corr / xsec_lo;
  }

  const string& mode() const { return mode_; }
  const string& scheme() const { return scheme_; }

 private:
  string mode_;
  string scheme_;
  unique_ptr<inval> input_;
};

class FlavRegMap {
 public:
  bool load(const string& path) {
    ifstream in(path.c_str());
    if (!in) return false;

    uborn_in_.clear();
    alr_to_uborn_.clear();
    uborn_in_.push_back({0, 0});
    alr_to_uborn_.push_back(0);

    int real_idx = 0;
    int pending_real_idx = 0;

    string line;
    while (getline(in, line)) {
      if (line.find("==>") == string::npos) continue;

      int f1 = 0, f2 = 0;
      if (!parse_channel_incoming(line, f1, f2)) continue;

      bool is_uborn = (line.find("uborn") != string::npos);
      if (is_uborn) {
        uborn_in_.push_back({f1, f2});
        int ubidx = static_cast<int>(uborn_in_.size()) - 1;
        // In FlavRegList layout, a real/alr entry is followed by its underlying Born
        // (uborn) channel. Keep that association so #rwgt ALR indices can be mapped to
        // the corresponding Born incoming flavors.
        if (pending_real_idx > 0) {
          if (static_cast<int>(alr_to_uborn_.size()) <= pending_real_idx) {
            alr_to_uborn_.resize(pending_real_idx + 1, 0);
          }
          alr_to_uborn_[pending_real_idx] = ubidx;
          pending_real_idx = 0;
        }
      } else {
        real_idx++;
        if (static_cast<int>(alr_to_uborn_.size()) <= real_idx) {
          alr_to_uborn_.resize(real_idx + 1, 0);
        }
        pending_real_idx = real_idx;
      }
    }

    return uborn_in_.size() > 1;
  }

  bool uborn_incoming(int uborn_idx, int& f1, int& f2) const {
    if (uborn_idx <= 0 || uborn_idx >= static_cast<int>(uborn_in_.size())) return false;
    f1 = uborn_in_[uborn_idx].first;
    f2 = uborn_in_[uborn_idx].second;
    return true;
  }

  bool rwgt_to_incoming(int rwgt_type, int rwgt_index, int& f1, int& f2) const {
    if (rwgt_type == 1) {
      return uborn_incoming(rwgt_index, f1, f2);
    }
    if (rwgt_type == 2 || rwgt_type == 3) {
      if (rwgt_index <= 0 || rwgt_index >= static_cast<int>(alr_to_uborn_.size())) {
        return false;
      }
      int ubidx = alr_to_uborn_[rwgt_index];
      return uborn_incoming(ubidx, f1, f2);
    }
    return false;
  }

 private:
  static string sanitize_token(string tok) {
    string out;
    for (char c : tok) {
      if (isalnum(static_cast<unsigned char>(c)) || c == '~' || c == '+' || c == '-') {
        out.push_back(c);
      }
    }
    return out;
  }

  static int parse_parton_token(const string& tok) {
    string t = tolower_str(sanitize_token(tok));
    if (t == "g") return 21;
    if (t == "d") return 1;
    if (t == "d~") return -1;
    if (t == "u") return 2;
    if (t == "u~") return -2;
    if (t == "s") return 3;
    if (t == "s~") return -3;
    if (t == "c") return 4;
    if (t == "c~") return -4;
    if (t == "b") return 5;
    if (t == "b~") return -5;
    if (t == "t") return 6;
    if (t == "t~") return -6;
    return 0;
  }

  static bool parse_channel_incoming(const string& line, int& f1, int& f2) {
    size_t p = line.find("==>");
    if (p == string::npos) return false;
    string left = line.substr(0, p);
    vector<string> toks = split_ws(left);

    vector<int> found;
    found.reserve(2);
    for (const string& tok : toks) {
      int flav = parse_parton_token(tok);
      if (flav != 0) {
        found.push_back(flav);
        if (found.size() == 2) break;
      }
    }

    if (found.size() != 2) return false;
    f1 = found[0];
    f2 = found[1];
    return true;
  }

  vector<pair<int, int>> uborn_in_;
  vector<int> alr_to_uborn_;
};

static bool is_quark(int id) { return id > 0 && id <= 6; }
static bool is_antiquark(int id) { return id < 0 && id >= -6; }
static bool is_gluon(int id) { return id == 21; }

static QuarkAssignment infer_quark_assignment(int f1, int f2) {
  QuarkAssignment qa;

  if (is_quark(f1) && is_antiquark(f2) && abs(f1) == abs(f2)) {
    qa.ok = true;
    qa.abs_flav = abs(f1);
    qa.quark_from_beam1 = true;
    return qa;
  }
  if (is_antiquark(f1) && is_quark(f2) && abs(f1) == abs(f2)) {
    qa.ok = true;
    qa.abs_flav = abs(f2);
    qa.quark_from_beam1 = false;
    return qa;
  }

  if (is_quark(f1) && is_gluon(f2)) {
    qa.ok = true;
    qa.abs_flav = abs(f1);
    qa.quark_from_beam1 = true;
    return qa;
  }
  if (is_gluon(f1) && is_quark(f2)) {
    qa.ok = true;
    qa.abs_flav = abs(f2);
    qa.quark_from_beam1 = false;
    return qa;
  }

  if (is_antiquark(f1) && is_gluon(f2)) {
    qa.ok = true;
    qa.abs_flav = abs(f1);
    qa.quark_from_beam1 = false;
    return qa;
  }
  if (is_gluon(f1) && is_antiquark(f2)) {
    qa.ok = true;
    qa.abs_flav = abs(f2);
    qa.quark_from_beam1 = true;
    return qa;
  }

  return qa;
}

static vector<double> extract_numbers(const string& s) {
  string cleaned = s;
  for (char& c : cleaned) {
    if (c == 'd' || c == 'D') c = 'E';
  }

  vector<double> vals;
  const char* cur = cleaned.c_str();
  while (*cur != '\0') {
    char* endptr = nullptr;
    double v = strtod(cur, &endptr);
    if (endptr != cur) {
      vals.push_back(v);
      cur = endptr;
    } else {
      ++cur;
    }
  }
  return vals;
}

static Vec4 particle_p4(const Particle& p) { return Vec4(p.e, p.px, p.py, p.pz); }

static int lept_pdg_to_outtype(int abs_pdg) {
  if (abs_pdg == 11) return ELE;
  if (abs_pdg == 13) return MUO;
  if (abs_pdg == 15) return TAU;
  return 0;
}

static bool is_charged_lepton(int id) {
  int a = abs(id);
  return a == 11 || a == 13 || a == 15;
}

static LeptonPair find_leptons(const vector<Particle>& particles) {
  LeptonPair out;
  int z_index = -1;
  for (size_t i = 0; i < particles.size(); ++i) {
    if (particles[i].id == 23) {
      z_index = static_cast<int>(i) + 1;  // LHE is 1-based
      break;
    }
  }

  vector<const Particle*> cand;
  if (z_index > 0) {
    for (const auto& p : particles) {
      if (p.status != 1 || !is_charged_lepton(p.id)) continue;
      if (p.m1 == z_index || p.m2 == z_index) cand.push_back(&p);
    }
  }

  if (cand.size() < 2) {
    cand.clear();
    for (const auto& p : particles) {
      if (p.status == 1 && is_charged_lepton(p.id)) cand.push_back(&p);
    }
  }

  double best_dm = numeric_limits<double>::infinity();
  const Particle* best_a = nullptr;
  const Particle* best_b = nullptr;

  for (size_t i = 0; i < cand.size(); ++i) {
    for (size_t j = i + 1; j < cand.size(); ++j) {
      int id1 = cand[i]->id;
      int id2 = cand[j]->id;
      if (id1 * id2 >= 0) continue;
      if (abs(id1) != abs(id2)) continue;
      Vec4 q = particle_p4(*cand[i]) + particle_p4(*cand[j]);
      double m2 = q.m2();
      if (m2 <= 0) continue;
      double dm = fabs(sqrt(m2) - 91.1876);
      if (dm < best_dm) {
        best_dm = dm;
        best_a = cand[i];
        best_b = cand[j];
      }
    }
  }

  if (!best_a || !best_b) return out;

  if (best_a->id > 0) {
    out.lminus = *best_a;
    out.lplus = *best_b;
  } else {
    out.lminus = *best_b;
    out.lplus = *best_a;
  }

  out.outtype = lept_pdg_to_outtype(abs(out.lminus.id));
  out.ok = (out.outtype != 0);
  return out;
}

static Vec4 boost(const Vec4& p, double bx, double by, double bz) {
  double b2 = bx * bx + by * by + bz * bz;
  if (b2 >= 1.0) return p;
  double gamma = 1.0 / sqrt(1.0 - b2);
  double bp = bx * p.px + by * p.py + bz * p.pz;
  double gamma2 = (b2 > 0.0) ? (gamma - 1.0) / b2 : 0.0;

  Vec4 out;
  out.e = gamma * (p.e - bp);
  out.px = p.px + gamma2 * bp * bx - gamma * bx * p.e;
  out.py = p.py + gamma2 * bp * by - gamma * by * p.e;
  out.pz = p.pz + gamma2 * bp * bz - gamma * bz * p.e;
  return out;
}

static bool unit3(double x, double y, double z, double& ux, double& uy, double& uz) {
  double n2 = x * x + y * y + z * z;
  if (n2 <= 0.0) return false;
  double inv = 1.0 / sqrt(n2);
  ux = x * inv;
  uy = y * inv;
  uz = z * inv;
  return true;
}

// Exact Born-angle reconstruction from POWHEG "# uub" four-vectors and flavors.
static bool costh_from_uub(const UubData& uub, const QuarkAssignment& qa, double& costh,
                           int& outtype) {
  if (!(uub.has_p1 && uub.has_p2 && uub.has_p3 && uub.has_p4 && uub.has_flav)) return false;

  Vec4 lminus;
  Vec4 lplus;
  if (uub.flav[2] > 0 && lept_pdg_to_outtype(abs(uub.flav[2])) != 0) {
    lminus = uub.p3;
    lplus = uub.p4;
    outtype = lept_pdg_to_outtype(abs(uub.flav[2]));
  } else if (uub.flav[3] > 0 && lept_pdg_to_outtype(abs(uub.flav[3])) != 0) {
    lminus = uub.p4;
    lplus = uub.p3;
    outtype = lept_pdg_to_outtype(abs(uub.flav[3]));
  } else {
    return false;
  }

  Vec4 pquark = qa.quark_from_beam1 ? uub.p1 : uub.p2;
  Vec4 qtot = uub.p1 + uub.p2;
  if (qtot.e <= 0.0) return false;

  double bx = qtot.px / qtot.e;
  double by = qtot.py / qtot.e;
  double bz = qtot.pz / qtot.e;

  Vec4 pquark_star = boost(pquark, bx, by, bz);
  Vec4 lminus_star = boost(lminus, bx, by, bz);

  double qx = 0.0, qy = 0.0, qz = 0.0;
  double lx = 0.0, ly = 0.0, lz = 0.0;
  if (!unit3(pquark_star.px, pquark_star.py, pquark_star.pz, qx, qy, qz)) return false;
  if (!unit3(lminus_star.px, lminus_star.py, lminus_star.pz, lx, ly, lz)) return false;

  costh = qx * lx + qy * ly + qz * lz;
  if (costh > 1.0) costh = 1.0;
  if (costh < -1.0) costh = -1.0;
  (void)lplus;
  return true;
}

static bool costh_collins_soper(const Particle& lminus, const Particle& lplus,
                                bool quark_from_beam1, double& costh) {
  // Fallback reconstruction when exact "# uub" Born kinematics are unavailable:
  // compute cos(theta*) in the Collins-Soper frame from leptons only.
  constexpr double inv_sqrt2 = 0.70710678118654752440;

  Vec4 lm = particle_p4(lminus);
  Vec4 lp = particle_p4(lplus);
  Vec4 q = lm + lp;

  double m2 = q.m2();
  if (m2 <= 0.0) return false;
  double pt2 = q.px * q.px + q.py * q.py;
  double den = sqrt(m2 * (m2 + pt2));
  if (den == 0.0) return false;

  double lm_plus = (lm.e + lm.pz) * inv_sqrt2;
  double lm_minus = (lm.e - lm.pz) * inv_sqrt2;
  double lp_plus = (lp.e + lp.pz) * inv_sqrt2;
  double lp_minus = (lp.e - lp.pz) * inv_sqrt2;

  double c = 2.0 * (lm_plus * lp_minus - lm_minus * lp_plus) / den;
  if (!quark_from_beam1) c = -c;

  if (c > 1.0) c = 1.0;
  if (c < -1.0) c = -1.0;
  costh = c;
  return true;
}

static bool parse_particle_line(const string& line, Particle& p) {
  vector<string> t = split_ws(line);
  if (t.size() < 10) return false;
  p.id = atoi(t[0].c_str());
  p.status = atoi(t[1].c_str());
  p.m1 = atoi(t[2].c_str());
  p.m2 = atoi(t[3].c_str());
  p.px = atof(t[6].c_str());
  p.py = atof(t[7].c_str());
  p.pz = atof(t[8].c_str());
  p.e = atof(t[9].c_str());
  return true;
}

static bool parse_rwgt_line(const string& line, int& type, int& index) {
  string s = ltrim(line);
  if (s.rfind("#rwgt", 0) != 0) return false;
  istringstream iss(s.substr(5));
  if (!(iss >> type >> index)) return false;
  return true;
}

static bool parse_uub_line(const string& line, UubData& uub) {
  string s = ltrim(line);
  if (s.rfind("# uub", 0) != 0) return false;

  if (s.find("p1") != string::npos) {
    vector<double> v = extract_numbers(s);
    if (v.size() >= 4) {
      uub.p1 = Vec4(v[v.size() - 4], v[v.size() - 3], v[v.size() - 2], v[v.size() - 1]);
      uub.has_p1 = true;
    }
    return true;
  }
  if (s.find("p2") != string::npos) {
    vector<double> v = extract_numbers(s);
    if (v.size() >= 4) {
      uub.p2 = Vec4(v[v.size() - 4], v[v.size() - 3], v[v.size() - 2], v[v.size() - 1]);
      uub.has_p2 = true;
    }
    return true;
  }
  if (s.find("p3") != string::npos) {
    vector<double> v = extract_numbers(s);
    if (v.size() >= 4) {
      uub.p3 = Vec4(v[v.size() - 4], v[v.size() - 3], v[v.size() - 2], v[v.size() - 1]);
      uub.has_p3 = true;
    }
    return true;
  }
  if (s.find("p4") != string::npos) {
    vector<double> v = extract_numbers(s);
    if (v.size() >= 4) {
      uub.p4 = Vec4(v[v.size() - 4], v[v.size() - 3], v[v.size() - 2], v[v.size() - 1]);
      uub.has_p4 = true;
    }
    return true;
  }
  if (s.find("flav") != string::npos) {
    vector<double> v = extract_numbers(s);
    if (v.size() >= 4) {
      uub.flav[0] = static_cast<int>(llround(v[v.size() - 4]));
      uub.flav[1] = static_cast<int>(llround(v[v.size() - 3]));
      uub.flav[2] = static_cast<int>(llround(v[v.size() - 2]));
      uub.flav[3] = static_cast<int>(llround(v[v.size() - 1]));
      uub.has_flav = true;
    }
    return true;
  }

  return true;
}

static bool parse_event_info(const vector<string>& block, EventInfo& info) {
  if (block.size() < 3) return false;

  int nup = -1;
  size_t hdr_idx = 0;
  for (size_t i = 1; i < block.size(); ++i) {
    string s = trim(block[i]);
    if (s.empty()) continue;
    if (s[0] == '#') continue;
    vector<string> toks = split_ws(s);
    if (toks.empty()) continue;
    nup = atoi(toks[0].c_str());
    hdr_idx = i;
    break;
  }
  if (nup <= 0) return false;

  info.particles.clear();
  size_t got = 0;
  for (size_t i = hdr_idx + 1; i < block.size() && got < static_cast<size_t>(nup); ++i) {
    string s = trim(block[i]);
    if (s.empty()) continue;
    if (s[0] == '#') continue;
    Particle p;
    if (!parse_particle_line(s, p)) continue;
    info.particles.push_back(p);
    got++;
  }

  for (const string& line : block) {
    int t = -1, idx = -1;
    if (parse_rwgt_line(line, t, idx)) {
      info.rwgt_type = t;
      info.rwgt_index = idx;
    }
    parse_uub_line(line, info.uub);
  }

  return !info.particles.empty();
}

static bool build_projection(const EventInfo& ev, const FlavRegMap* fmap, Projection& prj) {
  // Main qqbar->Z Born projection used by the reweighting code.
  // Source priority:
  // 1) exact "# uub" kinematics from POWHEG comments in the event block;
  // 2) if missing, incoming flavors from #rwgt + --flavreglist mapping;
  // 3) last resort, incoming status==-1 partons in the LHE record.
  prj = Projection();

  bool have_uub = ev.uub.has_p1 && ev.uub.has_p2 && ev.uub.has_p3 && ev.uub.has_p4
                  && ev.uub.has_flav;

  if (have_uub) {
    QuarkAssignment qa = infer_quark_assignment(ev.uub.flav[0], ev.uub.flav[1]);
    int intype = pdg_quark_to_griffin(qa.abs_flav);
    int outtype = 0;
    double costh = numeric_limits<double>::quiet_NaN();
    Vec4 q = ev.uub.p1 + ev.uub.p2;
    double shat = q.m2();

    // Exact projection path: use POWHEG's stored Born momenta/flavors from "# uub".
    if (qa.ok && intype != 0 && shat > 0.0 && costh_from_uub(ev.uub, qa, costh, outtype)
        && outtype != 0) {
      prj.ok = true;
      prj.intype = intype;
      prj.outtype = outtype;
      prj.shat = shat;
      prj.costh = costh;
      prj.source = "uub";
      prj.detail = "exact_uub";
      return true;
    }
  }

  LeptonPair lp = find_leptons(ev.particles);
  if (!lp.ok) {
    prj.detail = "missing_lepton_pair";
    return false;
  }

  int f1 = 0, f2 = 0;
  bool from_rwgt = false;
  if (fmap && ev.rwgt_type > 0 && ev.rwgt_index > 0) {
    // "Taken from" external FlavRegList: map this #rwgt channel back to its uborn
    // incoming quark flavors when possible.
    if (fmap->rwgt_to_incoming(ev.rwgt_type, ev.rwgt_index, f1, f2)) from_rwgt = true;
  }

  if (!from_rwgt) {
    vector<int> incoming;
    for (const auto& p : ev.particles) {
      if (p.status == -1) incoming.push_back(p.id);
    }
    if (incoming.size() >= 2) {
      f1 = incoming[0];
      f2 = incoming[1];
    } else {
      prj.detail = "missing_incoming_partons";
      return false;
    }
  }

  QuarkAssignment qa = infer_quark_assignment(f1, f2);
  int intype = pdg_quark_to_griffin(qa.abs_flav);
  if (!qa.ok || intype == 0) {
    prj.detail = from_rwgt ? "rwgt_flavor_not_projectable" : "incoming_flavor_not_projectable";
    return false;
  }

  Vec4 q = particle_p4(lp.lminus) + particle_p4(lp.lplus);
  double shat = q.m2();
  if (shat <= 0.0) {
    prj.detail = "nonpositive_mll2";
    return false;
  }

  double costh = numeric_limits<double>::quiet_NaN();
  // Approximate projection path: infer Born scattering angle from final-state leptons.
  if (!costh_collins_soper(lp.lminus, lp.lplus, qa.quark_from_beam1, costh)) {
    prj.detail = "costh_collins_soper_failed";
    return false;
  }

  prj.ok = true;
  prj.intype = intype;
  prj.outtype = lp.outtype;
  prj.shat = shat;
  prj.costh = costh;
  prj.source = from_rwgt ? "rwgt_flavreg" : "incoming";
  prj.detail = from_rwgt ? "mapped_from_rwgt" : "heuristic_from_incoming";
  return true;
}

static string format_weight(double w) {
  ostringstream oss;
  oss.setf(ios::scientific);
  oss << setprecision(16) << w;
  return oss.str();
}

static bool append_weight_to_event_block(vector<string>& block, double weight,
                                         const string& weight_id) {
  string wstr = format_weight(weight);

  bool has_weights = false;
  bool has_rwgt = false;
  for (const string& line : block) {
    string s = ltrim(line);
    if (s.rfind("<weights>", 0) == 0) has_weights = true;
    if (s.rfind("<rwgt>", 0) == 0) has_rwgt = true;
  }

  if (has_weights) {
    vector<string> out;
    out.reserve(block.size() + 1);
    bool inserted = false;
    for (const string& line : block) {
      string s = ltrim(line);
      if (!inserted && s.rfind("</weights>", 0) == 0) {
        out.push_back(wstr);
        inserted = true;
      }
      out.push_back(line);
    }
    if (!inserted) return false;
    block.swap(out);
    return true;
  }

  if (has_rwgt) {
    vector<string> out;
    out.reserve(block.size() + 1);
    bool inserted = false;
    for (const string& line : block) {
      string s = ltrim(line);
      if (!inserted && s.rfind("</rwgt>", 0) == 0) {
        out.push_back(" <wgt id='" + weight_id + "'>" + wstr + "</wgt>");
        inserted = true;
      }
      out.push_back(line);
    }
    if (!inserted) return false;
    block.swap(out);
    return true;
  }

  vector<string> out;
  out.reserve(block.size() + 3);
  bool inserted = false;
  for (const string& line : block) {
    string s = ltrim(line);
    if (!inserted && s.rfind("</event>", 0) == 0) {
      out.push_back("<rwgt>");
      out.push_back(" <wgt id='" + weight_id + "'>" + wstr + "</wgt>");
      out.push_back("</rwgt>");
      inserted = true;
    }
    out.push_back(line);
  }
  if (!inserted) return false;
  block.swap(out);
  return true;
}

static bool ensure_header_weight_line(vector<string>& output_lines, const string& weight_id,
                                      const string& weight_desc) {
  bool in_header = false;
  bool in_initrwgt = false;
  bool has_initrwgt = false;
  bool has_weight = false;
  bool inserted = false;

  vector<string> out;
  out.reserve(output_lines.size() + 8);

  for (const string& line : output_lines) {
    string s = ltrim(line);
    if (s.rfind("<header>", 0) == 0) in_header = true;
    if (s.rfind("<initrwgt>", 0) == 0) {
      in_initrwgt = true;
      has_initrwgt = true;
    }

    if (in_initrwgt && line.find("id='" + weight_id + "'") != string::npos) {
      has_weight = true;
    }
    if (in_initrwgt && line.find("id=\"" + weight_id + "\"") != string::npos) {
      has_weight = true;
    }

    if (in_initrwgt && !has_weight && !inserted && s.rfind("</weightgroup>", 0) == 0) {
      out.push_back("<weight id='" + weight_id + "' > " + weight_desc + " </weight>");
      inserted = true;
      has_weight = true;
    }

    if (in_initrwgt && !has_weight && !inserted && s.rfind("</initrwgt>", 0) == 0) {
      out.push_back("<weight id='" + weight_id + "' > " + weight_desc + " </weight>");
      inserted = true;
      has_weight = true;
    }

    out.push_back(line);

    if (s.rfind("</initrwgt>", 0) == 0) in_initrwgt = false;

    if (in_header && s.rfind("</header>", 0) == 0) {
      if (!has_initrwgt) {
        vector<string> add;
        add.push_back("<initrwgt>");
        add.push_back("<weightgroup name='GRIFFIN-EW' combine='none'>");
        add.push_back("<weight id='" + weight_id + "' > " + weight_desc + " </weight>");
        add.push_back("</weightgroup>");
        add.push_back("</initrwgt>");

        out.pop_back();
        for (const string& x : add) out.push_back(x);
        out.push_back(line);
      }
      in_header = false;
    }
  }

  output_lines.swap(out);
  return true;
}

static void print_usage() {
  cerr << "Usage:\n"
       << "  zewgt --input in.lhe --output out.lhe --card ewvirt_card.dat\\n"
       << "      [--flavreglist FlavRegList] [--mode nlo|nnlo] [--scheme alpha|gmu]\\n"
       << "      [--weight-id zewgt_griffin] [--weight-desc 'ZEWGT GRIFFIN EW virtual']\\n"
       << "      [--skip-unresolved]\n";
}

}  // namespace

int main(int argc, char** argv) {
  string in_path;
  string out_path;
  string card_path;
  string flavreg_path;
  string mode_override;
  string scheme_override;
  string weight_id = "zewgt_griffin";
  string weight_desc = "ZEWGT GRIFFIN EW virtual";
  bool strict = true;

  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      in_path = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--card" && i + 1 < argc) {
      card_path = argv[++i];
    } else if (arg == "--flavreglist" && i + 1 < argc) {
      flavreg_path = argv[++i];
    } else if (arg == "--mode" && i + 1 < argc) {
      mode_override = tolower_str(argv[++i]);
    } else if (arg == "--scheme" && i + 1 < argc) {
      scheme_override = tolower_str(argv[++i]);
    } else if (arg == "--weight-id" && i + 1 < argc) {
      weight_id = argv[++i];
    } else if (arg == "--weight-desc" && i + 1 < argc) {
      weight_desc = argv[++i];
    } else if (arg == "--skip-unresolved") {
      strict = false;
    } else if (arg == "--help" || arg == "-h") {
      print_usage();
      return 0;
    } else {
      cerr << "Unknown argument: " << arg << "\n";
      print_usage();
      return 1;
    }
  }

  if (in_path.empty() || out_path.empty() || card_path.empty()) {
    print_usage();
    return 1;
  }

  Card card = load_card(card_path);
  GriffinWeightComputer calculator(card, mode_override, scheme_override);

  FlavRegMap fmap;
  FlavRegMap* fmap_ptr = nullptr;
  if (!flavreg_path.empty()) {
    if (!fmap.load(flavreg_path)) {
      cerr << "Failed to load FlavRegList map from: " << flavreg_path << "\n";
      return 1;
    }
    fmap_ptr = &fmap;
  }

  ifstream in(in_path.c_str());
  if (!in) {
    cerr << "Failed to open input file: " << in_path << "\n";
    return 1;
  }

  ofstream out(out_path.c_str());
  if (!out) {
    cerr << "Failed to open output file: " << out_path << "\n";
    return 1;
  }

  Stats stats;

  string line;
  vector<string> pre_event_lines;
  pre_event_lines.reserve(4096);

  while (getline(in, line)) {
    if (line.find("<event>") == string::npos) {
      pre_event_lines.push_back(line);
      continue;
    }

    ensure_header_weight_line(pre_event_lines, weight_id, weight_desc);
    for (const string& l : pre_event_lines) out << l << '\n';
    pre_event_lines.clear();

    vector<string> block;
    block.push_back(line);
    while (getline(in, line)) {
      block.push_back(line);
      if (line.find("</event>") != string::npos) break;
    }

    stats.nevents++;

    EventInfo ev;
    if (!parse_event_info(block, ev)) {
      cerr << "Failed parsing event " << stats.nevents << "\n";
      return 1;
    }

    Projection prj;
    // Projection to Born variables happens here (build_projection above).
    if (!build_projection(ev, fmap_ptr, prj) || !prj.ok) {
      stats.n_unresolved++;
      if (strict) {
        cerr << "Failed to reconstruct qqbar->Z Born projection at event " << stats.nevents
             << " (" << prj.detail << ").\n"
             << "If this file has no '# uub' lines, provide --flavreglist FlavRegList or use"
             << " --skip-unresolved.\n";
        return 2;
      }

      if (!append_weight_to_event_block(block, 1.0, weight_id)) {
        cerr << "Failed inserting fallback weight in event " << stats.nevents << "\n";
        return 1;
      }
      for (const string& l : block) out << l << '\n';
      continue;
    }

    if (prj.source == "uub") stats.n_uub++;
    if (prj.source == "rwgt_flavreg") stats.n_rwgt_flavreg++;
    if (prj.source == "incoming") stats.n_incoming++;

    double w = calculator.compute_weight(prj.intype, prj.outtype, prj.shat, prj.costh);
    if (!isfinite(w)) {
      cerr << "Non-finite EW virtual weight at event " << stats.nevents << "\n";
      return 1;
    }

    if (!append_weight_to_event_block(block, w, weight_id)) {
      cerr << "Failed inserting weight in event " << stats.nevents << "\n";
      return 1;
    }

    for (const string& l : block) out << l << '\n';
  }

  ensure_header_weight_line(pre_event_lines, weight_id, weight_desc);
  for (const string& l : pre_event_lines) out << l << '\n';

  cerr << "Processed events: " << stats.nevents << "\n";
  cerr << "Projection source counts:"
       << " uub=" << stats.n_uub << ", rwgt_flavreg=" << stats.n_rwgt_flavreg
       << ", incoming=" << stats.n_incoming << ", unresolved=" << stats.n_unresolved << "\n";
  cerr << "EW mode=" << calculator.mode() << ", scheme=" << calculator.scheme() << "\n";

  return 0;
}
