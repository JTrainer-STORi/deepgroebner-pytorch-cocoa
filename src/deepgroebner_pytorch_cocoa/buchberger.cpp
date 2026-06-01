#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <typeinfo>
#include <random>
#include <functional>
#include <tuple>
#include <cctype>
#include <stdexcept>

#include "CoCoA/library.H"
#include "buchberger.h"
#include "ideals.h"

namespace CoCoA {
  namespace {

    std::string trim_copy(const std::string& s) {
      size_t begin = 0;
      while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) ++begin;
      size_t end = s.size();
      while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
      return s.substr(begin, end - begin);
    }

    ring parse_coeff_ring(const std::string& coeff_ring_str) {
      const std::string s = trim_copy(coeff_ring_str);
      if (s == "QQ") return RingQQ();
      if (s == "ZZ") return RingZZ();

      const std::string prefix = "ZZmod(";
      if (s.rfind(prefix, 0) == 0 && !s.empty() && s.back() == ')') {
        const std::string p_str = s.substr(prefix.size(), s.size() - prefix.size() - 1);
        const long p = std::stol(trim_copy(p_str));
        return NewZZmod(p);
      }

      throw std::invalid_argument("Unsupported coeff_ring: " + s + ". Supported: QQ, ZZ, ZZmod(p)");
    }

    ideal parse_fixed_ideal(const std::string& coeff_ring_str,
                            const std::string& symbols_str,
                            const std::vector<std::string>& basis_strs,
                            int& out_nvars) {
      if (basis_strs.empty()) {
        throw std::invalid_argument("basis_strs must be non-empty for fixed ideal input");
      }

      ring coeff = parse_coeff_ring(coeff_ring_str);
      SparsePolyRing P = NewPolyRing(coeff, symbols(symbols_str));

      std::vector<RingElem> generators;
      generators.reserve(basis_strs.size());
      for (const auto& basis_elem : basis_strs) {
        generators.emplace_back(P, basis_elem);
      }

      out_nvars = NumIndets(P);
      return ideal(generators);
    }

  } // namespace


  GlobalManager CoCoAFoundations;
  SignalWatcher MonitorInterrupt(SIGINT); // you must also call CheckForInterrupt every so often

  static const std::map<std::string, EliminationType> elimination_map = {
    {"gebauermoeller", EliminationType::GebauerMoeller},
    {"lcm", EliminationType::LCM},
    {"none", EliminationType::None},
  };

  static const std::map<std::string, RewardType> reward_map = {
    {"additions", RewardType::Additions},
    {"reductions", RewardType::Reductions},
  };

  static const std::map<std::string, SelectionType> strategy_map = {
    {"first",  SelectionType::First},
    {"last",   SelectionType::Last},
    {"degree", SelectionType::Degree},
    {"normal", SelectionType::Normal},
    {"sugar",  SelectionType::Sugar},
    {"policy-mimic", SelectionType::PolicyMimic},
    {"random", SelectionType::Random},
  };

  namespace {

    int leading_overlap_degree(const SPair& p, const std::vector<RingElem>& G) {
      const int deg_i = deg(LPP(G[p.i]));
      const int deg_j = deg(LPP(G[p.j]));
      return deg_i + deg_j - p.lcmdeg;
    }

    std::vector<long> kth_lead_exponents(ConstRefRingElem f, int term_index, long n_vars) {
      std::vector<long> exps(static_cast<std::size_t>(std::max<long>(n_vars, 0)), 0);
      SparsePolyIter it = BeginIter(f);
      for (int idx = 0; idx < term_index && !IsEnded(it); ++idx) {
        ++it;
      }
      if (IsEnded(it)) {
        return exps;
      }

      std::vector<long> raw_exps;
      exponents(raw_exps, PP(it));
      for (long var_idx = 0; var_idx < std::min<long>(n_vars, static_cast<long>(raw_exps.size())); ++var_idx) {
        exps[static_cast<std::size_t>(var_idx)] = raw_exps[static_cast<std::size_t>(var_idx)];
      }
      return exps;
    }

    std::vector<long> kth_pair_lcm_exponents(const SPair& p, const std::vector<RingElem>& G, int term_index) {
      const long n_vars = NumIndets(owner(G[p.i]));
      const std::vector<long> exps_i = kth_lead_exponents(G[p.i], term_index, n_vars);
      const std::vector<long> exps_j = kth_lead_exponents(G[p.j], term_index, n_vars);
      std::vector<long> lcm_exps(exps_i.size(), 0);
      for (std::size_t idx = 0; idx < lcm_exps.size(); ++idx) {
        lcm_exps[idx] = std::max(exps_i[idx], exps_j[idx]);
      }
      return lcm_exps;
    }

    int total_degree_from_exponents(const std::vector<long>& exps) {
      long total = 0;
      for (long exp : exps) {
        total += exp;
      }
      return static_cast<int>(total);
    }

  } // namespace

  inline std::ostream& print_pretty(std::ostream& os,
                                  const state_features& f)
  {
      os << "State Features\n";
      os << "-----------------------------\n";
      os << "G size:      " << f.G_cardinality << "\n";
      os << "P size:      " << f.P_cardinality << "\n";
      os << "Pair dens:   " << f.pair_density << "\n";
      os << "Mean deg:    " << f.mean_degree << "\n";
      os << "Max deg:     " << f.max_degree << "\n";
      os << "Min deg:     " << f.min_degree << "\n";
      os << "Std deg:     " << f.st_dev_degree << "\n";
      os << "Mean LCM:    " << f.mean_lcm_degree << "\n";
      os << "Std LCM:     " << f.std_dev_lcm_degree << "\n";
      os << "Pairs Processed: " << f.n_pairs_processed << "\n";
      os << "Pairs Pruned:  " << f.n_pairs_pruned << "\n";
      os << "Steps since: " << f.steps_since_basis_update << "\n";
      return os;
  }

  std::vector<float> state_features_to_vector(const state_features& s) {
    return {
        static_cast<float>(s.G_cardinality),
        static_cast<float>(s.P_cardinality),
        s.pair_density,

        s.mean_degree,
        static_cast<float>(s.max_degree),
        static_cast<float>(s.min_degree),
        s.st_dev_degree,

        s.mean_lcm_degree,
        static_cast<float>(s.max_lcm_degree),
        static_cast<float>(s.min_lcm_degree),
        s.std_dev_lcm_degree,

        static_cast<float>(s.n_pairs_pruned),
        static_cast<float>(s.n_pairs_processed),
        static_cast<float>(s.steps_since_basis_update)
    };
}

  static void update_running_stats(
    double x,
    int& count,
    double& mean,
    double& M2)
  {
    count++;
    double delta = x - mean;
    mean += delta / count;
    double delta2 = x - mean;
    M2 += delta * delta2;
  }

  void FeatureTracker::reset(state_features& f) {
      *this = FeatureTracker(); // reset internal stats
      f = state_features();     // reset features
  }

  void FeatureTracker::on_pair_removed(
    const SPair& pair,
    state_features& f)
  {
    f.P_cardinality--;

    // update LCM stats (pair processed)
    update_running_stats(
        pair.lcmdeg, lcm_count, lcm_mean, lcm_M2);

    f.mean_lcm_degree = lcm_mean;
    f.std_dev_lcm_degree =
        (lcm_count > 1) ? std::sqrt(lcm_M2 / lcm_count) : 0.0;

    f.max_lcm_degree =
        std::max(f.max_lcm_degree, pair.lcmdeg);

    if (lcm_count == 1)
        f.min_lcm_degree = pair.lcmdeg;
    else
        f.min_lcm_degree =
            std::min(f.min_lcm_degree, pair.lcmdeg);

    f.n_pairs_processed++;

  }


  void FeatureTracker::on_update(
    const RingElem& g,
    const std::vector<SPair>& new_pairs,
    int n_pairs_pruned,
    int n_old_pairs_pruned,
    state_features& f)
  {
    // update |G|
    f.G_cardinality++;

    // degree stats
    int d = deg(g);
    update_running_stats(d, deg_count, deg_mean, deg_M2);

    f.mean_degree = deg_mean;
    f.st_dev_degree =
        (deg_count > 1) ? std::sqrt(deg_M2 / deg_count) : 0.0;

    f.max_degree = std::max(f.max_degree, d);
    f.min_degree = (deg_count == 1) ? d : std::min(f.min_degree, d);

    // new pairs added
    f.P_cardinality += (int)new_pairs.size();

    // old pairs removed
    f.P_cardinality -= n_old_pairs_pruned;

    // update pair density
    if (f.G_cardinality >= 2) {
      double max_pairs =
          (double)f.G_cardinality * (f.G_cardinality - 1) / 2.0;
      f.pair_density = f.P_cardinality / max_pairs;
    }

    f.n_pairs_pruned += n_pairs_pruned;
  }

  void FeatureTracker::on_step_completed(state_features& f)
{
    // ---------- pair density ----------
    if (f.G_cardinality >= 2) {
        const double max_pairs =
            (double)f.G_cardinality * (f.G_cardinality - 1) / 2.0;

        f.pair_density =
            (max_pairs > 0.0)
                ? static_cast<float>(f.P_cardinality / max_pairs)
                : 0.0f;
    } else {
        f.pair_density = 0.0f;
    }

    // NOTE:
    // steps_since_basis_update is managed by the environment:
    //   - incremented at step start
    //   - reset to 0 when basis grows
}



  template <class InputIt, class RNG>
  std::vector<typename std::iterator_traits<InputIt>::value_type>
  random_unique(InputIt begin, InputIt end, std::size_t num_random, RNG& rng)
  {
      using T = typename std::iterator_traits<InputIt>::value_type;

      std::vector<T> buffer(begin, end);   // copy input (out of place)
      const std::size_t n = buffer.size();

      if (num_random > n)
          num_random = n;

      // Partial Fisher–Yates shuffle on the buffer
      for (std::size_t i = 0; i < num_random; ++i) {
          std::uniform_int_distribution<std::size_t> dist(i, n - 1);
          std::size_t r = dist(rng);
          std::swap(buffer[i], buffer[r]);
      }

      // Return the first num_random random unique elements
      return std::vector<T>(buffer.begin(), buffer.begin() + num_random);
  };

  void EmitDigits(long n, std::vector<std::string>& out)
  {
      std::string s = std::to_string(n);
      for (char c : s) {
          out.push_back(std::string(1, c));
      }
  }

  void TokenizePolynomial(
  const RingElem& f,
  std::vector<std::string>& out
  )
  {
      PolyRing P = owner(f);

      for (SparsePolyIter it = BeginIter(f); !IsEnded(it); ++it) {

          out.push_back("<TERM>");

          std::vector<long> exps;
          exponents(exps, PP(it));

          for (long i = 0; i < NumIndets(P); ++i) {
              if (exps[i] != 0) {
                  out.push_back("<VAR>");
                  EmitDigits(i, out);

                  out.push_back("<EXP>");
                  EmitDigits(exps[i], out);
              }
          }
      }
  }

  std::vector<std::string> truncate_tokens_at_term_start(
      const std::vector<std::string>& tokens,
      std::size_t max_len,
      const std::string& term_start_token = "<TERM>",
      const std::string& trunc_token = "<TRUNC>"
  ) {
      if (tokens.size() <= max_len) {
          return tokens;
      }

      std::size_t allowed_len = max_len - 1;
      std::vector<std::size_t> term_starts;

      for (std::size_t i = 0; i < tokens.size(); ++i) {
          if (tokens[i] == term_start_token) {
              term_starts.push_back(i);
          }
      }

      std::size_t cut = std::string::npos;

      for (std::size_t i = 0; i < term_starts.size(); ++i) {
          std::size_t end =
              (i + 1 < term_starts.size())
                  ? term_starts[i + 1]
                  : tokens.size();

          if (end <= allowed_len) {
              cut = end;
          } else {
              break;
          }
      }

      if (cut == std::string::npos) {
          return {trunc_token};
      }

      std::vector<std::string> result;
      result.reserve(cut + 1);
      result.insert(result.end(), tokens.begin(), tokens.begin() + cut);
      result.push_back(trunc_token);

      return result;
  }


    std::vector<std::string> TokenizePolynomialFixedLength(
    const RingElem& poly,
    std::size_t max_len,
    const std::string& poly_start,
    const std::string& poly_end
  )
  {
    if (max_len < 2) {
        throw std::invalid_argument("max_len too small for polynomial markers");
    }

    std::vector<std::string> out;

    std::vector<std::string> result;
    result.reserve(max_len);

    result.push_back(poly_start);

    TokenizePolynomial(poly, out);

    std::vector<std::string> body =
    truncate_tokens_at_term_start(
        out,
        max_len - 2
    );

    result.insert(result.end(), body.begin(), body.end());
    result.push_back(poly_end);

    // Pad AFTER polynomial is complete
    while (result.size() < max_len) {
        result.push_back("<PAD>");
    }

    return result;

  }

  SelectionResult SPairComparator(SelectionType selection, const std::vector<RingElem>& G, const std::vector<int>& Sugars, std::optional<unsigned> seed, std::mt19937& rng) {
    SelectionResult out;

    switch (selection) {

      case SelectionType::First:
          out.cmp = [](const SPair &p1, const SPair &p2) {
              return std::tie(p1.j, p1.i) < std::tie(p2.j, p2.i);
          };
          break;

        case SelectionType::Last:
          // Inverse of First: prioritize most recently added pairs (larger j, then i).
          out.cmp = [](const SPair &p1, const SPair &p2) {
            return std::tie(p1.j, p1.i) > std::tie(p2.j, p2.i);
          };
          break;

      case SelectionType::Degree:
          out.cmp = [&G](const SPair &p1, const SPair &p2) {
              return std::tie(p1.lcmdeg, p1.j, p1.i) < std::tie(p2.lcmdeg, p2.j, p2.i);
          };
          break;

      case SelectionType::Normal:
          out.cmp = [&G](const SPair &p1, const SPair &p2) {
              return std::tie(p1.Lij, p1.j, p1.i) < std::tie(p2.Lij, p2.j, p2.i);
          };
          break;

      case SelectionType::Sugar:
          out.cmp = [&G, &Sugars](const SPair &p1, const SPair &p2) {
              return std::tie(p1.Sugar, p1.Lij, p1.j, p1.i) < std::tie(p2.Sugar, p2.Lij, p2.j, p2.i);
          };
          break;

      case SelectionType::PolicyMimic:
          out.cmp = [&G](const SPair &p1, const SPair &p2) {
              const int overlap1 = leading_overlap_degree(p1, G);
              const int overlap2 = leading_overlap_degree(p2, G);
              const std::vector<long> lm1_lcm1 = kth_pair_lcm_exponents(p1, G, 1);
              const std::vector<long> lm1_lcm2 = kth_pair_lcm_exponents(p2, G, 1);
              const int lm1_lcm_deg1 = total_degree_from_exponents(lm1_lcm1);
              const int lm1_lcm_deg2 = total_degree_from_exponents(lm1_lcm2);

              return std::make_tuple(
                  p1.Lij,
                  -overlap1,
                  lm1_lcm_deg1,
                  lm1_lcm1,
                  -p1.Sugar,
                  p1.j,
                  p1.i
              ) < std::make_tuple(
                  p2.Lij,
                  -overlap2,
                  lm1_lcm_deg2,
                  lm1_lcm2,
                  -p2.Sugar,
                  p2.j,
                  p2.i
              );
          };
          break;

      case SelectionType::TrueDegree:
          out.cmp = [&G](const SPair &p1, const SPair &p2) {
              RingElem SPoly1 = SPoly(G[p1.i], G[p1.j]);
              RingElem SPoly2 = SPoly(G[p2.i], G[p2.j]);

              int td1 = IsZero(SPoly1) ? 0 : deg(LPP(SPoly1));
              int td2 = IsZero(SPoly2) ? 0 : deg(LPP(SPoly2));

              int d1 = deg(lcm(LPP(G[p1.i]), LPP(G[p1.j])));
              int d2 = deg(lcm(LPP(G[p2.i]), LPP(G[p2.j])));

              return std::tie(td1, d1, p1.j, p1.i) < std::tie(td2, d2, p2.j, p2.i);
          };
          break;
      case SelectionType::Random:
        out.is_random = true;
        if (seed) rng.seed(*seed);
        else      rng.seed(rand());
        break;

    }

    return out;

  };

  // take a polynomial f and a vector of polynomials v return the index of the first polynomial in the 
  // vector which divides f
  long FindReducer(ConstRefRingElem f,
                 const std::vector<RingElem>& v,
                 const std::vector<PPMonoidElem>& LMv)
  {
      if (IsZero(f)) return -1;

      const PPMonoidElem LMf = LPP(f); // Calculate once before loop

      const long n = static_cast<long>(v.size());
      for (long j = 0; j < n; ++j) {
          if (IsDivisible(LMf, LMv[j])) // using cached LM
              return j;

          CheckForInterrupt("Buchberger");
      }
      return -1;
  }

  std::tuple <RingElem, long, long> NRLM(ConstRefRingElem f, 
  const std::vector <RingElem> &v,
  const std::vector<PPMonoidElem>& LMv,
  long sugar,
  const std::vector<int>& Sugars) 
  {

    const SparsePolyRing P = owner(f);
    long steps = 0;
    RingElem m(P);
    RingElem r = f;

    long j = FindReducer(r, v, LMv);

    while (j != -1) // i.e as long as we exit the FindReducer function having found a divisor
    {
        // m = LM(r)/LM(v[i]);
        P->myDivLM(raw(m), raw(r),
                    raw(v[j])); // no checks Creates the result of the previous division in a very optimised way
        r -= m * v[j]; // remove the product of m * by the divisor polynomial from the polynomial being reduced
        sugar = std::max(sugar, deg(m) + Sugars[j]);
        ++steps; // increasing the number of steps because a polynomial addition has occured
        if (IsZero(r)) return {zero(P), {steps}, sugar}; // if the remainder is zero then we are done
        j = FindReducer(r, v, LMv);
        CheckForInterrupt("Buchberger");
    }
    return {r, steps, sugar}; // if we exit from the loop not having found r, then the leading monomial of r is no longer reducible
  }

  std::tuple <RingElem, long, long> NormalRemainder(ConstRefRingElem f,
  const std::vector <RingElem> &v,
  const std::vector<PPMonoidElem>& LMv,
  long sugar,
  const std::vector<int>& Sugars) {
    long totalsteps = 0;
    if (IsZero(f)) return {f, {totalsteps}, sugar};
    const SparsePolyRing P = owner(f);
    long tempsteps;
    RingElem ansNR(P);
    RingElem tmpNR;

    std::tie(tmpNR, totalsteps, sugar) = NRLM(f, v, LMv, sugar, Sugars);
    while (!IsZero(tmpNR)) {
        P->myMoveLMToBack(raw(ansNR),
                          raw(tmpNR)); // because the leading monomial of the temporary remainder is not reducible, send it to the back of the polynomial answer
        std::tie(tmpNR, tempsteps, sugar) = NRLM(tmpNR,
              v, LMv, sugar, Sugars); // Recompute the normal remainder of the temporary value without its leading term
        totalsteps += tempsteps;
        CheckForInterrupt("Buchberger");
    }
    return {ansNR, totalsteps, sugar};
  }

  RingElem SPoly(ConstRefRingElem g, ConstRefRingElem h) {
    const ring P = owner(g);
    PPMonoidElem m = lcm(LPP(g), LPP(h));
    return monomial(P, 1 / LC(g), m / LPP(g)) * g - monomial(P, 1 / LC(h), m / LPP(h)) * h;
  }

  std::pair <RingElem, long> reduce(ConstRefRingElem g, const std::vector <RingElem> &F) {
    long steps = 0;
    const ring P = owner(g);
    RingElem r(P);
    RingElem h = g;

    while (!IsZero(h)) {
        bool found_divisor = false;

        for (RingElem f: F) {
            if (IsDivisible(LPP(h), LPP(f))) {
                h -= monomial(P, LC(h) / LC(f), (LPP(h) / LPP(f))) * f;
                found_divisor = true;
                ++steps;
                break;
            }
        }

        if (!found_divisor) {
            r += monomial(P, LC(h), LPP(h));
            h -= monomial(P, LC(h), LPP(h));
        }
    }

    return {r + h, steps};
  }

  void update(std::vector<RingElem>& G,
            std::vector<PPMonoidElem>& LM,   // cache: LM[k] = LPP(G[k])
            std::vector<int>& degLM,
            std::vector<int>& Sugars,
            std::vector<SPair>& P,           // cached pairs: p.Lij, p.lcmdeg
            ConstRefRingElem f,
            EliminationType elimination)
{
    const int m = static_cast<int>(G.size()); // new polynomial will become index m
    const PPMonoidElem lpp_f = LPP(f);
    const int deg_lpp_f = deg(lpp_f);

    // Precompute Lif[i] = lcm(LM[i], lpp_f) once (used in all modes)
    std::vector<PPMonoidElem> Lif;
    Lif.reserve(m);

    std::vector<SPair> newPairs;
    newPairs.reserve(m);

    switch (elimination) {
      case EliminationType::None:
        for (int i = 0; i < m; i++) {
          PPMonoidElem L_if = lcm(LM[i], lpp_f);
          Lif.push_back(L_if);
          newPairs.emplace_back(i, m, std::move(L_if), Sugars[i], Sugars[m], degLM[i], deg_lpp_f);
        }
        break;
      case EliminationType::LCM:
        for (int i = 0; i < m; ++i) {
          PPMonoidElem L_if = lcm(LM[i], lpp_f);
          Lif.push_back(L_if);

          if (L_if != LM[i] * lpp_f) {
              newPairs.emplace_back(i, m, std::move(L_if), Sugars[i], Sugars[m], degLM[i], deg_lpp_f);
          }
        }
        break;
      case EliminationType::GebauerMoeller:

        for (int i = 0; i < m; i++) {
          PPMonoidElem L_if = lcm(LM[i], lpp_f);
          Lif.push_back(L_if);
        }

        auto redundant_old_pair = [&](const SPair& p) -> bool {
              if (!IsDivisible(*p.Lij, lpp_f)) return false; // lpp_f does NOT divide L_ij

              const PPMonoidElem& L_if = Lif[p.i];
              const PPMonoidElem& L_jf = Lif[p.j];

              if (p.Lij == L_if) return false;
              if (p.Lij == L_jf) return false;

              return true;
        };

        // Compact-in-place removal for P (no erase shifting)
        std::size_t w = 0;
        for (std::size_t r = 0; r < P.size(); ++r) {
            if (!redundant_old_pair(P[r])) {
                if (w != r) P[w] = std::move(P[r]);
                ++w;
            }
        }
        P.resize(w);    

        std::map <PPMonoidElem, std::vector<int>> lcms;
        for (int i = 0; i < m; i ++) {
          lcms[Lif[i]].push_back(i);
        }

        std::vector<PPMonoidElem> min_lcms;
        for (const auto &pair: lcms) {
          PPMonoidElem mon = pair.first;
          std::vector<int> v = pair.second;
          if (std::all_of(min_lcms.begin(), min_lcms.end(), [&mon](const PPMonoidElem &m) { return !IsDivisible(mon, m); })) {
              min_lcms.push_back(mon);
              if (std::none_of(v.begin(), v.end(), [&Lif, &LM, &lpp_f](int i) { return Lif[i] == LM[i] * lpp_f; })) {
                  newPairs.emplace_back(v[0], m, Lif[v[0]], Sugars[v[0]], Sugars[m], degLM[v[0]], deg_lpp_f);
              }
          }
        }

        break;
    }

    // Optional deterministic ordering
    std::sort(newPairs.begin(), newPairs.end(),
              [](const SPair& a, const SPair& b) {
                  return (a.i < b.i) || (a.i == b.i && a.j < b.j);
              });

    // Commit new polynomial and its LM cache
    G.push_back(f);
    LM.push_back(lpp_f);
    degLM.push_back(deg(lpp_f));

    // Append new pairs
    P.insert(P.end(),
             std::make_move_iterator(newPairs.begin()),
             std::make_move_iterator(newPairs.end()));

}

void update(std::vector<RingElem>& G,
            std::vector<PPMonoidElem>& LM,   // cache: LM[k] = LPP(G[k])
            std::vector<int>& degLM,
            std::vector<int>& Sugars,
            std::vector<SPair>& P,           // cached pairs: p.Lij, p.lcmdeg
            ConstRefRingElem f,
            EliminationType elimination,
            state_features& state_feats,
            FeatureTracker& feat_tracker)
{
    const int m = static_cast<int>(G.size()); // new polynomial will become index m

    int n_pruned = 0;
    int n_old_pairs_pruned = 0;

    const PPMonoidElem lpp_f = LPP(f);
    const int deg_lpp_f = deg(lpp_f);

    // Precompute Lif[i] = lcm(LM[i], lpp_f) once (used in all modes)
    std::vector<PPMonoidElem> Lif;
    Lif.reserve(m);

    std::vector<SPair> newPairs;
    newPairs.reserve(m);

    switch (elimination) {
      case EliminationType::None:
        for (int i = 0; i < m; i++) {
          PPMonoidElem L_if = lcm(LM[i], lpp_f);
          Lif.push_back(L_if);
          newPairs.emplace_back(i, m, std::move(L_if), Sugars[i], Sugars[m], degLM[i], deg_lpp_f);
        }
        break;
      case EliminationType::LCM:
        for (int i = 0; i < m; ++i) {
          PPMonoidElem L_if = lcm(LM[i], lpp_f);
          Lif.push_back(L_if);

          if (L_if != LM[i] * lpp_f) {
              newPairs.emplace_back(i, m, std::move(L_if), Sugars[i], Sugars[m], degLM[i], deg_lpp_f);
          }
        }
        n_pruned = (m - newPairs.size());
        break;
      case EliminationType::GebauerMoeller:

        for (int i = 0; i < m; i++) {
          PPMonoidElem L_if = lcm(LM[i], lpp_f);
          Lif.push_back(L_if);
        }

        auto redundant_old_pair = [&](const SPair& p) -> bool {
              if (!IsDivisible(*p.Lij, lpp_f)) return false; // lpp_f does NOT divide L_ij

              const PPMonoidElem& L_if = Lif[p.i];
              const PPMonoidElem& L_jf = Lif[p.j];

              if (p.Lij == L_if) return false;
              if (p.Lij == L_jf) return false;

              return true;
        };

        // Compact-in-place removal for P (no erase shifting)
        std::size_t w = 0;
        for (std::size_t r = 0; r < P.size(); ++r) {
            if (!redundant_old_pair(P[r])) {
                if (w != r) P[w] = std::move(P[r]);
                ++w;
            }
        }

        n_old_pairs_pruned = P.size() - w;

        P.resize(w);    

        std::map <PPMonoidElem, std::vector<int>> lcms;
        for (int i = 0; i < m; i ++) {
          lcms[Lif[i]].push_back(i);
        }

        std::vector<PPMonoidElem> min_lcms;
        for (const auto &pair: lcms) {
          PPMonoidElem mon = pair.first;
          std::vector<int> v = pair.second;
          if (std::all_of(min_lcms.begin(), min_lcms.end(), [&mon](const PPMonoidElem &m) { return !IsDivisible(mon, m); })) {
              min_lcms.push_back(mon);
              if (std::none_of(v.begin(), v.end(), [&Lif, &LM, &lpp_f](int i) { return Lif[i] == LM[i] * lpp_f; })) {
                  newPairs.emplace_back(v[0], m, Lif[v[0]], Sugars[v[0]], Sugars[m], degLM[v[0]], deg_lpp_f);
              }
          }
        }

        n_pruned = (m - newPairs.size()) + n_old_pairs_pruned;

        break;
    }

    // Optional deterministic ordering
    std::sort(newPairs.begin(), newPairs.end(),
              [](const SPair& a, const SPair& b) {
                  return (a.i < b.i) || (a.i == b.i && a.j < b.j);
              });

    // Commit new polynomial and its LM cache
    G.push_back(f);
    LM.push_back(lpp_f);
    degLM.push_back(deg(lpp_f));

    // Append new pairs
    P.insert(P.end(),
             std::make_move_iterator(newPairs.begin()),
             std::make_move_iterator(newPairs.end()));

            
    feat_tracker.on_update(f, newPairs, n_pruned, n_old_pairs_pruned, state_feats);

}


  std::vector <RingElem> minimalize(const std::vector <RingElem> &G) {
    std::vector <RingElem> G_ = G;
    std::sort(G_.begin(), G_.end(), [](ConstRefRingElem f, ConstRefRingElem g) { return LPP(f) < LPP(g); });
    std::vector <RingElem> Gmin;
    for (ConstRefRingElem g: G_) {
        if (std::none_of(Gmin.begin(), Gmin.end(),
                          [&g](ConstRefRingElem f) { return IsDivisible(LPP(g), LPP(f)); })) {
            Gmin.push_back(g);
        }
    }
    return Gmin;
  }

  std::vector <RingElem> interreduce(const std::vector <RingElem> &G) {
    std::vector <RingElem> Gred;
    for (ConstRefRingElem g: G) {
        const ring P = owner(g);
        Gred.push_back((reduce(g - monomial(P, LPP(g)), G).first + monomial(P, LPP(g))));
    }
    return Gred;
  }

  std::pair<std::vector<RingElem>, BuchbergerStats> buchberger(const std::vector<RingElem>& F,
  SelectionType selection,
  EliminationType elimination,
  RewardType rewards,
  bool sort_input,
  bool sort_reducers,
  double gamma,
  std::optional<int> seed)
  {
    std::vector <RingElem> G;
    std::vector <PPMonoidElem> LM;
    std::vector <int> degLM;
    std::vector <SPair> P;
    std::vector<RingElem> F_;

    // Sort input generators by cached LPP (optional)
    if (sort_input) {
        std::vector<std::pair<PPMonoidElem, RingElem>> keyed;
        keyed.reserve(F.size());
        for (auto& f : F) keyed.emplace_back(LPP(f), f);

        std::sort(keyed.begin(), keyed.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        for (size_t i = 0; i < F.size(); ++i)
            F_[i] = std::move(keyed[i].second);
    } else {
      F_ = F;
    }

    std::vector <int> Sugars;
    for (ConstRefRingElem f: F_) {
      Sugars.push_back(deg(f));
    }

    for (ConstRefRingElem f: F_) {
        update(G, LM, degLM, Sugars, P, f, elimination);
    }
    return buchberger(G, LM, degLM, Sugars, P, selection, elimination, rewards, sort_reducers, gamma, seed);
  }

  std::pair <std::vector<RingElem>, BuchbergerStats> buchberger(const std::vector <RingElem> &F,
    const std::vector <PPMonoidElem>& LM_F,
    const std::vector <int>& degLM_F,
    const std::vector <int>& Sugars_F,
    const std::vector <SPair> &S,
    SelectionType selection, 
    EliminationType elimination, 
    RewardType rewards,
    bool sort_reducers,
    double gamma,
    std::optional<int> seed) {

    std::vector <RingElem> G = F;
    std::vector <RingElem> G_ = F;
    std::vector <PPMonoidElem> LM = LM_F;
    std::vector <PPMonoidElem> LM_G_ = LM;
    std::vector <int> degLM = degLM_F;
    std::vector <int> Sugars = Sugars_F;
    std::vector <SPair> P = S;
    BuchbergerStats stats = {};
    double discount = 1.0;
    int run_length = 0;

    if (sort_reducers) {
        std::sort(G_.begin(), G_.end(), [](ConstRefRingElem f, ConstRefRingElem g) { return LPP(f) < LPP(g); });
        std::sort(LM_G_.begin(), LM_G_.end(), [](ConstRefPPMonoidElem f, ConstRefPPMonoidElem g) { return f < g; });
    };
    std::vector <int> SugarsG_;
    for (ConstRefRingElem g: G_) {
      SugarsG_.push_back(deg(g));
    }
    // Creates a ''empty'' function that takes two variables of type SPair and returns true of false

    std::mt19937 rng;
    
    auto select = SPairComparator(selection, G, Sugars, seed, rng);

    while (!P.empty()) {
      // The pair belong to the pair set is chosen that is minimal according to the heuristic
      // Note for the future, can define a sort function in a similar manner to this
      auto iter = select.is_random ? choice(P.begin(), P.end(), rng) : std::min_element(P.begin(), P.end(), select.cmp);
      SPair p = *iter;
      P.erase(iter);
      RingElem r;
      long s;
      int r_sugar;
      // std::tie(r,s) = reduce(SPoly(G[p.i], G[p.j]), G_);
      std::tie(r, s, r_sugar) = NormalRemainder(SPoly(G[p.i], G[p.j]), G_, LM_G_, p.Sugar, SugarsG_);
      double reward = (rewards == RewardType::Additions) ? (-1.0 - s) : -1.0;
      stats.polynomial_additions += s + 1;
      stats.total_reward += reward;
      stats.discounted_return += discount * reward;
      discount *= gamma;
      if (!IsZero(r)) {
        Sugars.push_back(r_sugar);
        update(G, LM, degLM, Sugars, P, r, elimination);
        stats.nonzero_reductions++;
        if (sort_reducers) {
          auto pos = std::upper_bound(
          LM_G_.begin(), LM_G_.end(), LM.back());

          size_t idx = std::distance(LM_G_.begin(), pos);

          LM_G_.insert(pos, LM.back());
          G_.insert(G_.begin() + idx, r);
          SugarsG_.insert(SugarsG_.begin() + idx, r_sugar);
        }
        else {
          G_.push_back(r);
          LM_G_.push_back(LM.back());
          SugarsG_.push_back(r_sugar);
        }
      } else {
          stats.zero_reductions++;
      }
      ++run_length;
    }

    return {interreduce(minimalize(G)), stats};

  }


  BuchbergerEnv::BuchbergerEnv(std::string ideal_dist,
                                EliminationType elimination,
                                RewardType rewards,
                                bool sort_input,
                                bool sort_reducers,
                                bool fix_ideal) : elimination(elimination), rewards(RewardType::Additions), sort_input(sort_input),
    sort_reducers(sort_reducers), fix_ideal(fix_ideal) {
    ideal_gen = parse_ideal_dist(ideal_dist);
    }


  BuchbergerEnv::BuchbergerEnv(std::string coeff_ring,
                                std::string symbols_str,
                                const std::vector<std::string>& basis_strs,
                                EliminationType elimination,
                                RewardType rewards,
                                bool sort_input,
                                bool sort_reducers)
      : elimination(elimination), rewards(RewardType::Additions), sort_input(sort_input),
        sort_reducers(sort_reducers), fix_ideal(true) {
      cached_ideal = parse_fixed_ideal(coeff_ring, symbols_str, basis_strs, fixed_nvars);
    }


  BuchbergerEnv::BuchbergerEnv(const BuchbergerEnv &other)
          : G(other.G), LM(other.LM), degLM(other.degLM), P(other.P), Sugars(other.Sugars), state_feats(other.state_feats), elimination(other.elimination), rewards(other.rewards), sort_input(other.sort_input), sort_reducers(other.sort_reducers), fix_ideal(other.fix_ideal), fixed_nvars(other.fixed_nvars), cached_ideal(other.cached_ideal), G_(other.G_), LM_G_(other.LM_G_), SugarsG_(other.SugarsG_), feat_tracker(other.feat_tracker) {
      if (other.ideal_gen) {
        ideal_gen = other.ideal_gen->copy();
      }
  }


  BuchbergerEnv &BuchbergerEnv::operator=(const BuchbergerEnv &other) {
      // the only non-default copy is ideal_gen
      if (other.ideal_gen) {
        ideal_gen = other.ideal_gen->copy();
      } else {
        ideal_gen.reset();
      }
      G = other.G;
      LM = other.LM;
      degLM = other.degLM;
      P = other.P;
      elimination = other.elimination;
      rewards = other.rewards;
      sort_input = other.sort_input;
      sort_reducers = other.sort_reducers;
      fix_ideal = other.fix_ideal;
      fixed_nvars = other.fixed_nvars;
      cached_ideal = other.cached_ideal;
      G_ = other.G_;
      LM_G_ = other.LM_G_;
      Sugars = other.Sugars;
      SugarsG_ = other.SugarsG_;
      state_feats = other.state_feats;
      feat_tracker = other.feat_tracker;
      return *this;
  }

  void BuchbergerEnv::reset() {
    std::vector<RingElem> F;

    if (!fix_ideal) {
        ideal I = ideal_gen->next();
        F = gens(I);
    } else {
        if (!cached_ideal.has_value()) {
          cached_ideal = ideal_gen->next();
          std::cout << "Fixed Ideal:" << std::endl;
          std::cout << *cached_ideal << std::endl;
        }
        ideal I = *cached_ideal;
        F = gens(I);
    }

    // Sort input generators by cached LPP (optional)
    if (sort_input) {
        std::vector<std::pair<PPMonoidElem, RingElem>> keyed;
        keyed.reserve(F.size());
        for (auto& f : F) keyed.emplace_back(LPP(f), f);

        std::sort(keyed.begin(), keyed.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        for (size_t i = 0; i < F.size(); ++i)
            F[i] = std::move(keyed[i].second);
    }

    last_reset_basis_elements.clear();
    last_reset_basis_elements.reserve(F.size());
    for (ConstRefRingElem f : F) {
      std::ostringstream oss;
      oss << f;
      last_reset_basis_elements.push_back(oss.str());
    }

    feat_tracker.reset(state_feats);
    G.clear();
    G_.clear();
    P.clear();
    Sugars.clear();
    SugarsG_.clear();
    LM.clear();
    degLM.clear();
    LM_G_.clear();   // add this member if you want sorted reducers cheaply

    for (ConstRefRingElem f : F)   Sugars.push_back(deg(f));

    for (ConstRefRingElem f : F) {
        update(G, LM, degLM, Sugars, P, f, elimination, state_feats, feat_tracker); // update pushes f into G and LM

        if (sort_reducers) {
            // insert the *newly added* basis element into G_
            const RingElem& gnew = G.back();
            const PPMonoidElem& lmnew = LM.back();

            auto pos = std::upper_bound(LM_G_.begin(), LM_G_.end(), lmnew);
            size_t idx = pos - LM_G_.begin();
            LM_G_.insert(pos, lmnew);
            G_.insert(G_.begin() + idx, gnew);
        } else {
            // if you don't keep a separate sorted reducer list, you can just mirror G
            G_.push_back(G.back());
            LM_G_.push_back(LM.back());
        }
    }

    if (P.empty()) {
        std::cout << "p-empty";
        for (ConstRefRingElem f : F) {
          std::cout << f << "\n";
        }
        reset();
        return;
    }

    for (ConstRefRingElem g : G_)  SugarsG_.push_back(deg(g));
  }

  double BuchbergerEnv::step(int action) {
    state_feats.steps_since_basis_update++;
    SPair active = P[(size_t)action];

    // Remove chosen pair by swap-pop
    P[(size_t)action] = std::move(P.back());
    P.pop_back();

    feat_tracker.on_pair_removed(active, state_feats);

    // Compute S-polynomial and remainder
    RingElem r;
    long stats;
    int r_sugar;
    std::tie(r, stats, r_sugar) =
        NormalRemainder(SPoly(G[active.i], G[active.j]),
                        G_, LM_G_, active.Sugar, SugarsG_);

    if (!IsZero(r)) {
        // update appends r to G and LM, and inserts new pairs into P
        Sugars.push_back(r_sugar);
        update(G, LM, degLM, Sugars, P, r, elimination, state_feats, feat_tracker);

        // Grab the appended polynomial + its LM from caches
        const RingElem& r_added = G.back();
        const PPMonoidElem& lm_added = LM.back();

        if (sort_reducers) {
            // Insert into reducer list using cached LM_G_
            auto pos = std::upper_bound(LM_G_.begin(), LM_G_.end(), lm_added);
            const size_t idx = (size_t)std::distance(LM_G_.begin(), pos);

            LM_G_.insert(pos, lm_added);
            G_.insert(G_.begin() + idx, r_added);
            SugarsG_.insert(SugarsG_.begin() + idx, r_sugar);
        } else {
            G_.push_back(r_added);
            LM_G_.push_back(lm_added);      // keep LM_G_ consistent even if unsorted
            SugarsG_.push_back(r_sugar);
        }
        state_feats.steps_since_basis_update = 0;
    }

    feat_tracker.on_step_completed(state_feats);

    return (rewards == RewardType::Additions) ? (-1.0 - stats) : -1.0;
  }


  double BuchbergerEnv::value(std::string strategy, double gamma) const {
      std::vector <RingElem> G_;
      BuchbergerStats stats;
      std::tie(G_, stats) = buchberger(G, LM, degLM, Sugars, P, strategy_map.at(strategy), elimination, rewards, sort_reducers, gamma);
      return stats.discounted_return;
  }

    std::vector<long> lead_monomials_vector(ConstRefRingElem f, long k, long n_target) {
      std::vector<long> lead;
      const long n_ring = NumIndets(owner(f));
      const long n_cols = (n_target > 0) ? n_target : n_ring;
      SparsePolyIter it = BeginIter(f); // it points to the first term in f
      for (int j{0}; j < std::min(k, NumTerms(f)); j++) {
          std::vector<long> expv; // a vector of long ints
          exponents(expv, PP(it)); // puts the exponent vector of the first term into a vector called expv
        for (long &exp: expv) {
              lead.push_back(exp); // adds the elements of expv to lead
          }
        for (long pad = (long)expv.size(); pad < n_cols; ++pad) {
          lead.push_back(0);
        }
          if (!IsEnded(it)) { // only increment the iterator if it isn't finished
              ++it;
          }
      }
      if (NumTerms(f) < k) {
          for (int i{0}; i < k - NumTerms(f); i++) {
          for (long j{0}; j < n_cols; j++) {
                  lead.push_back(0);
              }
          }
      }
      return lead;
  }

  LeadMonomialsEnv::LeadMonomialsEnv(std::string ideal_dist, 
                                      std::string elimination,
                                      std::string reward,
                                      std::string pq_strategy,
                                      bool sort_input,
                                      bool sort_reducers,
                                      bool fix_ideal,
                                      int N,
                                      int k): env{ideal_dist, elimination_map.at(elimination), reward_map.at(reward), sort_input, sort_reducers, fix_ideal}, N(N), k(k) {
      SPairPriorityStrategy = strategy_map.at(pq_strategy);
      n = env.nvars();
      cols = 2 * env.nvars() * k;
  }

  void LeadMonomialsEnv::reset() {
      env.reset();
      state.clear();
      leads.clear();
      for (ConstRefRingElem g: env.G) {
          leads.push_back(lead_monomials_vector(g, k, n));
      }

      std::mt19937 rng;
      auto select = SPairComparator(SPairPriorityStrategy, env.G, env.Sugars, seed, rng);
      
      N_Pairs = env.P.size();
      int true_N = std::min(N, N_Pairs);

      top_N_P.clear();
      top_N_P.resize(true_N);

      if (select.is_random) {
        top_N_P = random_unique(env.P.begin(), env.P.end(), true_N, rng);
      } else {
        std::partial_sort_copy(env.P.begin(), env.P.end(), top_N_P.begin(), top_N_P.end(), select.cmp);
      }

      for (const auto &p: top_N_P) {
        state.insert(state.end(), leads[p.i].begin(), leads[p.i].end());
        state.insert(state.end(), leads[p.j].begin(), leads[p.j].end());
      }
      if (true_N < N){
        state.insert(state.end(), 2 * (N - true_N) * n * k , 0);
      }
  }

  double LeadMonomialsEnv::step(int action) {
      // Need to find the index of top_N_P[action] in env.P
      auto P_action = std::distance(env.P.begin(), std::find(env.P.begin(), env.P.end(), top_N_P[action]));
      double reward = env.step(P_action);
      try {
          if (leads.size() < env.G.size()) {
              leads.push_back(lead_monomials_vector(env.G[env.G.size() - 1], k, n));
          }
          state.clear();
      } catch (const CoCoA::ErrorInfo &err) {
          std::cerr << "***ERROR***  UNCAUGHT CoCoA error";
          ANNOUNCE(std::cerr, err);
      }

      std::mt19937 rng;
      auto select = SPairComparator(SPairPriorityStrategy, env.G, env.Sugars, seed, rng);
      N_Pairs = env.P.size();
      int true_N = std::min(N, N_Pairs);


      top_N_P.clear();
      top_N_P.resize(true_N);


      if (select.is_random) {
        top_N_P = random_unique(env.P.begin(), env.P.end(), true_N, rng);
      } else {
        std::partial_sort_copy(env.P.begin(), env.P.end(), top_N_P.begin(), top_N_P.end(), select.cmp);
      }

      for (const auto &p: top_N_P) {
          state.insert(state.end(), leads[p.i].begin(), leads[p.i].end());
          state.insert(state.end(), leads[p.j].begin(), leads[p.j].end());
      }
      if (true_N < N){
        state.insert(state.end(), 2 * (N - true_N) * n * k , 0);
      }
      return reward;

  }

  // std::vector<int> LeadMonomialsEnv::features() {
  //     std::vector<int> my_features;
  //     int n_pairs;
  //     n_pairs = env.P.size();
  //     my_features.push_back(n_pairs);
  //     return my_features;
  // }

  void LeadMonomialsEnv::pp_state() {

    std::mt19937 rng;
    SPair first_p;
    SPair last_p;
    SPair degree_p;
    SPair normal_p;
    SPair sugar_p;
    SPair policy_mimic_p;
    SPair true_degree_p;

    const std::array<SelectionType, 6> selection_types = {
        SelectionType::First,
        SelectionType::Degree,
        SelectionType::Normal,
        SelectionType::Sugar,
        SelectionType::PolicyMimic,
        SelectionType::TrueDegree
    };

    for (SelectionType type : selection_types) {
        auto comparator = SPairComparator(type, env.G, env.Sugars, seed, rng);
        auto iter = std::min_element(top_N_P.begin(), top_N_P.end(), comparator.cmp);
        switch (type) {
          case SelectionType::First:
              first_p = *iter;
              break;
            case SelectionType::Last:
              last_p = *iter;
              break;
          case SelectionType::Degree:
              degree_p = *iter;
              break;
          case SelectionType::Normal:
              normal_p = *iter;
              break;
          case SelectionType::Sugar:
              sugar_p = *iter;
              break;
          case SelectionType::PolicyMimic:
              policy_mimic_p = *iter;
              break;
          case SelectionType::Random:
              break;
          case SelectionType::TrueDegree:
              true_degree_p = *iter;
              break;
    }
    }

    std::cout << "----------------------" << "\n";
    std::cout << "Pair Set:" << "\n";
    std::cout << "----------------------" << "\n";

    int counter = 1;

    for (const auto &p: top_N_P) {
        std::cout << counter << ": P_i: " << env.G[p.i] << ", P_j: " << env.G[p.j] << ", x-gamma: " << lcm(LPP(env.G[p.i]), LPP(env.G[p.j])) << ", S-poly: " << SPoly(env.G[p.i], env.G[p.j]);
        if (p == first_p){
          std::cout << " <-- First";
        }
        if (p == last_p){
          std::cout << " <-- Last";
        }
        if (p == degree_p){
          std::cout << " <-- Degree";
        } 
        if (p == normal_p){
          std::cout << " <-- Normal";
        } 
        if (p == sugar_p){
          std::cout << " <-- Sugar";
        } 
        if (p == policy_mimic_p){
          std::cout << " <-- PolicyMimic";
        }
        if (p == true_degree_p){
          std::cout << " <-- TrueDegree";
        } 
        std::cout << "\n\n";
        counter = counter + 1;
      }

    

  }

  // Creating an environment for returning lead exponent vector of G rather than G[p.i], G[p.j]

  GLeadMonomialsEnv::GLeadMonomialsEnv(std::string ideal_dist, 
                                      std::string elimination,
                                      std::string reward,
                                      std::string pq_strategy,
                                      bool sort_input,
                                      bool sort_reducers,
                                      bool fix_ideal,
                                      int N,
                                      int k,
                                      int N_P,
                                      int N_G): env{ideal_dist, elimination_map.at(elimination), reward_map.at(reward), sort_input, sort_reducers, fix_ideal}, N_P(N_P > 0 ? N_P : N), N_G(N_G > 0 ? N_G : N), k(k) {
      SPairPriorityStrategy = strategy_map.at(pq_strategy);
      if (this->N_P <= 0 || this->N_G <= 0) {
        throw std::invalid_argument("GLeadMonomialsEnv requires positive N_P and N_G budgets");
      }
      n = env.nvars();
      n_vars = n;
      cols = env.nvars() * k;
      g_cols = cols;
      p_cols = 2 * cols;
  } 

  GLeadMonomialsEnv::GLeadMonomialsEnv(std::string coeff_ring,
                                      std::string symbols_str,
                                      const std::vector<std::string>& basis_strs,
                                      std::string elimination,
                                      std::string reward,
                                      std::string pq_strategy,
                                      bool sort_input,
                                      bool sort_reducers,
                                      int N,
                                      int k,
                                      int N_P,
                                      int N_G): env{coeff_ring, symbols_str, basis_strs, elimination_map.at(elimination), reward_map.at(reward), sort_input, sort_reducers}, N_P(N_P > 0 ? N_P : N), N_G(N_G > 0 ? N_G : N), k(k) {
      SPairPriorityStrategy = strategy_map.at(pq_strategy);
      if (this->N_P <= 0 || this->N_G <= 0) {
        throw std::invalid_argument("GLeadMonomialsEnv requires positive N_P and N_G budgets");
      }
      n = env.nvars();
      n_vars = n;
      cols = env.nvars() * k;
      g_cols = cols;
      p_cols = 2 * cols;
  }

  void GLeadMonomialsEnv::build_padded_observation() {
    G_padded.assign(N_G * g_cols, 0);
    P_padded.assign(N_P * p_cols, 0);
    P_idx_padded.assign(N_P * 2, -1);
    P_sugars_padded.assign(N_P, 0);
    P_normal_ranks_padded.assign(N_P, 0);
    G_mask.assign(N_G, 0);
    P_mask.assign(N_P, 0);

    const int n_basis = static_cast<int>(leads.size() / g_cols);
    std::vector<int> global_to_local(n_basis, -1);
    std::vector<int> visible_g_indices;
    visible_g_indices.reserve(std::min(N_G, n_basis));

    auto add_visible_g = [&](int global_idx) {
      if (global_idx < 0 || global_idx >= n_basis) {
        return;
      }
      if (global_to_local[global_idx] >= 0) {
        return;
      }
      if (static_cast<int>(visible_g_indices.size()) >= N_G) {
        return;
      }
      global_to_local[global_idx] = static_cast<int>(visible_g_indices.size());
      visible_g_indices.push_back(global_idx);
    };

    // Pair endpoints are the highest-value state context for the visible action set.
    for (const SPair& p : top_N_P) {
      add_visible_g(p.i);
      add_visible_g(p.j);
    }
    // Fill spare context slots with the original basis order for a stable baseline.
    for (int global_idx = 0; global_idx < n_basis; ++global_idx) {
      add_visible_g(global_idx);
      if (static_cast<int>(visible_g_indices.size()) >= N_G) {
        break;
      }
    }

    for (int local_idx = 0; local_idx < static_cast<int>(visible_g_indices.size()); ++local_idx) {
      G_mask[local_idx] = 1;
      const int global_idx = visible_g_indices[local_idx];
      const int src_offset = global_idx * g_cols;
      const int dst_offset = local_idx * g_cols;
      std::copy(
        leads.begin() + src_offset,
        leads.begin() + src_offset + g_cols,
        G_padded.begin() + dst_offset
      );
    }

    const int n_p = std::min(N_P, static_cast<int>(top_N_P.size()));
    for (int i = 0; i < n_p; ++i) {
      const SPair& p = top_N_P[i];
      P_mask[i] = 1;

      P_idx_padded[2 * i] = (p.i >= 0 && p.i < n_basis) ? global_to_local[p.i] : -1;
      P_idx_padded[2 * i + 1] = (p.j >= 0 && p.j < n_basis) ? global_to_local[p.j] : -1;

      if (i < static_cast<int>(P_sugars.size())) {
        P_sugars_padded[i] = P_sugars[i];
      }
      if (i < static_cast<int>(P_normal_ranks.size())) {
        P_normal_ranks_padded[i] = P_normal_ranks[i];
      }

      const int dst_offset = i * p_cols;
      const int src_i = p.i * g_cols;
      const int src_j = p.j * g_cols;

      if (src_i + g_cols <= static_cast<int>(leads.size())) {
        std::copy(
          leads.begin() + src_i,
          leads.begin() + src_i + g_cols,
          P_padded.begin() + dst_offset
        );
      }
      if (src_j + g_cols <= static_cast<int>(leads.size())) {
        std::copy(
          leads.begin() + src_j,
          leads.begin() + src_j + g_cols,
          P_padded.begin() + dst_offset + g_cols
        );
      }
    }
  }

  void GLeadMonomialsEnv::reset(){
    env.reset();
    basis_elements = env.last_reset_basis_elements;
    leads.clear();
    G_terms.clear();
    G_term_counts.clear();
    P_idx.clear();
    for (ConstRefRingElem g: env.G) {
      std::vector<long> lead = lead_monomials_vector(g, k, n);
      leads.insert(leads.end(), lead.begin(), lead.end());

      const long n_terms = NumTerms(g);
      std::vector<long> full_terms = lead_monomials_vector(g, n_terms, n);
      G_terms.insert(G_terms.end(), full_terms.begin(), full_terms.end());
      G_term_counts.push_back(static_cast<int>(n_terms));
    }

    std::mt19937 rng;
    auto select = SPairComparator(SPairPriorityStrategy, env.G, env.Sugars, seed, rng);
    
    N_Pairs = env.P.size();
    int true_N = std::min(N_P, N_Pairs);

    top_N_P.clear();
    top_N_P.resize(true_N);

    if (select.is_random) {
      top_N_P = random_unique(env.P.begin(), env.P.end(), true_N, rng);
    } else {
      std::partial_sort_copy(env.P.begin(), env.P.end(), top_N_P.begin(), top_N_P.end(), select.cmp);
    }

    for (const SPair &p: top_N_P) {
      P_idx.push_back(p.i);
      P_idx.push_back(p.j);
    }

    P_sugars.clear();
    for (const SPair &p: top_N_P) {
      P_sugars.push_back(p.Sugar);
    }

    P_normal_ranks.clear();
    P_normal_ranks.resize(top_N_P.size(), 0);
    if (!top_N_P.empty()) {
      std::vector<int> order(top_N_P.size());
      std::iota(order.begin(), order.end(), 0);
      std::mt19937 normal_rng;
      auto normal_select = SPairComparator(SelectionType::Normal, env.G, env.Sugars, seed, normal_rng);
      std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return normal_select.cmp(top_N_P[a], top_N_P[b]);
      });
      for (int rank = 0; rank < static_cast<int>(order.size()); ++rank) {
        P_normal_ranks[order[rank]] = rank;
      }
    }

    state_feature_vector = state_features_to_vector(env.state_feats);
    build_padded_observation();

  }

  double GLeadMonomialsEnv::step(int action) {
    if (action < 0 || action >= static_cast<int>(top_N_P.size())) {
      throw std::out_of_range("GLeadMonomialsEnv::step received invalid action index");
    }
    size_t prev_G_size = env.G.size();
    auto P_action = std::distance(env.P.begin(), std::find(env.P.begin(), env.P.end(), top_N_P[action]));
    double reward = env.step(P_action);
    // Here we can quickly update leads because we know that G will have grown by at most 1 new element
    if (prev_G_size < env.G.size()) {
      std::vector<long> lead = lead_monomials_vector(env.G[env.G.size() - 1], k, n);
      leads.insert(leads.end(), lead.begin(), lead.end());

      const auto& g_new = env.G[env.G.size() - 1];
      const long n_terms = NumTerms(g_new);
      std::vector<long> full_terms = lead_monomials_vector(g_new, n_terms, n);
      G_terms.insert(G_terms.end(), full_terms.begin(), full_terms.end());
      G_term_counts.push_back(static_cast<int>(n_terms));
    }
    // For P_idx it is maybe not best but it is simplest to rebuild from scratch as the set of pairs can have varied a lot during one step
    P_idx.clear();
    P_sugars.clear();
    
    std::mt19937 rng;
    auto select = SPairComparator(SPairPriorityStrategy, env.G, env.Sugars, seed, rng);
    
    N_Pairs = env.P.size();
    int true_N = std::min(N_P, N_Pairs);

    top_N_P.clear();
    top_N_P.resize(true_N);

    if (select.is_random) {
      top_N_P = random_unique(env.P.begin(), env.P.end(), true_N, rng);
    } else {
      std::partial_sort_copy(env.P.begin(), env.P.end(), top_N_P.begin(), top_N_P.end(), select.cmp);
    }

    for (const SPair &p: top_N_P) {
      P_idx.push_back(p.i);
      P_idx.push_back(p.j);
    }

    for (const SPair &p: top_N_P) {
      P_sugars.push_back(p.Sugar);
    }

    P_normal_ranks.clear();
    P_normal_ranks.resize(top_N_P.size(), 0);
    if (!top_N_P.empty()) {
      std::vector<int> order(top_N_P.size());
      std::iota(order.begin(), order.end(), 0);
      std::mt19937 normal_rng;
      auto normal_select = SPairComparator(SelectionType::Normal, env.G, env.Sugars, seed, normal_rng);
      std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return normal_select.cmp(top_N_P[a], top_N_P[b]);
      });
      for (int rank = 0; rank < static_cast<int>(order.size()); ++rank) {
        P_normal_ranks[order[rank]] = rank;
      }
    }

    state_feature_vector = state_features_to_vector(env.state_feats);
    build_padded_observation();

    return reward;
  }

  NoPQEnv::NoPQEnv( 
                  std::string ideal_dist,
                  std::string elimination,
                  std::string reward,
                  bool sort_input,
                  bool sort_reducers,
                  bool fix_input,
                  int k): env{ideal_dist, elimination_map.at(elimination), reward_map.at(reward), sort_input, sort_reducers, fix_input}, k(k) {
      n = env.nvars();
      cols = 2 * env.nvars() * k;
      g_cols = env.nvars() * k;
  }

  NoPQEnv::NoPQEnv(
                  std::string coeff_ring,
                  std::string symbols_str,
                  const std::vector<std::string>& basis_strs,
                  std::string elimination,
                  std::string reward,
                  bool sort_input,
                  bool sort_reducers,
                  int k): env{coeff_ring, symbols_str, basis_strs, elimination_map.at(elimination), reward_map.at(reward), sort_input, sort_reducers}, k(k) {
      n = env.nvars();
      cols = 2 * env.nvars() * k;
              g_cols = env.nvars() * k;
  }

  void NoPQEnv::reset() {
      env.reset();
      state.clear();
      P_idx.clear();
      G_leads.clear();
      leads.clear();
      for (ConstRefRingElem g: env.G) {
        std::vector<long> g_lead = lead_monomials_vector(g, k, n);
        leads.push_back(g_lead);
        G_leads.insert(G_leads.end(), g_lead.begin(), g_lead.end());
      }

      N_Pairs = env.P.size();
      for (const auto &p: env.P) {
        P_idx.push_back(p.i);
        P_idx.push_back(p.j);
        state.insert(state.end(), leads[p.i].begin(), leads[p.i].end());
        state.insert(state.end(), leads[p.j].begin(), leads[p.j].end());
      }

      state_feature_vector = state_features_to_vector(env.state_feats);
  }

  double NoPQEnv::step(int action) {
      double reward = env.step(action);
      try {
          if (leads.size() < env.G.size()) {
              leads.push_back(lead_monomials_vector(env.G[env.G.size() - 1], k, n));
          }
          state.clear();
        P_idx.clear();
        G_leads.clear();
      } catch (const CoCoA::ErrorInfo &err) {
          std::cerr << "***ERROR***  UNCAUGHT CoCoA error";
          ANNOUNCE(std::cerr, err);
      }

      N_Pairs = env.P.size();

      for (const auto& g_lead : leads) {
        G_leads.insert(G_leads.end(), g_lead.begin(), g_lead.end());
      }

      for (const auto &p: env.P) {
          P_idx.push_back(p.i);
          P_idx.push_back(p.j);
          state.insert(state.end(), leads[p.i].begin(), leads[p.i].end());
          state.insert(state.end(), leads[p.j].begin(), leads[p.j].end());
      }

      state_feature_vector = state_features_to_vector(env.state_feats);

      return reward;

  }

}
