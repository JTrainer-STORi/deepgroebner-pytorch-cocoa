/*
 * An Environment for computing Groebner bases with Buchberger's algorithm (+ CoCoAlib)
 */

#ifndef BUCHBERGER_H
#define BUCHBERGER_H

#include <map>
#include <memory>
#include <optional>
#include <vector>
#include <queue>
#include <iterator>

#include "ideals.h"
#include "CoCoA/library.H"

namespace CoCoA
{

  RingElem SPoly(ConstRefRingElem g, ConstRefRingElem h);

  struct SPair {
    int i = -1;
    int j = -1;
    std::optional<PPMonoidElem> Lij;  // may be empty
    int lcmdeg = 0;
    int Sugar = 0;

    SPair() = default;

    static int compute_pair_sugar(int sugar_i, int sugar_j, int deg_i, int deg_j, int lcm_degree) {
      return std::max(sugar_i - deg_i, sugar_j - deg_j) + lcm_degree;
    }

    // For pairs (i,j) where both indices already exist in LM
    SPair(int ii, int jj, const std::vector<PPMonoidElem>& LM, const std::vector<int>& Sugars, const std::vector<int>& degLM)
        : i(ii), j(jj),
          Lij(lcm(LM[ii], LM[jj])),
          lcmdeg(deg(*Lij)),
          Sugar(compute_pair_sugar(Sugars[ii], Sugars[jj], degLM[ii], degLM[jj], lcmdeg))
    {}

    // Even cleaner: construct directly from a precomputed LCM (no extra lcm call)
    SPair(int ii, int jj, PPMonoidElem L, const std::vector<int>& Sugars, const std::vector<int>& degLM)
        : i(ii), j(jj),
          Lij(std::move(L)),
          lcmdeg(deg(*Lij)),
          Sugar(compute_pair_sugar(Sugars[ii], Sugars[jj], degLM[ii], degLM[jj], lcmdeg))
    {}

    // Construct a pair against a newly added basis element whose lead degree is known
    // but not yet appended to the cached degLM vector.
    SPair(
        int ii,
        int jj,
        PPMonoidElem L,
        int sugar_i,
        int sugar_j,
        int deg_i,
        int deg_j
    )
        : i(ii), j(jj),
          Lij(std::move(L)),
          lcmdeg(deg(*Lij)),
          Sugar(compute_pair_sugar(sugar_i, sugar_j, deg_i, deg_j, lcmdeg))
    {}
};


  struct state_features {
    /* The purpose of this structure is to store the 'handcrafted' state features that will be updated during each step of Buchberger's env and passed to the RL agent to aid in decision making */

    // 1) Global Progress / Size Features
    int G_cardinality = 0; // size of the basis set 
    int P_cardinality = 0; // size of the pair set
    float pair_density = 0.0; // The current number of pairs in the pair set as a fraction of the maximum possible number (G choose 2)

    // 2) Degree-based summaries
    float mean_degree = 0.0; // mean degree of basis elements
    int max_degree = 0; // max degree of basis elements
    int min_degree = 0; // min degree of basis elements
    float st_dev_degree = 0.0; // standard deviation of degree of basis elements

    float mean_lcm_degree = 0.0; // mean degree of lcm of pair set elements
    int max_lcm_degree = 0;
    int min_lcm_degree = 0;
    float std_dev_lcm_degree = 0.0;

    // 3) Pruning/ Progress features
    int n_pairs_pruned = 0; // number of pairs that have been pruned so far
    int n_pairs_processed = 0; // number of pairs that have undergone reduction so dar, proxy for the timestep
    int steps_since_basis_update = 0; // The number of steps since a new basis element has been added

  };

  class FeatureTracker {
    public:
      void reset(state_features& f);

      void on_update(const RingElem& g,
                     const std::vector<SPair>& new_pairs,
                     int n_pairs_pruned,
                     int n_old_pairs_pruned,
                     state_features& f);

      void on_pair_removed(const SPair& p,
                          state_features& f);

      void on_step_completed(state_features& f);

    private:
      // running stats for basis degrees
      int deg_count = 0;
      double deg_mean = 0.0;
      double deg_M2 = 0.0;

      // running stats for lcm degrees
      int lcm_count = 0;
      double lcm_mean = 0.0;
      double lcm_M2 = 0.0;

      int total_pairs_seen = 0;
      int total_pairs_pruned = 0;
    };


inline bool operator==(const SPair& p1, const SPair& p2) { return (p1.i == p2.i) && (p1.j == p2.j); }


  enum class EliminationType {GebauerMoeller, LCM, None};

  enum class RewardType {Additions, Reductions};

  struct BuchbergerStats {
  int zero_reductions;
  int nonzero_reductions;
  int polynomial_additions;
  double total_reward;
  double discounted_return;
  };

  std::pair<RingElem, long> reduce(ConstRefRingElem g, const std::vector<RingElem>& F);

  void update(std::vector<RingElem>& G,
        std::vector<PPMonoidElem>& LM,
        std::vector<int>& degLM,
        std::vector<int>& Sugars,
	      std::vector<SPair>& P,
	      ConstRefRingElem f,
	      EliminationType elimination);

  void update(std::vector<RingElem>& G,
        std::vector<PPMonoidElem>& LM,
        std::vector<int>& degLM,
        std::vector<int>& Sugars,
	      std::vector<SPair>& P,
	      ConstRefRingElem f,
	      EliminationType elimination,
        state_features& state_feats,
        FeatureTracker& feat_tracker);

  /**
 * Selection option for selecting next pairs from the pair set.
 */
  enum class SelectionType {First, Last, Degree, Normal, Sugar, PolicyMimic, Random, TrueDegree};

  struct SelectionResult {
    std::function<bool(const SPair&, const SPair&)> cmp; // empty if random
    bool is_random = false;
  };
  
  SelectionResult SPairComparator(SelectionType selection, const std::vector<RingElem>& G, const std::vector<int>& Sugars,  std::optional<unsigned> seed, std::mt19937& rng);

  /**
 * Return a minimal Groebner basis from arbitrary Groebner basis G.
 */
  std::vector<RingElem> minimalize(const std::vector<RingElem>& G);

  /**
 * Return the reduced Groebner basis from minimal Groebner basis G.
 */
  std::vector<RingElem> interreduce(const std::vector<RingElem>& G);

  /**
   * Return the Groebner basis for the ideal generated by F using Buchberger's algorithm.
   *
   * @param F Polynomial generators of the ideal.
   * @param selection Strategy for pair selection.
   * @param elimination Strategy for pair elimination.
   * @param rewards Reward value for each step.
   * @param sort_input Whether to sort the initial generating set by lead monomial.
   * @param sort_reducers Whether to choose reducers in sorted order by lead monomial.
   * @param gamma Discount rate for rewards.
   * @param seed Optional seed if using Random selection.
   */
  std::pair<std::vector<RingElem>, BuchbergerStats> buchberger(const std::vector<RingElem>& F,
							       SelectionType selection = SelectionType::Degree,
							       EliminationType elimination = EliminationType::GebauerMoeller,
							       RewardType rewards = RewardType::Additions,
							       bool sort_input = false,
							       bool sort_reducers = true,
							       double gamma = 1.0,
							       std::optional<int> seed = std::nullopt);



  /**
 * Return the Groebner basis for the ideal generated by F and starting pair set S using Buchberger's algorithm.
 *
 * @param F Generating set of the ideal.
 * @param S Current set of remaining s-pairs.
 * @param selection Strategy for pair selection.
 * @param elimination Strategy for pair elimination.
 * @param rewards Reward value for each step.
 * @param sort_reducers Whether to choose reducers in sorted order by lead monomial.
 * @param gamma Discount rate for rewards.
 * @param seed Optional seed if using Random selection.
 */
std::pair<std::vector<RingElem>, BuchbergerStats> buchberger(const std::vector<RingElem>& F,
  const std::vector<PPMonoidElem>& LM_F,
  const std::vector<int>& degLM_F,
  const std::vector<int>& Sugars_F,
							     const std::vector<SPair>& S,
							     SelectionType selection = SelectionType::Degree,
							     EliminationType elimination = EliminationType::GebauerMoeller,
							     RewardType rewards = RewardType::Additions,
							     bool sort_reducers = true,
							     double gamma = 1.0,
							     std::optional<int> seed = std::nullopt);

 /**
 * An environment for computing a Groebner basis using Buchberger's algorithm.
 */
  class BuchbergerEnv {

  public:

    /**
     * @param ideal_dist String naming the ideal distribution.
     * @param elimination Strategy for pair elimination.
     * @param rewards Reward value for each step.
     * @param sort_input Whether to sort the initial generating set by lead monomial.
     * @param sort_reducers Whether to choose reducers in sorted order by lead monomial.
     */
    BuchbergerEnv(std::string ideal_dist = "3-20-10-weighted",
		  EliminationType elimination = EliminationType::GebauerMoeller,
		  RewardType rewards = RewardType::Reductions,
		  bool sort_input = false,
		  bool sort_reducers = true,
      bool fix_ideal = false);

    /**
     * @param coeff_ring String naming the coefficient ring (e.g. "QQ", "ZZ", "ZZmod(32003)").
     * @param symbols_str Comma-separated list of indeterminate names, e.g. "x,y,z".
     * @param basis_strs List of polynomial strings defining the fixed ideal generators.
     */
    BuchbergerEnv(std::string coeff_ring,
      std::string symbols_str,
      const std::vector<std::string>& basis_strs,
      EliminationType elimination = EliminationType::GebauerMoeller,
      RewardType rewards = RewardType::Reductions,
      bool sort_input = false,
      bool sort_reducers = true);

    // define destructor/copy/move since std::unique_ptr has no copy constructor
    ~BuchbergerEnv() = default;
    BuchbergerEnv(const BuchbergerEnv& other);
    BuchbergerEnv& operator=(const BuchbergerEnv& other);
    BuchbergerEnv(BuchbergerEnv&& other) = default;
    BuchbergerEnv& operator=(BuchbergerEnv&& other) = default;

    void reset();

    double step(int action);

    void seed(int seed) {
      if (ideal_gen) ideal_gen->seed(seed);
    }

    int nvars() const {
      if (fixed_nvars > 0) return fixed_nvars;
      return ideal_gen->nvars();
    }

    double value(std::string strategy = "degree", double gamma = 1.0) const;

    std::vector<RingElem> G;
    std::vector<PPMonoidElem> LM;
    std::vector<int> degLM;
    // std::vector<int> nterms;
    std::vector<SPair> P;
    std::vector<int> Sugars;

    // Exact generator polynomials sampled/loaded at the most recent reset().
    std::vector<std::string> last_reset_basis_elements;

    state_features state_feats;

    private:

    std::unique_ptr<IdealGenerator> ideal_gen;
    EliminationType elimination;
    RewardType rewards;
    bool sort_input;
    bool sort_reducers;
    bool fix_ideal;
    int fixed_nvars = 0;

    std::optional<ideal> cached_ideal;
    std::vector<RingElem> G_;  // the reducers
    std::vector<PPMonoidElem> LM_G_;
    std::vector<int> SugarsG_;

    FeatureTracker feat_tracker;

  };

  /**
   * A BuchbergerEnv with state the matrix of pairs' lead monomials.
   */
  class LeadMonomialsEnv {

  public:

    /**
     * @param ideal_dist The Distribution from which ideals will be sampled
     * @param elimination The strategy for elimination of S-Pairs 
     * @param reward How reward will be assigned to states and actions
     * @param strategy The strategy for prioritizing S-Pairs
     * @param N Backward-compatible default for both pair and basis budgets.
     * @param N_P The maximum number of S-Pairs/actions shown in the state.
     * @param N_G The maximum number of basis elements shown as state context.
     * @param sort_input Whether to sort the initial generating set by lead monomial.
     * @param sort_reducers Whether to choose reducers in sorted order by lead monomial.
     * @param k The number of lead monomials shown for each polynomial.
     */

    LeadMonomialsEnv(std::string ideal_dist = "3-20-10-weighted",
        std::string elimination = "gebauermoeller",
        std::string reward = "additions",
        std::string pq_strategy = "first",
		    bool sort_input = false,
		    bool sort_reducers = true,
        bool fix_ideal = false,
        int N = 20,
		    int k = 2);

    void reset();

    double step(int action);

    void set_seed(int s) { 
      seed = s;
      env.seed(s); 
    };

    double value(std::string strategy = "degree", double gamma = 1.0) const { return env.value(strategy, gamma); };

    void pp_state();

    std::vector<int> state;
    std::vector<SPair> top_N_P;
    int cols;
    int N_Pairs;

  private:

    BuchbergerEnv env;
    std::optional<int> seed;
    int N;
    int n;
    int k;
    SelectionType SPairPriorityStrategy;
    std::vector<std::vector<long>> leads;

  };

  class GLeadMonomialsEnv {

  public:

    /**
     * @param ideal_dist The Distribution from which ideals will be sampled
     * @param elimination The strategy for elimination of S-Pairs 
     * @param reward How reward will be assigned to states and actions
     * @param strategy The strategy for prioritizing S-Pairs
     * @param N The maximum number of S-Pairs that will be in the state
     * @param sort_input Whether to sort the initial generating set by lead monomial.
     * @param sort_reducers Whether to choose reducers in sorted order by lead monomial.
     * @param k The number of lead monomials shown for each polynomial.
     */

    GLeadMonomialsEnv(std::string ideal_dist = "3-20-10-weighted",
        std::string elimination = "gebauermoeller",
        std::string reward = "additions",
        std::string pq_strategy = "first",
        bool sort_input = false,
        bool sort_reducers = true,
        bool fix_ideal = false,
        int N = 100,
        int k = 2,
        int N_P = -1,
        int N_G = -1);

      GLeadMonomialsEnv(std::string coeff_ring,
        std::string symbols_str,
        const std::vector<std::string>& basis_strs,
        std::string elimination = "gebauermoeller",
        std::string reward = "additions",
        std::string pq_strategy = "first",
        bool sort_input = false,
        bool sort_reducers = true,
        int N = 100,
        int k = 2,
        int N_P = -1,
        int N_G = -1);

    void reset();

    double step(int action);

    void set_seed(int s) { 
      seed = s;
      env.seed(s); 
    };

    double value(std::string strategy = "degree", double gamma = 1.0) const { return env.value(strategy, gamma); };

    std::vector<int> leads;
    std::vector<int> G_terms;
    std::vector<int> G_term_counts;
    int n_vars;
    int cols;
    int p_cols;
    int g_cols;

    std::vector<SPair> top_N_P;
    std::vector<int> P_idx;
    std::vector<int> P_sugars;
    std::vector<int> P_normal_ranks;
    std::vector<std::string> basis_elements;
    std::vector<int> G_padded;
    std::vector<int> P_padded;
    std::vector<int> P_idx_padded;
    std::vector<int> P_sugars_padded;
    std::vector<int> P_normal_ranks_padded;
    std::vector<int> G_mask;
    std::vector<int> P_mask;
    int N_Pairs;

    std::vector<float> state_feature_vector;

  private:

    BuchbergerEnv env;
    std::optional<int> seed;
    int N_P;
    int N_G;
    int n;
    int k;
    SelectionType SPairPriorityStrategy;

    void build_padded_observation();

  };

  class NoPQEnv {

    public:

      /**
       * @param elimination The strategy for elimination of S-Pairs 
       * @param reward How reward will be assigned to states and actions
       * @param strategy The strategy for prioritizing S-Pairs
       * @param sort_input Whether to sort the initial generating set by lead monomial.
       * @param sort_reducers Whether to choose reducers in sorted order by lead monomial.
       * @param k The number of lead monomials shown for each polynomial.
       */

      NoPQEnv(
          std::string ideal_dist = "3-20-10-weighted",
          std::string elimination = "gebauermoeller",
          std::string reward = "additions",
          bool sort_input = false,
          bool sort_reducers = true,
          bool fix_input = false,
          int k = 2);

        NoPQEnv(
          std::string coeff_ring,
          std::string symbols_str,
          const std::vector<std::string>& basis_strs,
          std::string elimination = "gebauermoeller",
          std::string reward = "additions",
          bool sort_input = false,
          bool sort_reducers = true,
          int k = 2);

      void reset();

      double step(int action);

      void set_seed(int s) { 
        seed = s;
        env.seed(s); 
      };

      double value(std::string strategy = "degree", double gamma = 1.0) const { return env.value(strategy, gamma); };

      std::vector<int> state;
      std::vector<int> P_idx;
      std::vector<int> G_leads;
      std::vector<float> state_feature_vector;
      int cols;
      int g_cols;
      int N_Pairs;


    private:

      BuchbergerEnv env;
      std::optional<int> seed;
      int n;
      int k;
      std::vector<std::vector<long>> leads;

  };


}

#endif
