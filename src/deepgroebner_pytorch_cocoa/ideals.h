/*
 *Generators of ideals with CoCoAlib (Just GeomIdealGenerator for now)
 */

#ifndef IDEALS_H
#define IDEALS_H

#include <memory>
#include <random>
#include <vector>
#include <string>
#include <utility>
#include <map>
#include "CoCoA/library.H"

namespace CoCoA
{

  int binomial(int n, int k);

  /*
   * Class definitions
   */

  enum class DistributionType {Uniform, Weighted, Maximum};

  enum class IdealClass {Cyclic, Acyclic, Eco, Katsura, Noon, Reimer};

  enum class GenerationMethod {Random, All, One};

  class IdealGenerator // Abstract base class for all ideal generators
  {

  public:

    virtual ideal next() = 0; // next() function returns an CoCoA::ideal

    virtual ~IdealGenerator() {}; // destructor: not really necessary here

    virtual void seed(int seed) {} 

    virtual int nvars() const = 0;

    auto copy() const { return std::unique_ptr<IdealGenerator>(copy_deriv());}

  protected:

    virtual IdealGenerator* copy_deriv() const = 0;

  };

  /**
 * Generator of random examples of binomial ideals.
 */
  class RandomBinomialIdealGenerator : public IdealGenerator {

  public:

    /**
     * @param n Number of variables in the polynomial ring.
     * @param d Maximum degree of a monomial.
     * @param s Number of generators of each ideal.
     * @param dist Type of distribution on monomials.
     * @param constants Whether to include constants as monomials.
     * @param homogeneous Whether the binomials are homogeneous.
     * @param pure Whether the binomials are pure.
     */
    RandomBinomialIdealGenerator(int n = 3, int d = 20, int s = 10, DistributionType dist = DistributionType::Uniform, bool constants = false, bool homogeneous = false, bool pure = false);

    ideal next() override;

    void seed(int seed) override { rng.seed(seed); }

    int nvars() const override { return n; }

  protected:

    RandomBinomialIdealGenerator* copy_deriv() const override { return new RandomBinomialIdealGenerator(*this); }

  private:

    int n;
    int s;
    bool homogeneous;
    bool pure;
    std::vector<std::vector<PPMonoidElem>> bases;
    std::mt19937 rng;
    std::discrete_distribution<int> degree_dist;
    std::uniform_int_distribution<int> coeff_dist{1, 32003-1};

  };

  /**
   * Generator for random binomial ideals where (n, d, s) are sampled independently
   * from integer ranges on each call to next().
   */
  class RangeRandomBinomialIdealGenerator : public IdealGenerator {

  public:

    RangeRandomBinomialIdealGenerator(
      int n_min,
      int n_max,
      int d_min,
      int d_max,
      int s_min,
      int s_max,
      DistributionType dist = DistributionType::Uniform,
      bool constants = false,
      bool homogeneous = false,
      bool pure = false
    );

    ideal next() override;

    void seed(int seed) override { rng.seed(seed); }

    // Return max n so downstream environments can allocate a fixed observation shape.
    int nvars() const override { return n_max; }

  protected:

    RangeRandomBinomialIdealGenerator* copy_deriv() const override { return new RangeRandomBinomialIdealGenerator(*this); }

  private:

    struct PrecomputedNDData {
      ring poly_ring;
      std::vector<std::vector<PPMonoidElem>> bases;
      std::discrete_distribution<int> degree_dist;
    };

    PrecomputedNDData& get_precomputed(int n, int d);

    int n_min;
    int n_max;
    int d_min;
    int d_max;
    int s_min;
    int s_max;
    DistributionType dist;
    bool constants;
    bool homogeneous;
    bool pure;
    std::mt19937 rng;
    std::uniform_int_distribution<int> coeff_dist{1, 32003-1};
    std::map<std::pair<int, int>, PrecomputedNDData> precomputed_by_nd;

  };

  /**
   * Generator for geometric/Laman ideals where n is sampled from [n_min, n_max]
   * on each call to next().
   */
  class RangeGeomIdealGenerator : public IdealGenerator {

  public:

    RangeGeomIdealGenerator(long n_min, long n_max, std::string laman_graph_directory);

    ideal next() override;

    void seed(int seed) override { rng.seed(seed); }

    // Return max variable count so downstream environments can keep fixed shapes.
    int nvars() const override { return static_cast<int>(2*n_max - 3); }

  protected:

    RangeGeomIdealGenerator* copy_deriv() const override { return new RangeGeomIdealGenerator(*this); }

  private:

    long n_min;
    long n_max;
    std::string laman_graph_directory;
    std::mt19937 rng;
    std::uniform_int_distribution<long> n_dist;

  };

    class GeomIdealGenerator : public IdealGenerator // Generator of polynomial Ideals corresponding to the set of polynomial equations describing a two vertex laman graph
  {

  public:

    /*
     * @param n Numberof vertices of Laman graph.
     */

    GeomIdealGenerator(long n, std::string laman_graph_directory);

    ideal next() override;

    void seed(int seed) override { rng.seed(seed); }

    int nvars() const override { return static_cast<int>(2*n - 3); }

    std::vector<std::pair<int, int>> edges;

    ring P;

  protected:

    GeomIdealGenerator* copy_deriv() const override { return new GeomIdealGenerator(*this); }

  private:

    long n;
    std::string laman_graph_directory;
    long index;
    GenerationMethod gmethod;
    std::mt19937 rng;
    std::uniform_int_distribution<int> coeff_dist{1, 32003-1};

  };

  /**
   * Generator for benchmark families (katsura/cyclic/acyclic/eco/noon/reimer)
   * where both family and n can be sampled from user-specified sets/ranges.
   */
  class BenchmarkIdealGenerator : public IdealGenerator {

  public:

    BenchmarkIdealGenerator(std::vector<IdealClass> classes, int n_min, int n_max);

    ideal next() override;

    void seed(int seed) override { rng.seed(seed); }

    int nvars() const override;

  protected:

    BenchmarkIdealGenerator* copy_deriv() const override { return new BenchmarkIdealGenerator(*this); }

  private:

    std::vector<IdealClass> classes;
    int n_min;
    int n_max;
    std::vector<std::pair<IdealClass, int>> class_n_pairs;
    std::mt19937 rng;
    std::uniform_int_distribution<size_t> pair_dist;

  };

  std::unique_ptr<IdealGenerator> parse_ideal_dist(
    const std::string& ideal_dist,
    const std::string& laman_graph_directory = ""
  );

  template<class Iter>
  Iter choice(Iter begin, Iter end, std::mt19937& rng) {
    std::uniform_int_distribution<> dist(0, std::distance(begin, end) - 1);
    std::advance(begin, dist(rng));
    return begin;
  }

  // std::string SampleIdealDist(std::string ideal_dist, int seed);

} // end of namespace CoCoA



#endif
