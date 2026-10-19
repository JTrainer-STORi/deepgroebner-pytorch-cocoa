/* This program reads in numerical representations of Laman Graphs from file
and converts them into ideals that can be passed to Buchberger's algorithm */

// TODO: Consider replacing the method by which binomials are randomly generated to be more efficient
// i.e. instead of generating every permutation of monomials, just generate one monomial


#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <typeinfo>
#include <array>
#include <map>
#include <set>
#include <cctype>
#include <stdexcept>
#include <unordered_map>

#include "CoCoA/library.H"
#include "ideals.h"

// helper functions for translating Laman graphs from their integer representation in the .txt files to a basis
std::string toBinary(long n) //converts an long int (base 2) to string of that int (base 10)
{
  std::string r;
  while (n != 0)
  {
    r += ( n % 2 == 0 ? "0" : "1");
    n /= 2;
  }
  return r;
}

long matDim(const long& g)
{
  double temp = std::floor((3 + std::sqrt(8*std::floor(std::log2(g)+1)))/2); // calculates the dimension of a matrix based on its integer (base 10) represntation
  long dim;
  dim = long (temp);
  return dim;
}

long sizeUD(const long& n)
{
  long s = n*(n-1)/2; // calculates the number of entries in the upper diagonal of a matrix based on its dimension
  return s;
}

std::vector<std::pair<int, int>> intToEdges(
    const std::string& nVertices,
    const std::string& laman_graph_directory,
    std::mt19937& rng)
{
  std::string line;
  // Avoid std::filesystem's macOS 10.15 minimum when only joining path components.
  const std::string filepath =
      laman_graph_directory + "/LamanGraphs" + nVertices + ".txt";
  std::ifstream myfile (filepath);
  std::vector<long> record;
  std::vector<std::pair<int, int>> edges; // initialize vector of edges

  if (myfile.is_open()) {
      while( std::getline (myfile, line) )
      {
        // Input lines are stored like "{3326, 4011, ...}" or "{7}".
        // Parse integers directly instead of slicing by fixed offsets.
        for (char& ch : line) {
          if (ch == '{' || ch == '}') ch = ' ';
        }

        std::stringstream ss(line);
        std::string field;
        while (std::getline( ss, field, ','))
          {
            // for each field, we wish to convert it to an int
            std::stringstream fs(field);
            long f;
            if (fs >> f)
              record.push_back(f);
          }

      }
      if (record.empty()) {
        throw std::runtime_error("No Laman graph encodings found in " + filepath);
      }

      std::uniform_int_distribution<int> dist(0, (record.size() - 1));


      int index = dist(rng);

      const size_t chosen = static_cast<size_t>(index) % record.size();
      std::string q = toBinary(record[chosen]); // convert integer rep to binary
      std::reverse(q.begin(),q.end());
      int d = matDim(record[chosen]); // calculate dimension of graph from int rep
      int sz = sizeUD(d); // calculate size of upper diagonal from int rep
      q.insert(q.begin(), sz - q.size(), '0'); // pad binary rep with zeros until it is the correct size
      int currentPosition = 0; // keeps track of current position in UD matrix
      for (int m {1}; m < d; m++){ // iterates over rows of matrix, first row has dimension d-1, second d-2, ... etc
        std::string row = q.substr(currentPosition, (d-m)); // splits the binary rep corresponding to entries in UD matrix
        auto it {row.begin()};
        for(unsigned j {0}; j < row.size(); j++){ //iterates over each row
          if (*it == '1'){ // if entry is a 1 then this corresponds to an edge
              edges.push_back(std::make_pair(m, j+m+1)); // bit of fudging here to do with the fact that the size of the row above the diagonal is getting smaller but the entries correspond to later entries
            }
            advance(it, 1);
          }
          currentPosition = currentPosition + (d-m); // update current position in the UDmatrix to correspond to the start of the next row
      }

      myfile.close();
    } else {
      throw std::runtime_error("Unable to open Laman graph data file: " + filepath);
    }

  return edges;

}

namespace CoCoA
{

  // functions for parsing input distributions 
  namespace {

    struct IntRange {
      int lo;
      int hi;
    };

    std::string trim_copy(const std::string& s) {
      size_t begin = 0;
      while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) ++begin;
      size_t end = s.size();
      while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
      return s.substr(begin, end - begin);
    }

    std::vector<std::string> split_top_level_dash(const std::string& s) {
      std::vector<std::string> out;
      std::string cur;
      int paren_depth = 0;
      for (char c : s) {
        if (c == '(') {
          ++paren_depth;
          cur.push_back(c);
        } else if (c == ')') {
          --paren_depth;
          if (paren_depth < 0) {
            throw std::invalid_argument("Malformed ideal_dist: unmatched ')' in " + s);
          }
          cur.push_back(c);
        } else if (c == '-' && paren_depth == 0) {
          out.push_back(trim_copy(cur));
          cur.clear();
        } else {
          cur.push_back(c);
        }
      }
      if (paren_depth != 0) {
        throw std::invalid_argument("Malformed ideal_dist: unmatched '(' in " + s);
      }
      out.push_back(trim_copy(cur));
      return out;
    }

    IntRange parse_int_or_range_token(const std::string& raw, const std::string& name) {
      const std::string token = trim_copy(raw);
      if (token.empty()) {
        throw std::invalid_argument("Malformed ideal_dist: empty token for " + name);
      }

      // Single integer.
      if (token.front() != '(') {
        const int v = std::stoi(token);
        return {v, v};
      }

      // Parenthesized range in the form (a-b).
      if (token.back() != ')') {
        throw std::invalid_argument("Malformed range for " + name + ": " + token);
      }
      const std::string inner = trim_copy(token.substr(1, token.size() - 2));
      const size_t sep = inner.find('-');
      if (sep == std::string::npos) {
        throw std::invalid_argument("Malformed range for " + name + ": " + token);
      }

      const int lo = std::stoi(trim_copy(inner.substr(0, sep)));
      const int hi = std::stoi(trim_copy(inner.substr(sep + 1)));
      if (lo > hi) {
        throw std::invalid_argument("Invalid range for " + name + ": lower bound > upper bound in " + token);
      }
      return {lo, hi};
    }

    std::string to_lower_copy(std::string s) {
      std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
      return s;
    }

    bool class_supports_n(IdealClass cls, int n) {
      switch (cls) {
        case IdealClass::Katsura:
          return n >= 1;
        case IdealClass::Cyclic:
        case IdealClass::Acyclic:
        case IdealClass::Eco:
        case IdealClass::Noon:
        case IdealClass::Reimer:
          return n >= 2;
      }
      return false;
    }

    int class_nvars(IdealClass cls, int n) {
      return n;
    }

    std::vector<IdealClass> parse_ideal_classes_token(const std::string& raw) {
      std::string token = trim_copy(raw);
      if (token.empty()) {
        throw std::invalid_argument("Malformed benchmark ideal_dist: empty class token");
      }

      if (token.front() == '(') {
        if (token.back() != ')') {
          throw std::invalid_argument("Malformed benchmark class token: " + token);
        }
        token = trim_copy(token.substr(1, token.size() - 2));
      }

      std::map<std::string, IdealClass> class_map = {
          {"cyclic", IdealClass::Cyclic},
          {"acyclic", IdealClass::Acyclic},
          {"eco", IdealClass::Eco},
          {"katsura", IdealClass::Katsura},
          {"noon", IdealClass::Noon},
          {"reimer", IdealClass::Reimer},
      };

      std::vector<IdealClass> out;
      std::set<IdealClass> seen;
      std::string part;

      auto flush_part = [&]() {
        const std::string piece = to_lower_copy(trim_copy(part));
        if (piece.empty()) {
          throw std::invalid_argument("Malformed benchmark class token: " + raw);
        }
        auto it = class_map.find(piece);
        if (it == class_map.end()) {
          throw std::invalid_argument("Unknown benchmark ideal class: " + piece);
        }
        if (seen.insert(it->second).second) {
          out.push_back(it->second);
        }
        part.clear();
      };

      for (char c : token) {
        if (c == '+' || c == ',') {
          flush_part();
        } else {
          part.push_back(c);
        }
      }
      flush_part();

      if (out.empty()) {
        throw std::invalid_argument("Malformed benchmark ideal_dist: no classes specified");
      }
      return out;
    }

    ring make_poly_ring_with_x0_to_xn(int nvars) {
      return NewPolyRing(NewZZmod(32003), SymbolRange("x", 0, nvars - 1));
    }

    std::vector<RingElem> build_katsura_polys(int nvars) {
      const ring P = make_poly_ring_with_x0_to_xn(nvars);
      std::vector<RingElem> x(indets(P).begin(), indets(P).end());
      std::vector<RingElem> polys;
      polys.reserve(static_cast<size_t>(nvars));
      const int order = nvars - 1;

      for (int i = 0; i < order; ++i) {
        RingElem poly = -x[static_cast<size_t>(i)];
        for (int j = -order; j <= order; ++j) {
          const int a = std::abs(j);
          const int b = std::abs(i - j);
          if (a <= order && b <= order) {
            poly += x[static_cast<size_t>(a)] * x[static_cast<size_t>(b)];
          }
        }
        polys.push_back(poly);
      }

      RingElem linear = x[0] - RingElem(P, 1);
      for (int i = 1; i < nvars; ++i) {
        linear += RingElem(P, 2) * x[static_cast<size_t>(i)];
      }
      polys.push_back(linear);

      return polys;
    }

    std::vector<RingElem> build_cyclic_polys(int n, bool include_product_term) {
      const ring P = make_poly_ring_with_x0_to_xn(n);
      std::vector<RingElem> x(indets(P).begin(), indets(P).end());
      std::vector<RingElem> polys;
      polys.reserve(static_cast<size_t>(n));

      for (int degree = 1; degree < n; ++degree) {
        RingElem sum = zero(P);
        for (int i = 0; i < n; ++i) {
          RingElem term = RingElem(P, 1);
          for (int offset = 0; offset < degree; ++offset) {
            term *= x[static_cast<size_t>((i + offset) % n)];
          }
          sum += term;
        }
        polys.push_back(sum);
      }

      if (include_product_term) {
        RingElem prod = RingElem(P, 1);
        for (int i = 0; i < n; ++i) {
          prod *= x[static_cast<size_t>(i)];
        }
        polys.push_back(prod - 1);
      }

      return polys;
    }

    std::vector<RingElem> build_eco_polys(int n) {
      const ring P = make_poly_ring_with_x0_to_xn(n);
      std::vector<RingElem> x(indets(P).begin(), indets(P).end());
      std::vector<RingElem> polys;
      polys.reserve(static_cast<size_t>(n));

      for (int i = 0; i < n - 1; ++i) {
        RingElem inner = x[static_cast<size_t>(i)];
        for (int j = i; j <= n - 2; ++j) {
          inner += x[static_cast<size_t>(j)] * x[static_cast<size_t>(j + 1)];
        }
        polys.push_back(inner * x[static_cast<size_t>(n - 1)] - RingElem(P, i + 1));
      }

      RingElem linear = RingElem(P, 1);
      for (int i = 0; i < n - 1; ++i) {
        linear += x[static_cast<size_t>(i)];
      }
      polys.push_back(linear);

      return polys;
    }

    std::vector<RingElem> build_noon_polys(int n) {
      const ring P = make_poly_ring_with_x0_to_xn(n);
      std::vector<RingElem> x(indets(P).begin(), indets(P).end());
      std::vector<RingElem> polys;
      polys.reserve(static_cast<size_t>(n));

      RingElem sumsq = zero(P);
      for (int j = 0; j < n; ++j) {
        sumsq += power(x[static_cast<size_t>(j)], 2);
      }

      for (int i = 0; i < n; ++i) {
        const RingElem xi = x[static_cast<size_t>(i)];
        const RingElem xi_sq = power(xi, 2);
        polys.push_back(
          RingElem(P, 10) * xi * (sumsq - xi_sq)
          - RingElem(P, 11) * xi
          + RingElem(P, 10)
        );
      }

      return polys;
    }

    std::vector<RingElem> build_reimer_polys(int n) {
      const ring P = make_poly_ring_with_x0_to_xn(n);
      std::vector<RingElem> x(indets(P).begin(), indets(P).end());
      std::vector<RingElem> polys;
      polys.reserve(static_cast<size_t>(n));

      for (int degree = 2; degree <= n + 1; ++degree) {
        RingElem poly = RingElem(P, -1);
        for (int i = 0; i < n; ++i) {
          const RingElem coeff = RingElem(P, (i % 2 == 0) ? 2 : -2);
          poly += coeff * power(x[static_cast<size_t>(i)], degree);
        }
        polys.push_back(poly);
      }

      return polys;
    }

    std::vector<RingElem> build_benchmark_polys(IdealClass cls, int n) {
      switch (cls) {
        case IdealClass::Katsura:
          return build_katsura_polys(n);
        case IdealClass::Cyclic:
          return build_cyclic_polys(n, true);
        case IdealClass::Acyclic:
          return build_cyclic_polys(n, false);
        case IdealClass::Eco:
          return build_eco_polys(n);
        case IdealClass::Noon:
          return build_noon_polys(n);
        case IdealClass::Reimer:
          return build_reimer_polys(n);
      }
      throw std::invalid_argument("Unknown benchmark ideal class");
    }

  } // namespace

  std::vector<PPMonoidElem> basis(const PPMonoid& PPM, int n, int d) {

    // Start by populating a vector with the appropriate numbers of bars (1) and stars (0)
    std::vector<int> a;
    for (int i = 0; i < d; i++)
      a.push_back(0);
    for (int i = 0; i < n - 1; i++)
      a.push_back(1);

    // all permutations of vector correspond to all monomials in degree d: This is way too big for large d or n
    std::vector<PPMonoidElem> B;
    do {
      std::vector<long> expv(n, 0);
      int index = 0;
      // step through the bars and stars vector and increment the exponent of the first variable for each star,
      // when a bar is encountered, move to the exponent of the next variable
      for (std::size_t i = 0; i < a.size(); i++) {
        if (a[i] == 0)
          expv[index]++;
        else
          index++;
      }
      PPMonoidElem t(PPM, expv);
      B.push_back(t);
    } while (std::next_permutation(a.begin(), a.end())); // For each permutation of bars and stars
    return B;
  }

  int binomial(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    k = std::min(k, n - k);
    long long c = 1;
    for (int i = 1; i <= k; ++i) {
      c = (c * (n - k + i)) / i;
    }
    return static_cast<int>(c);
  }

  std::discrete_distribution<int> degree_distribution(int n, int d, DistributionType dist) {
    std::vector<int> count{0};

    switch (dist) {
        case DistributionType::Uniform:
          for (int i = 1; i < d + 1; i++) {
              count.push_back(binomial(n + i - 1, n - 1));
          }
          break;
        case DistributionType::Weighted:
          for (int i = 0; i < d; i++) {
              count.push_back(1);
          }
          break;
        case DistributionType::Maximum:
          for (int i = 0; i < d - 1; i++) {
              count.push_back(0);
          }
          count.push_back(1);
          break;
    }

    return {count.begin(), count.end()};
  }

  RandomBinomialIdealGenerator::RandomBinomialIdealGenerator(int n, int d, int s, DistributionType dist) : n(n), s(s) {
    
    PPOrdering ord = StdDegRevLex(n);

    std::vector<symbol> IndetNames = SymbolRange("x", 0, NumIndets(ord)-1);

    PPMonoid PPM = NewPPMonoidEvOv(IndetNames, ord);

    for (int i = 0; i < d + 1; i++) {
        bases.push_back(basis(PPM, n, i));
    }
    degree_dist = degree_distribution(n, d, dist);
    std::random_device rand;
    rng.seed(rand());

  }


  ideal RandomBinomialIdealGenerator::next() {
    std::vector<RingElem> mygens;
    PPOrdering ord = StdDegRevLex(n);
    std::vector<symbol> IndetNames = SymbolRange("x", 0, NumIndets(ord)-1);
    ring P = NewPolyRing(NewZZmod(32003), IndetNames);

    for (int i = 0; i < s; i++) {
      int c = coeff_dist(rng);
      int d1 = degree_dist(rng);
      int d2 = degree_dist(rng);

      bool success = false;
      for (int trials = 0; trials < 1000; trials++) {
        PPMonoidElem m1 = *choice(bases[d1].begin(), bases[d1].end(), rng);
        PPMonoidElem m2 = *choice(bases[d2].begin(), bases[d2].end(), rng);
        std::vector<long> expvm1;
        exponents(expvm1, m1);
        std::vector<long> expvm2;
        exponents(expvm2, m2);
        if (m1 < m2) {
          RingElem binomial = monomial(P,1,expvm2) + monomial(P,c,expvm1);
          mygens.push_back(binomial);
          success = true;
          break;
        } else if (m1 > m2) {
          RingElem binomial = monomial(P,1,expvm1) + monomial(P,c,expvm2);
          mygens.push_back(binomial);
          success = true;
          break;
        }
      }
      if (!success)
        throw std::runtime_error("failed to generate two distinct random monomials after 1000 trials");
    }

    return ideal(mygens);

  }


  RangeRandomBinomialIdealGenerator::RangeRandomBinomialIdealGenerator(
      int n_min,
      int n_max,
      int d_min,
      int d_max,
      int s_min,
      int s_max,
      DistributionType dist)
      : n_min(n_min), n_max(n_max), d_min(d_min), d_max(d_max), s_min(s_min), s_max(s_max),
        dist(dist) {
    if (n_min < 1 || d_min < 1 || s_min < 1) {
      throw std::invalid_argument("Range bounds for n, d, s must all be >= 1");
    }
    if (n_min > n_max || d_min > d_max || s_min > s_max) {
      throw std::invalid_argument("Invalid range bounds: min must be <= max");
    }

    std::random_device rand;
    rng.seed(rand());

    // Prewarm all (n, d) combinations so reset-time sampling stays fast.
    for (int n = n_min; n <= n_max; ++n) {
      for (int d = d_min; d <= d_max; ++d) {
        (void)get_precomputed(n, d);
      }
    }
  }


  RangeRandomBinomialIdealGenerator::PrecomputedNDData&
  RangeRandomBinomialIdealGenerator::get_precomputed(int n, int d) {
    const std::pair<int, int> key{n, d};
    auto found = precomputed_by_nd.find(key);
    if (found != precomputed_by_nd.end()) {
      return found->second;
    }

    PPOrdering ord = StdDegRevLex(n);
    std::vector<symbol> indet_names = SymbolRange("x", 0, NumIndets(ord)-1);
    PPMonoid ppm = NewPPMonoidEvOv(indet_names, ord);

    PrecomputedNDData precomputed{
      NewPolyRing(NewZZmod(32003), indet_names),
      {},
      degree_distribution(n, d, dist)
    };

    precomputed.bases.reserve(static_cast<size_t>(d + 1));
    for (int i = 0; i < d + 1; i++) {
      precomputed.bases.push_back(basis(ppm, n, i));
    }

    auto inserted = precomputed_by_nd.emplace(key, std::move(precomputed));
    return inserted.first->second;
  }


  ideal RangeRandomBinomialIdealGenerator::next() {
    std::uniform_int_distribution<int> n_dist(n_min, n_max);
    std::uniform_int_distribution<int> d_dist(d_min, d_max);
    std::uniform_int_distribution<int> s_dist(s_min, s_max);

    const int n = n_dist(rng);
    const int d = d_dist(rng);
    const int s = s_dist(rng);

    auto& precomputed = get_precomputed(n, d);
    std::vector<RingElem> mygens;

    for (int i = 0; i < s; i++) {
      int c = coeff_dist(rng);
      int d1 = precomputed.degree_dist(rng);
      int d2 = precomputed.degree_dist(rng);

      bool success = false;
      for (int trials = 0; trials < 1000; trials++) {
        PPMonoidElem m1 = *choice(precomputed.bases[d1].begin(), precomputed.bases[d1].end(), rng);
        PPMonoidElem m2 = *choice(precomputed.bases[d2].begin(), precomputed.bases[d2].end(), rng);
        std::vector<long> expvm1;
        exponents(expvm1, m1);
        std::vector<long> expvm2;
        exponents(expvm2, m2);
        if (m1 < m2) {
          RingElem binomial = monomial(precomputed.poly_ring, 1, expvm2) + monomial(precomputed.poly_ring, c, expvm1);
          mygens.push_back(binomial);
          success = true;
          break;
        } else if (m1 > m2) {
          RingElem binomial = monomial(precomputed.poly_ring, 1, expvm1) + monomial(precomputed.poly_ring, c, expvm2);
          mygens.push_back(binomial);
          success = true;
          break;
        }
      }
      if (!success) {
        throw std::runtime_error("failed to generate two distinct random monomials after 1000 trials");
      }
    }

    return ideal(mygens);

  }

  GeomIdealGenerator::GeomIdealGenerator(long n, std::string laman_graph_directory)
      : n(n), laman_graph_directory(std::move(laman_graph_directory))
  {
    if (n < 3 || n > 10) {
      throw std::invalid_argument("GeomIdealGenerator supports n in [3, 10] (LamanGraphs3-10 data)");
    }
    if (this->laman_graph_directory.empty()) {
      throw std::invalid_argument("Geometric ideals require a Laman graph data directory");
    }

    std::string xsyms; // Initialize vector to store x co-ordinates/symbols
    std::string ysyms; // Initialize vector to store y co-ordinates/symbols
    std::string syms;


    for(int i {2}; i <= n; i++){  // This for loop creates a list of symbols that should be in the ring for a given number of vertices, we start at 2 because x1 and y1 are fixed = 0 to account for translation
      auto s = std::to_string(i);
      xsyms = xsyms + "x" + s + ",";
      if(i == n){
        ysyms = ysyms + "y" + s;
      } else if(i==2) { 
        continue; // We don't include a symbol for y2 as it does not appear in any of the basis polynomials as a result of being fixed = 0 to account for rotation
      } else {
        ysyms = ysyms + "y" + s + ",";
      }
    }

    syms = xsyms + ysyms; // Assigning the symbols like this ensures that all symbols x_i are before y_i in the ordering

    P = NewPolyRing(NewZZmod(32003), symbols(syms));

  }

  RangeGeomIdealGenerator::RangeGeomIdealGenerator(long n_min, long n_max, std::string laman_graph_directory)
      : n_min(n_min), n_max(n_max), laman_graph_directory(std::move(laman_graph_directory)), n_dist(n_min, n_max) {
    if (n_min > n_max) {
      throw std::invalid_argument("Invalid geom range bounds: min must be <= max");
    }
    if (n_min < 3 || n_max > 10) {
      throw std::invalid_argument("RangeGeomIdealGenerator supports n in [3, 10] (LamanGraphs3-10 data)");
    }
    if (this->laman_graph_directory.empty()) {
      throw std::invalid_argument("Geometric ideals require a Laman graph data directory");
    }

    std::random_device rand;
    rng.seed(rand());
  }

  ideal RangeGeomIdealGenerator::next() {
    const long sampled_n = n_dist(rng);
    GeomIdealGenerator geom_gen(sampled_n, laman_graph_directory);
    geom_gen.seed(static_cast<int>(rng()));
    return geom_gen.next();
  }


  ideal GeomIdealGenerator::next() // next method for GeomIdealGenerator
  {

    edges = intToEdges(std::to_string(n), laman_graph_directory, rng);

    std::vector<RingElem> xindets(indets(P).begin(), indets(P).begin() + (n-1)); // Makes a vector of all the x variables accounting for the fact that x1 is missing 
    std::vector<RingElem> yindets(indets(P).begin() + (n-1), indets(P).end()); // Likewise for y variables

    std::vector<RingElem> myGens;

    // sample coordinates
    std::uniform_int_distribution<int> coord_dist(-5,5);
    std::vector<std::pair<int,int>> coords(n+1); // 1-indexed

    coords[1] = {0,0};                 // vertex 1 fixed at origin
    coords[2] = {coord_dist(rng), 0};  // vertex 2 on x-axis
    for (int i = 3; i <= n; i++)
        coords[i] = {coord_dist(rng), coord_dist(rng)};

    // Step 4: compute edge lengths
    std::map<std::pair<int,int>, int> L;

    for (auto &e : edges) {
        int i = e.first, j = e.second;
        int dx = coords[i].first - coords[j].first;
        int dy = coords[i].second - coords[j].second;
        L[{i,j}] = dx*dx + dy*dy;
    }

    // generate polynomials
    std::vector<RingElem> polys;
    for (auto &e : edges) {
        int i = e.first, j = e.second;

        // x variables: x2..xn
        RingElem xi = (i==1 ? zero(P) : xindets[i-2]);
        RingElem xj = (j==1 ? zero(P) : xindets[j-2]);

        // y variables: y3..yn, y1=0, y2=0
        RingElem yi;
        if (i==1 || i==2) yi = zero(P);
        else yi = yindets[i-3];

        RingElem yj;
        if (j==1 || j==2) yj = zero(P);
        else yj = yindets[j-3];

        int Lij = L[{i,j}];
        polys.push_back(power(xi - xj, 2) + power(yi - yj, 2) - Lij);
    }

    ideal I = ideal(polys);
    return I;
  }

  BenchmarkIdealGenerator::BenchmarkIdealGenerator(std::vector<IdealClass> classes, int n_min, int n_max)
      : classes(std::move(classes)), n_min(n_min), n_max(n_max), pair_dist(0, 0) {
    if (this->classes.empty()) {
      throw std::invalid_argument("BenchmarkIdealGenerator requires at least one ideal class");
    }
    if (n_min > n_max) {
      throw std::invalid_argument("Invalid benchmark n range bounds: min must be <= max");
    }

    for (IdealClass cls : this->classes) {
      for (int n = n_min; n <= n_max; ++n) {
        if (class_supports_n(cls, n)) {
          class_n_pairs.push_back(std::make_pair(cls, n));
        }
      }
    }

    if (class_n_pairs.empty()) {
      throw std::invalid_argument("No valid (class, n) combinations for requested benchmark ideal_dist");
    }

    pair_dist = std::uniform_int_distribution<size_t>(0, class_n_pairs.size() - 1);

    std::random_device rand;
    rng.seed(rand());
  }

  int BenchmarkIdealGenerator::nvars() const {
    int max_vars = 0;
    for (const auto& cls_n : class_n_pairs) {
      max_vars = std::max(max_vars, class_nvars(cls_n.first, cls_n.second));
    }
    return max_vars;
  }

  ideal BenchmarkIdealGenerator::next() {
    const auto& cls_n = class_n_pairs[pair_dist(rng)];
    return ideal(build_benchmark_polys(cls_n.first, cls_n.second));
  }

  std::unique_ptr<IdealGenerator> parse_ideal_dist(
      const std::string& ideal_dist, const std::string& laman_graph_directory) {

    std::vector<std::string> dist_args = split_top_level_dash(ideal_dist);

    if (!dist_args.empty() && dist_args[0] == "geom") {
      // Geometric/Laman ideals: geom-n or geom-(n0-n1)
      if (dist_args.size() != 2) {
        throw std::invalid_argument("Malformed geometric ideal_dist. Expected format: geom-n or geom-(n0-n1)");
      }

      IntRange n_range = parse_int_or_range_token(dist_args[1], "n");
      if (n_range.lo == n_range.hi) {
        return std::make_unique<GeomIdealGenerator>(n_range.lo, laman_graph_directory);
      }
      return std::make_unique<RangeGeomIdealGenerator>(n_range.lo, n_range.hi, laman_graph_directory);
    }

    if (dist_args.size() == 2 && !dist_args[0].empty() &&
      (std::isalpha(static_cast<unsigned char>(dist_args[0][0])) || 
       (dist_args[0][0] == '(' && dist_args[0].size() > 1 && 
        std::isalpha(static_cast<unsigned char>(dist_args[0][1]))))) {
      // Benchmark families:
      //  - katsura-6
      //  - cyclic+eco-(4-7)
      //  - (katsura,reimer)-5
      std::vector<IdealClass> classes = parse_ideal_classes_token(dist_args[0]);
      IntRange n_range = parse_int_or_range_token(dist_args[1], "n");
      return std::make_unique<BenchmarkIdealGenerator>(classes, n_range.lo, n_range.hi);
    }

    if (dist_args.size() != 4) {
      throw std::invalid_argument("Malformed binomial ideal_dist. Expected exactly 4 fields: n-d-s-type or (n0-n1)-(d0-d1)-(s0-s1)-type. Optional suffixes are not supported.");
    }

    std::map<std::string, DistributionType> dist_types = {
        {"uniform", DistributionType::Uniform},
        {"weighted", DistributionType::Weighted},
        {"maximum", DistributionType::Maximum}
    };

    IntRange n_range = parse_int_or_range_token(dist_args[0], "n");
    IntRange d_range = parse_int_or_range_token(dist_args[1], "d");
    IntRange s_range = parse_int_or_range_token(dist_args[2], "s");

    const auto dist_it = dist_types.find(dist_args[3]);
    if (dist_it == dist_types.end()) {
      throw std::invalid_argument("Unknown ideal distribution type: " + dist_args[3]);
    }
    DistributionType dist = dist_it->second;

    if (n_range.lo == n_range.hi && d_range.lo == d_range.hi && s_range.lo == s_range.hi) {
      return std::make_unique<RandomBinomialIdealGenerator>(n_range.lo, d_range.lo, s_range.lo, dist);
    }

    return std::make_unique<RangeRandomBinomialIdealGenerator>(
      n_range.lo, n_range.hi,
      d_range.lo, d_range.hi,
      s_range.lo, s_range.hi,
      dist
    );

  }

}
