from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "buchberger.cpp" namespace "CoCoA":
    pass

cdef extern from "buchberger.h" namespace "CoCoA":
    cdef cppclass BuchbergerStats:
        int zero_reductions
        int nonzero_reductions
        int polynomial_additions
        double total_reward

    cdef cppclass NoPQEnv:
        NoPQEnv() except +
        NoPQEnv(string, string, string, bint, bint, bint, int, string) except +
        NoPQEnv(string, string, vector[string], string, string, bint, bint, int) except +
        NoPQEnv(const NoPQEnv&)
        void reset() except +
        double step(int)
        void set_seed(int)
        double value(string, double) except +
        BuchbergerStats heuristic_stats(string, int) except +
        bint is_groebner_basis() except +
        void set_ideal_corpus(vector[string], vector[string], vector[vector[string]]) except +
        int corpus_index()

        vector[int] state
        vector[int] G_leads
        vector[float] state_feature_vector
        int cols
        int g_cols
        int N_Pairs

cdef extern from "buchberger.h" namespace "CoCoA":
    cdef cppclass GLeadMonomialsEnv:
        GLeadMonomialsEnv() except +
        GLeadMonomialsEnv(string, string, string, string, bint, bint, bint, int, int, int, int, string) except +
        GLeadMonomialsEnv(string, string, vector[string], string, string, string, bint, bint, int, int, int, int) except +
        GLeadMonomialsEnv(const GLeadMonomialsEnv&)
        void reset()
        double step(int)
        void set_seed(int)
        double value(string, double) except +
        bint is_groebner_basis() except +
        vector[string] initial_ideal() except +
        void set_ideal_corpus(vector[string], vector[string], vector[vector[string]]) except +
        int corpus_index()

        vector[int] leads
        vector[int] G_padded
        vector[int] P_padded
        vector[int] P_sugars_padded
        vector[int] P_normal_ranks_padded
        vector[int] G_mask
        vector[int] P_mask
        vector[int] P_idx
        vector[int] P_sugars
        vector[int] P_normal_ranks
        vector[string] basis_elements
        vector[float] state_feature_vector
        int cols
        int p_cols
        int g_cols
        int N_Pairs
