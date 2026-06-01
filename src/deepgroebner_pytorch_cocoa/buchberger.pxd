from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "buchberger.cpp" namespace "CoCoA":
    pass

cdef extern from "buchberger.h" namespace "CoCoA":
    cdef cppclass NoPQEnv:
        NoPQEnv() except +
        NoPQEnv(string, string, string, bint, bint, bint, int) except +
        NoPQEnv(string, string, vector[string], string, string, bint, bint, int) except +
        NoPQEnv(const NoPQEnv&)
        void reset()
        double step(int)
        void set_seed(int)
        double value(string, double)

        vector[int] state
        vector[int] P_idx
        vector[int] G_leads
        vector[float] state_feature_vector
        int cols
        int g_cols
        int N_Pairs

cdef extern from "buchberger.h" namespace "CoCoA":
    cdef cppclass GLeadMonomialsEnv:
        GLeadMonomialsEnv() except +
        GLeadMonomialsEnv(string, string, string, string, bint, bint, bint, int, int, int, int) except +
        GLeadMonomialsEnv(string, string, vector[string], string, string, string, bint, bint, int, int, int, int) except +
        GLeadMonomialsEnv(const GLeadMonomialsEnv&)
        void reset()
        double step(int)
        void set_seed(int)
        double value(string, double)

        vector[int] leads
        vector[int] G_terms
        vector[int] G_term_counts
        int n_vars
        vector[int] G_padded
        vector[int] P_padded
        vector[int] P_idx_padded
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
