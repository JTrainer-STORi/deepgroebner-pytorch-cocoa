# distutils: language=c++
# cython: language_level=3

import numpy as np
import cython
from os import fsencode
from libcpp.vector cimport vector
from libcpp.string cimport string

from .buchberger cimport NoPQEnv, GLeadMonomialsEnv
from .package_resources import laman_graph_directory

cdef class CNoPQEnv:
    cdef NoPQEnv c_env
    cdef bint return_dict

    def __cinit__(self, ideal_dist="3-20-10-weighted", elimination="gebauermoeller", rewards="additions", sort_input=False, sort_reducers=True, fix_ideal=False, k=1, coeff_ring=None, symbols=None, basis=None, return_dict=False):
        cdef vector[string] basis_cpp
        cdef string elimination_cpp = elimination.encode()
        cdef string rewards_cpp = rewards.encode()
        cdef string ideal_dist_cpp
        cdef string graph_directory_cpp
        cdef string coeff_ring_cpp
        cdef string symbols_cpp
        self.return_dict = return_dict
        if basis is not None:
            for basis_elem in basis:
                basis_cpp.push_back((<str>basis_elem).encode())

        if coeff_ring is not None and symbols is not None and basis is not None:
            coeff_ring_cpp = (<str>coeff_ring).encode()
            symbols_cpp = (<str>symbols).encode()
            self.c_env = NoPQEnv(coeff_ring_cpp, symbols_cpp, basis_cpp, elimination_cpp, rewards_cpp, <bint>sort_input, <bint>sort_reducers, <int>k)
        else:
            ideal_dist_cpp = ideal_dist.encode()
            graph_directory_cpp = fsencode(laman_graph_directory())
            self.c_env = NoPQEnv(ideal_dist_cpp, elimination_cpp, rewards_cpp, <bint>sort_input, <bint>sort_reducers, <bint>fix_ideal, <int>k, graph_directory_cpp)

    def reset(self):
        self.c_env.reset()
        P = np.array(self.c_env.state, dtype=np.int32).reshape(-1, self.c_env.cols)
        G = np.array(self.c_env.G_leads, dtype=np.int32).reshape(-1, self.c_env.g_cols)
        state_feats = np.array(self.c_env.state_feature_vector, dtype=np.float32)
        n_pairs = P.shape[0]
        if self.return_dict:
            return {"P": P, "G": G, "state_feats": state_feats}, {'N_Pairs': n_pairs}
        return P, {'N_Pairs': n_pairs}

    def step(self, action):
        reward = self.c_env.step(action)
        P = np.array(self.c_env.state, dtype=np.int32).reshape(-1, self.c_env.cols)
        G = np.array(self.c_env.G_leads, dtype=np.int32).reshape(-1, self.c_env.g_cols)
        state_feats = np.array(self.c_env.state_feature_vector, dtype=np.float32)
        n_pairs = P.shape[0]
        terminated = (n_pairs == 0)
        if self.return_dict:
            return {"P": P, "G": G, "state_feats": state_feats}, reward, terminated, {'N_Pairs': n_pairs}
        return P, reward, terminated, {'N_Pairs': n_pairs}

    def seed(self, seed=None):
        if seed is not None:
            self.c_env.set_seed(seed)

    def value(self, strategy='degree', gamma=0.99):
        return self.c_env.value(strategy.encode(), gamma)

    def copy(self):
        copy = CNoPQEnv()
        copy.c_env = NoPQEnv(self.c_env)
        return copy

cdef class CGLeadMonomialsEnv:
    cdef GLeadMonomialsEnv c_env
    cdef object _G_buf
    cdef object _P_buf
    cdef object _P_sugar_buf
    cdef object _P_normal_rank_buf
    cdef object _G_mask_buf
    cdef object _P_mask_buf
    cdef object _P_idx_raw_buf
    cdef object _state_feats_buf
    cdef object _obs
    cdef bint _buffers_ready

    def __cinit__(self, ideal_dist="3-20-10-weighted", elimination="gebauermoeller", rewards="additions", pq_strategy='first', sort_input=False, sort_reducers=True, fix_ideal=False, N=20, k=1, N_P=None, N_G=None, coeff_ring=None, symbols=None, basis=None):
        cdef vector[string] basis_cpp
        cdef string elimination_cpp = elimination.encode()
        cdef string rewards_cpp = rewards.encode()
        cdef string pq_strategy_cpp = pq_strategy.encode()
        cdef string ideal_dist_cpp
        cdef string graph_directory_cpp
        cdef string coeff_ring_cpp
        cdef string symbols_cpp
        cdef int n_pair
        cdef int n_basis
        if N_P is None:
            n_pair = <int>N
        else:
            n_pair = <int>N_P
        if N_G is None:
            n_basis = <int>N
        else:
            n_basis = <int>N_G
        if basis is not None:
            for basis_elem in basis:
                basis_cpp.push_back((<str>basis_elem).encode())

        if coeff_ring is not None and symbols is not None and basis is not None:
            coeff_ring_cpp = (<str>coeff_ring).encode()
            symbols_cpp = (<str>symbols).encode()
            self.c_env = GLeadMonomialsEnv(coeff_ring_cpp, symbols_cpp, basis_cpp, elimination_cpp, rewards_cpp, pq_strategy_cpp, <bint>sort_input, <bint>sort_reducers, <int>N, <int>k, n_pair, n_basis)
        else:
            ideal_dist_cpp = ideal_dist.encode()
            graph_directory_cpp = fsencode(laman_graph_directory())
            self.c_env = GLeadMonomialsEnv(ideal_dist_cpp, elimination_cpp, rewards_cpp, pq_strategy_cpp, <bint>sort_input, <bint>sort_reducers, <bint>fix_ideal, <int>N, <int>k, n_pair, n_basis, graph_directory_cpp)
        self._buffers_ready = False

    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _allocate_buffers(self):
        cdef int n_g_slots = <int>self.c_env.G_mask.size()
        cdef int n_p_slots = <int>self.c_env.P_mask.size()
        cdef int feat_len = <int>self.c_env.state_feature_vector.size()
        self._G_buf = np.empty((n_g_slots, self.c_env.g_cols), dtype=np.float32)
        self._P_buf = np.empty((n_p_slots, self.c_env.p_cols), dtype=np.float32)
        self._P_sugar_buf = np.empty((n_p_slots,), dtype=np.float32)
        self._P_normal_rank_buf = np.empty((n_p_slots,), dtype=np.float32)
        self._G_mask_buf = np.empty((n_g_slots,), dtype=np.bool_)
        self._P_mask_buf = np.empty((n_p_slots,), dtype=np.bool_)
        self._P_idx_raw_buf = np.full((n_p_slots, 2), -1, dtype=np.int64)
        self._state_feats_buf = np.empty((feat_len,), dtype=np.float32)
        self._obs = {
            "G": self._G_buf,
            "P": self._P_buf,
            "G_mask": self._G_mask_buf,
            "P_mask": self._P_mask_buf,
            "state_feats": self._state_feats_buf,
        }
        self._buffers_ready = True

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _refresh_buffers_from_cpp(self):
        cdef Py_ssize_t i, j, idx
        cdef int n_g_slots
        cdef int n_p_slots
        cdef int g_cols
        cdef int p_cols
        cdef int feat_len
        cdef object G_buf
        cdef object P_buf
        cdef object P_sugar_buf
        cdef object P_normal_rank_buf
        cdef object G_mask_buf
        cdef object P_mask_buf
        cdef object P_idx_raw_buf
        cdef object state_feats_buf

        n_g_slots = self.c_env.G_mask.size()
        n_p_slots = self.c_env.P_mask.size()
        g_cols = self.c_env.g_cols
        p_cols = self.c_env.p_cols
        feat_len = self.c_env.state_feature_vector.size()

        if (
            (not self._buffers_ready)
            or self._G_buf.shape[0] != n_g_slots
            or self._G_buf.shape[1] != g_cols
            or self._P_buf.shape[0] != n_p_slots
            or self._P_buf.shape[1] != p_cols
            or self._P_sugar_buf.shape[0] != n_p_slots
            or self._P_normal_rank_buf.shape[0] != n_p_slots
            or self._G_mask_buf.shape[0] != n_g_slots
            or self._P_mask_buf.shape[0] != n_p_slots
            or self._P_idx_raw_buf.shape[0] != n_p_slots
            or self._state_feats_buf.shape[0] != feat_len
        ):
            self._allocate_buffers()

        G_buf = self._G_buf
        P_buf = self._P_buf
        P_sugar_buf = self._P_sugar_buf
        P_normal_rank_buf = self._P_normal_rank_buf
        G_mask_buf = self._G_mask_buf
        P_mask_buf = self._P_mask_buf
        P_idx_raw_buf = self._P_idx_raw_buf
        state_feats_buf = self._state_feats_buf

        idx = 0
        for i in range(n_g_slots):
            for j in range(g_cols):
                G_buf[i, j] = self.c_env.G_padded[idx]
                idx += 1

        idx = 0
        for i in range(n_p_slots):
            for j in range(p_cols):
                P_buf[i, j] = self.c_env.P_padded[idx]
                idx += 1

        for i in range(n_p_slots):
            P_sugar_buf[i] = self.c_env.P_sugars_padded[i]
            P_normal_rank_buf[i] = self.c_env.P_normal_ranks_padded[i]
            P_mask_buf[i] = self.c_env.P_mask[i]

        for i in range(n_g_slots):
            G_mask_buf[i] = self.c_env.G_mask[i]

        for i in range(n_p_slots):
            P_idx_raw_buf[i, 0] = -1
            P_idx_raw_buf[i, 1] = -1

        idx = 0
        for i in range(min(n_p_slots, <int>self.c_env.P_idx.size() // 2)):
            P_idx_raw_buf[i, 0] = self.c_env.P_idx[idx]
            P_idx_raw_buf[i, 1] = self.c_env.P_idx[idx + 1]
            idx += 2

        for i in range(feat_len):
            state_feats_buf[i] = self.c_env.state_feature_vector[i]

    cdef dict _build_info(self):
        return {
            "P_idx_raw": self._P_idx_raw_buf.copy(),
            "P_sugar": self._P_sugar_buf.copy(),
            "P_normal_rank": self._P_normal_rank_buf.copy(),
            "basis_elements": [s.decode("utf-8") for s in self.c_env.basis_elements],
        }

    def reset(self):
        self.c_env.reset()
        self._refresh_buffers_from_cpp()
        return self._obs, self._build_info()

    def step(self, action):
        reward = self.c_env.step(action)
        self._refresh_buffers_from_cpp()
        return self._obs, reward, self.c_env.N_Pairs == 0, self._build_info()

    def seed(self, seed=None):
        if seed is not None:
            self.c_env.set_seed(seed)

    def value(self, strategy='degree', gamma=0.99):
        return self.c_env.value(strategy.encode(), gamma)

    def copy(self):
        copy = CGLeadMonomialsEnv()
        copy.c_env = GLeadMonomialsEnv(self.c_env)
        return copy
