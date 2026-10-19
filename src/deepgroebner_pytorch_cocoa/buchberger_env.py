import gymnasium as gym
import numpy as np
from typing import Optional, Sequence
from deepgroebner_pytorch_cocoa.wrapped import CGLeadMonomialsEnv, CNoPQEnv
from .ideal_corpus import load_ideal_corpus, split_ring_symbols, validate_corpus_options


_BENCHMARK_CLASSES = {"katsura", "cyclic", "acyclic", "eco", "noon", "reimer"}


def split_top_level_dash(s: str) -> list[str]:
    tokens: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
            cur.append(ch)
            continue
        if ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Malformed ideal_dist: unmatched ')' in {s}")
            cur.append(ch)
            continue
        if ch == "-" and depth == 0:
            tokens.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)

    if depth != 0:
        raise ValueError(f"Malformed ideal_dist: unmatched '(' in {s}")

    tokens.append("".join(cur).strip())
    return tokens


def parse_int_or_range_token(token: str, name: str) -> tuple[int, int]:
    token = token.strip()
    if not token:
        raise ValueError(f"Malformed ideal_dist: empty token for {name}")

    if token.startswith("("):
        if not token.endswith(")"):
            raise ValueError(f"Malformed ideal_dist {name}-range token: {token}")
        lo_hi = [part.strip() for part in token[1:-1].split("-", 1)]
        if len(lo_hi) != 2:
            raise ValueError(f"Malformed ideal_dist {name}-range token: {token}")
        lo = int(lo_hi[0])
        hi = int(lo_hi[1])
        if lo > hi:
            raise ValueError(f"Malformed ideal_dist {name}-range token: lower bound > upper bound in {token}")
        return lo, hi

    value = int(token)
    return value, value


def parse_class_list_token(token: str) -> list[str]:
    token = token.strip()
    if token.startswith("("):
        if not token.endswith(")"):
            raise ValueError(f"Malformed benchmark class token: {token}")
        token = token[1:-1].strip()

    parts = [part.strip().lower() for part in token.replace(",", "+").split("+")]
    parts = [part for part in parts if part]
    if not parts:
        raise ValueError("Malformed benchmark ideal_dist: no classes specified")

    unknown = [part for part in parts if part not in _BENCHMARK_CLASSES]
    if unknown:
        raise ValueError(f"Unknown benchmark ideal class(es): {', '.join(unknown)}")

    seen: set[str] = set()
    deduped: list[str] = []
    for part in parts:
        if part not in seen:
            deduped.append(part)
            seen.add(part)
    return deduped


def parse_nvars_from_ideal_dist(ideal_dist: str) -> int:
    ideal_dist = ideal_dist.strip()
    dist_args = split_top_level_dash(ideal_dist)

    if len(dist_args) == 2 and dist_args[0] == "geom":
        _, hi = parse_int_or_range_token(dist_args[1], "n")
        if hi < 3:
            raise ValueError(f"Malformed geom ideal_dist: {ideal_dist}. Expected n >= 3")
        # GeomIdealGenerator uses variables x2..xn and y3..yn.
        return 2 * hi - 3

    if len(dist_args) == 2 and dist_args[0] and (dist_args[0][0].isalpha() or 
        (dist_args[0][0] == "(" and len(dist_args[0]) > 1 and dist_args[0][1].isalpha())):
        _, hi = parse_int_or_range_token(dist_args[1], "n")
        return hi

    if len(dist_args) != 4:
        raise ValueError(
            "Malformed binomial ideal_dist. Expected exactly 4 fields: n-d-s-type; "
            "optional suffixes are not supported."
        )
    _, hi = parse_int_or_range_token(dist_args[0], "n")
    return hi


def _resolve_pq_budgets(N: int, N_P: Optional[int], N_G: Optional[int]) -> tuple[int, int]:
    n_pair = N if N_P is None else N_P
    n_basis = N if N_G is None else N_G
    if n_pair <= 0 or n_basis <= 0:
        raise ValueError("Priority-queue observation budgets N_P and N_G must be positive.")
    return int(n_pair), int(n_basis)


def _resolve_observation_nvars(n_vars: int, observation_nvars: Optional[int]) -> int:
    if observation_nvars is None:
        return n_vars
    if not isinstance(observation_nvars, (int, np.integer)) or observation_nvars < n_vars:
        raise ValueError("observation_nvars must be an integer at least as large as the ideal's variable count.")
    return int(observation_nvars)


def _pad_exponent_matrix(matrix: np.ndarray, n_vars: int, target_n_vars: int, n_monomials: int) -> np.ndarray:
    """Pad each monomial separately, keeping term and pair endpoint boundaries."""
    if n_vars == target_n_vars:
        return matrix
    monomials = matrix.reshape(matrix.shape[0], n_monomials, n_vars)
    return np.pad(monomials, ((0, 0), (0, 0), (0, target_n_vars - n_vars))).reshape(
        matrix.shape[0], n_monomials * target_n_vars
    )


class BuchbergerEnvExponentsv1(gym.Env):
    def __init__(self, ideal_dist: Optional[str] = "3-20-10-weighted", elimination: str = "gebauermoeller", rewards: str = "additions", pq_strategy: str = 'first', sort_input: bool = False, sort_reducers: bool = True, fix_ideal: bool = False, N: int = 20, k: int = 2, N_P: Optional[int] = None, N_G: Optional[int] = None, coeff_ring: Optional[str] = None, symbols: Optional[str] = None, basis_elements: Optional[Sequence[str]] = None, ideal_corpus: Optional[str] = None, observation_nvars: Optional[int] = None):
      super().__init__()
      n_pair, n_basis = _resolve_pq_budgets(N, N_P, N_G)
      self.ideal_corpus = ideal_corpus
      corpus = None
      if ideal_corpus is not None:
          corpus = load_ideal_corpus(ideal_corpus)
          validate_corpus_options(corpus, coeff_ring=coeff_ring, symbols=symbols,
                                  basis_elements=basis_elements, fix_ideal=fix_ideal)
          coeff_ring, symbols = corpus.coeff_ring, corpus.symbols
          # Initialize observation buffers at the corpus's largest variable count.
          basis_elements = [split_ring_symbols(symbols)[0]]

      has_fixed_ideal_args = (coeff_ring is not None) or (symbols is not None) or (basis_elements is not None)
      has_all_fixed_ideal_args = (coeff_ring is not None) and (symbols is not None) and (basis_elements is not None)
      if has_fixed_ideal_args and not has_all_fixed_ideal_args:
          raise ValueError("To use a fixed input ideal, provide coeff_ring, symbols, and basis_elements together.")

      if has_all_fixed_ideal_args:
          if len(basis_elements) == 0:
              raise ValueError("basis_elements must be non-empty when using a fixed input ideal.")

          self.env = CGLeadMonomialsEnv(
              elimination=elimination,
              rewards=rewards,
              pq_strategy=pq_strategy,
              sort_input=sort_input,
              sort_reducers=sort_reducers,
              N=N,
              k=k,
              N_P=n_pair,
              N_G=n_basis,
              coeff_ring=coeff_ring,
              symbols=symbols,
              basis=list(basis_elements),
          )
          self.n_vars = len(split_ring_symbols(symbols))
          if self.n_vars == 0:
              raise ValueError("symbols must contain at least one variable name.")
      else:
          self.env = CGLeadMonomialsEnv(ideal_dist=ideal_dist, elimination=elimination, rewards=rewards, pq_strategy=pq_strategy, sort_input=sort_input, sort_reducers=sort_reducers, fix_ideal=fix_ideal, N=N, k=k, N_P=n_pair, N_G=n_basis)
          self.n_vars = parse_nvars_from_ideal_dist(ideal_dist)

      if corpus is not None:
          self.env.set_ideal_corpus(corpus.coeff_rings, corpus.ring_symbols, corpus.bases)

      self._native_n_vars = self.n_vars
      self.n_vars = _resolve_observation_nvars(self.n_vars, observation_nvars)

      self.N = N
      self.N_P = n_pair
      self.N_G = n_basis
      self.k = k

      # This observation space is based on having a matrix of exponent vectors for S-Polynomials
      self.observation_space = gym.spaces.Dict({
          "G": gym.spaces.Box(low=0, high=np.inf, shape=(n_basis, k * self.n_vars), dtype=np.float32),
          "P": gym.spaces.Box(low=0, high=np.inf, shape=(n_pair, 2 * k * self.n_vars), dtype=np.float32),
          "G_mask": gym.spaces.Box(low=0, high=1, shape=(n_basis,), dtype=np.bool_),
          "P_mask": gym.spaces.Box(low=0, high=1, shape=(n_pair,), dtype=np.bool_),
          "state_features": gym.spaces.Box(low=0, high =np.inf, shape =(14,), dtype=np.float32)
      })

      # Action space is simply choosing one of the visible S-Pairs
      self.action_space = gym.spaces.Discrete(n_pair)

      self.state_feature_names = [
          "G_cardinality",
          "P_cardinality",
          "pair_density",
          "mean_degree",
          "max_degree",
          "min_degree",
          "st_dev_degree",
          "mean_lcm_degree",
          "max_lcm_degree",
          "min_lcm_degree",
          "std_dev_lcm_degree",
          "n_pairs_pruned",
          "n_pairs_processed",
          "steps_since_basis_update"
      ]
    
    def _build_observation(self, obs_dict: dict) -> dict:
        """Copy native buffers so later steps and resets cannot change this observation."""
        return {
            "G": _pad_exponent_matrix(np.array(obs_dict["G"], dtype=np.float32, copy=True),
                                      self._native_n_vars, self.n_vars, self.k),
            "P": _pad_exponent_matrix(np.array(obs_dict["P"], dtype=np.float32, copy=True),
                                      self._native_n_vars, self.n_vars, 2 * self.k),
            "G_mask": np.array(obs_dict["G_mask"], dtype=np.bool_, copy=True),
            "P_mask": np.array(obs_dict["P_mask"], dtype=np.bool_, copy=True),
            "state_features": np.array(obs_dict["state_feats"], dtype=np.float32, copy=True),
        }

    def _build_info(self, cenv_info):
        info = {name: cenv_info[name] for name in ("P_idx_raw", "P_sugar", "P_normal_rank")}
        if self.ideal_corpus is not None:
            info["ideal_corpus_index"] = self.env.corpus_index()
        return info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.env.seed(seed)

        obs_dict, cenv_info = self.env.reset()
        self.total_reward = 0
        self.episode_length = 0

        self.state = self._build_observation(obs_dict)

        return self.state, self._build_info(cenv_info)

    def step(self, action):

        obs_dict, reward, terminated, cenv_info = self.env.step(action)

        self.total_reward += reward
        self.episode_length += 1

        truncated = False

        self.state = self._build_observation(obs_dict)
        return self.state, reward, bool(terminated), truncated, self._build_info(cenv_info)
    
    def value(self, strategy: str = 'degree', gamma: float = 1.0) -> float:
        return self.env.value(strategy=strategy, gamma=gamma)


class BuchbergerEnvExponentsNoPQv1(gym.Env):
    def __init__(
        self,
        ideal_dist: Optional[str] = "3-20-10-weighted",
        elimination: str = "gebauermoeller",
        rewards: str = "additions",
        sort_input: bool = False,
        sort_reducers: bool = True,
        fix_ideal: bool = False,
        k: int = 2,
        coeff_ring: Optional[str] = None,
        symbols: Optional[str] = None,
        basis_elements: Optional[Sequence[str]] = None,
        ideal_corpus: Optional[str] = None,
        observation_nvars: Optional[int] = None,
    ):
        super().__init__()
        self.ideal_corpus = ideal_corpus
        corpus = None
        if ideal_corpus is not None:
            corpus = load_ideal_corpus(ideal_corpus)
            validate_corpus_options(corpus, coeff_ring=coeff_ring, symbols=symbols,
                                    basis_elements=basis_elements, fix_ideal=fix_ideal)
            coeff_ring, symbols = corpus.coeff_ring, corpus.symbols
            basis_elements = [split_ring_symbols(symbols)[0]]

        has_fixed_ideal_args = (coeff_ring is not None) or (symbols is not None) or (basis_elements is not None)
        has_all_fixed_ideal_args = (coeff_ring is not None) and (symbols is not None) and (basis_elements is not None)
        if has_fixed_ideal_args and not has_all_fixed_ideal_args:
            raise ValueError("To use a fixed input ideal, provide coeff_ring, symbols, and basis_elements together.")

        if has_all_fixed_ideal_args:
            if len(basis_elements) == 0:
                raise ValueError("basis_elements must be non-empty when using a fixed input ideal.")

            self.env = CNoPQEnv(
                elimination=elimination,
                rewards=rewards,
                sort_input=sort_input,
                sort_reducers=sort_reducers,
                k=k,
                coeff_ring=coeff_ring,
                symbols=symbols,
                basis=list(basis_elements),
                return_dict=True,
            )
            self.n_vars = len(split_ring_symbols(symbols))
            if self.n_vars == 0:
                raise ValueError("symbols must contain at least one variable name.")
        else:
            if ideal_dist is None:
                raise ValueError("ideal_dist must be provided for BuchbergerEnvExponentsNoPQv1.")

            self.env = CNoPQEnv(
                ideal_dist=ideal_dist,
                elimination=elimination,
                rewards=rewards,
                sort_input=sort_input,
                sort_reducers=sort_reducers,
                fix_ideal=fix_ideal,
                k=k,
                return_dict=True,
            )
            self.n_vars = parse_nvars_from_ideal_dist(ideal_dist)
        if corpus is not None:
            self.env.set_ideal_corpus(corpus.coeff_rings, corpus.ring_symbols, corpus.bases)
        self._native_n_vars = self.n_vars
        self.n_vars = _resolve_observation_nvars(self.n_vars, observation_nvars)
        self.k = k
        self.g_dim = self.n_vars * self.k
        self.pair_dim = 2 * self.n_vars * self.k

        # Variable-size observations: basis elements G and pair list P.
        self.observation_space = gym.spaces.Dict({
            "P": gym.spaces.Sequence(
                gym.spaces.Box(low=0, high=np.inf, shape=(self.pair_dim,), dtype=np.int32),
                stack=True,
            ),
            "G": gym.spaces.Sequence(
                gym.spaces.Box(low=0, high=np.inf, shape=(self.g_dim,), dtype=np.int32),
                stack=True,
            ),
            "P_mask": gym.spaces.Sequence(
                gym.spaces.Box(low=0, high=1, shape=(), dtype=np.int8),
                stack=True,
            ),
            "G_mask": gym.spaces.Sequence(
                gym.spaces.Box(low=0, high=1, shape=(), dtype=np.int8),
                stack=True,
            ),
            "state_features": gym.spaces.Box(low=0, high=np.inf, shape=(14,), dtype=np.float32),
        })
        # Action is the integer index into the current pair list.
        self.action_space = gym.spaces.Box(
            low=0,
            high=np.iinfo(np.int32).max,
            shape=(),
            dtype=np.int32,
        )

    def _coerce_matrix(self, state: np.ndarray, width: int) -> np.ndarray:
        state = np.asarray(state, dtype=np.int32)
        if state.ndim == 1:
            if state.size == 0:
                return np.zeros((0, width), dtype=np.int32)
            return state.reshape(-1, width)
        return state

    def _build_observation(self, obs_dict: dict) -> dict:
        P = _pad_exponent_matrix(self._coerce_matrix(obs_dict["P"], 2 * self.k * self._native_n_vars),
                                 self._native_n_vars, self.n_vars, 2 * self.k)
        G = _pad_exponent_matrix(self._coerce_matrix(obs_dict["G"], self.k * self._native_n_vars),
                                 self._native_n_vars, self.n_vars, self.k)

        state_features = np.asarray(obs_dict["state_feats"], dtype=np.float32)
        if state_features.shape != (14,):
            state_features = state_features.reshape(14,)
        return {
            "P": P,
            "G": G,
            "P_mask": np.ones((P.shape[0],), dtype=np.int8),
            "G_mask": np.ones((G.shape[0],), dtype=np.int8),
            "state_features": state_features,
        }

    def _build_info(self, state: dict) -> dict:
        n_pairs = int(state["P"].shape[0])
        n_basis = int(state["G"].shape[0])
        info = {
            "n_pairs": n_pairs,
            "n_basis": n_basis,
        }
        if self.ideal_corpus is not None:
            info["ideal_corpus_index"] = self.env.corpus_index()
        return info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.env.seed(seed)
        obs_dict, _ = self.env.reset()
        self.state = self._build_observation(obs_dict)
        self.total_reward = 0.0
        self.episode_length = 0
        return self.state, self._build_info(self.state)

    def step(self, action):
        action = int(action)
        n_pairs = int(self.state["P"].shape[0])
        if action < 0 or action >= n_pairs:
            raise ValueError(f"Invalid action {action}; expected 0 <= action < {n_pairs}.")

        obs_dict, reward, terminated, _ = self.env.step(action)
        self.state = self._build_observation(obs_dict)

        self.total_reward += reward
        self.episode_length += 1

        truncated = False
        return self.state, reward, bool(terminated), truncated, self._build_info(self.state)

    def value(self, strategy: str = "degree", gamma: float = 1.0) -> float:
        return self.env.value(strategy=strategy, gamma=gamma)
    
