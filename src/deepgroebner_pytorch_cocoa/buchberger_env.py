import gymnasium as gym
import numpy as np
from typing import Optional, Sequence
from deepgroebner_pytorch_cocoa.wrapped import CGLeadMonomialsEnv, CNoPQEnv


_BENCHMARK_CLASSES = {"katsura", "cyclic", "acyclic", "eco", "noon", "reimer"}


def _split_top_level_dash(s: str) -> list[str]:
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


def _parse_int_or_range_token(token: str, name: str) -> tuple[int, int]:
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


def _parse_class_list_token(token: str) -> list[str]:
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


def _parse_nvars_from_ideal_dist(ideal_dist: str) -> int:
    ideal_dist = ideal_dist.strip()
    dist_args = _split_top_level_dash(ideal_dist)

    if len(dist_args) == 2 and dist_args[0] == "geom":
        _, hi = _parse_int_or_range_token(dist_args[1], "n")
        if hi < 3:
            raise ValueError(f"Malformed geom ideal_dist: {ideal_dist}. Expected n >= 3")
        # GeomIdealGenerator uses variables x2..xn and y3..yn.
        return 2 * hi - 3

    if len(dist_args) == 2 and dist_args[0] and (dist_args[0][0].isalpha() or 
        (dist_args[0][0] == "(" and len(dist_args[0]) > 1 and dist_args[0][1].isalpha())):
        _, hi = _parse_int_or_range_token(dist_args[1], "n")
        return hi

    if not dist_args:
        raise ValueError(f"Malformed ideal_dist: {ideal_dist}")
    _, hi = _parse_int_or_range_token(dist_args[0], "n")
    return hi


def _resolve_pq_budgets(N: int, N_P: Optional[int], N_G: Optional[int]) -> tuple[int, int]:
    n_pair = N if N_P is None else N_P
    n_basis = N if N_G is None else N_G
    if n_pair <= 0 or n_basis <= 0:
        raise ValueError("Priority-queue observation budgets N_P and N_G must be positive.")
    return int(n_pair), int(n_basis)


class BuchbergerEnvExponentsv1(gym.Env):
    def __init__(self, ideal_dist: Optional[str] = "3-20-10-weighted", elimination: str = "gebauermoeller", rewards: str = "additions", pq_strategy: str = 'first', sort_input: bool = False, sort_reducers: bool = True, fix_ideal: bool = False, N: int = 20, k: int = 2, N_P: Optional[int] = None, N_G: Optional[int] = None, coeff_ring: Optional[str] = None, symbols: Optional[str] = None, basis_elements: Optional[Sequence[str]] = None):
      super().__init__()
      n_pair, n_basis = _resolve_pq_budgets(N, N_P, N_G)

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
          self.n_vars = len([s.strip() for s in symbols.split(",") if s.strip()])
          if self.n_vars == 0:
              raise ValueError("symbols must contain at least one variable name.")
      else:
          self.env = CGLeadMonomialsEnv(ideal_dist=ideal_dist, elimination=elimination, rewards=rewards, pq_strategy=pq_strategy, sort_input=sort_input, sort_reducers=sort_reducers, fix_ideal=fix_ideal, N=N, k=k, N_P=n_pair, N_G=n_basis)
          self.n_vars = _parse_nvars_from_ideal_dist(ideal_dist)

      self.N = N
      self.N_P = n_pair
      self.N_G = n_basis
      self.k = k

      # This observation space is based on having a matrix of exponent vectors for S-Polynomials
      self.observation_space = gym.spaces.Dict({
          "G": gym.spaces.Box(low=0, high=np.inf, shape=(n_basis, k * self.n_vars), dtype=np.float32),
          "P": gym.spaces.Box(low=0, high=np.inf, shape=(n_pair, 2 * k * self.n_vars), dtype=np.float32),
          "P_idx": gym.spaces.Box(low=-1, high=np.iinfo(np.int64).max, shape=(n_pair, 2), dtype=np.int64),
          "P_idx_raw": gym.spaces.Box(low=-1, high=np.iinfo(np.int64).max, shape=(n_pair, 2), dtype=np.int64),
          "P_sugar": gym.spaces.Box(low=0, high=np.inf, shape=(n_pair,), dtype=np.float32),
          "P_normal_rank": gym.spaces.Box(low=0, high=np.inf, shape=(n_pair,), dtype=np.float32),
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
    
    def _get_info(self):
        action_mask = np.array(self.state.any(axis=1), dtype=np.int8)
        token_mask = np.array(self.state != 0, dtype=np.int8)
        return {'action_mask': action_mask, 'token_mask': token_mask}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.env.seed(seed)

        obs_dict, cenv_info = self.env.reset()
        self.total_reward = 0
        self.episode_length = 0

        #info = self._get_info()
        info = {}

        self.state = {
            "G": np.asarray(obs_dict["G"], dtype=np.float32),
            "P": np.asarray(obs_dict["P"], dtype=np.float32),
            "P_idx": np.asarray(obs_dict["P_idx"], dtype=np.int64),
            "P_idx_raw": np.asarray(obs_dict.get("P_idx_raw", obs_dict["P_idx"]), dtype=np.int64),
            "P_sugar": np.asarray(obs_dict.get("P_sugar", np.zeros((self.N_P,), dtype=np.float32)), dtype=np.float32),
            "P_normal_rank": np.asarray(obs_dict.get("P_normal_rank", np.zeros((self.N_P,), dtype=np.float32)), dtype=np.float32),
            "G_mask": np.asarray(obs_dict["G_mask"], dtype=np.bool_),
            "P_mask": np.asarray(obs_dict["P_mask"], dtype=np.bool_),
            "state_features": np.asarray(obs_dict["state_feats"], dtype=np.float32),
        }

        info["pair_sugar"] = self.state["P_sugar"].copy()
        info["pair_normal_rank"] = self.state["P_normal_rank"].copy()

        return self.state, info

    def step(self, action):

        obs_dict, reward, terminated, cenv_info = self.env.step(action)

        self.total_reward += reward
        self.episode_length += 1

        truncated = False

        self.state = {
            "G": np.asarray(obs_dict["G"], dtype=np.float32),
            "P": np.asarray(obs_dict["P"], dtype=np.float32),
            "P_idx": np.asarray(obs_dict["P_idx"], dtype=np.int64),
            "P_idx_raw": np.asarray(obs_dict.get("P_idx_raw", obs_dict["P_idx"]), dtype=np.int64),
            "P_sugar": np.asarray(obs_dict.get("P_sugar", np.zeros((self.N_P,), dtype=np.float32)), dtype=np.float32),
            "P_normal_rank": np.asarray(obs_dict.get("P_normal_rank", np.zeros((self.N_P,), dtype=np.float32)), dtype=np.float32),
            "G_mask": np.asarray(obs_dict["G_mask"], dtype=np.bool_),
            "P_mask": np.asarray(obs_dict["P_mask"], dtype=np.bool_),
            "state_features": np.asarray(obs_dict["state_feats"], dtype=np.float32),
        }

        #info = self._get_info()
        info = {
            "pair_sugar": self.state["P_sugar"].copy(),
            "pair_normal_rank": self.state["P_normal_rank"].copy(),
        }

        return self.state, reward, bool(terminated), truncated, info
    
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
    ):
        super().__init__()

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
            self.n_vars = len([s.strip() for s in symbols.split(",") if s.strip()])
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
            self.n_vars = _parse_nvars_from_ideal_dist(ideal_dist)
        self.k = k
        self.g_dim = self.n_vars * self.k
        self.pair_dim = 2 * self.n_vars * self.k

        # Variable-size observations: basis elements G and pair list P.
        self.observation_space = gym.spaces.Dict({
            "P": gym.spaces.Sequence(
                gym.spaces.Box(low=0, high=np.inf, shape=(self.pair_dim,), dtype=np.int32),
                stack=True,
            ),
            "P_idx": gym.spaces.Sequence(
                gym.spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.int32),
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
        P = self._coerce_matrix(obs_dict["P"], self.pair_dim)
        P_idx = self._coerce_matrix(obs_dict["P_idx"], 2)
        G = self._coerce_matrix(obs_dict["G"], self.g_dim)

        state_features = np.asarray(obs_dict["state_feats"], dtype=np.float32)
        if state_features.shape != (14,):
            state_features = state_features.reshape(14,)
        return {
            "P": P,
            "P_idx": P_idx,
            "G": G,
            "P_mask": np.ones((P.shape[0],), dtype=np.int8),
            "G_mask": np.ones((G.shape[0],), dtype=np.int8),
            "state_features": state_features,
        }

    def _build_info(self, state: dict) -> dict:
        n_pairs = int(state["P"].shape[0])
        n_basis = int(state["G"].shape[0])
        return {
            "n_pairs": n_pairs,
            "n_basis": n_basis,
        }

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
    
