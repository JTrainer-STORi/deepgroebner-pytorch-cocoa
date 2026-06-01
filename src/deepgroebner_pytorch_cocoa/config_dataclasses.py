from typing import Optional, Literal
from dataclasses import dataclass, field
import torch
import os
import tyro

@dataclass
class ModelInputs:
  # For most setups, D = 2*E will be true.

  P: torch.Tensor = None # [B, N, D]
  P_idx: torch.Tensor = None # [B, N, 2]
  P_idx_raw: torch.Tensor = None # [B, N, 2]
  P_mask: torch.Tensor = None # [B, N]
  P_sugar: torch.Tensor = None # [B, N]
  P_normal_rank: torch.Tensor = None # [B, N]
  G: torch.Tensor = None # [B, M, E]
  G_mask: torch.Tensor = None # [B, M]
  G_poly_terms: torch.Tensor = None # [B, M, T, E_term]
  G_poly_term_mask: torch.Tensor = None # [B, M, T]
  state_features: torch.Tensor = None # [B, F]
  h_value: torch.Tensor = None # [B]

@dataclass
class ModelConfig:
  """Configuration operations shared between the actor and critic networks"""

  # Fields which must be set at runtime based on the environment's observation space
  state_embedding_head_in_dim: int | None = field(init=False, default=None) # This will be set in the __post_init__ method based on the shape of the state features in the environment's observation space
  state_features_embedding_head_in_dim: int | None = field(init=False, default=None) # This will be set in the __post_init__ method based on the shape of the state features in the environment's observation space

  # State Embedding Head Related Config
  state_embedding_head: Literal['deep-set', 'gnn', 'invariant-set-transformer', 'polynomial-rnn-deep-set', None] = 'deep-set' # The type of state embedding head to use
  state_embedding_head_out_dim: int = 64 # Out dimension of the state embedding head

  # Polynomial RNN state/shared encoder config
  polynomial_rnn_cell: Literal['gru', 'lstm'] = 'gru' # Recurrent cell used to encode variable-length polynomial term sequences.
  polynomial_rnn_hidden_dim: int = 128 # Hidden dimension of the recurrent polynomial encoder.
  polynomial_rnn_num_layers: int = 1 # Number of recurrent layers in the polynomial encoder.
  polynomial_rnn_bidirectional: tyro.conf.FlagConversionOff[bool] = True # Whether the recurrent polynomial encoder is bidirectional.
  polynomial_rnn_dropout: float = 0.0 # Dropout between recurrent layers when polynomial_rnn_num_layers > 1.
  polynomial_rnn_layer_norm: tyro.conf.FlagConversionOff[bool] = True # Whether to apply layer norm after the recurrent summary projection.

  # Deep Set Related Config
  deep_set_phi_n_layers: int = 2 # Number of hidden layers in the phi MLP in the deep set
  deep_set_phi_hidden_dims: int = 128 # Shared hidden dimension of the phi MLP in the deep set
  deep_set_phi_out_dim: int = 64 # Out dimension of the phi MLP in the deep set
  deep_set_phi_activation: Literal['relu', 'gelu', 'selu', 'tanh'] = 'gelu' # Activations used in the phi MLP in the deep set
  deep_set_phi_layer_norm: tyro.conf.FlagConversionOff[bool] = True # Whether to use layer normalization in the phi MLP in the deep set
  deep_set_rho_n_layers: int = 2 # Number of hidden layers in the rho MLP in the deep set
  deep_set_rho_hidden_dims: int = 128 # Shared hidden dimension of the rho MLP in the deep set
  deep_set_rho_activation: Literal['relu', 'gelu', 'selu', 'tanh'] = 'gelu' # Activations used in the rho MLP in the deep set
  deep_set_rho_layer_norm: tyro.conf.FlagConversionOff[bool] = True # Whether to use layer normalization in the rho MLP in the deep set
  deep_set_pooling: Literal['sum', 'mean', 'sqrt'] = 'mean' # Pooling method for the deep set
  deep_set_post_pool_layer_norm: tyro.conf.FlagConversionOff[bool] = True # Whether to use layer normalization after pooling in the deep set

  # Invariant Set Transformer Related Config
  invariant_st_hidden_dim: int = 128 # Hidden dimensions for the invariant set transformer
  invariant_st_num_heads: int = 4 # Number of attention heads for the invariant set transformer
  invariant_st_num_ab: int = 2 # Number of attention blocks in the invariant set transformer encoder; uses ISAB when num_inds is set, otherwise MAB
  invariant_st_layer_norm: tyro.conf.FlagConversionOff[bool] = True # Whether to use layer normalization in the invariant set transformer
  invariant_st_pma_hidden_dim: int = 128 # Hidden dimension for the PMA layer in the invariant set transformer
  invariant_st_pma_num_heads: int = 4 # Number of attention heads for the PMA layer in the invariant set transformer
  invariant_st_final_n_layers: int = 2 # Number of hidden layers for the MLP after the PMA layer in the invariant set transformer
  invariant_st_final_hidden_dims: int = 128 # Shared hidden dimension for the MLP after the PMA layer in the invariant set transformer
  invariant_st_final_activation: Literal['relu', 'gelu', 'selu', 'tanh'] = 'gelu' # Activation function for the MLP after the PMA layer in the invariant set transformer
  invariant_st_final_layer_norm: tyro.conf.FlagConversionOff[bool] = True # Whether to use layer normalization for the MLP after the PMA layer in the invariant set transformer
  invariant_st_num_inds: Optional[int] = 16 # Number of inducing points for ISAB; set to None to use MAB instead

  # GNN Related Config
  gnn_message_passing_layers: int = 4 # Number of message passing layers in the GNN
  gnn_n_layers: int = 1 # Number of hidden layers in each GNN MLP
  gnn_hidden_dims: int = 256 # Shared hidden dimension in each GNN MLP
  gnn_node_encoding_dim: int = 128 # Node encoding dimension for the GNN
  gnn_edge_encoding_dim: int = 128 # Edge encoding dimension for the GNN
  gnn_activation: Literal['relu', 'gelu', 'selu', 'tanh'] = 'gelu' # Activation function for the GNN
  gnn_layer_norm_mlp: tyro.conf.FlagConversionOff[bool] = True # Whether to use layer normalization in the GNN MLP
  gnn_make_undirected: tyro.conf.FlagConversionOff[bool] = False # Whether to make the GNN undirected

  # State Features Embedding Head Related Config
  state_features_embedding_head: Literal['mlp', None] = 'mlp' # The type of state features embedding head to use
  state_features_embedding_head_n_layers: int = 2 # Number of hidden layers for MLP in state features embedding head
  state_features_embedding_head_hidden_dims: int = 128 # Shared hidden dimension for MLP in state features embedding head
  state_features_embedding_head_out_dim: int = 64 # Output dimension for state features embedding head
  state_features_embedding_head_activation: Literal['relu', 'gelu', 'selu', 'tanh'] = 'gelu' # Activation function for MLP in state features embedding head
  state_features_embedding_head_layer_norm: tyro.conf.FlagConversionOff[bool] = True # Whether to use layer normalization in MLP in state features embedding head

  # Combiner Related Config
  state_combiner: Literal['film', 'concat'] = 'film' # Which type of combiner to use when combining the state and state feature heads

  # FiLM Related Config
  film_n_layers: int = 1 # Number of hidden layers for FiLM
  film_hidden_dims: int = 128 # Shared hidden dimension for FiLM
  film_activation: Literal['relu', 'gelu', 'selu', 'tanh'] = 'gelu' # Activation function for FiLM
  film_layer_norm: tyro.conf.FlagConversionOff[bool] = False # Whether to use layer normalization in FiLM
  film_final_n_layers: int = 2 # Number of hidden layers for final FiLM MLP
  film_final_hidden_dims: int = 128 # Shared hidden dimension for final FiLM MLP
  film_final_activation: Literal['relu', 'gelu', 'selu', 'tanh'] = 'gelu' # Activation function for FiLM
  film_final_layer_norm: tyro.conf.FlagConversionOff[bool] = True # Whether to use layer normalization in FiLM
  film_pair_mod_layer_norm: tyro.conf.FlagConversionOff[bool] = True # Whether to use layer normalization in pair modification for FiLM
  film_state_mod_layer_norm: tyro.conf.FlagConversionOff[bool] = True # Whether to use layer normalization in state modification for FiLM

  # Concat Related config
  concat_n_layers: int = 2 # Number of hidden layers for concat MLPs
  concat_hidden_dims: int = 128 # Shared hidden dimension for concat MLPs
  concat_activation: Literal['relu', 'gelu', 'selu', 'tanh'] = 'gelu' # Activation function for concat
  concat_layer_norm: tyro.conf.FlagConversionOff[bool] = False # Whether to use layer normalization in concat

@dataclass
class ActorConfig(ModelConfig):
  """Configuration options for the Actor network."""

  pair_embedding_head_in_dim: int | None = field(init=False, default=None) # This will be set in the __post_init__ method based on the shape of the pair set P in the environment's observation space

  share_polynomial_encoder_between_heads: tyro.conf.FlagConversionOff[bool] = False # Whether to share the item/polynomial encoder between the state head and pair embedding head.
  shared_polynomial_encoder_type: Literal['mlp', 'rnn'] = 'mlp' # Which encoder to use for the shared polynomial backbone.
  pair_combiner: Literal['sqa', 'film', 'concat', 'cross-attn', 'basis-cross-attn', 'bipartite-mp', 'pair-only'] = 'film' # Which type of combiner to use when combining the pair and pair state heads; `basis-cross-attn` lets each pair attend directly to all visible basis elements, while `pair-only` is for setups where state is only used upstream, e.g. for routing a pair head.
  cross_attn_num_heads: int = 4 # Number of attention heads for the cross-attention pair combiner.
  cross_attn_post_attn_layer_norm: tyro.conf.FlagConversionOff[bool] = True # Whether to apply layer norm after residual cross-attention update in pair combiner.
  bipartite_hidden_dim: int = 128 # Hidden dimension used by the bipartite message passing pair combiner.
  bipartite_layers: int = 3 # Number of alternating G->P and P->G message passing layers in bipartite combiner.
  # Single Query Attention Related Config
  single_query_attention_dim: int = 128 # The dimension to use for the single query attention mechanism when the pair combiner is set to 'sqa'
  single_query_attention_n_glimpses: int  = 0 # The number of glimpses to use for the single query attention mechanism
  single_query_glimpse_attention_mode: str = "additive" # The attention mode to use for the single query glimpse attention
  single_query_attention_final_attention: str  = "scaled dot-product" # The final attention mechanism to use for the single query attention
  single_query_attention_norm_qk: tyro.conf.FlagConversionOff[bool] = False # Whether to use normalization on the query and key vectors in the single query attention

  # Pair Embedding Head Related Config
  pair_embedding_head: Literal['per-entity-mlp', 'symmetric-per-entity-mlp', 'moe-symmetric-per-entity-mlp', 'equivariant-set-transformer', 'equivariant-pairset-gnn', None] = 'symmetric-per-entity-mlp' # The type of pair embedding head to use
  pair_embedding_head_n_layers: int = 2 # Number of hidden layers in MLPs in pair embedding head
  pair_embedding_head_hidden_dims: int = 128 # Shared hidden dimension in MLPs in pair embedding head
  pair_embedding_head_out_dim: int = 64 # Out dimension of the pair embedding head
  pair_embedding_head_activation: Literal['relu', 'gelu', 'selu', 'tanh'] = 'gelu' # Activations used in MLPs in pair embedding head
  pair_embedding_head_layer_norm: tyro.conf.FlagConversionOff[bool] = True # Whether to use layer normalization in MLPs in pair embedding head
  pair_scoring_head_activation: Literal['identity', 'relu', 'gelu', 'selu', 'tanh'] = 'gelu' # Activation used before the final scalar scorer in pair-only/no-state policy heads
  pair_scoring_head_layer_norm: tyro.conf.FlagConversionOff[bool] = True # Whether to apply layer normalization before the final scalar scorer in pair-only/no-state policy heads
  use_lcm_gcd: tyro.conf.FlagConversionOff[bool] = True # Whether to use Lowest Common Multiple/Greatest Common Divisor features in pair embedding head
  
  moe_num_experts: int = 4 # Number of symmetric per-entity MLP experts used by the routed MoE pair embedding head.
  moe_top_k: int = 2 # Number of experts selected per state by the routed MoE pair embedding head. Set to 1 or 2 for sparse MoE-style routing.
  moe_router_n_layers: int = 1 # Number of hidden layers in the state-conditioned router MLP for the routed MoE pair embedding head.
  moe_router_hidden_dims: int = 128 # Shared hidden dimension in the routed MoE router MLP.
  moe_router_activation: Literal['relu', 'gelu', 'selu', 'tanh'] = 'gelu' # Activation used in the routed MoE router MLP.
  moe_router_layer_norm: tyro.conf.FlagConversionOff[bool] = True # Whether to use layer normalization in the routed MoE router MLP.

  # config related to the item encoding in the SymmetricPerEntityMLP
  item_embedding_head_n_layers: int = 2 # Number of hidden layers in MLPs in item embedding head
  item_embedding_head_hidden_dims: int = 128 # Shared hidden dimension in MLPs in item embedding head
  item_embedding_head_out_dim: int = 128 # Out dimension of the item embedding head
  item_embedding_head_activation: Literal['relu', 'gelu', 'selu', 'tanh'] = 'gelu' # Activations used in MLPs in item embedding head
  item_embedding_head_layer_norm: tyro.conf.FlagConversionOff[bool] = True # Whether to use layer normalization in MLPs in item embedding head

  # Equivariant Set Transformer Related Config
  equivariant_st_hidden_dim: int = 128 # Hidden dimensions for the equivariant set transformer
  equivariant_st_num_heads: int = 4 # Number of attention heads for the equivariant set transformer
  equivariant_st_num_ab: int = 2 # Number of attention blocks in the equivariant set transformer encoder; applies to ISAB or MAB depending on encoder type
  equivariant_st_layer_norm: tyro.conf.FlagConversionOff[bool] = False # Whether to use layer normalization in the equivariant set transformer
  equivariant_st_final_n_layers: int = 2 # Number of hidden layers for the MLP after the PMA layer in the equivariant set transformer
  equivariant_st_final_hidden_dims: int = 128 # Shared hidden dimension for the MLP after the PMA layer in the equivariant set transformer
  equivariant_st_final_activation: Literal['relu', 'gelu', 'selu', 'tanh'] = 'gelu' # Activation function for the MLP after the PMA layer in the equivariant set transformer
  equivariant_st_final_layer_norm: tyro.conf.FlagConversionOff[bool] = True # Whether to use layer normalization for the MLP after the PMA layer in the equivariant set transformer
  equivariant_st_num_inds: Optional[int] = 16 # Number of inducing points for ISAB; set to None to use MAB instead

  # Arguments for the equivariant pairset gnn
  pair_gnn_make_undirected: tyro.conf.FlagConversionOff[bool] = False # Whether to make the pairset GNN undirected by adding reverse edges
  pair_gnn_n_layers: int = 1 # Number of hidden layers in pairset GNN MLPs
  pair_gnn_hidden_dims: int = 128 # Shared hidden dimension in pairset GNN MLPs
  pair_gnn_node_embedding_dim: int = 128 # Node embedding dimension for the pairset GNN
  pair_gnn_activation: Literal['relu', 'gelu', 'selu', 'tanh'] = 'gelu' # Activation function for the pairset GNN
  pair_gnn_layer_norm_mlp: tyro.conf.FlagConversionOff[bool] = False # Whether to use layer normalization in the MLPs in the pairset GNN
  pair_gnn_n_heads: int = 4 # Number of attention heads for the pairset GNN, if using a GAT-based architecture
  pair_gnn_message_passing_layers: int = 4 # Number of message passing layers for the pairset GNN

@dataclass
class CriticConfig(ModelConfig):
  """Configuration options for the Critic network."""
  
  # value head related config
  value_function_heuristic: Literal["random", "first", "degree", "normal", "sugar", "policy-mimic"] = "degree" # The heuristic to use for the critic's value estimates. When the critic strategy is `heuristic-only` or `heuristic-plus-learned-residual`
  critic_strategy: Literal["heuristic-only", "heuristic-plus-learned-residual", "learned-only"] = "heuristic-plus-learned-residual" # The strategy to use for the critic's value estimates. "heuristic-only" uses only a heuristic value based on the number of pairs and basis elements, "learned-only" uses only the learned value function, and "heuristic-plus-learned" combines both by adding the heuristic value to the learned value function's output.
  value_head_n_layers: int = 2 # Number of hidden layers for MLP in value head
  value_head_hidden_dims: int = 128 # Shared hidden dimension for MLP in value head
  value_head_activation: Literal['relu', 'gelu', 'selu', 'tanh'] = 'gelu' # Activation function for MLP in value head
  value_head_layer_norm: tyro.conf.FlagConversionOff[bool] = False # Whether to use layer normalization in MLP in value head

@dataclass
class EnvConfig:
  """Configuration options for the Buchberger Environment."""
  gym_id: str = "BuchbergerExponents-v1" # The id of the Gym environment to use, should be registered in deepgroebner_pytorch_cocoa/envs/__init__.py
  ideal_dist: Optional[str] = "3-20-10-weighted" # The distribution to sample ideals from.
  env_seed: int = 12 # The seed for the environment's random number generator, which controls the sampling of ideals and other stochastic aspects of the environment. Setting this to a fixed value can help ensure reproducibility across runs.
  elimination: Literal["gebauermoeller", "lcm", "none"] = "gebauermoeller" # The elimination strategy to use in the environment.
  rewards: Literal["additions", "reductions"] = "additions" # Whether to use a reward signal based on the number of additions or reductions.
  priority_queue_strategy: Literal["first", "last", "random", "degree", "normal", "sugar", "policy-mimic"] = "degree" # The strategy to use for ordering the priority queue which determines which N pairs are included in the agent's observation.
  sort_input: tyro.conf.FlagConversionOff[bool] = False # Whether to sort the input polynomials to Buchberger's algorithm by leading monomial
  sort_reducers: tyro.conf.FlagConversionOff[bool] = True # Whether to sort the reducers in the reduction step of Buchberger's algorithm by leading monomial
  fix_ideal: tyro.conf.FlagConversionOff[bool] = False # Whether to fix the ideal across episodes, useful for evaluation
  N: int = 100 # Default observation budget for both visible pairs/actions and visible basis elements, unless N_P or N_G is set.
  N_P: Optional[int] = None # Optional override for the number of visible S-pairs/actions. If None, uses N.
  N_G: Optional[int] = None # Optional override for the number of visible basis elements. If None, uses N.
  k: int = 2 # The number of lead monomials to show per polynomial in the agent's observation
  coeff_ring: Optional[str] = None # The coefficient ring to use for the polynomials in the environment, e.g. "QQ" for rational numbers, "ZZ" for integers or "ZZmod(101)" for integers modulo 101. If None, it will default to using rational numbers.
  symbols: Optional[str] = None # A comma-separated string of variable names to use for the polynomials in the environment, e.g. "x,y,z".
  basis_elements: Optional[str] = None # A list of strings representing the basis elements to use in the environment, where each string is a polynomial such as "x^2 + y^2 - 1". If provided, this will fix the ideal used in the environment to be the one generated by these basis elements. If None, the environment will sample ideals from the distribution specified by ideal_dist.

@dataclass
class TrainingConfig:
  """Configuration options for training the agent."""
  batch_size: int = field(init=False, default=None) # This will be set in the training script based on the number of environments and the number of steps per update
  minibatch_size: int = field(init=False, default=None) # This will be set in the training script based on the batch size and the number of minibatches

  experiment_name: str = os.path.basename(__file__).rstrip(".py") # The name of the experiment, used for naming files related to this experiment such as model checkpoints and TensorBoard logs.
  learning_rate: float = 2.5e-4 # The learning rate for the optimizer.
  anneal_learning_rate: tyro.conf.FlagConversionOff[bool] = False # Whether to anneal the learning rate over the course of training, typically using a linear schedule that decreases the learning rate from its initial value down to zero by the end of training.
  use_gae: tyro.conf.FlagConversionOff[bool] = True # Whether to use Generalized Advantage Estimation (GAE) for computing the advantage estimates during training.
  normalize_advantage: tyro.conf.FlagConversionOff[bool] = True # Whether to normalize the advantage estimates to have mean zero and standard deviation one across the batch of data used for each update.
  normalize_value_function: tyro.conf.FlagConversionOff[bool] = True # Whether to apply PopArt-style normalization to learned value targets while preserving the critic's raw-value outputs.
  value_normalization_epsilon: float = 1e-8 # Numerical stability epsilon and minimum std used in PopArt value normalization.
  value_normalization_clip: float = 10.0 # Clip range applied after value normalization. Set <= 0 to disable clipping.
  normalize_reward: tyro.conf.FlagConversionOff[bool] = False # Whether to normalize rewards online using running statistics of discounted returns.
  reward_normalization_epsilon: float = 1e-8 # Numerical stability epsilon used in reward normalization.
  reward_normalization_clip: float = 0.0 # Clip range applied after reward normalization. Set <= 0 to disable clipping.
  normalize_observation_state_features: bool = True # Whether to normalize only the floating-point `state_features` observation key.
  observation_normalization_epsilon: float = 1e-8 # Numerical stability epsilon used in state_features observation normalization.
  observation_normalization_clip: float = 10.0 # Clip range applied after state_features normalization. Set <= 0 to disable clipping.
  clip_coef: float = 0.1 # The clipping coefficient for PPO
  clip_critic_loss: tyro.conf.FlagConversionOff[bool] = True # Whether to use a clipped value loss, which can help prevent large updates to the value function that can destabilize training.
  critic_clip_coef: float = 0.1 # The clipping coefficient to use if clip_value_loss is True, which determines how much the value function updates are clipped.
  entropy_coefficient: float = 0.01 # The coefficient for the entropy bonus, which encourages exploration by adding a term to the loss that is proportional to the entropy of the policy's action distribution.
  normalize_masked_action_entropy: tyro.conf.FlagConversionOff[bool] = False # Whether to normalize the entropy bonus by log(number of valid actions) in masked-action states.
  critic_loss_coefficient: float = 0.1 # The coefficient for the value loss term in the overall loss function used for training. 
  share_actor_critic_trunk: tyro.conf.FlagConversionOff[bool] = False # Whether to share the actor's and critic's state-side trunk modules (state embedding, optional state-feature embedding, and state combiner).
  clip_grad_norm_actor: tyro.conf.FlagConversionOff[bool] = True # Whether to apply gradient norm clipping to actor parameters.
  clip_grad_norm_critic: tyro.conf.FlagConversionOff[bool] = True # Whether to apply gradient norm clipping to critic parameters.
  max_grad_norm_actor: float = 0.5 # The maximum gradient norm for the actor (policy) network, used for gradient clipping to help stabilize training.
  max_grad_norm_critic: float = 0.5 # The maximum gradient norm for the critic (value) network, used for gradient clipping to help stabilize training.
  target_kl: Optional[float] = None # The target KL divergence between the old and new policy after an update. If not None, this can be used to implement early stopping during each update if the KL divergence exceeds this target, which can help prevent excessively large updates that can destabilize training.
  training_seed: int = 24 # The seed for the training process
  torch_deterministic: tyro.conf.FlagConversionOff[bool] = False # Ensuring GPU convolution operations produce exactly identical outputs across runs by using deterministic algorithms and controlling parallel execution behavior. Setting this to True can help with reproducibility, but may have performance implications.
  pytorch_num_threads: Optional[int] = None # Optional override for PyTorch intra-op CPU thread pool size (torch.set_num_threads). If None, PyTorch default/thread env vars are used.
  pytorch_num_interop_threads: Optional[int] = None # Optional override for PyTorch inter-op CPU thread pool size (torch.set_num_interop_threads). If None, PyTorch default is used.
  cuda: tyro.conf.FlagConversionOff[bool] = True # Whether to use CUDA (GPU) for training if available.
  enable_torch_compile: tyro.conf.FlagConversionOff[bool] = False # Whether to enable torch.compile for actor/critic training forward passes.
  torch_compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default" # torch.compile mode; use "default" to defer to PyTorch defaults.
  torch_compile_backend: str = "inductor" # Backend passed to torch.compile.
  torch_compile_fullgraph: tyro.conf.FlagConversionOff[bool] = False # Whether to require full-graph capture in torch.compile.
  torch_compile_dynamic: tyro.conf.FlagConversionOff[bool] = False # Whether to enable dynamic-shape support in torch.compile.
  track: tyro.conf.FlagConversionOff[bool] = True # Whether to track the experiment with Weights and Biases (wandb). If True, make sure to set up wandb with your project and API key.
  console_verbosity: Literal["quiet", "normal", "verbose"] = "normal" # Controls trainer terminal output. Use "quiet" for minimal logs, "normal" for update summaries, or "verbose" for config/model dumps and eval progress details.
  wandb_project: str = "ppo-implementation-details" # The name of the Weights and Biases project to use for tracking this experiment.
  wandb_entity: Optional[str] = None # The name of the Weights and Biases entity (team) to use for tracking this experiment. If None, it will default to your personal account.
  wandb_group: Optional[str] = None # Optional W&B group name used to group related runs together in the dashboard.
  wandb_init_timeout: int = 180 # Seconds to wait for wandb.init() before treating startup as timed out.
  wandb_offline_on_timeout: tyro.conf.FlagConversionOff[bool] = True # Whether to retry wandb initialization in offline mode after an init timeout.
  output_directory: Optional[str] = None # The path to a directory to save wandb artifacts if needed
  system_metrics_enabled: tyro.conf.FlagConversionOff[bool] = True # Whether to periodically write process/device memory samples to system_metrics.jsonl in the run directory.
  system_metrics_interval_seconds: float = 10.0 # Wall-clock sampling interval for system_metrics.jsonl. Set <= 0 to record once per PPO update.
  save_actor: tyro.conf.FlagConversionOff[bool] = True # Whether to save the actor model after training
  save_critic: tyro.conf.FlagConversionOff[bool] = False # Whether to save the critic model after training
  max_runtime_seconds: Optional[int] = None # Optional wall-clock budget for a single training run; training stops early when exceeded.
  hparam_metrics_out: Optional[str] = None # Optional JSON path where final training metrics are written for hyperparameter tuning.
  hparam_eval_interval_steps: int = 250_000 # Run fixed-seed greedy evaluation every this many environment steps. Set <= 0 to disable.
  hparam_eval_num_examples: int = 100 # Number of fixed evaluation episodes used for the greedy hyperparameter metric. Set <= 0 to disable.
  hparam_eval_seed: int = 10_000 # Base seed for fixed evaluation examples; episode i uses seed hparam_eval_seed + i.
  hparam_eval_on_train_env: tyro.conf.FlagConversionOff[bool] = True # Whether to run fixed greedy evaluation on the main training env. When explicitly set, this works independently from NoPQ evaluation.
  hparam_eval_on_nopq_env: tyro.conf.FlagConversionOff[bool] = True # Whether to run fixed greedy evaluation on the configured NoPQ env. When explicitly set, this works independently from main-env evaluation.
  hparam_eval_nopq_gym_id: str = "BuchbergerExponentsNoPQ-v1" # Gym id used when evaluating on the NoPQ environment.
  hparam_eval_progress_interval_seconds: float = 30.0 # Seconds between progress messages while greedy evaluation is running. Set <= 0 to disable.
  hparam_eval_max_pairs_processed_multiplier: Optional[float] = None # Legacy no-op. Retained only for backwards-compatible CLI/config loading; greedy eval no longer truncates on processed-pair budgets.
  hparam_eval_max_pairs_processed_slack: int = 0 # Legacy no-op paired with hparam_eval_max_pairs_processed_multiplier.
  hparam_eval_nopq_benchmark_pairs_processed_multiplier: Optional[float] = 5.0 # Legacy no-op. Retained only for backwards-compatible CLI/config loading; greedy eval no longer truncates NoPQ episodes using benchmark-derived processed-pair caps.
  hparam_eval_nopq_benchmark_pairs_processed_slack: int = 50 # Legacy no-op paired with hparam_eval_nopq_benchmark_pairs_processed_multiplier.
  hparam_eval_max_steps_since_basis_update: Optional[int] = None # Legacy no-op. Retained only for backwards-compatible CLI/config loading; greedy eval no longer truncates on basis-update stagnation.
  hparam_eval_max_episode_seconds: Optional[float] = None # Optional wall-clock timeout per greedy-eval episode. Uses a subprocess watchdog so a hung native step can be terminated.
  total_timesteps: int = 1_000_000 # The total number of timesteps to train the agent for.
  num_envs: int = 12 # The number of parallel environments to use during training. 
  num_steps_per_update: int = 1024 # The number of steps to run in each environment before performing an update to the model. This is often referred to as the "rollout length" or "horizon".
  num_epochs_per_update: int = 4 # How many times the model will iterate over the collected data during each update step.
  num_minibatches: int = 4 # The number of mini-batches to use during training. 
  gamma: float = 0.99 # The discount factor for future rewards in reinforcement learning.
  gae_lambda: float = 0.95 # The lambda parameter for Generalized Advantage Estimation (GAE). 
  vectorization_mode: Literal["sync", "async"] = "async" # The mode to use for vectorizing the environments during training. "sync" runs multiple environment copies in the same process, "async" runs each environment copy in a seperate process which can lead to big speedups when many CPU cores are available.
  counterfactual_diagnostic_frequency: int = 20 # How often (in terms of number of updates) to run the counterfactual diagnostic function during training. Set this to 0 to disable the counterfactual diagnostic.
  gradient_sensitivity_diagnostic_frequency: int = 100 # How often (in terms of number of updates) to run the gradient sensitivity diagnostic function during training. Set this to 0 to disable the gradient sensitivity diagnostic.
  ranking_diagnostic_frequency: int = 20 # How often (in terms of number of updates) to run policy-vs-heuristic ranking diagnostics. Set this to 0 to disable.

@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
