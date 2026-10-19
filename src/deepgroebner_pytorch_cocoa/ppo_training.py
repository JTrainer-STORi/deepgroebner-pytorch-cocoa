"""PPO training orchestration with explicit run state and managed resources."""

from __future__ import annotations

from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
import json
import os
import pprint as pp
import time

import numpy as np
import torch
import torch.optim as optim

from .config_dataclasses import Config, ModelInputs
from .diagnostics import run_actor_global_diagnostics, run_actor_ranking_diagnostics
from .model_input_utils import (
    bind_obs_dict_as_model_inputs_views,
    build_model_inputs_from_batched_obs,
    copy_obs_dict_to_model_inputs,
    get_space_last_dim,
)
from .ppo_agent import Agent
from .ppo_evaluation import TrainingEvaluation
from .ppo_normalization import RunningMeanStd, RewardNormalizer, build_value_normalizer
from .ppo_optimization import compute_returns_and_advantages, optimize_update
from .ppo_rollout import RolloutBuffer
from .system_metrics import SystemMetricsRecorder, summarize_state_feature_rollout
from .training_env import create_training_envs
from .training_summary import build_training_summary
from .training_utils import (
    console_allows,
    console_print,
    log_system_metrics,
    print_update_summary,
    tensor_stat_to_float,
    configure_training_runtime,
    create_run_writer,
    print_runtime_cpu_thread_info,
    resolve_training_config,
    save_model_checkpoints,
)


@dataclass
class TrainingResult:
    summary: dict
    checkpoint_paths: dict[str, str]
    run_directory: str


class PPOTrainer:
    """Main class responsible for training the policy(actor) and value(critic) networks using the PPO algorithm"""

    def __init__(self, cfg: Config):

        self.cfg = resolve_training_config(cfg)

    def _setup(self, resources: ExitStack):
        
        cfg = self.cfg

        # Print the training configuration to console
        if console_allows(cfg, "verbose"):
            pp.pprint(vars(cfg))
            print("\n\n\n")

        # Setup writer for local logging and weights and bias tracking (if enabled using --training.track True)
        self.run_name, self.writer = create_run_writer(cfg)
        resources.callback(self.writer.close)

        self.device = configure_training_runtime(cfg)

        # Resposible for logging useful metrics from the hardware 
        self.system_metrics_recorder = SystemMetricsRecorder(
            os.path.join(self.writer.log_dir, "system_metrics.jsonl"),
            device=self.device,
            enabled=cfg.training.system_metrics_enabled,
            sample_interval_seconds=cfg.training.system_metrics_interval_seconds,
        )
        resources.callback(self.system_metrics_recorder.close)

        # Initialize environments for training rollouts
        self.envs = create_training_envs(cfg)
        resources.callback(self.envs.close)

        # If using a training run in which the priority queue isn't being used then the buffers used in training will need to be modified to accomodate states of varying size (unknown a-priori)
        self.ragged = "nopq" in cfg.env.gym_id.lower()
        self.pq_budget = (
            None if self.ragged else cfg.env.N if cfg.env.N_P is None else cfg.env.N_P
        )

        # Prints information about how many CPU threads will be used by PyTorch and the environment during training.
        print_runtime_cpu_thread_info(self.envs, cfg)

        # Set up greedy evaluation during training based on config settings
        self.evaluation = TrainingEvaluation(cfg)

        space = self.envs.single_observation_space

        # Set up state feature and reward normalization if enabled in config
        self.obs_sf_normalizer = (
            RunningMeanStd(
                shape=space["state_features"].shape, epsilon=1e-4, device=self.device
            )
            if cfg.training.normalize_observation_state_features
            else None
        )
        self.reward_normalizer = (
            RewardNormalizer(
                cfg.training.num_envs,
                cfg.training.gamma,
                epsilon=cfg.training.reward_normalization_epsilon,
                clip_range=cfg.training.reward_normalization_clip,
                device=self.device,
            )
            if cfg.training.normalize_reward
            else None
        )

        # Set the config arguments for the models that can only be set once the environment has been initialized
        for model in (cfg.actor, cfg.critic):
            model.state_embedding_head_in_dim = get_space_last_dim(space, "G")
            model.state_features_embedding_head_in_dim = get_space_last_dim(
                space, "state_features"
            )
        cfg.actor.pair_embedding_head_in_dim = get_space_last_dim(space, "P")

        self.agent = Agent(cfg, self.envs).to(self.device)
        self.value_normalizer = build_value_normalizer(
            cfg, self.agent, device=self.device
        )
        if (
            cfg.training.normalize_value_function
            and cfg.critic.critic_strategy == "heuristic-only"
        ):
            console_print(
                cfg,
                "Skipping value normalization because "
                "cfg.critic.critic_strategy='heuristic-only' has no learned critic.",
            )
        for name in ("actor", "critic"):
            if name == "critic" and cfg.critic.critic_strategy == "heuristic-only":
                continue
            model = getattr(self.agent, name)
            console_print(cfg, f"{name.title()} Network:", level="verbose")
            if console_allows(cfg, "verbose"):
                print(model)
            console_print(
                cfg,
                f"{name.title()} parameters: {sum(p.numel() for p in model.parameters())}",
            )
            console_print(cfg, "\n\n\n", level="verbose")
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=cfg.training.learning_rate, eps=1e-5
        )
        # Initialize buffer for storing rollouts
        self.rollout = RolloutBuffer(cfg, self.envs, self.device, ragged=self.ragged)
        self.global_step = 0
        self.policy_updates = 0
        self.episode_counter = 0
        self.max_episodic_return = float("-inf")
        self.min_episodic_return = float("inf")
        self.recent_returns = deque(maxlen=100)
        self.last_k_avg_return = float("nan")
        self.timed_out = False
        self.start_time = time.time()
        seed = (
            [cfg.env.env_seed] * cfg.training.num_envs
            if cfg.env.fix_ideal
            else cfg.env.env_seed
        )
        # Prepare each environment by calling the rest method
        obs, self.info = self.envs.reset(seed=seed)
        self.inputs = None
        self.inputs = self._observation_to_input(obs)
        self._record_system_metrics(force=True, elapsed_seconds=0.0)
        self.next_done = torch.zeros(
            cfg.training.num_envs, device=self.device, dtype=torch.float32
        )
        self.num_updates = cfg.training.total_timesteps // cfg.training.batch_size

    def _observation_to_input(self, obs: dict) -> ModelInputs:
        """Convert observations to model inputs and normalize state features."""
        cfg = self.cfg
        if self.ragged:
            inputs = build_model_inputs_from_batched_obs(obs, self.device)
        elif self.device.type == "cpu":
            inputs = bind_obs_dict_as_model_inputs_views(obs)
        elif self.inputs is None:
            inputs = build_model_inputs_from_batched_obs(obs, self.device)
        else:
            inputs = self.inputs
            copy_obs_dict_to_model_inputs(obs, inputs)
        if self.obs_sf_normalizer is not None:
            self.obs_sf_normalizer.update(inputs.state_features)
            inputs.state_features.copy_(
                self.obs_sf_normalizer.normalize(
                    inputs.state_features,
                    epsilon=cfg.training.observation_normalization_epsilon,
                    clip_range=cfg.training.observation_normalization_clip,
                )
            )
        return inputs

    def _record_system_metrics(
        self, *, force=False, elapsed_seconds=None, state_features=None
    ):
        recorder = self.system_metrics_recorder
        if not recorder.enabled:
            return
        sample_time = recorder.clock_fn()
        if (
            not force
            and recorder.last_sample_time is not None
            and recorder.sample_interval_seconds > 0
            and (
                sample_time - recorder.last_sample_time
                < recorder.sample_interval_seconds
            )
        ):
            return

        extra_metrics = summarize_state_feature_rollout(
            self.inputs.state_features if state_features is None else state_features,
            pq_budget=self.pq_budget,
        )
        record = recorder.record(
            global_step=self.global_step,
            policy_updates=self.policy_updates,
            elapsed_seconds=(
                time.time() - self.start_time
                if elapsed_seconds is None
                else elapsed_seconds
            ),
            extra_metrics=extra_metrics,
            sample_time=sample_time,
        )
        log_system_metrics(self.writer, record, self.global_step)

    def _collect_rollout(self):
        """ Collect num_steps_per_update transitions using policy rollout in the environment"""
        cfg = self.cfg
        for step in range(cfg.training.num_steps_per_update):
            self.global_step += cfg.training.num_envs
            self.rollout.store_observation(
                step, self.inputs, self.next_done, info=self.info
            )
            with torch.no_grad():
                action, logprob, _, value, heuristic_value = (
                    self.agent.get_action_and_value(self.inputs)
                )
            self.rollout.store_action(step, action, logprob, value, heuristic_value)
            obs, reward, terminations, truncations, infos = self.envs.step(
                action.cpu().numpy()
            )
            done = torch.logical_or(
                torch.from_numpy(terminations).to(device=self.device, dtype=torch.bool),
                torch.from_numpy(truncations).to(device=self.device, dtype=torch.bool),
            ).to(dtype=torch.float32)
            self.next_done.copy_(done)
            reward = torch.from_numpy(reward).to(
                device=self.device, dtype=torch.float32
            )
            if self.reward_normalizer is not None:
                reward = self.reward_normalizer.normalize(reward, done)
            self.rollout.rewards[step].copy_(reward)
            self.inputs = self._observation_to_input(obs)
            self.info = infos
            self._log_completed_episodes(infos)

    def _log_completed_episodes(self, infos):
        # Same-step autoreset puts completed-episode statistics in final_info.
        infos = infos.get("final_info", infos)
        if "_episode" in infos:
            for i, entry in enumerate(infos["_episode"]):
                if entry:
                    self.episode_counter += 1
                    episodic_return = infos["episode"]["r"][i].item()
                    self.recent_returns.append(episodic_return)
                    self.last_k_avg_return = float(np.mean(self.recent_returns))
                    self.max_episodic_return = max(
                        self.max_episodic_return, episodic_return
                    )
                    self.min_episodic_return = min(
                        self.min_episodic_return, episodic_return
                    )
                    self.writer.add_scalar(
                        "training_rollouts/episodic_return", episodic_return, self.global_step
                    )
                    self.writer.add_scalar(
                        "training_rollouts/max_episodic_return",
                        self.max_episodic_return,
                        self.global_step,
                    )
                    self.writer.add_scalar(
                        "training_rollouts/min_episodic_return",
                        self.min_episodic_return,
                        self.global_step,
                    )
                    self.writer.add_scalar(
                        "training_rollouts/episodic_return_last_100_avg",
                        self.last_k_avg_return,
                        self.global_step,
                    )
                    self.writer.add_scalar(
                        "training_rollouts/episodic_length",
                        infos["episode"]["l"][i].item(),
                        self.global_step,
                    )
                    self.writer.add_scalar(
                        "training_rollouts/episodes_completed",
                        self.episode_counter,
                        self.global_step,
                    )

    def _prepare_batch(self):
        """Use the rollout data to calculate returns and advantages for each timestep and flatten the time-step and then environment dimensions of the rollouts"""
        cfg = self.cfg
        with torch.no_grad():
            next_value, _ = self.agent.get_value(self.inputs)
        advantages, returns = compute_returns_and_advantages(
            self.rollout.rewards,
            self.rollout.values,
            self.rollout.dones,
            next_value,
            self.next_done,
            gamma=cfg.training.gamma,
            gae_lambda=cfg.training.gae_lambda,
            use_gae=cfg.training.use_gae,
        )
        return self.rollout.flatten(advantages, returns)

    def _run_diagnostics(self, batch, indices):
        cfg = self.cfg
        diagnostics_input = batch.model_inputs(indices)
        update_idx = self.policy_updates + 1
        return_attention = (
            cfg.actor.pair_embedding_head == "alphastar-buchberger-pair-transformer"
        )
        has_global_state = (
            cfg.actor.state_embedding_head is not None
            or cfg.actor.state_features_embedding_head is not None
        )

        if has_global_state or return_attention:
            run_actor_global_diagnostics(
                actor=self.agent.actor,
                input=diagnostics_input,
                writer=self.writer,
                update_idx=update_idx,
                global_step=self.global_step,
                diag_every=(
                    cfg.training.counterfactual_diagnostic_frequency if has_global_state else 0
                ),
                grad_every=(
                    cfg.training.gradient_sensitivity_diagnostic_frequency if has_global_state else 0
                ),
                return_attention=return_attention,
                attention_every=cfg.training.attention_diagnostic_frequency,
            )

            self.agent.actor.zero_grad(set_to_none=True)

        ranking_every = cfg.training.ranking_diagnostic_frequency
        if ranking_every > 0 and update_idx % ranking_every == 0:
            run_actor_ranking_diagnostics(
                actor=self.agent.actor,
                input=diagnostics_input,
                writer=self.writer,
                update_idx=update_idx,
                global_step=self.global_step,
                ranking_every=ranking_every,
                **batch.ranking_metadata(indices),
            )

    def _log_update(self, metrics, batch, update):
        cfg = self.cfg
        self.policy_updates += 1
        scalars = {
            "algorithm_general/policy_updates": self.policy_updates,
        }
        if cfg.training.anneal_learning_rate:
            scalars["algorithm_general/learning_rate"] = self.optimizer.param_groups[0][
                "lr"
            ]
        if cfg.critic.critic_strategy != "heuristic-only":
            scalars.update(
                {
                    "value_function/loss": metrics.value_loss,
                    "value_function/weighted_loss": cfg.training.critic_loss_coefficient
                    * metrics.value_loss,
                    "value_function/clip_fraction": np.mean(metrics.value_clipfracs),
                    "value_function/grad_norm_before_clip": metrics.critic_grad_norm,
                }
            )
        scalars.update(
            {
                "value_function/explained_variance": metrics.explained_variance,
                "policy_function/loss": metrics.policy_loss,
                "policy_function/entropy_raw": metrics.entropy_raw,
                "policy_function/entropy_normalized": metrics.entropy_normalized,
                "policy_function/entropy_bonus": metrics.entropy_bonus,
                "policy_function/weighted_entropy_bonus": cfg.training.entropy_coefficient
                * metrics.entropy_bonus,
                "policy_function/old_approx_kl": metrics.old_approx_kl,
                "policy_function/approx_kl": metrics.approx_kl,
                "value_function/advantage_mean": (
                    sum(metrics.advantage_means) / len(metrics.advantage_means)
                    if metrics.advantage_means
                    else float("nan")
                ),
                "value_function/advantage_std": (
                    (
                        sum(s**2 for s in metrics.advantage_stds)
                        / len(metrics.advantage_stds)
                    )
                    ** 0.5
                    if metrics.advantage_stds
                    else float("nan")
                ),
                "policy_function/grad_norm_before_clip": metrics.actor_grad_norm,
                "policy_function/clip_fraction": np.mean(metrics.clipfracs),
            }
        )
        if self.obs_sf_normalizer is not None:
            std = torch.sqrt(
                self.obs_sf_normalizer.var
                + cfg.training.observation_normalization_epsilon
            )
            scalars.update(
                {
                    "policy_function/state_features_running_mean": tensor_stat_to_float(
                        self.obs_sf_normalizer.mean
                    ),
                    "policy_function/state_features_running_std": tensor_stat_to_float(
                        std
                    ),
                    "policy_function/state_features_running_std_max": std.detach()
                    .max()
                    .item(),
                    "policy_function/state_features_running_std_min": std.detach()
                    .min()
                    .item(),
                }
            )
        if self.reward_normalizer is not None:
            std = torch.sqrt(
                self.reward_normalizer.rms.var
                + cfg.training.reward_normalization_epsilon
            )
            scalars.update(
                {
                    "algorithm_general/reward_running_return_mean": tensor_stat_to_float(
                        self.reward_normalizer.rms.mean
                    ),
                    "algorithm_general/reward_running_return_std": tensor_stat_to_float(
                        std
                    ),
                }
            )
        if self.value_normalizer is not None:
            std = torch.sqrt(
                self.value_normalizer.var + cfg.training.value_normalization_epsilon
            )
            scalars.update(
                {
                    "value_function/running_target_mean": tensor_stat_to_float(
                        self.value_normalizer.mean
                    ),
                    "value_function/running_target_std": tensor_stat_to_float(
                        std
                    ),
                }
            )
        sps = int(self.global_step / (time.time() - self.start_time))
        scalars["performance/steps_per_second"] = sps
        for tag, value in scalars.items():
            self.writer.add_scalar(tag, value, self.global_step)
        self._record_system_metrics(
            state_features=batch.observations["state_features"],
        )
        print_update_summary(
            cfg=cfg,
            update_idx=update,
            num_updates=self.num_updates,
            global_step=self.global_step,
            sps=sps,
            recent_returns=self.recent_returns,
            last_k_avg_return=self.last_k_avg_return,
            policy_loss=metrics.policy_loss,
            entropy=metrics.entropy_bonus,
            approx_kl=metrics.approx_kl,
            critic_loss=(
                metrics.value_loss
                if cfg.critic.critic_strategy != "heuristic-only"
                else None
            ),
        )

    def _evaluate_if_due(self):
        if (
            self.evaluation.enabled
            and self.evaluation.next_step is not None
            and self.global_step >= self.evaluation.next_step
        ):
            self.evaluation.evaluate(
                agent=self.agent,
                writer=self.writer,
                global_step=self.global_step,
                run_name=self.run_name,
                device=self.device,
                obs_sf_normalizer=self.obs_sf_normalizer,
            )

    def run(self) -> TrainingResult:
        """Train, save final artifacts, and close resources on success or failure."""
        cfg = self.cfg
        with ExitStack() as resources:
            self._setup(resources)
            self._evaluate_if_due()
            for update in range(1, self.num_updates + 1):
                if cfg.training.max_runtime_seconds is not None:
                    elapsed = time.time() - self.start_time
                    if elapsed >= cfg.training.max_runtime_seconds:
                        self.timed_out = True
                        print(
                            f"Stopping early due to max runtime budget: elapsed={elapsed:.1f}s, "
                            f"limit={cfg.training.max_runtime_seconds}s"
                        )
                        break
                self.rollout.clear()
                if cfg.training.anneal_learning_rate:
                    fraction = 1.0 - (update - 1.0) / self.num_updates
                    self.optimizer.param_groups[0]["lr"] = (
                        fraction * cfg.training.learning_rate
                    )
                self._collect_rollout()
                batch = self._prepare_batch()
                metrics = optimize_update(
                    cfg, self.agent, self.optimizer, batch, self.value_normalizer
                )
                self._run_diagnostics(batch, metrics.last_indices)
                self._log_update(metrics, batch, update)
                self._evaluate_if_due()
            checkpoint_paths = save_model_checkpoints(cfg, self.agent, self.run_name)
            for kind, path in sorted(checkpoint_paths.items()):
                print(f"Final {kind} checkpoint saved to {path}")
            summary = build_training_summary(
                cfg,
                self.evaluation,
                run_name=self.run_name,
                global_step=self.global_step,
                policy_updates=self.policy_updates,
                episodes_completed=self.episode_counter,
                max_episodic_return=self.max_episodic_return,
                last_100_avg_return=(
                    self.last_k_avg_return if self.recent_returns else None
                ),
                timed_out=self.timed_out,
                elapsed_seconds=time.time() - self.start_time,
            )
            if cfg.training.hparam_metrics_out is not None:
                os.makedirs(
                    os.path.dirname(cfg.training.hparam_metrics_out) or ".",
                    exist_ok=True,
                )
                with open(
                    cfg.training.hparam_metrics_out, "w", encoding="utf-8"
                ) as handle:
                    json.dump(summary, handle, indent=2)
                print(
                    f"Wrote hparam metrics summary to {cfg.training.hparam_metrics_out}"
                )
            self._record_system_metrics(force=True)
            return TrainingResult(summary, checkpoint_paths, self.writer.log_dir)


def train(cfg: Config) -> TrainingResult:
    """Run PPO from Python using an independent copy of the supplied configuration."""
    return PPOTrainer(cfg).run()
