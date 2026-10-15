import pytest

from deepgroebner_pytorch_cocoa import ppo_evaluation, ppo_training
from deepgroebner_pytorch_cocoa.config_dataclasses import Config


def _training_config(tmp_path, *, steps_per_update=4, updates=6):
    cfg = Config()
    cfg.env.coeff_ring = "QQ"
    cfg.env.symbols = "x,y"
    cfg.env.basis_elements = "x^2 - y,x*y - 1,y^2 - x"
    cfg.env.elimination = "none"
    cfg.training.num_envs = 1
    cfg.training.num_steps_per_update = steps_per_update
    cfg.training.total_timesteps = steps_per_update * updates
    cfg.training.num_epochs_per_update = 1
    cfg.training.num_minibatches = 2
    cfg.training.cuda = False
    cfg.training.track = False
    cfg.training.pytorch_num_threads = 1
    cfg.training.system_metrics_enabled = False
    cfg.training.console_verbosity = "quiet"
    cfg.training.save_actor = False
    cfg.training.save_critic = False
    cfg.training.output_directory = str(tmp_path)
    return cfg


def test_disabled_evaluation_is_never_invoked(tmp_path, monkeypatch):
    #arrange
    cfg = _training_config(tmp_path)
    cfg.training.hparam_eval_interval_steps = 1
    cfg.training.hparam_eval_num_examples = 0

    def unexpected_evaluation(self, **kwargs):
        pytest.fail("Training invoked disabled evaluation")

    monkeypatch.setattr(
        ppo_evaluation.TrainingEvaluation, "evaluate", unexpected_evaluation
    )
    trainer = ppo_training.PPOTrainer(cfg)

    #act
    result = trainer.run()

    #assert
    assert result.summary["global_step"] == 24
    assert trainer.evaluation.next_step is None


@pytest.mark.parametrize(
    "interval,steps_per_update,updates,expected_schedule",
    [
        (5, 4, 6, [(8, 5, 10), (12, 10, 15), (16, 15, 20), (20, 20, 25)]),
        (3, 8, 3, [(8, 3, 9), (16, 9, 18), (24, 18, 27)]),
    ],
    ids=["skip_updates_and_exact_boundary", "cross_multiple_intervals"],
)
def test_evaluation_runs_only_when_due_and_advances_schedule(
    tmp_path, monkeypatch, interval, steps_per_update, updates, expected_schedule
):
    #arrange
    cfg = _training_config(
        tmp_path, steps_per_update=steps_per_update, updates=updates
    )
    cfg.training.hparam_eval_interval_steps = interval
    cfg.training.hparam_eval_num_examples = 1
    monkeypatch.setattr(
        ppo_evaluation,
        "resolve_greedy_eval_corpus",
        lambda cfg, seeds: ([{"episode_seed": seeds[0]}], None, False),
    )
    monkeypatch.setattr(
        ppo_evaluation,
        "run_fixed_greedy_eval",
        lambda **kwargs: {
            "mean_return": 2.0,
            "mean_return_completed_only": 2.0,
            "median_return": 2.0,
            "min_return": 2.0,
            "max_return": 2.0,
            "mean_length": 1.0,
            "max_length": 1,
            "num_examples": 1,
            "num_completed": 1,
            "completion_rate": 1.0,
            "num_truncated_total": 0,
            "num_timed_out": 0,
            "num_worker_exited": 0,
            "elapsed_seconds": 0.0,
        },
    )
    observed_schedule = []
    evaluate = ppo_evaluation.TrainingEvaluation.evaluate

    def record_evaluation(self, **kwargs):
        previous_next_step = self.next_step
        evaluate(self, **kwargs)
        observed_schedule.append(
            (kwargs["global_step"], previous_next_step, self.next_step)
        )

    monkeypatch.setattr(
        ppo_evaluation.TrainingEvaluation, "evaluate", record_evaluation
    )
    trainer = ppo_training.PPOTrainer(cfg)

    #act
    result = trainer.run()

    #assert
    assert observed_schedule == expected_schedule
    assert trainer.evaluation.next_step == expected_schedule[-1][2]
    assert result.summary["policy_updates"] == updates
    assert result.summary["greedy_eval_last_mean_return"] == 2.0
