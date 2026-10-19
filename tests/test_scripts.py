from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest

import evaluate_model
import evaluate_heuristics
import generate_corpus
import run_training
from deepgroebner_pytorch_cocoa.config_dataclasses import Config
from deepgroebner_pytorch_cocoa.evaluation_resources import (
    RESOURCE_FIELDS,
    RESOURCE_STATISTIC_FIELDS,
)
from deepgroebner_pytorch_cocoa.model_evaluation import (
    TestCorpusConfig as CorpusTestConfig,
    EvaluationResult,
    evaluate,
)


def test_training_script_passes_default_config_directly_to_trainer(monkeypatch):
    #arrange
    configs = []
    result = object()
    expected_config = asdict(Config())

    def capture(cfg):
        configs.append(cfg)
        return result

    monkeypatch.setattr(run_training, "train", capture)

    #act
    actual_result = run_training.main([])

    #assert
    assert actual_result is result
    assert type(configs[0]) is Config
    assert asdict(configs[0]) == expected_config


@pytest.mark.parametrize(
    "pair_head",
    [
        "per-entity-mlp",
        "symmetric-per-entity-mlp",
        "alphastar-buchberger-pair-transformer",
    ],
)
def test_training_script_passes_nested_options_to_trainer(monkeypatch, pair_head):
    #arrange
    configs = []
    monkeypatch.setattr(run_training, "train", configs.append)
    args = [
        f"--actor.pair-embedding-head={pair_head}",
        "--actor.state-embedding-head=deep-set",
        "--actor.state-features-embedding-head=mlp",
        "--actor.pair-combiner=sqa",
        "--actor.pair-embedding-head-out-dim=32",
        "--actor.single-query-attention-final-attention=scaled dot-product",
        "--actor.use-lcm-gcd=False",
        "--critic.state-features-embedding-head=None",
        "--critic.value-head-hidden-dims=96",
        "--env.ideal-dist=None",
        "--env.coeff-ring=QQ",
        "--env.symbols=x,y",
        "--env.basis-elements=x^2 - y,x*y - 1",
        "--env.N-P=24",
        "--training.learning-rate=0.0001",
        "--training.total-timesteps=64",
        "--training.track=True",
        "--training.normalize-observation-state-features=False",
        "--training.greedy-eval-on-nopq-env=True",
        "--training.greedy-eval-first-step=12000",
        "--training.greedy-eval-interval-steps=4000",
        "--training.greedy-eval-num-examples=20",
        "--training.greedy-eval-corpus=/tmp/shared_evaluation.jsonl",
        "--training.greedy-eval-seed=100",
        "--training.greedy-eval-progress-interval-seconds=5",
        "--training.greedy-eval-max-episode-seconds=60",
        "--training.counterfactual-diagnostic-frequency=5",
        "--training.attention-diagnostic-frequency=2",
        "--training.output-directory=/tmp/path with spaces",
    ]

    #act
    run_training.main(args)
    (cfg,) = configs

    #assert
    assert cfg.actor.pair_embedding_head == pair_head
    assert cfg.actor.state_embedding_head == "deep-set"
    assert cfg.actor.state_features_embedding_head == "mlp"
    assert cfg.actor.pair_combiner == "sqa"
    assert cfg.actor.pair_embedding_head_out_dim == 32
    assert cfg.actor.single_query_attention_final_attention == "scaled dot-product"
    assert cfg.actor.use_lcm_gcd is False
    assert cfg.critic.state_features_embedding_head is None
    assert cfg.critic.value_head_hidden_dims == 96
    assert cfg.env.ideal_dist is None
    assert cfg.env.coeff_ring == "QQ"
    assert cfg.env.symbols == "x,y"
    assert cfg.env.basis_elements == "x^2 - y,x*y - 1"
    assert cfg.env.N_P == 24
    assert cfg.training.learning_rate == 0.0001
    assert cfg.training.total_timesteps == 64
    assert cfg.training.track is True
    assert cfg.training.normalize_observation_state_features is False
    assert cfg.training.greedy_eval_on_nopq_env is True
    assert cfg.training.greedy_eval_first_step == 12000
    assert cfg.training.greedy_eval_interval_steps == 4000
    assert cfg.training.greedy_eval_num_examples == 20
    assert cfg.training.greedy_eval_corpus == "/tmp/shared_evaluation.jsonl"
    assert cfg.training.greedy_eval_seed == 100
    assert cfg.training.greedy_eval_progress_interval_seconds == 5
    assert cfg.training.greedy_eval_max_episode_seconds == 60
    assert cfg.training.counterfactual_diagnostic_frequency == 5
    assert cfg.training.attention_diagnostic_frequency == 2
    assert cfg.training.output_directory == "/tmp/path with spaces"
    assert cfg.actor.pair_embedding_head_in_dim is None
    assert cfg.training.batch_size is None


def test_training_script_reads_process_arguments(monkeypatch):
    #arrange
    configs = []
    monkeypatch.setattr(run_training, "train", configs.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_training.py",
            "--actor.state-features-embedding-head=mlp",
            "--training.learning-rate=0.001",
        ],
    )

    #act
    run_training.main()

    #assert
    assert configs[0].actor.state_features_embedding_head == "mlp"
    assert configs[0].actor.state_embedding_head is None
    assert configs[0].training.learning_rate == 0.001


@pytest.mark.parametrize(
    "option",
    [
        "--env.priority-queue-strategy=last",
        "--training.hparam-eval-interval-steps=4",
        "--training.greedy-eval-nopq-gym-id=BuchbergerExponentsNoPQ-v1",
        "--training.greedy-eval-max-pairs-processed-multiplier=5",
        "--training.greedy-eval-max-pairs-processed-slack=0",
        "--training.greedy-eval-nopq-benchmark-pairs-processed-multiplier=5",
        "--training.greedy-eval-nopq-benchmark-pairs-processed-slack=50",
        "--training.greedy-eval-max-steps-since-basis-update=100",
    ],
)
def test_training_script_rejects_retired_options(option, monkeypatch):
    #arrange
    monkeypatch.setattr(
        run_training, "train", lambda cfg: pytest.fail("Unexpected training")
    )

    #act
    #assert
    with pytest.raises(SystemExit) as error:
        run_training.main([option])
    assert error.value.code != 0


@pytest.mark.parametrize(
    "script,option",
    [
        (run_training, "--training.total-timesteps"),
        (evaluate_model, "--model-name"),
        (evaluate_heuristics, "--corpus"),
        (generate_corpus, "--examples"),
    ],
)
def test_scripts_expose_help_without_running_jobs(script, option, capsys):
    #arrange
    args = ["--help"]

    #act
    #assert
    with pytest.raises(SystemExit) as exc:
        script.main(args)
    output = capsys.readouterr().out
    assert exc.value.code == 0
    assert option in output


def test_scripts_generate_train_and_evaluate_a_fixed_ideal(tmp_path, monkeypatch):
    #arrange
    basis = "x^2 - y,x*y - 1,y^2 - x"
    expected_basis = [entry.replace(" ", "") for entry in basis.split(",")]
    corpus_path = tmp_path / "test_corpora" / "ideals.jsonl"
    corpus_args = [
        "--ideal-distribution=None",
        "--coeff-ring=QQ",
        "--symbols=x,y",
        f"--basis-elements={basis}",
        "--elimination=none",
        "--examples=1",
        "--seed=123",
        f"--output-path={corpus_path}",
    ]

    #act
    generate_corpus.main(corpus_args)
    corpus = [json.loads(line) for line in corpus_path.read_text().splitlines()]

    #assert
    assert len(corpus) == 1
    assert corpus[0]["episode_seed"] == 123
    assert [entry.replace(" ", "") for entry in corpus[0]["basis_elements"]] == expected_basis

    #act
    #assert
    with pytest.raises(FileExistsError):
        generate_corpus.main(corpus_args)

    #arrange
    training_args = [
        "--env.coeff-ring=QQ",
        "--env.symbols=x,y",
        f"--env.basis-elements={basis}",
        "--env.elimination=none",
        "--training.num-envs=1",
        "--training.num-steps-per-update=4",
        "--training.total-timesteps=4",
        "--training.num-epochs-per-update=1",
        "--training.num-minibatches=2",
        "--training.greedy-eval-num-examples=0",
        "--training.cuda=False",
        "--training.track=False",
        "--training.pytorch-num-threads=1",
        "--training.system-metrics-enabled=False",
        "--training.console-verbosity=quiet",
        f"--training.output-directory={tmp_path}",
    ]

    #act
    result = run_training.main(training_args)
    actor_path = Path(result.checkpoint_paths["actor"])

    #assert
    assert result.summary["global_step"] == 4
    assert actor_path.is_file()

    #arrange
    other_project = tmp_path / "other_project"
    other_project.mkdir()
    monkeypatch.chdir(other_project)

    for gym_id in ("BuchbergerExponents-v1", "BuchbergerExponentsNoPQ-v1"):
        #arrange
        output = tmp_path / gym_id
        evaluation_args = [
            f"--corpus={corpus_path}",
            f"--model-name={actor_path}",
            f"--gym-id={gym_id}",
            "--device=cpu",
            "--verbosity=quiet",
            f"--output-dir={output}",
        ]

        #act
        cli_result = evaluate_model.main(evaluation_args)
        summary = json.loads((output / "summary.json").read_text())
        episodes = [
            json.loads(line)
            for line in (output / "episode_results.jsonl").read_text().splitlines()
        ]
        metadata = json.loads((output / "metadata.json").read_text())

        #assert
        assert summary["num_evals"] == len(episodes) == 1
        assert summary["length_min"] > 0
        assert summary["return_mean"] == episodes[0]["return"]
        assert metadata["gym_id"] == gym_id
        assert Path(metadata["model_path"]) == actor_path.resolve()

        #arrange
        api_output = tmp_path / "api" / gym_id
        api_cfg = CorpusTestConfig(
            corpus=str(corpus_path),
            model_name=str(actor_path),
            gym_id=gym_id,
            device="cpu",
            verbosity="quiet",
            output_dir=str(api_output),
        )

        #act
        api_result = evaluate(api_cfg)
        api_summary = json.loads(Path(api_result.artifact_paths["summary"]).read_text())

        #assert
        assert isinstance(api_result, EvaluationResult)
        # Timing and process memory vary across the separate CLI and API runs.
        resource_keys = {*RESOURCE_FIELDS, *RESOURCE_STATISTIC_FIELDS, "resources"}
        deterministic = lambda record: {key: value for key, value in record.items() if key not in resource_keys}
        assert deterministic(api_result.summary) == deterministic(summary)
        assert cli_result.summary == summary
        assert [deterministic(row) for row in api_result.episode_results] == [
            deterministic(row) for row in episodes
        ]
        assert cli_result.episode_results == episodes
        assert api_summary["resources"]["elapsed_seconds"] >= 0
        assert api_summary["resources"]["cpu_seconds"] >= 0
        assert api_result.output_directory == str(api_output.resolve())
        assert set(api_result.artifact_paths) == {
            "episode_results_csv",
            "episode_results_jsonl",
            "summary",
            "metadata",
        }
        assert all(Path(path).is_file() for path in api_result.artifact_paths.values())
        assert api_summary == api_result.summary
