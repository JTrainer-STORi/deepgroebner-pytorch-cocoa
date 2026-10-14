from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest

import evaluate_model
import generate_corpus
import run_training
from deepgroebner_pytorch_cocoa.config_dataclasses import Config


def test_training_script_passes_default_config_directly_to_trainer(monkeypatch):
    configs = []
    result = object()

    def capture(cfg):
        configs.append(cfg)
        return result

    monkeypatch.setattr(run_training, "train", capture)
    assert run_training.main([]) is result
    assert type(configs[0]) is Config
    assert asdict(configs[0]) == asdict(Config())


@pytest.mark.parametrize(
    "pair_head",
    [
        "per-entity-mlp",
        "symmetric-per-entity-mlp",
        "alphastar-buchberger-pair-transformer",
    ],
)
def test_training_script_passes_nested_options_to_trainer(monkeypatch, pair_head):
    configs = []
    monkeypatch.setattr(run_training, "train", configs.append)
    run_training.main(
        [
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
            "--training.hparam-eval-on-nopq-env=True",
            "--training.counterfactual-diagnostic-frequency=5",
            "--training.output-directory=/tmp/path with spaces",
        ]
    )
    (cfg,) = configs
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
    assert cfg.training.hparam_eval_on_nopq_env is True
    assert cfg.training.counterfactual_diagnostic_frequency == 5
    assert cfg.training.output_directory == "/tmp/path with spaces"
    assert cfg.actor.pair_embedding_head_in_dim is None
    assert cfg.training.batch_size is None


def test_training_script_reads_process_arguments(monkeypatch):
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
    run_training.main()
    assert configs[0].actor.state_features_embedding_head == "mlp"
    assert configs[0].actor.state_embedding_head is None
    assert configs[0].training.learning_rate == 0.001


@pytest.mark.parametrize(
    "script,option",
    [
        (run_training, "--training.total-timesteps"),
        (evaluate_model, "--model-name"),
        (generate_corpus, "--examples"),
    ],
)
def test_scripts_expose_help_without_running_jobs(script, option, capsys):
    with pytest.raises(SystemExit) as exc:
        script.main(["--help"])
    assert exc.value.code == 0
    assert option in capsys.readouterr().out


def test_scripts_generate_train_and_evaluate_a_fixed_ideal(tmp_path):
    basis = "x^2 - y,x*y - 1,y^2 - x"
    corpus_path = tmp_path / "corpus.jsonl"
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
    generate_corpus.main(corpus_args)
    corpus = [json.loads(line) for line in corpus_path.read_text().splitlines()]
    assert len(corpus) == 1
    assert corpus[0]["episode_seed"] == 123
    assert [entry.replace(" ", "") for entry in corpus[0]["basis_elements"]] == [
        entry.replace(" ", "") for entry in basis.split(",")
    ]
    with pytest.raises(FileExistsError):
        generate_corpus.main(corpus_args)

    result = run_training.main(
        [
            "--env.coeff-ring=QQ",
            "--env.symbols=x,y",
            f"--env.basis-elements={basis}",
            "--env.elimination=none",
            "--training.num-envs=1",
            "--training.num-steps-per-update=4",
            "--training.total-timesteps=4",
            "--training.num-epochs-per-update=1",
            "--training.num-minibatches=2",
            "--training.hparam-eval-num-examples=0",
            "--training.cuda=False",
            "--training.track=False",
            "--training.pytorch-num-threads=1",
            "--training.system-metrics-enabled=False",
            "--training.console-verbosity=quiet",
            f"--training.output-directory={tmp_path}",
        ]
    )
    assert result.summary["global_step"] == 4
    actor_path = Path(result.checkpoint_paths["actor"])
    assert actor_path.is_file()

    for gym_id in ("BuchbergerExponents-v1", "BuchbergerExponentsNoPQ-v1"):
        output = tmp_path / gym_id
        evaluate_model.main(
            [
                f"--corpus={corpus_path}",
                f"--model-name={actor_path}",
                f"--gym-id={gym_id}",
                "--device=cpu",
                "--verbosity=quiet",
                f"--output-dir={output}",
            ]
        )
        summary = json.loads((output / "summary.json").read_text())
        episodes = [
            json.loads(line)
            for line in (output / "episode_results.jsonl").read_text().splitlines()
        ]
        assert summary["num_evals"] == len(episodes) == 1
        assert summary["length_min"] > 0
        assert summary["return_mean"] == episodes[0]["return"]
        metadata = json.loads((output / "metadata.json").read_text())
        assert metadata["gym_id"] == gym_id
        assert Path(metadata["model_path"]) == actor_path.resolve()
