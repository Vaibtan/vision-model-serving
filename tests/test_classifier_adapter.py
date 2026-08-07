from __future__ import annotations

import hashlib
from contextlib import contextmanager
from io import BytesIO
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import tarfile
from types import ModuleType
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vision_model_serving.classifier import (  # noqa: E402
    ClassifierArtifactIdentity,
    ClassifierInputError,
    ClassifierLoadError,
    ClassifierRuntimeOutput,
    LocalTokenizerIdentity,
    MmbcdClassifierAdapter,
    LocalRobertaTokenizer,
    TokenBatch,
    TokenizerVerificationError,
)
from vision_model_serving.dicom import DicomCanonicalizer  # noqa: E402


class TokenizerStub:
    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def encode(self, prompt: str, *, max_length: int) -> TokenBatch:
        self.calls.append((prompt, max_length))
        return TokenBatch(
            input_ids=np.array([[0, 2]], dtype=np.int64),
            attention_mask=np.array([[1, 1]], dtype=np.int64),
        )


class RobertaTokenizerStub:
    load_calls: list[tuple[str, dict[str, object]]] = []
    encode_calls: list[tuple[list[str], dict[str, object]]] = []

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: object) -> RobertaTokenizerStub:
        cls.load_calls.append((path, kwargs))
        return cls()

    def __call__(self, prompts: list[str], **kwargs: object) -> dict[str, np.ndarray]:
        self.encode_calls.append((prompts, kwargs))
        return {
            "input_ids": np.array([[0, 15248, 14086, 35, 2]], dtype=np.int64),
            "attention_mask": np.ones((1, 5), dtype=np.int64),
        }


class RuntimeStub:
    def __init__(self):
        self.calls: list[tuple[np.ndarray, TokenBatch]] = []

    def execute(self, crops: np.ndarray, tokens: TokenBatch) -> ClassifierRuntimeOutput:
        self.calls.append((crops, tokens))
        return ClassifierRuntimeOutput(
            logits=np.array([[2.0, -1.0]], dtype=np.float32),
            fused_embeddings=np.zeros((1, 768), dtype=np.float32),
            roi_attention=np.full((1, 1, 8), 0.125, dtype=np.float32),
            inference_ms=1.0,
        )


class TensorStub:
    def __init__(self, values: np.ndarray):
        self.values = values

    def to(self, **kwargs: object) -> TensorStub:
        return self

    def long(self) -> TensorStub:
        return self

    def detach(self) -> TensorStub:
        return self

    def cpu(self) -> TensorStub:
        return self

    def contiguous(self) -> TensorStub:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class LoadResultStub:
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()


class ModelStub:
    def __init__(self):
        self.load_calls: list[tuple[dict[str, object], bool]] = []
        self.to_calls: list[dict[str, object]] = []
        self.eval_called = False
        self.inputs: tuple[object, object, object] | None = None

    def load_state_dict(
        self,
        state: dict[str, object],
        *,
        strict: bool,
    ) -> LoadResultStub:
        self.load_calls.append((state, strict))
        return LoadResultStub()

    def to(self, **kwargs: object) -> ModelStub:
        self.to_calls.append(kwargs)
        return self

    def eval(self) -> ModelStub:
        self.eval_called = True
        return self

    def __call__(
        self,
        crops: object,
        input_ids: object,
        attention_mask: object,
    ) -> tuple[TensorStub, TensorStub, TensorStub]:
        self.inputs = (crops, input_ids, attention_mask)
        return (
            TensorStub(np.array([[2.0, -1.0]], dtype=np.float32)),
            TensorStub(np.zeros((1, 768), dtype=np.float32)),
            TensorStub(np.full((1, 1, 8), 0.125, dtype=np.float32)),
        )


class ArtifactStub:
    id = "mmbcd-classifier"
    role = "classifier"
    sha256 = "a" * 64
    repository_revision = "b" * 40
    strict_load_verified = True

    @contextmanager
    def open_checkpoint(self):
        yield BytesIO(b"verified classifier checkpoint")


def torch_stub(checkpoint: object) -> tuple[ModuleType, list[tuple[str, object]]]:
    module = ModuleType("torch")
    events: list[tuple[str, object]] = []
    module.float32 = "float32"
    module.int64 = "int64"
    module.manual_seed = lambda seed: events.append(("manual_seed", seed))
    module.set_float32_matmul_precision = lambda value: events.append(
        ("matmul_precision", value)
    )
    module.use_deterministic_algorithms = lambda value, **kwargs: events.append(
        ("deterministic_algorithms", (value, kwargs))
    )
    module.backends = SimpleNamespace(
        cudnn=SimpleNamespace(benchmark=True, deterministic=False, allow_tf32=True),
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
    )

    def load(stream: object, **kwargs: object) -> object:
        events.append(("load", kwargs))
        return checkpoint

    module.load = load
    module.equal = lambda left, right: left is right
    module.from_numpy = lambda values: TensorStub(np.asarray(values))

    @contextmanager
    def inference_mode():
        events.append(("inference_mode", True))
        yield

    module.inference_mode = inference_mode
    module.cuda = SimpleNamespace(
        synchronize=lambda device: events.append(("synchronize", device)),
        manual_seed_all=lambda seed: events.append(("cuda_manual_seed_all", seed)),
        is_initialized=lambda: False,
    )
    return module, events


def mammogram() -> SimpleNamespace:
    return SimpleNamespace(pixels=np.zeros((1024, 1024), dtype=np.uint8))


def classifier_rois() -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(canonical_xyxy=(0.0, 0.0, 1024.0, 1024.0))
        for _ in range(8)
    )


def tokenizer_identity() -> LocalTokenizerIdentity:
    return LocalTokenizerIdentity(
        id="roberta-base",
        revision="c" * 40,
        file_sha256=(
            ("config.json", "d" * 64),
            ("merges.txt", "e" * 64),
            ("tokenizer.json", "f" * 64),
            ("tokenizer_config.json", "0" * 64),
            ("vocab.json", "1" * 64),
        ),
    )


GOLDEN_DETECTIONS = (
    (0.3297639787197113, 0.5857259035110474, 0.2284022569656372, 0.37220728397369385),
    (0.5776022672653198, 0.7295639514923096, 0.4591044485569, 0.1594323068857193),
    (0.22187182307243347, 0.171395406126976, 0.07801583409309387, 0.12175251543521881),
    (0.7794584035873413, 0.8195061087608337, 0.10132759809494019, 0.14421264827251434),
    (0.8958551287651062, 0.19005624949932098, 0.17639052867889404, 0.15130801498889923),
    (0.6287335157394409, 0.5495498180389404, 0.2821159362792969, 0.15279290080070496),
    (0.5589895844459534, 0.43098220229148865, 0.17606845498085022, 0.14717312157154083),
    (0.3472023904323578, 0.11947508901357651, 0.13118338584899902, 0.06570783257484436),
)


def adapter(
    tokenizer: TokenizerStub | None = None,
    runtime: RuntimeStub | None = None,
) -> MmbcdClassifierAdapter:
    return MmbcdClassifierAdapter(
        runtime=runtime or RuntimeStub(),
        tokenizer=tokenizer or TokenizerStub(),
        artifact=ClassifierArtifactIdentity(
            id="mmbcd-classifier",
            sha256="a" * 64,
            repository_revision="b" * 40,
        ),
        tokenizer_identity=tokenizer_identity(),
    )


class ClassifierPromptTests(unittest.TestCase):
    def test_prompt_is_label_free_with_one_indication_prefix(self) -> None:
        tokenizer = TokenizerStub()
        classifier = adapter(tokenizer)

        empty = classifier.predict(mammogram(), classifier_rois(), "  \n\t ")
        populated = classifier.predict(
            mammogram(),
            classifier_rois(),
            "  screening\n  follow-up  ",
        )

        self.assertEqual(empty.input.prompt, "Indication:")
        self.assertEqual(populated.input.prompt, "Indication: screening follow-up")
        self.assertEqual(
            tokenizer.calls,
            [("Indication:", 90), ("Indication: screening follow-up", 90)],
        )
        self.assertFalse(empty.input.label_information_used)
        self.assertFalse(populated.input.label_information_used)


class OfflineTokenizerTests(unittest.TestCase):
    def test_verified_snapshot_loads_and_encodes_offline(self) -> None:
        tokenizer_root = (
            REPOSITORY_ROOT
            / "artifacts"
            / "roberta-base-tokenizer-e2da8e2f811d1448a5b465c236feacd80ffbac7b"
        )
        if not tokenizer_root.is_dir():
            self.skipTest("checksum-pinned tokenizer snapshot is not installed")
        fake_transformers = ModuleType("transformers")
        fake_transformers.RobertaTokenizer = RobertaTokenizerStub
        RobertaTokenizerStub.load_calls.clear()
        RobertaTokenizerStub.encode_calls.clear()

        with patch.dict(os.environ, {}, clear=False), patch.dict(
            sys.modules,
            {"transformers": fake_transformers},
        ):
            tokenizer = LocalRobertaTokenizer.from_manifest(
                tokenizer_root,
                REPOSITORY_ROOT / "config" / "model-artifacts.json",
            )
            tokens = tokenizer.encode("Indication:", max_length=90)
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")

        self.assertEqual(
            tokenizer.identity.revision,
            "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
        )
        self.assertEqual(
            RobertaTokenizerStub.load_calls,
            [
                (
                    str(tokenizer_root.resolve()),
                    {"local_files_only": True, "trust_remote_code": False},
                )
            ],
        )
        self.assertEqual(
            RobertaTokenizerStub.encode_calls,
            [
                (
                    ["Indication:"],
                    {
                        "padding": True,
                        "truncation": True,
                        "max_length": 90,
                        "return_tensors": "np",
                    },
                )
            ],
        )
        self.assertEqual(tokens.input_ids.tolist(), [[0, 15248, 14086, 35, 2]])

    def test_snapshot_checksum_mismatch_fails_before_transformers_loads(self) -> None:
        tokenizer_root = (
            REPOSITORY_ROOT
            / "artifacts"
            / "roberta-base-tokenizer-e2da8e2f811d1448a5b465c236feacd80ffbac7b"
        )
        if not tokenizer_root.is_dir():
            self.skipTest("checksum-pinned tokenizer snapshot is not installed")
        fake_transformers = ModuleType("transformers")
        fake_transformers.RobertaTokenizer = RobertaTokenizerStub
        RobertaTokenizerStub.load_calls.clear()

        with TemporaryDirectory() as temporary:
            copied_root = Path(temporary)
            for source in tokenizer_root.iterdir():
                if source.is_file():
                    shutil.copy2(source, copied_root / source.name)
            with (copied_root / "config.json").open("ab") as stream:
                stream.write(b"corrupt")
            with patch.dict(sys.modules, {"transformers": fake_transformers}):
                with self.assertRaises(TokenizerVerificationError) as raised:
                    LocalRobertaTokenizer.from_manifest(
                        copied_root,
                        REPOSITORY_ROOT / "config" / "model-artifacts.json",
                    )

        self.assertEqual(raised.exception.code, "classifier_tokenizer_invalid")
        self.assertEqual(RobertaTokenizerStub.load_calls, [])


class ClassifierInputTests(unittest.TestCase):
    def test_exactly_eight_rois_are_required_before_execution(self) -> None:
        tokenizer = TokenizerStub()
        runtime = RuntimeStub()
        classifier = adapter(tokenizer, runtime)

        valid = classifier_rois()
        for rois in (valid[:7], valid + (valid[0],)):
            with self.subTest(count=len(rois)):
                with self.assertRaises(ClassifierInputError) as raised:
                    classifier.predict(mammogram(), rois, "")
                self.assertEqual(raised.exception.code, "classifier_input_invalid")

        self.assertEqual(tokenizer.calls, [])
        self.assertEqual(runtime.calls, [])

    def test_crop_order_and_normalization_reach_runtime_unchanged(self) -> None:
        pixels = np.zeros((10, 80), dtype=np.uint8)
        rois = []
        for index in range(8):
            pixels[:, index * 10 : (index + 1) * 10] = index * 30
            rois.append(
                SimpleNamespace(
                    canonical_xyxy=(
                        float(index * 10),
                        0.0,
                        float((index + 1) * 10),
                        10.0,
                    )
                )
            )
        runtime = RuntimeStub()
        classifier = adapter(runtime=runtime)

        classifier.predict(SimpleNamespace(pixels=pixels), tuple(rois), "")

        crops = runtime.calls[0][0]
        self.assertEqual(crops.shape, (1, 8, 3, 224, 224))
        self.assertEqual(crops.dtype, np.float32)
        self.assertFalse(crops.flags.writeable)
        red_values = crops[0, :, 0, 0, 0]
        expected = np.array(
            [((index * 30) / 255.0 - 0.485) / 0.229 for index in range(8)],
            dtype=np.float32,
        )
        np.testing.assert_allclose(red_values, expected, rtol=0, atol=1e-6)

    def test_degenerate_or_out_of_bounds_rois_fail_before_tokenization(self) -> None:
        invalid_boxes = (
            (-1.0, 0.0, 10.0, 10.0),
            (0.0, 0.0, 0.0, 10.0),
            (0.0, 0.0, 1025.0, 10.0),
            (0.0, float("nan"), 10.0, 10.0),
        )
        for invalid in invalid_boxes:
            rois = list(classifier_rois())
            rois[0] = SimpleNamespace(canonical_xyxy=invalid)
            tokenizer = TokenizerStub()
            with self.subTest(box=invalid):
                with self.assertRaises(ClassifierInputError):
                    adapter(tokenizer=tokenizer).predict(mammogram(), rois, "")
                self.assertEqual(tokenizer.calls, [])


class GoldenClassifierInputTests(unittest.TestCase):
    def test_public_dicom_and_archived_boxes_reproduce_the_l4_crop_tensor(self) -> None:
        series_uid = (
            "1.3.6.1.4.1.9590.100.1.2."
            "100131208110604806117271735422083351547"
        )
        fixture = REPOSITORY_ROOT / "fixtures" / "cbis-ddsm" / series_uid / "1-1.dcm"
        if not fixture.is_file():
            self.skipTest("checksum-pinned CBIS-DDSM fixture is not installed")
        with fixture.open("rb") as stream:
            canonical = DicomCanonicalizer().decode(stream)
        rois = tuple(
            SimpleNamespace(
                canonical_xyxy=(
                    (center_x - width / 2.0) * 1024.0,
                    (center_y - height / 2.0) * 1024.0,
                    (center_x + width / 2.0) * 1024.0,
                    (center_y + height / 2.0) * 1024.0,
                )
            )
            for center_x, center_y, width, height in GOLDEN_DETECTIONS
        )
        runtime = RuntimeStub()

        result = adapter(runtime=runtime).predict(canonical, rois, "")

        observed = hashlib.sha256(
            np.ascontiguousarray(runtime.calls[0][0]).tobytes()
        ).hexdigest()
        self.assertEqual(
            observed,
            "89cda9694e3696f63eb70706a6ae4cc2dd16e9be214f27ba6f106479eac90155",
        )
        self.assertEqual(result.input.crop_tensor_sha256, observed)

    def test_archived_l4_outputs_reproduce_the_classifier_prediction_hash(self) -> None:
        archive = REPOSITORY_ROOT / "vision-model-serving-l4-fp32-20260807.tar.gz"
        if not archive.is_file():
            self.skipTest("checksum-pinned L4 evidence archive is not installed")
        with tarfile.open(archive, "r:gz") as bundle:
            member = bundle.getmember("./bundles/mmbcd-outputs.npz")
            extracted = bundle.extractfile(member)
            if extracted is None:
                self.fail("archive MMBCD output bundle cannot be read")
            with np.load(BytesIO(extracted.read()), allow_pickle=False) as outputs:
                logits = outputs["logits"].copy()
                embeddings = outputs["fused_embeddings"].copy()

        runtime = RuntimeStub()
        runtime.execute = lambda crops, tokens: ClassifierRuntimeOutput(
            logits=logits,
            fused_embeddings=embeddings,
            roi_attention=np.full((1, 1, 8), 0.125, dtype=np.float32),
            inference_ms=91.25580596923828,
        )

        result = adapter(runtime=runtime).predict(mammogram(), classifier_rois(), "")

        self.assertEqual(
            result.prediction_sha256,
            "43ec1c4593c0549510098ea082ea7092c7fd5631c95d8b912ecf31633185899b",
        )
        np.testing.assert_allclose(
            result.logits,
            (3.582984209060669, -4.886208534240723),
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            result.probabilities,
            (0.9997902512550354, 0.0002097902470268309),
            rtol=0,
            atol=0,
        )


class ClassifierResultTests(unittest.TestCase):
    def test_result_exposes_indices_and_labeled_inspection_data(self) -> None:
        result = adapter().predict(mammogram(), classifier_rois(), "")

        self.assertEqual(result.class_indices, (0, 1))
        self.assertEqual(result.logits, (2.0, -1.0))
        np.testing.assert_allclose(
            result.probabilities,
            (0.95257413, 0.04742587),
            rtol=0,
            atol=1e-7,
        )
        self.assertEqual(result.predicted_class_index, 0)
        self.assertFalse(hasattr(result, "class_names"))
        self.assertFalse(hasattr(result, "decision_threshold"))
        self.assertEqual(
            result.attention.kind,
            "model_inspection_not_causal_or_clinical_evidence",
        )
        self.assertEqual(result.attention.roi_weights, (0.125,) * 8)
        self.assertEqual(result.fused_embeddings.shape, (1, 768))
        self.assertFalse(result.fused_embeddings.flags.writeable)
        digest = hashlib.sha256()
        digest.update(np.array([[2.0, -1.0]], dtype="<f4").tobytes())
        digest.update(np.zeros((1, 768), dtype="<f4").tobytes())
        self.assertEqual(result.prediction_sha256, digest.hexdigest())

    def test_repeated_predictions_are_identical_and_auditable(self) -> None:
        classifier = adapter()

        first = classifier.predict(mammogram(), classifier_rois(), "")
        second = classifier.predict(mammogram(), classifier_rois(), "")

        self.assertEqual(first.logits, second.logits)
        self.assertEqual(first.probabilities, second.probabilities)
        self.assertEqual(first.prediction_sha256, second.prediction_sha256)
        self.assertEqual(
            first.input.crop_tensor_sha256,
            second.input.crop_tensor_sha256,
        )
        self.assertEqual(first.input.input_ids_sha256, second.input.input_ids_sha256)
        self.assertEqual(
            first.input.attention_mask_sha256,
            second.input.attention_mask_sha256,
        )
        self.assertEqual(first.provenance.artifact.id, "mmbcd-classifier")
        self.assertEqual(first.provenance.tokenizer.id, "roberta-base")
        self.assertTrue(first.provenance.offline_assets_only)
        self.assertFalse(first.provenance.strict_checkpoint_load)
        self.assertEqual(first.provenance.precision, "float32")
        self.assertEqual(
            first.warnings,
            (
                "class_semantics_and_decision_threshold_unverified",
                "attention_is_inspection_not_causal_or_clinical_evidence",
            ),
        )
        for value in (
            first.timings.load_ms,
            first.timings.crop_preprocess_ms,
            first.timings.tokenization_ms,
            first.timings.inference_ms,
            first.timings.result_ms,
        ):
            self.assertGreaterEqual(value, 0.0)


class ClassifierRuntimeTests(unittest.TestCase):
    def test_strict_loads_verified_artifact_and_executes_privately(self) -> None:
        shared_weight = object()
        checkpoint = {
            "module.img_fc1.weight": shared_weight,
            "module.img_fc_layer.1.weight": shared_weight,
            "module.img_fc1.bias": shared_weight,
            "module.img_fc_layer.1.bias": shared_weight,
            "module.txt_fc1.weight": shared_weight,
            "module.txt_fc_layer.1.weight": shared_weight,
            "module.txt_fc1.bias": shared_weight,
            "module.txt_fc_layer.1.bias": shared_weight,
        }
        model = ModelStub()
        fake_torch, events = torch_stub(checkpoint)

        with patch.dict(sys.modules, {"torch": fake_torch}):
            classifier = MmbcdClassifierAdapter.from_artifact(
                ArtifactStub(),
                tokenizer=TokenizerStub(),
                tokenizer_identity=tokenizer_identity(),
                model_factory=lambda: model,
                device="cuda:0",
            )
            result = classifier.predict(mammogram(), classifier_rois(), "")

        self.assertTrue(model.eval_called)
        self.assertEqual(model.to_calls, [{"device": "cuda:0", "dtype": "float32"}])
        self.assertTrue(model.load_calls[0][1])
        self.assertNotIn("module.img_fc1.weight", model.load_calls[0][0])
        self.assertIn("img_fc1.weight", model.load_calls[0][0])
        self.assertFalse(hasattr(classifier, "model"))
        self.assertEqual(result.artifact.sha256, ArtifactStub.sha256)
        self.assertTrue(result.provenance.strict_checkpoint_load)
        self.assertEqual(
            next(value for name, value in events if name == "load"),
            {"map_location": "cpu", "weights_only": True},
        )
        self.assertEqual(
            [name for name, _ in events].count("synchronize"),
            2,
        )
        self.assertIn("inference_mode", [name for name, _ in events])
        self.assertIn(("manual_seed", 0), events)
        self.assertIn(("cuda_manual_seed_all", 0), events)
        self.assertIn(("matmul_precision", "highest"), events)
        self.assertIn(
            ("deterministic_algorithms", (True, {"warn_only": False})),
            events,
        )

    def test_alias_and_strict_state_mismatches_fail_with_sanitized_errors(self) -> None:
        shared = object()
        valid_state = {
            "module.img_fc1.weight": shared,
            "module.img_fc_layer.1.weight": shared,
            "module.img_fc1.bias": shared,
            "module.img_fc_layer.1.bias": shared,
            "module.txt_fc1.weight": shared,
            "module.txt_fc_layer.1.weight": shared,
            "module.txt_fc1.bias": shared,
            "module.txt_fc_layer.1.bias": shared,
        }

        mismatched_aliases = dict(valid_state)
        mismatched_aliases["module.img_fc_layer.1.weight"] = object()
        fake_torch, _ = torch_stub(mismatched_aliases)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaises(ClassifierLoadError) as alias_error:
                MmbcdClassifierAdapter.from_artifact(
                    ArtifactStub(),
                    tokenizer=TokenizerStub(),
                    tokenizer_identity=tokenizer_identity(),
                    model_factory=ModelStub,
                )
        self.assertEqual(
            alias_error.exception.detail,
            "classifier checkpoint aliases differ",
        )

        model = ModelStub()

        def reject_shape(
            state: dict[str, object],
            *,
            strict: bool,
        ) -> LoadResultStub:
            raise ValueError("C:/patient/private/checkpoint shape mismatch")

        model.load_state_dict = reject_shape
        fake_torch, _ = torch_stub(valid_state)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaises(ClassifierLoadError) as shape_error:
                MmbcdClassifierAdapter.from_artifact(
                    ArtifactStub(),
                    tokenizer=TokenizerStub(),
                    tokenizer_identity=tokenizer_identity(),
                    model_factory=lambda: model,
                )
        self.assertEqual(shape_error.exception.code, "classifier_load_failed")
        self.assertNotIn("patient", str(shape_error.exception))
        self.assertIsNone(shape_error.exception.__cause__)

    def test_non_classifier_or_unverified_artifact_fails_closed(self) -> None:
        for artifact_value in (
            SimpleNamespace(
                id="focalnet-dino-detector",
                role="detector",
                sha256="a" * 64,
                repository_revision="b" * 40,
                strict_load_verified=True,
            ),
            SimpleNamespace(
                id="mmbcd-classifier",
                role="classifier",
                sha256="a" * 64,
                repository_revision="b" * 40,
                strict_load_verified=False,
            ),
        ):
            with self.subTest(artifact=artifact_value.id):
                with self.assertRaises(ClassifierLoadError):
                    MmbcdClassifierAdapter.from_artifact(
                        artifact_value,
                        tokenizer=TokenizerStub(),
                        tokenizer_identity=tokenizer_identity(),
                        model_factory=ModelStub,
                    )


if __name__ == "__main__":
    unittest.main()
