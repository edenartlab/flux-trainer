"""Tests for the file-based job hand-off (job_io.py).

Run with:  python -m unittest discover -s tests
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import job_io  # noqa: E402


def touch(path, content=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)


class ArgsAndResultTest(unittest.TestCase):
    def test_reads_args(self):
        with tempfile.TemporaryDirectory() as job:
            with open(os.path.join(job, "args.json"), "w") as f:
                json.dump({"task_id": "abc", "args": {"name": "Banny"}}, f)
            self.assertEqual(job_io.read_args(job)["args"], {"name": "Banny"})

    def test_args_without_an_args_object_are_rejected(self):
        with tempfile.TemporaryDirectory() as job:
            with open(os.path.join(job, "args.json"), "w") as f:
                json.dump({"task_id": "abc"}, f)
            with self.assertRaises(ValueError):
                job_io.read_args(job)

    def test_result_is_written_whole(self):
        with tempfile.TemporaryDirectory() as job:
            path = job_io.write_result(job, {"status": "failed", "error": "boom"})
            with open(path) as f:
                self.assertEqual(json.load(f), {"status": "failed", "error": "boom"})
            self.assertEqual(os.listdir(job), ["result.json"])


class ExportOutputsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.results = os.path.join(self.tmp.name, "results", "run")
        self.job = os.path.join(self.tmp.name, "job")
        self.config = {
            "output_dir": self.results,
            "output_name": "run",
            "lora_trigger_text": "a purple banana character",
            "mode": "object",
            "custom_validation_prompts": None,
        }

    def test_exports_lora_recent_checkpoints_and_thumbnail(self):
        touch(os.path.join(self.results, "run.safetensors"), b"final")
        for step in (750, 1500, 2250, 3000):
            touch(os.path.join(self.results, f"run-step{step:08d}.safetensors"))
        thumbnail = os.path.join(self.tmp.name, "EDEN.jpg")
        touch(thumbnail)

        result = job_io.export_outputs(self.job, self.config, thumbnail_path=thumbnail)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["lora"], "output/run.safetensors")
        self.assertEqual(result["thumbnail"], "output/thumbnail.jpg")
        self.assertEqual(result["lora_trigger_text"], "a purple banana character")
        self.assertEqual(result["lora_mode"], "object")
        self.assertNotIn("custom_validation_prompts", result)
        # the three most recent checkpoints, newest first
        self.assertEqual(
            [v["step"] for v in result["checkpoint_versions"]], [3000, 2250, 1500]
        )
        for name in [result["lora"], result["thumbnail"]] + [
            v["file"] for v in result["checkpoint_versions"]
        ]:
            self.assertTrue(os.path.isfile(os.path.join(self.job, name)), name)
        with open(os.path.join(self.job, result["lora"]), "rb") as f:
            self.assertEqual(f.read(), b"final")
        # the manifest must survive a JSON round trip
        json.loads(json.dumps(result, default=str))

    def test_custom_prompts_are_returned(self):
        touch(os.path.join(self.results, "run.safetensors"))
        self.config["custom_validation_prompts"] = ["one", "two"]
        result = job_io.export_outputs(self.job, self.config)
        self.assertEqual(result["custom_validation_prompts"], ["one", "two"])
        self.assertNotIn("thumbnail", result)

    def test_missing_lora_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            job_io.export_outputs(self.job, self.config)


if __name__ == "__main__":
    unittest.main()
