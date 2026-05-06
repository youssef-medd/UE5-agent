import pytest
from sandbox.code_validator import CodeValidator
from sandbox.dry_run import DryRunner
from orchestrator.task_parser import parse_task


class TestCodeValidator:
    def setup_method(self):
        self.v = CodeValidator()

    def test_valid_ue5_script(self):
        code = "import unreal\nunreal.log('hello')"
        result = self.v.validate(code)
        assert result.valid

    def test_rejects_eval(self):
        code = "eval('import os')"
        result = self.v.validate(code)
        assert not result.valid
        assert any("eval" in e for e in result.errors)

    def test_rejects_banned_import(self):
        code = "import subprocess\nsubprocess.run('rm -rf /')"
        result = self.v.validate(code)
        assert not result.valid

    def test_rejects_syntax_error(self):
        code = "def broken(:"
        result = self.v.validate(code)
        assert not result.valid
        assert any("Syntax" in e for e in result.errors)

    def test_warns_infinite_loop(self):
        code = "while True:\n    pass"
        result = self.v.validate(code)
        assert result.valid
        assert result.warnings


class TestDryRunner:
    def test_dry_run_valid_code(self):
        runner = DryRunner()
        report = runner.run("import unreal\nunreal.log('test')")
        assert report.valid
        assert "DRY RUN" in report.simulated_output

    def test_dry_run_invalid_code(self):
        runner = DryRunner()
        report = runner.run("eval('bad')")
        assert not report.valid


class TestTaskParser:
    def test_spawn_intent(self):
        task = parse_task("spawn a point light above the player")
        assert task.intent == "spawn_actor"
        assert "spawn" in task.flags

    def test_delete_intent(self):
        task = parse_task("delete all actors named TestCube")
        assert task.intent == "delete_actor"
        assert "destructive" in task.flags

    def test_dry_run_flag(self):
        task = parse_task("simulate adding 10 crates")
        assert "dry_run" in task.flags
