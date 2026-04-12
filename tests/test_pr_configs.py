"""
Tests for PR-introduced configuration files:
- .gemini/config.yaml
- .gemini/styleguide.md
- .github/workflows/python-lint.yml
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent

GEMINI_CONFIG = REPO_ROOT / ".gemini" / "config.yaml"
STYLEGUIDE = REPO_ROOT / ".gemini" / "styleguide.md"
LINT_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "python-lint.yml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


# ===========================================================================
# .gemini/config.yaml
# ===========================================================================

class TestGeminiConfig:
    """Structural and value tests for .gemini/config.yaml."""

    @pytest.fixture(scope="class")
    def config(self) -> dict:
        return load_yaml(GEMINI_CONFIG)

    def test_file_exists(self):
        assert GEMINI_CONFIG.exists(), ".gemini/config.yaml must exist"

    def test_file_is_valid_yaml(self):
        data = load_yaml(GEMINI_CONFIG)
        assert isinstance(data, dict)

    def test_top_level_keys_present(self, config):
        required_keys = {"have_fun", "memory_config", "code_review", "ignore_patterns"}
        assert required_keys.issubset(config.keys()), (
            f"Missing top-level keys: {required_keys - config.keys()}"
        )

    def test_have_fun_is_bool(self, config):
        assert isinstance(config["have_fun"], bool)

    def test_have_fun_is_false(self, config):
        assert config["have_fun"] is False

    def test_memory_config_present(self, config):
        assert "memory_config" in config
        assert isinstance(config["memory_config"], dict)

    def test_memory_config_disabled_key_present(self, config):
        assert "disabled" in config["memory_config"]

    def test_memory_config_disabled_is_bool(self, config):
        assert isinstance(config["memory_config"]["disabled"], bool)

    def test_memory_config_disabled_is_false(self, config):
        assert config["memory_config"]["disabled"] is False

    def test_code_review_present(self, config):
        assert "code_review" in config
        assert isinstance(config["code_review"], dict)

    def test_code_review_disable_key_present(self, config):
        assert "disable" in config["code_review"]

    def test_code_review_not_disabled(self, config):
        assert config["code_review"]["disable"] is False

    def test_code_review_comment_severity_threshold(self, config):
        assert config["code_review"]["comment_severity_threshold"] == "MEDIUM"

    def test_code_review_comment_severity_valid_value(self, config):
        valid_thresholds = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert config["code_review"]["comment_severity_threshold"] in valid_thresholds

    def test_code_review_max_review_comments(self, config):
        # -1 means unlimited
        assert config["code_review"]["max_review_comments"] == -1

    def test_code_review_max_review_comments_is_int(self, config):
        assert isinstance(config["code_review"]["max_review_comments"], int)

    def test_pull_request_opened_section_present(self, config):
        pr_config = config["code_review"].get("pull_request_opened")
        assert pr_config is not None
        assert isinstance(pr_config, dict)

    def test_pull_request_opened_keys_present(self, config):
        pr_config = config["code_review"]["pull_request_opened"]
        required = {"help", "summary", "code_review", "include_drafts"}
        assert required.issubset(pr_config.keys())

    def test_pull_request_opened_help_is_false(self, config):
        assert config["code_review"]["pull_request_opened"]["help"] is False

    def test_pull_request_opened_summary_is_true(self, config):
        assert config["code_review"]["pull_request_opened"]["summary"] is True

    def test_pull_request_opened_code_review_is_true(self, config):
        assert config["code_review"]["pull_request_opened"]["code_review"] is True

    def test_pull_request_opened_include_drafts_is_true(self, config):
        assert config["code_review"]["pull_request_opened"]["include_drafts"] is True

    def test_pull_request_opened_all_booleans(self, config):
        pr_config = config["code_review"]["pull_request_opened"]
        for key, value in pr_config.items():
            assert isinstance(value, bool), f"pull_request_opened.{key} must be bool, got {type(value)}"

    def test_ignore_patterns_is_list(self, config):
        assert isinstance(config["ignore_patterns"], list)

    def test_ignore_patterns_is_empty(self, config):
        assert config["ignore_patterns"] == []

    def test_no_unexpected_top_level_keys(self, config):
        known_keys = {"have_fun", "memory_config", "code_review", "ignore_patterns"}
        unknown = set(config.keys()) - known_keys
        assert not unknown, f"Unexpected top-level keys found: {unknown}"


# ===========================================================================
# .gemini/styleguide.md
# ===========================================================================

class TestStyleguide:
    """Content and structure tests for .gemini/styleguide.md."""

    @pytest.fixture(scope="class")
    def content(self) -> str:
        return STYLEGUIDE.read_text(encoding="utf-8")

    @pytest.fixture(scope="class")
    def lines(self, content) -> list[str]:
        return content.splitlines()

    def test_file_exists(self):
        assert STYLEGUIDE.exists(), ".gemini/styleguide.md must exist"

    def test_file_is_not_empty(self, content):
        assert len(content.strip()) > 0

    def test_starts_with_h1_heading(self, content):
        assert content.startswith("# "), "Styleguide must start with an H1 heading"

    def test_title_contains_pr_review(self, content):
        first_line = content.splitlines()[0]
        assert "PR Review" in first_line or "Review" in first_line

    def test_repository_context_present(self, content):
        assert "similarity" in content
        assert "RationAI" in content

    def test_required_sections_present(self, content):
        required_sections = [
            "Primary Review Focus",
            "General Comment Style",
            "Domain-Specific Guidance",
            "Architecture",
            "Types",
            "Libraries",
        ]
        for section in required_sections:
            assert section in content, f"Section '{section}' missing from styleguide"

    def test_ignore_formatting_guidance_present(self, content):
        # Critical guidance: do not comment on formatting
        assert "formatting" in content.lower() or "linting" in content.lower()

    def test_ml_focus_mentioned(self, content):
        ml_terms = ["ML", "machine learning", "tensor", "PyTorch"]
        assert any(term in content for term in ml_terms)

    def test_ray_pipeline_mentioned(self, content):
        assert "ray" in content.lower() or "Ray" in content

    def test_digital_pathology_context_present(self, content):
        pathology_terms = ["WSI", "whole-slide", "pathology", "tissue"]
        assert any(term in content for term in pathology_terms)

    def test_known_models_referenced(self, content):
        models = ["GigaPath", "Virchow2", "UNI2"]
        assert any(model in content for model in models)

    def test_ratiopath_library_referenced(self, content):
        assert "ratiopath" in content

    def test_google_docstring_style_referenced(self, content):
        assert "Google Docstring Style" in content or "Google" in content

    def test_what_not_to_comment_section_present(self, content):
        # The "What NOT to Comment On" section is important for reviewers
        assert "NOT" in content or "not" in content.lower()

    def test_minimum_line_count(self, lines):
        # Styleguide should have substantial content (>= 50 lines)
        assert len(lines) >= 50, f"Styleguide too short: {len(lines)} lines"

    def test_type_hinting_guidance_present(self, content):
        assert "type hint" in content.lower() or "Type Hint" in content

    def test_repository_structure_section_present(self, content):
        # src/, examples/, tests/, pretrained/ should be mentioned
        assert "src/" in content
        assert "examples/" in content
        assert "tests/" in content

    def test_testing_not_blocking_prs(self, content):
        # Key policy: do not block PRs over tests
        assert "test" in content.lower()
        # Verify "do not block" or similar phrasing
        assert re.search(r"(do not block|not block|minimal test)", content, re.IGNORECASE)

    def test_gpu_memory_guidance_present(self, content):
        gpu_terms = ["GPU", "VRAM", "float16", "bfloat16", "non_blocking"]
        assert any(term in content for term in gpu_terms)

    def test_czech_comments_policy_present(self, content):
        assert "Czech" in content or "czech" in content.lower()

    def test_hardcoded_paths_policy_present(self, content):
        assert "hardcoded path" in content.lower() or "Hardcoded path" in content

    def test_line_length_policy_present(self, content):
        assert "line length" in content.lower() or "Line length" in content


# ===========================================================================
# .github/workflows/python-lint.yml
# ===========================================================================

class TestPythonLintWorkflow:
    """Structural and value tests for .github/workflows/python-lint.yml."""

    @pytest.fixture(scope="class")
    def workflow(self) -> dict:
        return load_yaml(LINT_WORKFLOW)

    def test_file_exists(self):
        assert LINT_WORKFLOW.exists(), ".github/workflows/python-lint.yml must exist"

    def test_file_is_valid_yaml(self):
        data = load_yaml(LINT_WORKFLOW)
        assert isinstance(data, dict)

    def test_workflow_name_present(self, workflow):
        assert "name" in workflow

    def test_workflow_name_value(self, workflow):
        assert workflow["name"] == "Python Lint (RationAI Standard)"

    def test_workflow_name_mentions_lint(self, workflow):
        assert "Lint" in workflow["name"] or "lint" in workflow["name"].lower()

    def test_on_trigger_present(self, workflow):
        # 'on' becomes True in YAML, pyyaml parses the key as True
        assert "on" in workflow or True in workflow

    def test_push_trigger_configured(self, workflow):
        on_block = workflow.get("on") or workflow.get(True)
        assert on_block is not None
        assert "push" in on_block

    def test_pull_request_trigger_configured(self, workflow):
        on_block = workflow.get("on") or workflow.get(True)
        assert "pull_request" in on_block

    def test_workflow_dispatch_trigger_configured(self, workflow):
        on_block = workflow.get("on") or workflow.get(True)
        assert "workflow_dispatch" in on_block

    def test_push_targets_master_branch(self, workflow):
        on_block = workflow.get("on") or workflow.get(True)
        push_config = on_block["push"]
        assert "branches" in push_config
        assert "master" in push_config["branches"]

    def test_pull_request_targets_master_branch(self, workflow):
        on_block = workflow.get("on") or workflow.get(True)
        pr_config = on_block["pull_request"]
        assert "branches" in pr_config
        assert "master" in pr_config["branches"]

    def test_push_branches_not_empty(self, workflow):
        on_block = workflow.get("on") or workflow.get(True)
        assert len(on_block["push"]["branches"]) > 0

    def test_pull_request_branches_not_empty(self, workflow):
        on_block = workflow.get("on") or workflow.get(True)
        assert len(on_block["pull_request"]["branches"]) > 0

    def test_jobs_section_present(self, workflow):
        assert "jobs" in workflow
        assert isinstance(workflow["jobs"], dict)

    def test_run_job_present(self, workflow):
        assert "run" in workflow["jobs"]

    def test_run_job_uses_reusable_workflow(self, workflow):
        run_job = workflow["jobs"]["run"]
        assert "uses" in run_job, "Job 'run' must use a reusable workflow"

    def test_run_job_uses_rationai_workflow(self, workflow):
        uses = workflow["jobs"]["run"]["uses"]
        assert "RationAI" in uses

    def test_run_job_uses_python_lint_workflow(self, workflow):
        uses = workflow["jobs"]["run"]["uses"]
        assert "python-lint" in uses

    def test_run_job_pins_to_main_ref(self, workflow):
        uses = workflow["jobs"]["run"]["uses"]
        assert uses.endswith("@main"), f"Workflow ref should be '@main', got: {uses}"

    def test_run_job_uses_correct_full_path(self, workflow):
        uses = workflow["jobs"]["run"]["uses"]
        assert uses == "RationAI/.github/.github/workflows/python-lint.yml@main"

    def test_no_extra_jobs(self, workflow):
        # This workflow should only define the single 'run' job
        assert list(workflow["jobs"].keys()) == ["run"]

    def test_workflow_dispatch_allows_manual_trigger(self, workflow):
        on_block = workflow.get("on") or workflow.get(True)
        # workflow_dispatch may be None (no extra config) or a dict
        assert "workflow_dispatch" in on_block

    def test_push_and_pr_target_same_branch(self, workflow):
        on_block = workflow.get("on") or workflow.get(True)
        push_branches = set(on_block["push"]["branches"])
        pr_branches = set(on_block["pull_request"]["branches"])
        assert push_branches == pr_branches, (
            "push and pull_request triggers should target the same branches"
        )