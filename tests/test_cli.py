from yam_yabasha.cli import _default_project_root, _parser, _should_redraw_hint


def test_project_root_is_discovered_from_parent_folder(monkeypatch, tmp_path):
    project = tmp_path / "yam-yabasha"
    (project / "data").mkdir(parents=True)
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert _default_project_root() == project.resolve()


def test_workflow_defaults_to_both_camera_locations():
    args = _parser().parse_args(["workflow"])

    assert args.locations == ["entrance", "porch"]
    assert str(args.input_root) == "Test"
    assert args.redraw_hints is False
    assert args.reuse_hints is False


def test_missing_hint_is_always_annotated(tmp_path):
    hint_path = tmp_path / "missing.json"

    assert _should_redraw_hint(
        hint_path,
        redraw_all=False,
        reuse_all=True,
    )


def test_explicit_hint_modes_do_not_prompt(tmp_path):
    hint_path = tmp_path / "hint.json"
    hint_path.write_text("{}", encoding="utf-8")

    assert _should_redraw_hint(
        hint_path,
        redraw_all=True,
        reuse_all=False,
    )
    assert not _should_redraw_hint(
        hint_path,
        redraw_all=False,
        reuse_all=True,
    )


def test_interactive_hint_choice(monkeypatch, tmp_path):
    hint_path = tmp_path / "hint.json"
    hint_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")

    assert _should_redraw_hint(
        hint_path,
        redraw_all=False,
        reuse_all=False,
    )
