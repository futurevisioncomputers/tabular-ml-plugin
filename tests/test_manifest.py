"""Agents, commands and skills are markdown, so nothing else checks them.

Every failure this file catches is silent at runtime. A command whose YAML
frontmatter does not parse loads with *empty* metadata -- the description and
argument hint vanish and the command still appears to work. An agent naming a
skill that does not exist gets no skill. A skill advertising a script that was
never written sends the model looking for a file that is not there.

The last one is the failure this plugin is least entitled to ship: it publishes
`verify_rulebook.py` specifically to catch documented-vs-implemented drift.

Run: python -m tests.test_manifest   (plain asserts, no pytest dep)
"""

from __future__ import annotations

import json
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

AGENTS_DIR = os.path.join(ROOT, "agents")
COMMANDS_DIR = os.path.join(ROOT, "commands")
SKILLS_DIR = os.path.join(ROOT, "skills")
MANIFEST = os.path.join(ROOT, ".claude-plugin", "plugin.json")

FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def read_frontmatter(path):
    """Parse a file's YAML frontmatter, or raise with the file named.

    yaml.safe_load is used rather than a regex because the bug worth catching
    here -- an unquoted value starting with '[' -- is invisible to a regex and
    only shows up in a real parser.
    """
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    m = FRONTMATTER.match(text)
    assert m, f"{os.path.basename(path)}: no YAML frontmatter block"
    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError as e:
        raise AssertionError(
            f"{os.path.basename(path)}: frontmatter failed to parse -- at "
            f"runtime this loads with EMPTY metadata and every field is "
            f"silently dropped.\n  {e}") from None
    assert isinstance(data, dict), (
        f"{os.path.basename(path)}: frontmatter parsed to {type(data).__name__}, "
        "not a mapping")
    return data, text


def md_files(directory):
    return sorted(f for f in os.listdir(directory) if f.endswith(".md"))


# ---------------------------------------------------------------------------

def test_manifest_is_installable():
    with open(MANIFEST, encoding="utf-8") as fh:
        data = json.load(fh)

    for field in ("name", "version", "description"):
        assert data.get(field), f"plugin.json missing {field}"

    # `name` is the install identifier -- marketplaces and --plugin-dir key on
    # it. A space or capital here breaks resolution while the manifest still
    # looks fine to a human reader.
    assert re.fullmatch(r"[a-z0-9][a-z0-9-]*", data["name"]), (
        f"plugin.json name {data['name']!r} is not a valid slug. This field is "
        "the install identifier, not a display name -- use displayName for "
        "the human-readable title and author for credit.")

    assert re.fullmatch(r"\d+\.\d+\.\d+", data["version"]), (
        f"version {data['version']!r} is not semver")
    print(f"  manifest: {data['name']} v{data['version']}")


def test_every_markdown_frontmatter_parses():
    """The ml-start.md bug: `argument-hint: [a] [b]` is a YAML parse error."""
    checked = 0
    for directory in (AGENTS_DIR, COMMANDS_DIR):
        for name in md_files(directory):
            read_frontmatter(os.path.join(directory, name))
            checked += 1
    for skill in sorted(os.listdir(SKILLS_DIR)):
        path = os.path.join(SKILLS_DIR, skill, "SKILL.md")
        if os.path.exists(path):
            read_frontmatter(path)
            checked += 1
    print(f"  {checked} frontmatter blocks parse")


def test_argument_hints_are_strings():
    """`argument-hint: [model, e.g. ridge]` parses fine -- as a two-item LIST.

    No error anywhere, and the hint the user sees is wrong. Only a type check
    catches this class.
    """
    for name in md_files(COMMANDS_DIR):
        data, _ = read_frontmatter(os.path.join(COMMANDS_DIR, name))
        hint = data.get("argument-hint")
        if hint is None:
            continue
        assert isinstance(hint, str), (
            f"{name}: argument-hint parsed as {type(hint).__name__} -- quote "
            f"the value so YAML reads it as a string. Got: {hint!r}")
    print("  argument-hints are strings")


def test_agents_declare_required_fields():
    for name in md_files(AGENTS_DIR):
        data, body = read_frontmatter(os.path.join(AGENTS_DIR, name))
        stem = name[:-3]
        assert data.get("name") == stem, (
            f"{name}: frontmatter name {data.get('name')!r} does not match the "
            f"filename {stem!r}; the filename is what gets invoked")
        desc = data.get("description", "")
        assert len(desc) > 80, (
            f"{name}: description is {len(desc)} chars. It is the only thing "
            "deciding whether this agent gets invoked -- name the triggers.")
        assert body.strip(), f"{name}: no body after the frontmatter"
    print(f"  {len(md_files(AGENTS_DIR))} agents declare name + description")


def test_agent_skills_exist():
    """An agent listing a skill that does not exist silently gets no skill."""
    available = {d for d in os.listdir(SKILLS_DIR)
                 if os.path.exists(os.path.join(SKILLS_DIR, d, "SKILL.md"))}
    for name in md_files(AGENTS_DIR):
        data, _ = read_frontmatter(os.path.join(AGENTS_DIR, name))
        for skill in data.get("skills") or []:
            assert skill in available, (
                f"{name}: declares skill {skill!r}, which does not exist. "
                f"Available: {sorted(available)}")
    print(f"  agent skill references resolve ({len(available)} skills)")


def test_skill_names_match_directories():
    for skill in sorted(os.listdir(SKILLS_DIR)):
        path = os.path.join(SKILLS_DIR, skill, "SKILL.md")
        if not os.path.exists(path):
            continue
        data, _ = read_frontmatter(path)
        assert data.get("name") == skill, (
            f"{skill}/SKILL.md: frontmatter name {data.get('name')!r} does not "
            f"match its directory {skill!r}")
    print("  skill names match their directories")


SCRIPT_REF = re.compile(r"`?scripts/([A-Za-z0-9_]+\.py)`?")


def test_referenced_scripts_exist():
    """Docs promising a script that was never written is the drift this
    plugin exists to prevent. Skills own their own scripts/ directory."""
    missing = []
    for skill in sorted(os.listdir(SKILLS_DIR)):
        skill_md = os.path.join(SKILLS_DIR, skill, "SKILL.md")
        if not os.path.exists(skill_md):
            continue
        with open(skill_md, encoding="utf-8") as fh:
            text = fh.read()
        for script in set(SCRIPT_REF.findall(text)):
            if not os.path.exists(os.path.join(SKILLS_DIR, skill, "scripts", script)):
                missing.append(f"{skill}/SKILL.md references missing scripts/{script}")
    assert not missing, "documented scripts that do not exist:\n  " + "\n  ".join(missing)
    print("  every script named in a SKILL.md exists")


def test_every_script_is_documented():
    """The reverse drift: a script nobody is told about never gets used."""
    undocumented = []
    for skill in sorted(os.listdir(SKILLS_DIR)):
        scripts_dir = os.path.join(SKILLS_DIR, skill, "scripts")
        skill_md = os.path.join(SKILLS_DIR, skill, "SKILL.md")
        if not (os.path.isdir(scripts_dir) and os.path.exists(skill_md)):
            continue
        with open(skill_md, encoding="utf-8") as fh:
            text = fh.read()
        for script in sorted(os.listdir(scripts_dir)):
            if not script.endswith(".py") or script.startswith("_"):
                continue
            if script not in text:
                undocumented.append(f"{skill}/scripts/{script}")
    assert not undocumented, (
        "scripts not mentioned in their SKILL.md:\n  " + "\n  ".join(undocumented))
    print("  every script is named in its SKILL.md")


def test_commands_reference_real_agents():
    agents = {f[:-3] for f in md_files(AGENTS_DIR)}
    pattern = re.compile(r"`([a-z][a-z0-9-]+)` agent")
    for name in md_files(COMMANDS_DIR):
        with open(os.path.join(COMMANDS_DIR, name), encoding="utf-8") as fh:
            text = fh.read()
        for ref in set(pattern.findall(text)):
            assert ref in agents, (
                f"{name}: names the {ref!r} agent, which does not exist. "
                f"Available: {sorted(agents)}")
    print("  command agent references resolve")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"tests/test_manifest.py -- {len(tests)} checks\n")
    for fn in tests:
        fn()
    print(f"\nOK ({len(tests)} checks)")


if __name__ == "__main__":
    main()
