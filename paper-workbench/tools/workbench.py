#!/usr/bin/env python3
"""Small, dependency-free thesis workbench for the DiffSHEG repository."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "paper-workbench" / "config.json"
MARKER_RE = re.compile(r"\b(?:TODO|TBD|FIXME)\b|待补|待完善|待更新|待核验|占位", re.IGNORECASE)
OPEN_TASK_RE = re.compile(r"^\s*[-*]\s+\[ \]\s+", re.MULTILINE)
DONE_TASK_RE = re.compile(r"^\s*[-*]\s+\[[xX]\]\s+", re.MULTILINE)
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
CHAPTER_RE = re.compile(
    r"^##\s+(摘要与关键词|第[一二三四五六七八九十0-9]+章(?:\s+.*)?|参考文献|致谢|附录(?:\s+.*)?)\s*$",
    re.MULTILINE,
)

if sys.platform == "win32":
    # Keep Chinese status output readable when the script is launched by VS Code,
    # PowerShell, or an automated terminal capture.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


@dataclass
class Chapter:
    title: str
    units: int
    markers: int


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_text(relative_path: str) -> str:
    path = ROOT / relative_path
    return path.read_text(encoding="utf-8") if path.exists() else ""


def text_units(text: str) -> int:
    """Approximate length as CJK characters plus Latin words."""
    cjk = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
    latin_words = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text))
    return cjk + latin_words


def extract_chapters(text: str) -> list[Chapter]:
    matches = list(CHAPTER_RE.finditer(text))
    chapters: list[Chapter] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end]
        chapters.append(
            Chapter(
                title=match.group(1).strip(),
                units=text_units(body),
                markers=len(MARKER_RE.findall(body)),
            )
        )
    return chapters


def figure_summary(relative_dir: str) -> tuple[int, int, list[str]]:
    directory = ROOT / relative_dir
    files = [path for path in directory.iterdir() if path.is_file()] if directory.exists() else []
    accepted = [path for path in files if path.suffix.lower() in {".png", ".pdf", ".jpg", ".jpeg", ".svg"}]
    stems = sorted({path.stem for path in accepted})
    return len(stems), len(accepted), stems


def bibliography_count(relative_path: str) -> int:
    return len(re.findall(r"(?m)^\s*@\w+\s*\{", read_text(relative_path)))


def task_counts(relative_path: str) -> tuple[int, int]:
    text = read_text(relative_path)
    return len(OPEN_TASK_RE.findall(text)), len(DONE_TASK_RE.findall(text))


def markdown_files(patterns: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    return sorted(paths)


def broken_links(paths: Iterable[Path]) -> list[str]:
    problems: list[str] = []
    for source in paths:
        text = source.read_text(encoding="utf-8")
        for raw_target in LINK_RE.findall(text):
            target = raw_target.strip().strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:", "data:")):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            candidate = (source.parent / target).resolve()
            if not candidate.exists():
                problems.append(f"{source.relative_to(ROOT)} -> {raw_target}")
    return problems


def chapter_status(chapter: Chapter) -> str:
    if chapter.units < 200:
        return "待展开"
    if chapter.markers:
        return "草稿 / 有待办"
    return "已有草稿"


def collect(config: dict) -> dict:
    manuscript_text = read_text(config["manuscript"])
    chapters = extract_chapters(manuscript_text)
    figure_groups, figure_files, stems = figure_summary(config["figures_dir"])
    open_tasks, done_tasks = task_counts(config["tasks"])
    bib_count = bibliography_count(config["bibliography"])
    expected = config["expected_chapters"]
    found_titles = {chapter.title for chapter in chapters}
    missing = [title for title in expected if title not in found_titles]
    paths = markdown_files(config["scan_markdown"])
    links = broken_links(paths)
    return {
        "manuscript_units": text_units(manuscript_text),
        "manuscript_markers": len(MARKER_RE.findall(manuscript_text)),
        "chapters": chapters,
        "missing_chapters": missing,
        "figure_groups": figure_groups,
        "figure_files": figure_files,
        "figure_stems": stems,
        "bib_count": bib_count,
        "open_tasks": open_tasks,
        "done_tasks": done_tasks,
        "broken_links": links,
        "scanned_files": len(paths),
    }


def dashboard_markdown(config: dict, data: dict) -> str:
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %z")
    rows: list[str] = []
    for chapter in data["chapters"]:
        rows.append(
            f"| {chapter.title} | {chapter.units:,} | {chapter.markers} | {chapter_status(chapter)} |"
        )
    for missing in data["missing_chapters"]:
        rows.append(f"| {missing} | 0 | 0 | 缺失 |")

    warning_lines: list[str] = []
    if data["missing_chapters"]:
        warning_lines.append(f"- 正文尚缺 {len(data['missing_chapters'])} 个预期组成部分。")
    if data["manuscript_markers"]:
        warning_lines.append(f"- 正文中检测到 {data['manuscript_markers']} 个待办/占位标记。")
    if data["bib_count"] < 5:
        warning_lines.append(f"- BibTeX 当前只有 {data['bib_count']} 条记录，尚不足以支撑相关工作章节。")
    if data["broken_links"]:
        warning_lines.append(f"- 扫描到 {len(data['broken_links'])} 个本地断链，请运行 `check` 查看明细。")
    warning_lines.append("- 1-Step 加速同时出现 `33.3x` 与 `1100x` 两种口径，已在主张台账标为冲突。")

    chapter_table = "\n".join(rows) if rows else "| 暂无章节 | 0 | 0 | 缺失 |"
    warnings = "\n".join(warning_lines)
    return f"""# 论文动态总览

> 自动生成：{now}
> 生成命令：`.\\.venv\\Scripts\\python.exe paper-workbench\\tools\\workbench.py update`

## 总体状态

| 项目 | 当前值 |
|---|---:|
| 正文规模（中文字符 + 英文词，近似） | {data['manuscript_units']:,} |
| 已出现的正文组成部分 | {len(data['chapters'])} / {len(config['expected_chapters'])} |
| 图表组 / 图表文件 | {data['figure_groups']} / {data['figure_files']} |
| BibTeX 条目 | {data['bib_count']} |
| 开放任务 / 已完成任务 | {data['open_tasks']} / {data['done_tasks']} |
| 扫描的 Markdown 文件 | {data['scanned_files']} |

## 章节雷达

| 章节 | 规模 | 待办标记 | 状态 |
|---|---:|---:|---|
{chapter_table}

规模只是进度信号，不代表内容质量。章节最终状态以 [章节审校模板](templates/chapter-review.md) 的检查结果为准。

## 当前风险

{warnings}

## 快速入口

- [正文草稿](../{config['manuscript']})
- [任务看板](tasks.md)
- [主张与证据台账](claims.md)
- [实验台账](experiments.md)
- [决策日志](decisions.md) / [详细验证记录](records/README.md)
- [文献台账](literature.md) / [BibTeX](references.bib)
- [图表台账](figures.md) / [图表文件](../{config['figures_dir']})
- [写作路线](../{config['roadmap']})
- [实验结果](../{config['results']})
- [项目进展](../{config['progress']})

## 推荐推进顺序

1. 冻结实验统计口径，先解决 [C-003](claims.md) 的速度基线冲突。
2. 补齐参考文献并把绪论中的外部事实逐句绑定到原始来源。
3. 按“第二章基础 → 第三章方法 → 第四章实验 → 第五章总结”推进正文。
4. 全文完成后再重写摘要与创新点，避免前后数字漂移。
"""


def print_status(config: dict, data: dict) -> None:
    print(f"{config['project_name']}")
    print(f"  manuscript units : {data['manuscript_units']:,}")
    print(f"  sections found   : {len(data['chapters'])}/{len(config['expected_chapters'])}")
    print(f"  figure groups    : {data['figure_groups']} ({data['figure_files']} files)")
    print(f"  bibliography     : {data['bib_count']} entries")
    print(f"  tasks            : {data['open_tasks']} open / {data['done_tasks']} done")
    print(f"  manuscript flags : {data['manuscript_markers']}")
    print(f"  broken links     : {len(data['broken_links'])}")
    for chapter in data["chapters"]:
        print(f"    - {chapter.title}: {chapter.units:,} units, {chapter_status(chapter)}")
    for title in data["missing_chapters"]:
        print(f"    - {title}: missing")


def run_check(config: dict, data: dict, strict: bool) -> int:
    issues: list[str] = []
    issues.extend(f"missing chapter: {title}" for title in data["missing_chapters"])
    if data["manuscript_markers"]:
        issues.append(f"manuscript contains {data['manuscript_markers']} TODO/placeholder markers")
    if data["bib_count"] < 5:
        issues.append(f"bibliography has only {data['bib_count']} entries")
    issues.extend(f"broken link: {item}" for item in data["broken_links"])
    if data["open_tasks"]:
        issues.append(f"task board contains {data['open_tasks']} open tasks")

    if issues:
        print("Paper workspace check found items to resolve:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Paper workspace check passed with no tracked issues.")
    return 1 if strict and issues else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and refresh the DiffSHEG thesis workbench.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("status", help="Print a concise workspace status.")
    subparsers.add_parser("update", help="Regenerate paper-workbench/DASHBOARD.md.")
    check_parser = subparsers.add_parser("check", help="Check chapters, links, references and tracked tasks.")
    check_parser.add_argument("--strict", action="store_true", help="Return exit code 1 when issues exist.")
    args = parser.parse_args()

    config = load_config()
    data = collect(config)
    command = args.command or "status"
    if command == "status":
        print_status(config, data)
        return 0
    if command == "update":
        dashboard = ROOT / config["dashboard"]
        dashboard.write_text(dashboard_markdown(config, data), encoding="utf-8")
        print(f"Updated {dashboard.relative_to(ROOT)}")
        print_status(config, data)
        return 0
    if command == "check":
        return run_check(config, data, args.strict)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
