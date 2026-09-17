#!/usr/bin/env python3
"""每日 AI 资讯早餐 —— 运行即生成今天的简报。"""

from __future__ import annotations

import argparse
import os
import plistlib
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import feedparser
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# ---------- 配置 ----------
RSS_FEEDS = [
    ("少数派", "https://sspai.com/feed"),
    ("36氪", "https://www.36kr.com/feed"),
    ("量子位", "https://www.qbitai.com/feed"),
    ("Solidot", "https://www.solidot.org/index.rss"),
    ("InfoQ", "https://www.infoq.cn/feed"),
]

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
HOURS = 24
MAX_ITEMS = 40  # 送给 LLM 的文章上限，避免 prompt 过长
SERVERCHAN_SENDKEY = os.getenv("SERVERCHAN_SENDKEY", "").strip()
PUSHPLUS_TOKEN = os.getenv("PUSHPLUS_TOKEN", "").strip()
LAUNCH_LABEL = "com.ai-breakfast.daily"
SCHEDULE_HOUR = 8
SCHEDULE_MINUTE = 0


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_entry_time(entry) -> datetime | None:
    """尽量从 RSS entry 解析出带时区的发布时间。"""
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed:
            try:
                return datetime(*parsed[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                pass
    for key in ("published", "updated"):
        raw = entry.get(key)
        if not raw:
            continue
        try:
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (TypeError, ValueError, OverflowError):
            continue
    return None


def fetch_recent_articles() -> list[dict]:
    """抓取各 RSS 源最近 24 小时内的文章。"""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=HOURS)
    articles: list[dict] = []

    for source, url in RSS_FEEDS:
        print(f"📡 抓取：{source} ...", flush=True)
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            print(f"  ⚠️  解析失败：{e}", flush=True)
            continue

        if getattr(feed, "bozo", False) and not feed.entries:
            print(f"  ⚠️  源不可用：{getattr(feed, 'bozo_exception', '')}", flush=True)
            continue

        count = 0
        for entry in feed.entries:
            published = parse_entry_time(entry)
            # 没有时间戳时也收下，避免空源；有时间则过滤 24h
            if published is not None and published < cutoff:
                continue

            title = strip_html(entry.get("title") or "")
            summary = strip_html(entry.get("summary") or entry.get("description") or "")
            if len(summary) > 300:
                summary = summary[:300] + "…"
            link = entry.get("link") or ""
            if not title:
                continue

            articles.append(
                {
                    "source": source,
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "published": published.isoformat() if published else "",
                }
            )
            count += 1

        print(f"  ✅ {count} 篇", flush=True)

    # 按时间倒序，截断
    articles.sort(key=lambda a: a["published"] or "", reverse=True)
    return articles[:MAX_ITEMS]


def build_prompt(articles: list[dict]) -> str:
    lines = []
    for i, a in enumerate(articles, 1):
        lines.append(
            f"{i}. 【{a['source']}】{a['title']}\n"
            f"   摘要：{a['summary'] or '（无）'}\n"
            f"   链接：{a['link']}"
        )
    corpus = "\n".join(lines)
    today = datetime.now().strftime("%Y-%m-%d")

    return f"""你是一位犀利、简洁的科技/AI 资讯主编。今天是 {today}。

下面是最近 24 小时从多个 RSS 源抓到的文章（标题+摘要）。请从中挑出最值得关注的内容，整理成 **恰好 10 条**「资讯早餐」。

要求：
1. 每条包含：标题、一句话摘要、一个犀利观点（观点要有态度，不要空洞套话）
2. 优先 AI / 大模型 / 科技产品相关；不够则补科技/互联网要闻
3. 不要编造源里没有的事实；可合并同类话题
4. 用 Markdown 输出，格式严格如下：

# 🍳 每日 AI 资讯早餐 · {today}

## 1. <标题>
- **摘要**：<一句话>
- **观点**：<犀利一句>
- **来源**：<媒体名>（可选附链接）

## 2. ...
（共 10 条）

---
原文素材：
{corpus}
"""


def call_deepseek(prompt: str) -> str:
    if not DEEPSEEK_API_KEY:
        raise SystemExit(
            "缺少 DEEPSEEK_API_KEY。请复制 .env.example 为 .env 并填入密钥。"
        )

    resp = requests.post(
        DEEPSEEK_URL,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": DEEPSEEK_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "你是科技资讯主编，输出简洁有观点的 Markdown 简报。",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
        },
        timeout=90,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def save_markdown(content: str) -> Path:
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"breakfast_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    path.write_text(content + "\n", encoding="utf-8")
    return path


def briefing_title(content: str) -> str:
    for line in content.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:64]
    return f"每日 AI 资讯早餐 · {datetime.now().strftime('%Y-%m-%d')}"


def wechat_configured() -> bool:
    return bool(SERVERCHAN_SENDKEY or PUSHPLUS_TOKEN)


def push_serverchan(title: str, content: str) -> None:
    url = f"https://sctapi.ftqq.com/{SERVERCHAN_SENDKEY}.send"
    resp = requests.post(
        url,
        data={"title": title, "desp": content},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    code = data.get("code", data.get("errno"))
    if code not in (0, "0"):
        raise RuntimeError(f"Server酱返回异常：{data}")
    print("📲 已通过 Server酱 推送到微信", flush=True)


def push_pushplus(title: str, content: str) -> None:
    resp = requests.post(
        "https://www.pushplus.plus/send",
        json={
            "token": PUSHPLUS_TOKEN,
            "title": title,
            "content": content,
            "template": "markdown",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 200:
        raise RuntimeError(f"PushPlus 返回异常：{data}")
    print("📲 已通过 PushPlus 推送到微信", flush=True)


def push_to_wechat(content: str) -> None:
    """个人微信没有开放直连 API，走 Server酱 / PushPlus 中转。"""
    if not wechat_configured():
        print(
            "⚠️  未配置微信推送，已跳过。\n"
            "   1. 打开 https://sct.ftqq.com/ 微信扫码登录\n"
            "   2. 复制 SendKey，写入 .env 的 SERVERCHAN_SENDKEY=\n"
            "   3. 再运行一次本脚本即可收到测试推送",
            flush=True,
        )
        return

    title = briefing_title(content)
    if SERVERCHAN_SENDKEY:
        push_serverchan(title, content)
        return
    push_pushplus(title, content)


def launch_agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_LABEL}.plist"


def python_executable() -> str:
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def install_schedule() -> None:
    """用 macOS launchd 在每天本地时间 8:00 运行本脚本。"""
    if sys.platform != "darwin":
        raise SystemExit("当前定时任务安装仅支持 macOS。Linux 可用 crontab：0 8 * * *")

    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    dest = launch_agent_path()
    dest.parent.mkdir(parents=True, exist_ok=True)

    plist = {
        "Label": LAUNCH_LABEL,
        "ProgramArguments": [python_executable(), str(ROOT / "daily_breakfast.py")],
        "WorkingDirectory": str(ROOT),
        "StartCalendarInterval": {
            "Hour": SCHEDULE_HOUR,
            "Minute": SCHEDULE_MINUTE,
        },
        "StandardOutPath": str(logs / "breakfast.log"),
        "StandardErrorPath": str(logs / "breakfast.err.log"),
    }
    dest.write_bytes(plistlib.dumps(plist))

    uid = os.getuid()
    domain = f"gui/{uid}"
    target = f"{domain}/{LAUNCH_LABEL}"
    subprocess.run(["launchctl", "bootout", domain, str(dest)], check=False)
    loaded = subprocess.run(
        ["launchctl", "bootstrap", domain, str(dest)],
        capture_output=True,
        text=True,
    )
    if loaded.returncode != 0:
        fallback = subprocess.run(
            ["launchctl", "load", "-w", str(dest)],
            capture_output=True,
            text=True,
        )
        if fallback.returncode != 0:
            raise SystemExit(
                "安装 launchd 任务失败：\n"
                f"{loaded.stderr or loaded.stdout}\n{fallback.stderr or fallback.stdout}"
            )
    subprocess.run(["launchctl", "enable", target], check=False)
    print(
        f"✅ 已安装定时任务：每天 {SCHEDULE_HOUR:02d}:{SCHEDULE_MINUTE:02d} 自动整理并推送\n"
        f"   plist: {dest}\n"
        f"   日志: {logs / 'breakfast.log'}\n"
        "   注意：Mac 在 8:00 需开机且未休眠，否则会错过当次。",
        flush=True,
    )
    if not wechat_configured():
        print("⚠️  尚未配置 SERVERCHAN_SENDKEY，到点会生成简报但不会发到微信。", flush=True)


def uninstall_schedule() -> None:
    dest = launch_agent_path()
    uid = os.getuid()
    subprocess.run(["launchctl", "bootout", f"gui/{uid}", str(dest)], check=False)
    subprocess.run(["launchctl", "unload", str(dest)], check=False)
    if dest.exists():
        dest.unlink()
    print("✅ 已移除每天 8:00 的定时任务。", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="每日 AI 资讯早餐")
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="只生成简报，不推送微信",
    )
    parser.add_argument(
        "--install-schedule",
        action="store_true",
        help="安装 macOS 每天 8:00 定时任务",
    )
    parser.add_argument(
        "--uninstall-schedule",
        action="store_true",
        help="移除每天 8:00 定时任务",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.install_schedule:
        install_schedule()
        return
    if args.uninstall_schedule:
        uninstall_schedule()
        return

    print("🍳 开始制作今日 AI 资讯早餐...\n")
    articles = fetch_recent_articles()
    if not articles:
        print("最近 24 小时没有抓到文章，请检查网络或 RSS 源。")
        sys.exit(1)

    print(f"\n📝 共 {len(articles)} 篇素材，正在调用 DeepSeek 总结...\n")
    briefing = call_deepseek(build_prompt(articles))

    print("=" * 60)
    print(briefing)
    print("=" * 60)

    path = save_markdown(briefing)
    print(f"\n💾 已保存：{path}")

    if not args.no_push:
        print()
        try:
            push_to_wechat(briefing)
        except Exception as e:
            print(f"❌ 微信推送失败：{e}", flush=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
