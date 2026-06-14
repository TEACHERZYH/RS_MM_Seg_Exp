from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download password-protected Seafile shared file")
    parser.add_argument("--share-url", type=str, required=True)
    parser.add_argument("--password", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--retry-delay", type=int, default=5)
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


def login_and_resolve_download_url(share_url: str, password: str) -> str:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    first = session.get(share_url, timeout=30)
    first.raise_for_status()

    csrf = session.cookies.get("sfcsrftoken", "")
    data = {"password": password}
    if csrf:
        data["csrfmiddlewaretoken"] = csrf

    post = session.post(
        share_url,
        data=data,
        headers={"Referer": share_url},
        timeout=30,
        allow_redirects=True,
    )
    post.raise_for_status()

    if "请输入正确的密码" in post.text or "Please enter the password" in post.text:
        raise RuntimeError("Password rejected by Seafile share page.")

    dl = session.get(
        f"{share_url}?dl=1",
        headers={"Referer": share_url},
        timeout=30,
        allow_redirects=False,
    )
    if "Location" in dl.headers:
        return requests.compat.urljoin(share_url, dl.headers["Location"])

    match = re.search(r'https://[^"\']+/seafhttp/files/[^"\']+', dl.text)
    if match:
        return match.group(0)

    raise RuntimeError("Failed to resolve direct seafhttp download URL.")


def download_with_curl(url: str, output: Path, resume: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["curl.exe", "-L", "--fail"]
    if resume:
        command.extend(["-C", "-"])
    command.extend(["--output", str(output), url])
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    direct_url = login_and_resolve_download_url(args.share_url, args.password)
    print(f"Resolved direct URL: {direct_url}")
    if args.resolve_only:
        return
    command = ["curl.exe", "-L", "--fail", "--retry", str(args.retries), "--retry-delay", str(args.retry_delay), "--retry-all-errors"]
    if args.resume:
        command.extend(["-C", "-"])
    command.extend(["--output", str(output), direct_url])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
