"""Needle-TR Sürüm ve Etiketleme (Release) Yöneticisi.

Bu modül:
1. pyproject.toml dosyasındaki sürümü otomatik olarak artırır (patch, minor, major veya özel sürüm).
2. Değişiklikleri git ile commit eder.
3. Otomatik olarak 'vX.Y.Z' biçiminde git tag oluşturur.
4. İsteğe bağlı olarak doğrudan GitHub'a push ederek GitHub Actions PyPI dağıtımını tetikler.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


def parse_semver(version_str: str) -> Tuple[int, int, int]:
    """Semver sürüm dizesini (X.Y.Z) ayrıştırır."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version_str.strip())
    if not match:
        raise ValueError(f"Geçersiz semver formatı: '{version_str}'. Beklenen: X.Y.Z (örn: 0.1.1)")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def calculate_next_version(current: str, bump_type: str) -> str:
    """Verilen bump türüne (patch, minor, major) veya özel sürüme göre yeni sürümü belirler."""
    bump = bump_type.strip().lower()
    major, minor, patch = parse_semver(current)

    if bump == "patch":
        return f"{major}.{minor}.{patch + 1}"
    elif bump == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump == "major":
        return f"{major + 1}.0.0"
    elif re.match(r"^\d+\.\d+\.\d+", bump_type):
        return bump_type.strip()
    else:
        raise ValueError(
            f"Bilinmeyen sürüm artırma türü: '{bump_type}'. "
            "Seçenekler: 'patch', 'minor', 'major' veya doğrudan '0.2.0' gibi bir sürüm numarası."
        )


def get_current_version(pyproject_path: Path = Path("pyproject.toml")) -> str:
    """pyproject.toml dosyasından mevcut sürümü okur."""
    if not pyproject_path.exists():
        raise FileNotFoundError(f"{pyproject_path} bulunamadı.")
    content = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not match:
        raise ValueError("pyproject.toml içinde 'version' alanı bulunamadı.")
    return match.group(1)


def update_pyproject_version(new_version: str, pyproject_path: Path = Path("pyproject.toml")) -> None:
    """pyproject.toml dosyasındaki sürümü yeni sürümle değiştirir."""
    content = pyproject_path.read_text(encoding="utf-8")
    updated = re.sub(r'version\s*=\s*"[^"]+"', f'version = "{new_version}"', content, count=1)
    pyproject_path.write_text(updated, encoding="utf-8")


def run_git_command(args: list[str]) -> str:
    """Git komutunu çalıştırır ve çıktısını döner; hata durumunda exception fırlatır."""
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Git komutu başarısız oldu ({' '.join(args)}):\n{res.stderr.strip()}")
    return res.stdout.strip()


def load_env_tokens(env_path: Path = Path(".env")) -> dict[str, str]:
    """Loads environment variables from .env file if it exists."""
    tokens = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                tokens[k.strip()] = v.strip("\"' ")
    return tokens


def get_pypi_token() -> Optional[str]:
    """Returns PyPI publishing token from env or .env file."""
    for key in ("UV_PUBLISH_TOKEN", "PYPI_API_TOKEN", "PYPI_TOKEN", "PYPI_API_TOKKEN"):
        if key in sys.modules.get("os", __import__("os")).environ:
            return sys.modules.get("os", __import__("os")).environ[key]
    env_tokens = load_env_tokens()
    for key in ("UV_PUBLISH_TOKEN", "PYPI_API_TOKEN", "PYPI_TOKEN", "PYPI_API_TOKKEN"):
        if key in env_tokens:
            return env_tokens[key]
    return None


def run_release(
    bump_type: str = "patch",
    push: bool = False,
    publish: bool = False,
    custom_message: Optional[str] = None,
    pyproject_path: Path = Path("pyproject.toml"),
) -> str:
    """Sürüm yükseltme, commit, git tag ve opsiyonel PyPI dağıtımını yürütür."""
    # 1. Mevcut sürümü al ve yenisini hesapla
    current_version = get_current_version(pyproject_path)
    new_version = calculate_next_version(current_version, bump_type)
    tag_name = f"v{new_version}"

    print(f"\n[1/4] 📦 Sürüm yükseltiliyor: {current_version} -> {new_version}...")
    update_pyproject_version(new_version, pyproject_path)
    print(f"      {pyproject_path} güncellendi.")

    # 2. Git kontrol ve commit
    commit_msg = custom_message or f"chore: release {tag_name}"
    print(f"[2/4] 📝 Git commit oluşturuluyor: '{commit_msg}'...")
    run_git_command(["git", "add", str(pyproject_path)])
    # Eğer README veya svg varsa onları da stage'e ekle
    if Path("README.md").exists():
        run_git_command(["git", "add", "README.md"])
    if Path("benchmark_results.svg").exists():
        run_git_command(["git", "add", "benchmark_results.svg"])

    run_git_command(["git", "commit", "-m", commit_msg])

    # 3. Git tag oluştur
    print(f"[3/4] 🏷️ Git tag oluşturuluyor: {tag_name}...")
    run_git_command(["git", "tag", tag_name])
    print(f"      Tag {tag_name} başarıyla oluşturuldu.")

    # 4. İsteğe bağlı push
    if push:
        print(f"[4/4] 🚀 Değişiklikler ve etiket GitHub'a gönderiliyor...")
        branch = run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        run_git_command(["git", "push", "origin", branch, "--tags"])
        print(f"      ✨ GitHub'a push tamamlandı! GitHub Actions yayını tetiklendi.")
    else:
        print(f"[4/4] ℹ️ Değişiklikleri GitHub'a göndermek için şunu çalıştırın:")
        print(f"      git push origin main --tags\n")

    # 5. İsteğe bağlı doğrudan PyPI yayını (.env'den token alarak)
    if publish:
        print("\n[PyPI] Paket derleniyor ve PyPI'ye yayınlanıyor...")
        import shutil
        dist_dir = Path("dist")
        if dist_dir.exists():
            shutil.rmtree(dist_dir)
        subprocess.run(["uv", "build"], check=True)

        token = get_pypi_token()
        if not token:
            print("⚠️  .env dosyasında veya ortamda PyPI token (UV_PUBLISH_TOKEN / PYPI_API_TOKEN) bulunamadı.")
            token = input("🔑 Lütfen PyPI API token'ınızı girin: ").strip()

        if token:
            subprocess.run(["uv", "publish", "--token", token], check=True)
            print("✨ PyPI yayını başarıyla tamamlandı!")
        else:
            print("❌ Token girilmediği için PyPI yayını atlandı.")

    return new_version
