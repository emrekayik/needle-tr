#!/usr/bin/env bash
# ==============================================================================
# Needle-TR Otomatik Sürüm Artırma, Derleme, Git Push ve PyPI Yayınlama Scripti
# Kullanım:
#   ./release.sh           (Varsayılan: patch artırır, örn: 0.1.8 -> 0.1.9)
#   ./release.sh minor     (Minor artırır, örn: 0.1.8 -> 0.2.0)
#   ./release.sh major     (Major artırır, örn: 0.1.8 -> 1.0.0)
#   ./release.sh 0.3.0     (Belirtilen sürüme yükseltir)
# ==============================================================================
set -e

BUMP_TYPE="${1:-patch}"

echo ""
echo "=============================================================="
echo "🚀 NEEDLE-TR OTOMATİK SÜRÜM YAYINLAMA: [$BUMP_TYPE]"
echo "=============================================================="

# 1. .env dosyasından ortam değişkenlerini satır satır güvenle yükle
if [ -f ".env" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        # Yorum ve boş satırları atla
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue
        if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
            key="${BASH_REMATCH[1]}"
            val="${BASH_REMATCH[2]}"
            # Tırnak işaretlerini temizle
            val="${val%\"}"
            val="${val#\"}"
            val="${val%\'}"
            val="${val#\'}"
            export "$key"="$val"
        fi
    done < .env
fi

# PyPI token'ını olası tüm isimlerden kontrol et
PUBLISH_TOKEN="${UV_PUBLISH_TOKEN:-${PYPI_API_TOKEN:-${PYPI_TOKEN:-${PYPI_API_TOKKEN}}}}"
if [ -n "$PUBLISH_TOKEN" ]; then
    export UV_PUBLISH_TOKEN="$PUBLISH_TOKEN"
fi

# 2. Çalışma dizinindeki bekleyen değişiklikleri kontrol et ve commit'e hazırla
if [ -n "$(git status --porcelain)" ]; then
    echo "📝 [1/5] Bekleyen değişiklikler git'e ekleniyor..."
    git add .
    git commit -m "chore: pre-release preparations" || true
fi

# 3. Sürümü artır, commit et, git tag oluştur ve GitHub'a push et
echo ""
echo "📦 [2/5] Sürüm artırılıyor, etiketleniyor ve GitHub'a push ediliyor..."
uv run needle-tr release "$BUMP_TYPE" --push

# 4. Eski derleme artıklarını temizle
echo ""
echo "🧹 [3/5] dist/ klasörü temizleniyor..."
rm -rf dist/*

# 5. Yeni sürüm paketlerini derle
echo ""
echo "🔨 [4/5] Yeni paketler derleniyor (uv build)..."
uv build

# 6. PyPI'ye yayınla
echo ""
echo "📤 [5/5] PyPI'ye yayınlanıyor..."
if [ -n "$UV_PUBLISH_TOKEN" ]; then
    echo "✓ PyPI token .env dosyasından başarıyla yüklendi."
    uv publish --token "$UV_PUBLISH_TOKEN"
else
    echo "⚠️  .env dosyasında veya ortam değişkenlerinde PyPI token bulunamadı."
    read -p "🔑 Lütfen PyPI API token'ınızı girin (pypi-...): " USER_TOKEN
    if [ -n "$USER_TOKEN" ]; then
        uv publish --token "$USER_TOKEN"
        if [ ! -f ".env" ] || ! grep -q "UV_PUBLISH_TOKEN" .env; then
            echo "" >> .env
            echo "UV_PUBLISH_TOKEN=\"$USER_TOKEN\"" >> .env
            echo "✓ Token sonraki kullanımlar için .env dosyasına kaydedildi (.gitignore korumalı)."
        fi
    else
        echo "❌ Token girilmedi, PyPI yüklemesi atlandı (GitHub Actions üzerinden yayınlanabilir)."
    fi
fi

echo ""
echo "=============================================================="
echo "✨ BAŞARILI! Yeni sürüm hem GitHub'da hem de PyPI'de yayında!"
echo "=============================================================="
echo ""
