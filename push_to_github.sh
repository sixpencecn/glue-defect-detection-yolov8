#!/usr/bin/env bash
# ============================================================
# 一键推送脚本 - 将本项目推送到你的 GitHub 新仓库
# ============================================================
# 使用步骤:
#   1. 在 https://github.com/new 手动创建公开空仓库
#      仓库名建议: glue-defect-yolov8  (不要勾选初始化 README)
#   2. 生成 Personal Access Token (PAT):
#      GitHub 右上角头像 -> Settings -> Developer settings ->
#      Personal access tokens -> Tokens (classic) -> Generate new token
#      勾选 repo 权限, 复制生成的 token
#   3. 在本脚本所在目录执行:
#      GITHUB_TOKEN=你的token bash push_to_github.sh
# ============================================================
set -e

REPO_NAME="glue-defect-yolov8"
OWNER="sixpencecn"

if [ -z "$GITHUB_TOKEN" ]; then
  echo "错误: 请先设置环境变量 GITHUB_TOKEN"
  echo "用法: GITHUB_TOKEN=你的token bash push_to_github.sh"
  exit 1
fi

# 远程仓库地址 (token 仅用于本次推送, 不写入配置)
REMOTE="https://${GITHUB_TOKEN}@github.com/${OWNER}/${REPO_NAME}.git"

echo ">>> 推送到 https://github.com/${OWNER}/${REPO_NAME} ..."
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE"
git branch -M main
git push -u origin main

# 推送后移除含 token 的 remote 配置, 避免凭据泄露
git remote remove origin
echo ">>> 推送完成! 仓库地址: https://github.com/${OWNER}/${REPO_NAME}"
echo ">>> 提示: 已移除含 token 的远程配置, 下次推送需重新运行本脚本。"
