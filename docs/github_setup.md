# GitHub Setup

対象repo案:

```text
OzasaHiro/thermal-geopt
```

## Current State

確認済み:

- `git version 2.43.0`
- `git-lfs/3.4.1`
- `gh version 2.92.0`
- `gh auth status`: not logged in

Git global identity is not configured yet:

- `git config --global --get user.name`: empty
- `git config --global --get user.email`: empty

## One-Time Local Setup

Gitコミット前に、ローカル端末でユーザー情報を設定する。

```bash
git config --global user.name "OzasaHiro"
git config --global user.email "YOUR_EMAIL_OR_GITHUB_NOREPLY_EMAIL"
```

GitHub noreplyを使う場合の形式:

```text
<github-user-id>+OzasaHiro@users.noreply.github.com
```

`<github-user-id>` は GitHub の email settings で確認する。

## gh Authentication

この端末で実行する。

```bash
gh auth login
```

推奨選択:

- GitHub.com
- HTTPSまたはSSH。迷う場合はHTTPS。
- Authenticate Git with your GitHub credentials: Yes
- Browser loginまたはdevice code login

確認:

```bash
gh auth status
```

## Create Private Repository

`Thermal_GeoPT` ディレクトリで実行する。

```bash
gh repo create OzasaHiro/thermal-geopt \
  --private \
  --source=. \
  --remote=origin \
  --push
```

すでにrepoだけ作成済みの場合:

```bash
git remote add origin git@github.com:OzasaHiro/thermal-geopt.git
git branch -M main
git push -u origin main
```

HTTPSを使う場合:

```bash
git remote add origin https://github.com/OzasaHiro/thermal-geopt.git
git branch -M main
git push -u origin main
```

## Git LFS Policy

通常は大容量成果物をGitに入れない。どうしてもcheckpointを共有する必要が出た場合のみ、明示的にLFS対象へ入れる。

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
```

ただし初期repoでは `.pt`、`.npy`、`.npz`、`.vtp`、`.vtu`、`.stl`、Zarr shardは `.gitignore` で除外済み。
