# GitHub 上传指南

## 📋 准备工作

### 1. 检查项目状态

在开始之前，请确认：
- ✅ 已创建 `.gitignore` 文件（已自动创建）
- ✅ 已检查敏感信息（API密钥、密码等）
- ✅ 已确认要上传的文件

### 2. 需要排除的内容（已在.gitignore中配置）

以下内容**不会**上传到GitHub：
- `__pycache__/` - Python缓存文件
- `results/` - 仿真结果文件（可能很大）
- `logs/` - 日志文件
- `data/cles/*.parquet` - 数据文件（如果包含敏感信息）
- `校准用数据源/` - 原始数据文件
- `archive/` - 历史代码存档
- `*.code-workspace` - IDE配置文件

---

## 🚀 上传步骤

### 方法一：使用Git命令行（推荐）

#### Step 1: 初始化Git仓库（如果还没有）

```bash
# 在项目根目录下执行
cd "C:\Users\lenovo\Desktop\一些任务\浙江大学\鄢老师\多智能体\从0到1"
git init
```

#### Step 2: 添加文件到暂存区

```bash
# 添加所有文件（.gitignore会自动排除不需要的文件）
git add .

# 或者只添加特定文件
git add README.md
git add config/
git add agents/
git add core/
git add modules/
git add docs/
```

#### Step 3: 提交更改

```bash
# 创建第一次提交
git commit -m "Initial commit: Multi-agent agricultural policy model"
```

#### Step 4: 在GitHub上创建仓库

1. 登录 [GitHub](https://github.com)
2. 点击右上角 **"+"** → **"New repository"**
3. 填写仓库信息：
   - **Repository name**: `agricultural-policy-abm` (或您喜欢的名称)
   - **Description**: "Multi-agent model for agricultural policy evaluation"
   - **Visibility**: 选择 **Public** 或 **Private**
   - ⚠️ **不要**勾选 "Initialize this repository with a README"（因为本地已有）
4. 点击 **"Create repository"**

#### Step 5: 连接本地仓库到GitHub

```bash
# 添加远程仓库（将 YOUR_USERNAME 和 REPO_NAME 替换为实际值）
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 例如：
# git remote add origin https://github.com/yourusername/agricultural-policy-abm.git
```

#### Step 6: 推送代码到GitHub

```bash
# 推送主分支
git branch -M main
git push -u origin main
```

如果遇到认证问题，可能需要：
- 使用 **Personal Access Token**（推荐）
- 或配置 **SSH密钥**

---

### 方法二：使用GitHub Desktop（图形界面，更简单）

#### Step 1: 下载并安装GitHub Desktop

从 [GitHub Desktop官网](https://desktop.github.com/) 下载安装

#### Step 2: 登录GitHub账户

在GitHub Desktop中登录您的GitHub账户

#### Step 3: 添加本地仓库

1. 点击 **"File"** → **"Add Local Repository"**
2. 选择项目目录：`C:\Users\lenovo\Desktop\一些任务\浙江大学\鄢老师\多智能体\从0到1`
3. 如果提示"这不是一个Git仓库"，点击 **"Create a Repository"**

#### Step 4: 提交更改

1. 在左侧查看要提交的文件
2. 在底部填写提交信息：`Initial commit: Multi-agent agricultural policy model`
3. 点击 **"Commit to main"**

#### Step 5: 发布到GitHub

1. 点击 **"Publish repository"**
2. 填写仓库名称和描述
3. 选择 **Public** 或 **Private**
4. 点击 **"Publish Repository"**

---

## 🔐 认证设置

### 使用Personal Access Token（推荐）

如果使用HTTPS推送，需要Personal Access Token：

1. 登录GitHub → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
2. 点击 **"Generate new token"**
3. 选择权限：至少勾选 `repo`
4. 生成后**复制token**（只显示一次！）
5. 推送时使用token作为密码：
   ```bash
   Username: your_github_username
   Password: your_personal_access_token
   ```

### 使用SSH密钥（更安全）

```bash
# 1. 生成SSH密钥（如果还没有）
ssh-keygen -t ed25519 -C "your_email@example.com"

# 2. 复制公钥
cat ~/.ssh/id_ed25519.pub

# 3. 在GitHub上添加SSH密钥
# Settings → SSH and GPG keys → New SSH key

# 4. 使用SSH URL添加远程仓库
git remote set-url origin git@github.com:YOUR_USERNAME/REPO_NAME.git
```

---

## 📝 后续更新

### 日常更新流程

```bash
# 1. 查看更改状态
git status

# 2. 添加更改的文件
git add .

# 3. 提交更改
git commit -m "描述您的更改"

# 4. 推送到GitHub
git push
```

---

## ⚠️ 注意事项

### 1. 文件大小限制

- GitHub单个文件限制：**100MB**
- 仓库总大小建议：**<1GB**
- 如果`results/`文件夹很大，确保已在`.gitignore`中排除

### 2. 敏感信息检查

上传前请确认以下内容**不包含**敏感信息：
- ❌ API密钥
- ❌ 密码
- ❌ 个人身份信息
- ❌ 商业机密数据

### 3. 许可证

建议添加许可证文件（如MIT License）：
```bash
# 在项目根目录创建LICENSE文件
# 或从GitHub创建仓库时选择许可证
```

### 4. 中文路径问题

如果遇到中文路径问题，可以：
- 使用Git Bash（Windows）
- 或配置Git支持中文：
  ```bash
  git config --global core.quotepath false
  ```

---

## 🎯 推荐的仓库结构

上传后，您的GitHub仓库应该包含：

```
agricultural-policy-abm/
├── README.md                    # 项目说明（已有）
├── .gitignore                   # Git忽略文件（已创建）
├── config/                      # 配置文件
│   ├── master_config.yaml
│   └── ...
├── agents/                      # 智能体代码
│   ├── farmer_agent.py
│   ├── enterprise_agent.py
│   └── ...
├── core/                        # 核心模型
│   ├── model.py
│   └── ...
├── modules/                     # 功能模块
├── docs/                        # 文档
│   ├── GITHUB_UPLOAD_GUIDE.md  # 本文件
│   └── ...
├── analysis/                    # 分析脚本
├── examples/                    # 示例代码
└── requirements.txt            # 依赖包（建议创建）
```

---

## 📦 创建requirements.txt（可选但推荐）

```bash
# 在项目根目录创建requirements.txt
pip freeze > requirements.txt
```

或手动创建，包含主要依赖：
```
mesa>=2.0.0
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.3.0
scipy>=1.7.0
pyyaml>=5.4.0
```

---

## ✅ 检查清单

上传前请确认：

- [ ] `.gitignore` 已创建并正确配置
- [ ] 已检查敏感信息
- [ ] `README.md` 已更新（可选）
- [ ] 已创建 `requirements.txt`（可选）
- [ ] 已测试代码可以运行
- [ ] 已准备好提交信息

---

## 🆘 常见问题

### Q1: 推送时提示"Authentication failed"
**A**: 使用Personal Access Token替代密码，或配置SSH密钥

### Q2: 文件太大无法推送
**A**: 检查`.gitignore`是否正确排除了大文件，或使用Git LFS

### Q3: 中文文件名显示乱码
**A**: 配置 `git config --global core.quotepath false`

### Q4: 想删除已上传的敏感文件
**A**: 需要从Git历史中完全删除（使用`git filter-branch`或BFG Repo-Cleaner）

---

## 📚 参考资源

- [Git官方文档](https://git-scm.com/doc)
- [GitHub官方指南](https://docs.github.com/)
- [GitHub Desktop使用指南](https://docs.github.com/en/desktop)

---

**祝您上传顺利！** 🎉

