## 贡献代码

鼓励并感谢为**PaddleViT**项目提供贡献的研究人员和开发人员。

您需要同意[PaddleViT 参与者许可协议](https://cla-assistant.io/BR-IDL/PaddleViT)，方可以参与PaddlePaddle贡献。

本文档描述了我们的工作流程和代码风格。

## 工作流程

PaddleViT 使用这个[Git 分支模型](http://nvie.com/posts/a-successful-git-branching-model/).  您可以按照以下步骤提交代码并参与贡献.

### 1. Fork 

  请从您的fork中提交 `Pull Requests` . 
  
  只需要前往我们的 GitHub repo 页面并点击 ["Fork"](https://help.github.com/articles/fork-a-repo/) 按钮.
   
### 2. 克隆 (Clone)

   将您的fork复制到本地:

   ```bash
   $ git clone https://github.com/your-github-account/PPViT
   $ cd PPViT
   ```

### 3. 创建本地 `feature` 分支

   对于日常工作例如添加新功能或修复错误，请在编码之前基于`develop`分支创建一个 `feature` 分支:

   ```bash
   $ git checkout develop
   $ git checkout -b feature
   ```
   其中`feature` 可以替换为你正在处理的功能的名称.

### 4. 提交 (Commit)

   `during and after` 您的更改，将代码提交到本地存储库.

   ```shell
   $ git add -A
   $ git commit -m “message”
   ```
  
### 5. 测试

   - 我们鼓励编写`unittest` 来测试你编写的类与方法的实现.
   - 在开始合并之前，请在相关数据集上测试模型的性能。
 
### 6. 持续 pulling

   有经验的Git用户会经常从官方存储库中pull数据--每天甚至每小时，因此他们会尽早注意到与其他人的工作冲突，并且更容易解决较小的冲突。

   ```bash
   $ git remote add upstream https://github.com/BR-IDL/PaddleViT
   $ git pull upstream develop
   ```

### 7. Push 以及 file a `Pull Request`

   1. **Push** 您的本地工作到您的fork仓库中:

      ```bash
      $ git push origin my-cool-stuff
      ```
      push操作允许您创建一个pull request,请求此 [official repo](https://github.com/BR-IDL/PaddleViT) 将您的更改拉入到官方库中.

   2. 想要创建一个`Pull Request`, 请按照 [这些步骤](https://help.github.com/articles/creating-a-pull-request/).

      如果您的更改是`fixing an issue`, 请在pull request的描述部分写下["Fixes <issue-URL>"](https://help.github.com/articles/closing-issues-using-keywords/).  当合并您的 pull request时，Github将关闭该问题.

      请记住为您的pull request指定审阅者.  如果您不知道正确的选择，请遵循Github的推荐.

### 8. 删除本地和远程 `feature` 分支

   成功合并到`develop`分支后，删除您的`feature` 分支。
   为了保持您的本地工作区和fork简洁，您可能想要删除合并的分支：

   ```bash
   $ git push origin :my-cool-stuff
   $ git checkout develop
   $ git pull upstream develop
   $ git branch -d my-cool-stuff
   ```

## 代码审查

-  请随时通过 IM 或电子邮件来 ping 您的审阅者以发送您的pull request.

- 请回答审阅者的每一条评论. 如果您要关注评论，请写“完成”；否则请给出理由。

- 如果您不希望您的审阅者被电子邮件通知淹没，可以通过 [批量](https://help.github.com/articles/reviewing-proposed-changes-in-a-pull-request/) 回复评论.

- 减少非必要的提交.  存在一些开发人员经常提交，建议通过运行 `git commit --amend` 代替 `git commit`，将一系列小的变动附加到一个提交中.

## Coding Standard

### Code Style

我们的Python代码遵循 [PEP8 language guide](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_language_rules/) 以及 [PEP8 style guide](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/).
  
### Use Pylint

[Pylint](http://pylint.pycqa.org/en/latest/) 是一款 Python代码分析工具，可以分析Python代码中的错误，发现不符合编程标准或存在潜在问题的代码。 

### Comments and Annotations
  
为了让其他人更容易使用并生成在线文件，请在每个类方法的每个函数中包含文档的描述字符串。
  
### 单元测试

请记得添加相关的单元测试

- 对于 Python 代码, 请使用 [Python's standard `unittest` package](http://pythontesting.net/framework/unittest/unittest-introduction/).

尝试对每个类方法的每个函数都进行单元测试。
  
