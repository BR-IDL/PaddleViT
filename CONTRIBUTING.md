English | [简体中文](./CONTRIBUTING_cn.md)

# Contribute Code

You encourage and appreciate researchers and developers to contribute to project **PaddleViT**. 
To contribute to PaddlePaddle, you have to agree with the [PaddleViT Contributor License Agreement](https://cla-assistant.io/BR-IDL/PaddleViT).

This document explains our workflow and working style.

## Workflow

PaddleViT uses this [Git branching model](http://nvie.com/posts/a-successful-git-branching-model/).  You can follow the listed steps for common contributions.

### 1. Fork the repo

  Please **file `Pull Requests` from your fork**. 
  
  To make a fork, just head over to our GitHub repo page and click the ["Fork"](https://help.github.com/articles/fork-a-repo/) button.
   
### 2. Clone the repo

   To make a copy of your fork to your local env:

   ```bash
   $ git clone https://github.com/your-github-account/PPViT
   $ cd PPViT
   ```

### 3. Create local `feature` branch

   For daily works like adding a new feature or fixing a bug, open a `feature` branch based on `develop` branch before coding:

   ```bash
   $ git checkout develop
   $ git checkout -b feature
   ```
   wher `feature` can be replaced with the name of your feature you are working on.

### 4. Commit

   Commit your code to the local repository **during and after** your coding.

   ```shell
   $ git add -A
   $ git commit -m “message”
   ```
  
### 5. Test

   - We encourage writing `unittest` to test your class and method.
   - Please test and report model performance on related datasets before you start to merge.
 
### 6. Keep pulling

   An experienced Git user pulls from the official repo often -- daily or even hourly, so they notice conflicts with others work early, and it's easier to resolve smaller conflicts.

   ```bash
   $ git remote add upstream https://github.com/BR-IDL/PaddleViT
   $ git pull upstream develop
   ```

### 7. Push and file a `Pull Request`

   1. **Push** your local work into your forked repo:

      ```bash
      $ git push origin my-cool-stuff
      ```

      The push allows you to create a pull request, requesting owners of this [official repo](https://github.com/BR-IDL/PaddleViT) to pull your change into the official one.

   2. To create a `Pull Request`, please follow [these steps](https://help.github.com/articles/creating-a-pull-request/).

      If your change is for fixing an issue, please write ["Fixes <issue-URL>"](https://help.github.com/articles/closing-issues-using-keywords/) in the description section of your pull request.  Github would close the issue when the owners merge your pull request.

      Please remember to specify some reviewers for your pull request.  If you don't know who are the right ones, please follow Github's recommendation.

### 8. Delete local and remote `feature` branches

   After merging into `develop` branch successfully, delete your `feature` branch.
   To keep your local workspace and your fork clean, you might want to remove merged branches:

   ```bash
   $ git push origin :my-cool-stuff
   $ git checkout develop
   $ git pull upstream develop
   $ git branch -d my-cool-stuff
   ```

## Code Review

-  Please feel free to ping your reviewers by sending them the URL of your pull request via IM or email.

- Please answer reviewers' every comment.  If you are to follow the comment, please write "Done"; please give a reason otherwise.

- If you don't want your reviewers to get overwhelmed by email notifications, you might reply their comments by [in a batch](https://help.github.com/articles/reviewing-proposed-changes-in-a-pull-request/).

- Reduce the unnecessary commits.  Some developers commit often.  It is recommended to append a sequence of small changes into one commit by running `git commit --amend` instead of `git commit`.

## Coding Standard

### Code Style

Our Python code follows the [PEP8 language guide](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_language_rules/) and [PEP8 style guide](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/).
  
### Use Pylint

[Pylint](http://pylint.pycqa.org/en/latest/) is a Python code analysis tool that analyzes errors in Python code and finds code that does not meet coding style standards and has potential problems.

### Comments and Annotations
  
To make it easier for others to use and generate online documents, please include a docstring for each function on each class method.
  
### Unit Tests

Please remember to add related unit tests.

- For Python code, please use [Python's standard `unittest` package](http://pythontesting.net/framework/unittest/unittest-introduction/).

Try to have unit tests for each function on each class method.
  
