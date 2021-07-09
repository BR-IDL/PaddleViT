# Contribute Code

You are welcome to contribute to project PPViT. 

We sincerely appreciate your contribution.  This document explains our workflow and work style.

## Workflow

PPViT uses this [Git branching model](http://nvie.com/posts/a-successful-git-branching-model/).  The following steps guide usual contributions.

1. Fork

   Our development community has been growing fastly; it doesn't make sense for everyone to write into the official repo.  So, please file Pull Requests from your fork.  To make a fork,  just head over to the GitHub page and click the ["Fork" button](https://help.github.com/articles/fork-a-repo/).
   
1. Clone

   To make a copy of your fork to your local computers, please run

   ```bash
   git clone https://github.com/your-github-account/PPViT
   cd PPViT
   ```

1. Create the local feature branch

   For daily works like adding a new feature or fixing a bug, please open your feature branch based on develop before coding:

   ```bash
   git checkout develop
   git checkout -b my-cool-stuff
   ```

1. Commit

   Commit to the local repository.

   ```shell
   git add -A
   git commit -m “message”
   ```
  
1. Build and test

   Users can build PPViT natively on Linux and Mac OS X.  But to unify the building environment and to make it easy for debugging.
1. Keep pulling

   An experienced Git user pulls from the official repo often -- daily or even hourly, so they notice conflicts with others work early, and it's easier to resolve smaller conflicts.

   ```bash
   git remote add upstream https://github.com/xperzy/PPViT
   git pull upstream develop
   ```

1. Push and file a pull request

   You can "push" your local work into your forked repo:

   ```bash
   git push origin my-cool-stuff
   ```

   The push allows you to create a pull request, requesting owners of this [official repo](https://github.com/xperzy/PPViT) to pull your change into the official one.

   To create a pull request, please follow [these steps](https://help.github.com/articles/creating-a-pull-request/).

   If your change is for fixing an issue, please write ["Fixes <issue-URL>"](https://help.github.com/articles/closing-issues-using-keywords/) in the description section of your pull request.  Github would close the issue when the owners merge your pull request.

   Please remember to specify some reviewers for your pull request.  If you don't know who are the right ones, please follow Github's recommendation.

1. Delete local and remote branches

   After merging into develop branch successfully, delete feature branch.
   To keep your local workspace and your fork clean, you might want to remove merged branches:

   ```bash
   git push origin :my-cool-stuff
   git checkout develop
   git pull upstream develop
   git branch -d my-cool-stuff
   ```

### Code Review

-  Please feel free to ping your reviewers by sending them the URL of your pull request via IM or email.  Please do this after your pull request passes the CI.

- Please answer reviewers' every comment.  If you are to follow the comment, please write "Done"; please give a reason otherwise.

- If you don't want your reviewers to get overwhelmed by email notifications, you might reply their comments by [in a batch](https://help.github.com/articles/reviewing-proposed-changes-in-a-pull-request/).

- Reduce the unnecessary commits.  Some developers commit often.  It is recommended to append a sequence of small changes into one commit by running `git commit --amend` instead of `git commit`.

## Coding Standard

### Code Style

Our Python code follows the [PEP8 language guide](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_language_rules/) and [PEP8 style guide](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/).
  
Our build process helps to check the code style.  In [`build.sh`](https://github.com/PaddlePaddle/Paddle/blob/b84e8226514b8bb4405c3c28e54aa5077193d179/paddle/scripts/docker/build.sh#L42), the entry point of our [builder Docker image](https://github.com/PaddlePaddle/Paddle/blob/b84e8226514b8bb4405c3c28e54aa5077193d179/Dockerfile#L88), the CMake argument `WITH_STYLE_CHECK` is set to `ON` by default.  This flag is on

Please install pre-commit, which automatically reformat the changes to C/C++ and Python code whenever we run `git commit`.  To check the whole codebase, we can run the command `pre-commit run -a`, as in the [`check_style.sh` file](https://github.com/PaddlePaddle/Paddle/blob/b84e8226514b8bb4405c3c28e54aa5077193d179/paddle/scripts/travis/check_style.sh#L30), which is invoked by [our Travis CI configuration](https://github.com/PaddlePaddle/Paddle/blob/b84e8226514b8bb4405c3c28e54aa5077193d179/.travis.yml#L43).
  
### Use Pylint

[Pylint](http://pylint.pycqa.org/en/latest/) is a Python code analysis tool that analyzes errors in Python code and finds code that does not meet coding style standards and has potential problems.

### Code Annotation
  
To make it easier for others to use and generate online documents, try to have a docstring for each function on each class method.
  
### Unit Tests

Please remember to add related unit tests.

- For Python code, please use [Python's standard `unittest` package](http://pythontesting.net/framework/unittest/unittest-introduction/).

Try to have unit tests for each function on each class method.
  
