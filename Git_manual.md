# PPViT开发流程 #
The document describes how to use git to develop on ppvit.

## First Develop on PPViT

#### step 1. Git clone develop branch

```shell
git clone -b develop https://github.com/xperzy/PPViT.git
```

#### step 2. Build feature branch based on develop

```shell
git checkout -b feature
```

#### step 3. Commit to the remote repository

Commit to the temporary storage.

```shell
git add -A
```

Commit to the local repository.
```shell
git commit -m “message”
```

Commit to the remote repository.
```shell
git push --set-upstream origin feature
```

#### step 4. Commit merge request on github

![image](https://github.com/wflrz123/Document_classify/blob/master/%E5%9B%BE%E7%89%871.png)

> Note:
> - May cause conflicts when merging. Watting for reviewer review the request.

#### step 5. Delete feature branch

After merging into develop branch successfully, delete feature branch.
```shell
git push origin --delete feature
```

## Secondary Develop 

Pull the latest code to the local repository and merge into feature branch.

```shell
git checkout develop
git pull
git checkout feature
git merge develop
```

