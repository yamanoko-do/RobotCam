用于内参标定，手在眼外，双目外参标定，极线校正的存储库

# 关于分支，累积小的提交，最后合并进入主分支

- 克隆存储库：
```
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/yamanoko-do/RobotCam.git --recurse-submodules
```

- 下载参考标定数据：
```
git lfs pull
```

# Piper 

can_piper

- 查找can：bash ./thirdpart/piper_sdk/piper_sdk/find_all_can_port.sh
- 激活can：bash ./thirdpart/piper_sdk/piper_sdk/can_activate.sh can_piper 1000000 "3-1:1.0"

# 参考

- [PointCloudGeneration-git](https://github.com/musimab/PointCloudGeneration)
- [d435官方api+demo](https://dev.intelrealsense.com/docs/python2)
