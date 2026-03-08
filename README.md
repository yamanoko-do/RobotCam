用于内参标定，手在眼外，双目外参标定，极线校正的存储库

- 克隆存储库：
```
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/yamanoko-do/RobotCam.git --recurse-submodules
```

- 下载参考标定数据：
```
git lfs pull
```
# Usage
```bash
├── calibration #使用opencv实现标定流程的实现
│   ├── bino.py
│   ├── eye2hand.py
│   ├── mono.py
│   └── utils.py
├── data #存放标定数据
│   ├── 棋盘格27_27.pdf
│   ├── cali_pose.txt
│   ├── chessboard_images
│   ├── eye2hand_images
│   ├── output
│   └── piper_description.urdf
├── hardware #硬件接口
│   ├── camera
│   │   ├── basecamera.py
│   │   ├── binocam.py
│   │   └── d435.py
│   └── robotarm
│       └── piper.py
├── README.md
├── requirements.txt
├── thirdpart
│   └── piper_sdk
└── tools #标定工具入口
    ├── calibrate_bino.py
    ├── calibrate_eye2hand.py
    └── calibrate_intrinsic.py
```

# Piper 

can_piper

- 查找can：bash ./thirdpart/piper_sdk/piper_sdk/find_all_can_port.sh
- 激活can：bash ./thirdpart/piper_sdk/piper_sdk/can_activate.sh can_piper 1000000 "3-1:1.0"

# 参考

- [PointCloudGeneration-git](https://github.com/musimab/PointCloudGeneration)
- [d435官方api+demo](https://dev.intelrealsense.com/docs/python2)


