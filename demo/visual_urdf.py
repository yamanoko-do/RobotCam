import matplotlib.pyplot as plt

from pytransform3d.urdf import UrdfTransformManager

file_path = "data/piper_description.urdf"

try:
    # 打开文件并读取内容
    with open(file_path, "r", encoding="utf-8") as file:
        urdf_str = file.read()  # 将文件内容读取为字符串
    print("URDF 文件读取成功！")
except FileNotFoundError:
    print(f"错误：文件 {file_path} 未找到！")
except Exception as e:
    print(f"读取文件时发生错误：{e}")

tm = UrdfTransformManager()
tm.load_urdf(urdf_str)


joint_names = ["joint1","joint2","joint3","joint4","joint5","joint6","joint6_to_gripper_base","joint7","joint8"]
joint_angles = [0, 0, 0, 0, 0, 0, 0, 0, 0]
offset = [0, -172.22, -102.78, 0, 0, 0, 0, 0, 0]
joint_angles =  [-0.091, 4.551, -41.699, -1.131, 64.722, -85.087, 0, 0, 0]
result = [a + b for a, b in zip(offset, joint_angles)]
print(result)
for name, angle in zip(joint_names, joint_angles):
    tm.set_joint(name, angle)

whitelist=["base_link","link1","link2","link3","link4","link5","link6","gripper_base","link7","link8"]
whitelist=["gripper_base","link7"]

ax = tm.plot_frames_in(
    "piper",
    whitelist=whitelist,
    s=0.1,
    show_name=True,
)
ax = tm.plot_connections_in("piper", ax=ax)
ax.set_xlim((-0.2, 0.8))
ax.set_ylim((-0.5, 0.5))
ax.set_zlim((-0.2, 0.8))
plt.show()