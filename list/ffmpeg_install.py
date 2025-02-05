import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# FFmpeg 실행 파일 경로 명시 (필요한 경우)
ffmpeg_path = r"D:/ffmpeg-2025-01-15-git-4f3c9f2f03-full_build/ffmpeg-2025-01-15-git-4f3c9f2f03-full_build/bin/ffmpeg.exe"

# 데이터 생성 함수
def generate_data():
    x = np.linspace(0, 2 * np.pi, 100)
    for t in np.linspace(0, 2 * np.pi, 200):
        y = np.sin(x + t)  # y 값은 시간에 따라 변함
        yield x, y

# 업데이트 함수
def update(frame):
    x, y = frame
    line.set_data(x, y)
    return line,

# 초기화 함수
def init():
    line.set_data([], [])
    return line,

# Figure와 Line 생성
fig, ax = plt.subplots()
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-1.5, 1.5)
line, = ax.plot([], [], lw=2)

# 애니메이션 생성
ani = FuncAnimation(fig, update, frames=generate_data, init_func=init, blit=True)

# mp4로 저장
output_file = "D:/real_time_plot.mp4"
writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
ani.save(output_file, writer=writer)

print(f"애니메이션이 {output_file}로 저장되었습니다!")