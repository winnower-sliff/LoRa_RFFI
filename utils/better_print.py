import threading
import time


class TextAnimator:
    """在终端中显示动态文本动画的线程化工具"""

    def __init__(self):
        """初始化文本动画生成器"""
        self.__text_running = ""  # 初始化属性避免 AttributeError
        self.__text_finish = ""
        self.__anim_thread = None
        self.__stop_event = None

    def _animate(self):
        """执行动画的核心循环逻辑

        循环更新文本结尾的句号数量（1-3个循环变化），直到收到停止信号。
        当停止事件触发时，将结尾符号替换为三个感叹号并清除行尾。
        """
        dots = 1
        start_time = time.time()

        while not self.__stop_event.is_set():
            ending = "." * dots + " " * (5 - dots)
            print(f"\r{self.__text_running}{ending}", end="", flush=True)
            dots = dots + 1 if dots < 3 else 1
            self.__stop_event.wait(0.5)

        end_time = time.time()
        cost_time = end_time - start_time

        pad = ' ' * max(2, 30 - len(self.__text_finish))
        if len(self.__text_finish) >= 30:
            pad = '\n' + ' ' * 30
        print(
            f"\r{self.__text_finish}{pad}Cost Time: {cost_time:5.2f}s\033[K"
        )

    def start(self, text_run: str):
        """启动动画线程（支持多次调用）"""
        # 如果已有线程在运行，先停止它
        if self.__anim_thread and self.__anim_thread.is_alive():
            self.stop()

        # 初始化新的停止事件和线程
        self.__stop_event = threading.Event()
        self.__text_running = text_run
        self.__text_finish = text_run  # 默认与 text_run 相同
        self.__anim_thread = threading.Thread(target=self._animate)
        self.__anim_thread.start()

    def stop(self, text_finish: str = None):
        """停止动画并等待线程结束"""
        if self.__anim_thread and self.__anim_thread.is_alive():
            self.__text_finish = text_finish if text_finish else self.__text_running
            self.__stop_event.set()
            self.__anim_thread.join()
            self.__anim_thread = None  # 清理线程对象


def print_colored_text(text, color_code):
    """
    在终端中打印彩色文本。

    :param text: 要打印的文本。
    :param color_code: ANSI转义序列中的颜色代码（不包括开头的或\x1b[和结尾的m）。

    以下是一些常用的ANSI转义序列，用于设置文本颜色：

    * 重置/默认: 0
    * 黑色: 30
    * 红色: 31
    * 绿色: 32
    * 黄色: 33
    * 蓝色: 34
    * 洋红/紫色: 35
    * 青色: 36
    * 白色: 37
    """
    # 构建完整的ANSI转义序列
    color_sequence = f"\033[{color_code}m"
    # 使用print()函数输出彩色文本，并在末尾重置颜色
    print(f"{color_sequence}{text}\033[0m")


if __name__ == "__main__":
    print("颜色对应列表：")
    for i in range(30, 40):
        print_colored_text(f"123\t{i}", str(i))


# 使用示例
if __name__ == "__main__":
    # 初始化动画器
    animator = TextAnimator("Processing")

    try:
        # 启动动画线程
        animator.start()
        animator.__text_running = "hacker"

        # 模拟主进程工作（这里用sleep代替实际工作）
        time.sleep(5)

    finally:
        # 停止动画并等待线程结束
        animator.stop()

    # 主进程后续工作
    print("\nMain process completed!")
