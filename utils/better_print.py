import threading
import time


class TextAnimator:
    """在终端中显示动态文本动画的线程化工具

    Attributes:
        __text_running (str): 动画运行期间显示的基础文本
        __text_finish (str): 动画结束时显示的最终文本
    """

    def __init__(self, text_run: str, text_finish: str = None):
        """初始化文本动画生成器

        Args:
            text_run: 动画运行期间持续显示的基础文本
            text_finish: 动画结束时要显示的文本（默认与text_run相同）
        """
        self.__stop_event = threading.Event()
        self.__text_running = text_run
        self.__text_finish = text_finish if text_finish else text_run
        self.__anim_thread = threading.Thread(target=self._animate)

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

        print(
            f"\r{self.__text_finish}!!!   Cost Time: {cost_time:.2f}s\033[K"
        )  # \033[K 清除行尾

    def start(self):
        """启动动画线程"""
        self.__anim_thread.start()

    def stop(self):
        """停止动画并等待线程结束

        发送停止信号后，会等待动画线程完成最后的输出清理操作
        """
        self.__stop_event.set()
        self.__anim_thread.join()


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
