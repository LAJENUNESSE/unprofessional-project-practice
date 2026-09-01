"""PyInstaller 打包入口：python launcher.py 不直接使用，仅供 build_exe 打包。

同时处理 Windows 控制台编码：
- stdout/stderr 遇到 GBK 无法表示的字符（如 emoji）时替换而不是崩溃；
- stdin 容错解码，避免管道输入编码不一致产生 surrogate 后续请求崩溃。
真实控制台输入走 ReadConsoleW（Unicode），不受影响。
"""

import sys


def main() -> int:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass
    from qa_agent.__main__ import main as qa_main

    return qa_main()


if __name__ == "__main__":
    sys.exit(main())
