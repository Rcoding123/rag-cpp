from .cli import hard_exit, main

# Every ragrl process loads torch and _ragof; their native teardown at a normal exit intermittently crashed with
# 0xC0000005 after the work was done (LESSONS 63, 64), so a command that returns ends with hard_exit. A command that
# raises still exits through the interpreter with its traceback and a nonzero code.
hard_exit(main() or 0)
