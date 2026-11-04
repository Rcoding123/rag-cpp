// pid.hpp — the process id, for per-process temporary file names (portable).
#pragma once

#if defined(_WIN32)
#include <process.h>
inline int test_pid() { return _getpid(); }
#else
#include <unistd.h>
inline int test_pid() { return static_cast<int>(::getpid()); }
#endif
