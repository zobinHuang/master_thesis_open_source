#pragma once

#define PROFILING_ENABLED true

#if PROFILING_ENABLED
    #define PROFILE(profile_command) profile_command
#else
    #define PROFILE(profile_command)
#endif
