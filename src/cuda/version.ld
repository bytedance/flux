{
global:
    extern "C++" {
        bytedance::flux::*;
        nvshmemi_init_thread*;
        nvshmemi_finalize*;
    };

local:
    *;
};
