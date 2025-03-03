### FAQ (Frequently Asked Questions)


1. If you encounter a NCCL connection problem, that may be the problem inside the `launch.sh` script. A possible solution is to export a proper `NCCL_SOCKET_IFNAME` variable manually. Try to set it to the first word you get from `ifconfig`.

