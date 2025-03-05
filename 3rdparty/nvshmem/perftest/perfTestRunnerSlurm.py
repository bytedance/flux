#!/usr/bin/env python
from subprocess import check_call,CalledProcessError #2.6

check_call(["sbatch", "-N", "1", "--qos=short", "-p", "dgx-1p", "perfTestRunner.py"])
#check_call(["sbatch", "-N", "1", "--qos=short", "-p", "dgx-1v", "perfTestRunner.py"])
#check_call(["sbatch", "-N", "1", "--qos=short", "-p", "hsw_p100", "perfTestRunner.py"])
#check_call(["sbatch", "-N", "1", "--qos=short", "-p", "hsw_v100", "perfTestRunner.py"])
