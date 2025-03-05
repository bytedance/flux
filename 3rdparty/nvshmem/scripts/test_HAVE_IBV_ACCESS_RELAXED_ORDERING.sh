#!/bin/sh

if [ -n "$1" ]; then
  CXX=$1
fi

if [ -n "$2" ]; then
  CXXFLAGS=$2
fi

# use /tmp as fallback for systems without /dev/shm
# /tmp is slow so using it only if /dev/shm is not available
tmpdir=/tmp
if [ -w "/dev/shm" ]; then
  tmpdir=/dev/shm
fi
tmpfile="$(mktemp --suffix=.cpp ${tmpdir}/nvshmem.XXXXXXXXX)"

cat >${tmpfile} <<EOL
#include <infiniband/verbs.h>
int main(void) {
    int x = IBV_ACCESS_RELAXED_ORDERING;
    return 0;
}
EOL

${CXX} ${CXXFLAGS} ${tmpfile} -o conftest_HAVE_IBV_ACCESS_RELAXED_ORDERING > /dev/null 2>&1
ret=$?

rm -f conftest_HAVE_IBV_ACCESS_RELAXED_ORDERING ${tmpfile}

if [ "$ret" -eq 0 ]; then
    echo "1"
else
    echo "0"
fi
