/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SOCKET_UTILS_H_
#define NCCL_SOCKET_UTILS_H_

#include <stdlib.h>  // for atoi
#include <string.h>  // for strlen, strncmp

struct netIf {
  char prefix[64];
  int port;
};

static inline int parseStringList(const char* string, struct netIf* ifList, int maxList) {
  if (!string) return 0;

  const char* ptr = string;

  int ifNum = 0;
  int ifC = 0;
  char c;
  do {
    c = *ptr;
    if (c == ':') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = atoi(ptr+1);
        ifNum++; ifC = 0;
      }
      while (c != ',' && c != '\0') c = *(++ptr);
    } else if (c == ',' || c == '\0') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = -1;
        ifNum++; ifC = 0;
      }
    } else {
      ifList[ifNum].prefix[ifC] = c;
      ifC++;
    }
    ptr++;
  } while (ifNum < maxList && c);
  return ifNum;
}

static bool matchIf(const char* string, const char* ref, bool matchExact) {
  // Make sure to include '\0' in the exact case
  int matchLen = matchExact ? strlen(string) + 1 : strlen(ref);
  return strncmp(string, ref, matchLen) == 0;
}

static bool matchPort(const int port1, const int port2) {
  if (port1 == -1) return true;
  if (port2 == -1) return true;
  if (port1 == port2) return true;
  return false;
}

static inline bool matchIfList(const char* string, int port, struct netIf* ifList, int listSize, bool matchExact) {
  // Make an exception for the case where no user list is defined
  if (listSize == 0) return true;

  for (int i=0; i<listSize; i++) {
    if (matchIf(string, ifList[i].prefix, matchExact)
        && matchPort(port, ifList[i].port)) {
      return true;
    }
  }
  return false;
}

#endif
