/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */
/*TODO: NVIDIA copyright*/

#ifndef _SIMPLE_PMI_UTIL_H_
#define _SIMPLE_PMI_UTIL_H_

/* maximum sizes for arrays */
#define SPMIU_MAXLINE 1024
#define SPMIU_IDSIZE 32

/* we don't have access to MPIU_Assert and friends here in the PMI code */
#if defined(HAVE_ASSERT_H)
#include <assert.h>
#define SPMIU_Assert(expr) assert(expr)
#else
#define SPMIU_Assert(expr)
#endif

#if defined HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif /* HAVE_ARPA_INET_H */

/* prototypes for SPMIU routines */
void SPMIU_Set_rank(int PMI_rank);
void SPMIU_SetServer(void);
void SPMIU_printf(int print_flag, const char *fmt, ...);
int SPMIU_readline(int fd, char *buf, int max);
int SPMIU_writeline(int fd, char *buf);
int SPMIU_parse_keyvals(char *st);
void SPMIU_dump_keyvals(void);
char *SPMIU_getval(const char *keystr, char *valstr, int vallen);
void SPMIU_chgval(const char *keystr, char *valstr);

#endif
