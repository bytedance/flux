/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

/* Allow fprintf to logfile */
/* style: allow:fprintf:1 sig:0 */

/* Utility functions associated with PMI implementation, but not part of
   the PMI interface itself.  Reading and writing on pipes, signals, and parsing
   key=value messages
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

#include "simple_pmiutil.h"

#define MAXVALLEN 1024
#define MAXKEYLEN 32

/* These are not the keyvals in the keyval space that is part of the
   PMI specification.
   They are just part of this implementation's internal utilities.
*/
struct SPMIU_keyval_pairs {
    char key[MAXKEYLEN];
    char value[MAXVALLEN];
};
static struct SPMIU_keyval_pairs SPMIU_keyval_tab[64] = {{{0}, {0}}};
static int SPMIU_keyval_tab_idx = 0;

/* This is used to prepend printed output.  Set the initial value to
   "unset" */
static char SPMIU_print_id[SPMIU_IDSIZE] = "unset";

void SPMIU_Set_rank(int PMI_rank) { snprintf(SPMIU_print_id, SPMIU_IDSIZE, "cli_%d", PMI_rank); }
void SPMIU_SetServer(void) { strncpy(SPMIU_print_id, "server", SPMIU_IDSIZE); }

/* Note that vfprintf is part of C89 */

/* style: allow:fprintf:1 sig:0 */
/* style: allow:vfprintf:1 sig:0 */
/* This should be combined with the message routines */
void SPMIU_printf(int print_flag, const char *fmt, ...) {
    va_list ap;
    static FILE *logfile = 0;

    /* In some cases when we are debugging, the handling of stdout or
       stderr may be unreliable.  In that case, we make it possible to
       select an output file. */
    if (!logfile) {
        char *p;
        p = getenv("PMI_USE_LOGFILE");
        if (p) {
            char filename[1024];
            p = getenv("PMI_ID");
            if (p) {
                snprintf(filename, sizeof(filename), "testclient-%s.out", p);
                logfile = fopen(filename, "w");
            } else {
                logfile = fopen("testserver.out", "w");
            }
        } else
            logfile = stderr;
    }

    if (print_flag) {
        fprintf(logfile, "[%s]: ", SPMIU_print_id);
        va_start(ap, fmt);
        vfprintf(logfile, fmt, ap);
        va_end(ap);
        fflush(logfile);
    }
}

#define MAX_READLINE 1024
/*
 * Return the next newline-terminated string of maximum length maxlen.
 * This is a buffered version, and reads from fd as necessary.  A
 */
int SPMIU_readline(int fd, char *buf, int maxlen) {
    static char readbuf[MAX_READLINE];
    static char *nextChar = 0, *lastChar = 0; /* lastChar is really one past
                                                 last char */
    static int lastfd = -1;
    ssize_t n;
    int curlen;
    char *p, ch;

    /* Note: On the client side, only one thread at a time should
       be calling this, and there should only be a single fd.
       Server side code should not use this routine (see the
       replacement version in src/pm/util/pmiserv.c) */
    if (nextChar != lastChar && fd != lastfd) {
        fprintf(stderr, "Panic - buffer inconsistent\n");
        return -1;
    }

    p = buf;
    curlen = 1; /* Make room for the null */
    while (curlen < maxlen) {
        if (nextChar == lastChar) {
            lastfd = fd;
            do {
                n = read(fd, readbuf, sizeof(readbuf) - 1);
            } while (n == -1 && errno == EINTR);
            if (n == 0) {
                /* EOF */
                break;
            } else if (n < 0) {
                /* Error.  Return a negative value if there is no
                   data.  Save the errno in case we need to return it
                   later. */
                if (curlen == 1) {
                    curlen = 0;
                }
                break;
            }
            nextChar = readbuf;
            lastChar = readbuf + n;
            /* Add a null at the end just to make it easier to print
               the read buffer */
            readbuf[n] = 0;
            /* FIXME: Make this an optional output */
            /* printf( "Readline %s\n", readbuf ); */
        }

        ch = *nextChar++;
        *p++ = ch;
        curlen++;
        if (ch == '\n') break;
    }

    /* We null terminate the string for convenience in printing */
    *p = 0;

    /* Return the number of characters, not counting the null */
    return curlen - 1;
}

int SPMIU_writeline(int fd, char *buf) {
    ssize_t size, n;

    size = strlen(buf);
    if (size > SPMIU_MAXLINE) {
        buf[SPMIU_MAXLINE - 1] = '\0';
        SPMIU_printf(1, "write_line: message string too big: :%s:\n", buf);
    } else if (buf[strlen(buf) - 1] != '\n') /* error:  no newline at end */
        SPMIU_printf(1, "write_line: message string doesn't end in newline: :%s:\n", buf);
    else {
        do {
            n = write(fd, buf, size);
        } while (n == -1 && errno == EINTR);

        if (n < 0) {
            SPMIU_printf(1, "write_line error; fd=%d buf=:%s:\n", fd, buf);
            perror("system msg for write_line failure ");
            return (-1);
        }
        if (n < size) SPMIU_printf(1, "write_line failed to write entire message\n");
    }
    return 0;
}

/*
 * Given an input string st, parse it into internal storage that can be
 * queried by routines such as SPMIU_getval.
 */
int SPMIU_parse_keyvals(char *st) {
    char *p, *keystart, *valstart;
    int offset;

    if (!st) return (-1);

    SPMIU_keyval_tab_idx = 0;
    p = st;
    while (1) {
        while (*p == ' ') p++;
        /* got non-blank */
        if (*p == '=') {
            SPMIU_printf(1, "SPMIU_parse_keyvals:  unexpected = at character %d in %s\n", p - st,
                         st);
            return (-1);
        }
        if (*p == '\n' || *p == '\0') return (0); /* normal exit */
        /* got normal character */
        keystart = p; /* remember where key started */
        int length = 0;
        while (*p != ' ' && *p != '=' && *p != '\n' && *p != '\0') {
            p++;
            length++;
        }

        if (*p == ' ' || *p == '\n' || *p == '\0') {
            SPMIU_printf(1, "SPMIU_parse_keyvals: unexpected key delimiter at character %d in %s\n",
                         p - st, st);
            return (-1);
        }
        /* Null terminate the key */
        *p = '\0';
        /* store key */
        if (length >= MAXKEYLEN) { /*guarantee the string end with \0*/
            *(keystart + MAXKEYLEN - 1) = '\0';
        }
        strncpy(SPMIU_keyval_tab[SPMIU_keyval_tab_idx].key, keystart, MAXKEYLEN);

        valstart = ++p; /* start of value */
        while (*p != ' ' && *p != '\n' && *p != '\0') p++;
        /* store value */
        strncpy(SPMIU_keyval_tab[SPMIU_keyval_tab_idx].value, valstart, MAXVALLEN);
        offset = (int)(p - valstart);
        /* When compiled with -fPIC, the pgcc compiler generates incorrect
           code if "p - valstart" is used instead of using the
           intermediate offset */
        SPMIU_keyval_tab[SPMIU_keyval_tab_idx].value[offset] = '\0';
        SPMIU_keyval_tab_idx++;
        if (*p == ' ') continue;
        if (*p == '\n' || *p == '\0') return (0); /* value has been set to empty */
    }
}

void SPMIU_dump_keyvals(void) {
    int i;
    for (i = 0; i < SPMIU_keyval_tab_idx; i++)
        SPMIU_printf(1, "  %s=%s\n", SPMIU_keyval_tab[i].key, SPMIU_keyval_tab[i].value);
}

char *SPMIU_getval(const char *keystr, char *valstr, int vallen) {
    int i;
    char *rc = NULL;

    for (i = 0; i < SPMIU_keyval_tab_idx; i++) {
        if (strcmp(keystr, SPMIU_keyval_tab[i].key) == 0) {
            rc = strncpy(valstr, SPMIU_keyval_tab[i].value, vallen);
            if (rc == NULL) {
                SPMIU_printf(1, "strncpy failed in SPMIU_getval\n");
                return NULL;
            }
            return valstr;
        }
    }
    valstr[0] = '\0';
    return NULL;
}

void SPMIU_chgval(const char *keystr, char *valstr) {
    int i;

    for (i = 0; i < SPMIU_keyval_tab_idx; i++) {
        if (strcmp(keystr, SPMIU_keyval_tab[i].key) == 0) {
            strncpy(SPMIU_keyval_tab[i].value, valstr, MAXVALLEN - 1);
            SPMIU_keyval_tab[i].value[MAXVALLEN - 1] = '\0';
        }
    }
}
