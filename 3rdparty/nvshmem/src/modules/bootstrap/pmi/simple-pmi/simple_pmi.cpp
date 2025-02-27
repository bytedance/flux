/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

/*********************** PMI implementation ********************************/
/*
 * This file implements the client-side of the PMI interface.
 *
 * Note that the PMI client code must not print error messages (except
 * when an abort is required) because MPI error handling is based on
 * reporting error codes to which messages are attached.
 *
 * In v2, we should require a PMI client interface to use MPI error codes
 * to provide better integration with MPICH.
 */
/***************************************************************************/

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <errno.h>

#include "pmi_internal.h"
#include "simple_pmiutil.h"

#define MAXHOSTNAME 256
#define PMI_VERSION 1
#define PMI_SUBVERSION 0

static int PMI_fd = -1;
static int PMI_size = 1;
static int PMI_rank = 0;

/* Set PMI_initialized to 1 for singleton init but no process manager
   to help.  Initialized to 2 for normal initialization.  Initialized
   to values higher than 2 when singleton_init by a process manager.
   All values higher than 1 invlove a PM in some way.
*/
typedef enum {
    PMI_UNINITIALIZED = 0,
    SINGLETON_INIT_BUT_NO_PM = 1,
    NORMAL_INIT_WITH_PM,
    SINGLETON_INIT_WITH_PM
} PMIState;
static PMIState PMI_initialized = PMI_UNINITIALIZED;

/* ALL GLOBAL VARIABLES MUST BE INITIALIZED TO AVOID POLLUTING THE
   LIBRARY WITH COMMON SYMBOLS */
static int PMI_kvsname_max = 0;
static int PMI_keylen_max = 0;
static int PMI_vallen_max = 0;

static int PMI_debug = 0;

/* Function prototypes for internal routines */
static int PMII_getmaxes(int *kvsname_max, int *keylen_max, int *vallen_max);
static int PMII_Connect_to_pm(char *hostname, int portnum);
static int PMII_Set_from_port(int fd, int id);

static int GetResponse(const char[], const char[], int);
static int getPMIFD(int *);

static char cached_singinit_key[SPMIU_MAXLINE];
static char cached_singinit_val[SPMIU_MAXLINE];

/******************************** Group functions *************************/

int SPMI_Init(int *spawned) {
    char *p;
    int notset = 1;
    int rc;

    // INFO(NVSHMEM_BOOTSTRAP, "in PMI_Init");

    PMI_initialized = PMI_UNINITIALIZED;

    /* FIXME: Why is setvbuf commented out? */
    /* FIXME: What if the output should be fully buffered (directed to file)?
       unbuffered (user explicitly set?) */
    /* setvbuf(stdout,0,_IONBF,0); */
    setbuf(stdout, NULL);
    /* SPMIU_printf( 1, "PMI_INIT\n" ); */

    /* Get the value of PMI_DEBUG from the environment if possible, since
       we may have set it to help debug the setup process */
    p = getenv("PMI_DEBUG");
    if (p) PMI_debug = atoi(p);

    /* Get the fd for PMI commands; if none, we're a singleton */
    rc = getPMIFD(&notset);
    if (rc) {
        return rc;
    }

    if (PMI_fd == -1) {
        /* Singleton init: Process not started with mpiexec,
           so set size to 1, rank to 0 */
        PMI_size = 1;
        PMI_rank = 0;
        *spawned = 0;

        PMI_initialized = SINGLETON_INIT_BUT_NO_PM;
        /* 256 is picked as the minimum allowed length by the PMI servers */
        PMI_kvsname_max = 256;
        PMI_keylen_max = 256;
        PMI_vallen_max = 256;

        // INFO(NVSHMEM_BOOTSTRAP, "PMI_fd not found, returning spawned = 0");
        return (0);
    }

    /* If size, rank, and debug are not set from a communication port,
       use the environment */
    if (notset) {
        if ((p = getenv("PMI_SIZE")))
            PMI_size = atoi(p);
        else
            PMI_size = 1;

        if ((p = getenv("PMI_RANK"))) {
            PMI_rank = atoi(p);
            /* Let the util routine know the rank of this process for
               any messages (usually debugging or error) */
            SPMIU_Set_rank(PMI_rank);
        } else
            PMI_rank = 0;

        if ((p = getenv("PMI_DEBUG")))
            PMI_debug = atoi(p);
        else
            PMI_debug = 0;

        /* Leave unchanged otherwise, which indicates that no value
           was set */
    }

    // INFO(NVSHMEM_BOOTSTRAP, "rank: %d size: %d", PMI_rank, PMI_size);

    PMII_getmaxes(&PMI_kvsname_max, &PMI_keylen_max, &PMI_vallen_max);

    if (!PMI_initialized) PMI_initialized = NORMAL_INIT_WITH_PM;

    *spawned = 1;

    return (0);
}

int SPMI_Initialized(int *initialized) {
    /* Turn this into a logical value (1 or 0) .  This allows us
       to use PMI_initialized to distinguish between initialized with
       an PMI service (e.g., via mpiexec) and the singleton init,
       which has no PMI service */
    *initialized = (PMI_initialized != 0);
    return SPMI_SUCCESS;
}

int SPMI_Get_size(int *size) {
    if (PMI_initialized)
        *size = PMI_size;
    else
        *size = 1;
    return (0);
}

int SPMI_Get_rank(int *rank) {
    if (PMI_initialized)
        *rank = PMI_rank;
    else
        *rank = 0;
    return (0);
}

int SPMI_Barrier(void) {
    int err = SPMI_SUCCESS;

    if (PMI_initialized > SINGLETON_INIT_BUT_NO_PM) {
        err = GetResponse("cmd=barrier_in\n", "barrier_out", 0);
    }

    return err;
}

/* Inform the process manager that we're in finalize */
int SPMI_Finalize(void) {
    int err = SPMI_SUCCESS;

    if (PMI_initialized > SINGLETON_INIT_BUT_NO_PM) {
        err = GetResponse("cmd=finalize\n", "finalize_ack", 0);
        shutdown(PMI_fd, SHUT_RDWR);
        close(PMI_fd);
    }

    return err;
}

int SPMI_Abort(int exit_code, const char error_msg[]) {
    char buf[SPMIU_MAXLINE];

    /* include exit_code in the abort command */
    snprintf(buf, SPMIU_MAXLINE, "cmd=abort exitcode=%d\n", exit_code);

    SPMIU_printf(PMI_debug, "aborting job:\n%s\n", error_msg);
    GetResponse(buf, "", 0);

    /* the above command should not return */
    return -1;
}

/************************************* Keymap functions **********************/

/*FIXME: need to return an error if the value of the kvs name returned is
  truncated because it is larger than length */
/* FIXME: My name should be cached rather than re-acquired, as it is
   unchanging (after singleton init) */
int SPMI_KVS_Get_my_name(char kvsname[], int length) {
    int err;

    if (PMI_initialized == SINGLETON_INIT_BUT_NO_PM) {
        /* Return a dummy name */
        /* FIXME: We need to support a distinct kvsname for each
           process group */
        snprintf(kvsname, length, "singinit_kvs_%d_0", (int)getpid());
        return 0;
    }
    err = GetResponse("cmd=get_my_kvsname\n", "my_kvsname", 0);
    if (err == SPMI_SUCCESS) {
        SPMIU_getval("kvsname", kvsname, length);
    }
    return err;
}

int SPMI_KVS_Get_name_length_max(int *maxlen) {
    if (maxlen == NULL) return SPMI_ERR_INVALID_ARG;
    *maxlen = PMI_kvsname_max;
    return SPMI_SUCCESS;
}

int SPMI_KVS_Get_key_length_max(int *maxlen) {
    if (maxlen == NULL) return SPMI_ERR_INVALID_ARG;
    *maxlen = PMI_keylen_max;
    return SPMI_SUCCESS;
}

int SPMI_KVS_Get_value_length_max(int *maxlen) {
    if (maxlen == NULL) return SPMI_ERR_INVALID_ARG;
    *maxlen = PMI_vallen_max;
    return SPMI_SUCCESS;
}

int SPMI_KVS_Put(const char kvsname[], const char key[], const char value[]) {
    char buf[SPMIU_MAXLINE];
    int err = SPMI_SUCCESS;
    int r;
    char *rc;

    /* This is a special hack to support singleton initialization */
    if (PMI_initialized == SINGLETON_INIT_BUT_NO_PM) {
        rc = strncpy(cached_singinit_key, key, PMI_keylen_max);
        if (rc == NULL) return SPMI_FAIL;

        rc = strncpy(cached_singinit_val, value, PMI_vallen_max);
        if (rc == NULL) return SPMI_FAIL;

        return 0;
    }

    r = snprintf(buf, SPMIU_MAXLINE, "cmd=put kvsname=%s key=%s value=%s\n", kvsname, key, value);
    if (r < 0) return SPMI_FAIL;
    err = GetResponse(buf, "put_result", 1);
    return err;
}

int SPMI_KVS_Commit() {
    /* no-op in this implementation */
    return (0);
}

/*FIXME: need to return an error if the value returned is truncated
  because it is larger than length */
int SPMI_KVS_Get(const char kvsname[], const char key[], char value[], int length) {
    char buf[SPMIU_MAXLINE];
    int err = SPMI_SUCCESS;
    int rc;

    rc = snprintf(buf, SPMIU_MAXLINE, "cmd=get kvsname=%s key=%s\n", kvsname, key);
    if (rc < 0) return SPMI_FAIL;

    err = GetResponse(buf, "get_result", 0);
    if (err == SPMI_SUCCESS) {
        SPMIU_getval("rc", buf, SPMIU_MAXLINE);
        rc = atoi(buf);
        if (rc == 0) {
            SPMIU_getval("value", value, length);
            return (0);
        } else {
            return (-1);
        }
    }

    return err;
}

/***************** Internal routines not part of PMI interface ***************/

/* to get all maxes in one message */
/* FIXME: This mixes init with get maxes */
static int PMII_getmaxes(int *kvsname_max, int *keylen_max, int *vallen_max) {
    char buf[SPMIU_MAXLINE], cmd[SPMIU_MAXLINE], errmsg[SPMIU_MAXLINE];
    const int truncation_length = SPMIU_MAXLINE / 3;
    int err, rc;

    rc = snprintf(buf, SPMIU_MAXLINE, "cmd=init pmi_version=%d pmi_subversion=%d\n", PMI_VERSION,
                  PMI_SUBVERSION);
    if (rc < 0) {
        return SPMI_FAIL;
    }

    rc = SPMIU_writeline(PMI_fd, buf);
    if (rc != 0) {
        SPMIU_printf(1, "Unable to write to PMI_fd\n");
        return SPMI_FAIL;
    }
    buf[0] = 0; /* Ensure buffer is empty if read fails */
    err = SPMIU_readline(PMI_fd, buf, SPMIU_MAXLINE);
    if (err < 0) {
        SPMIU_printf(1, "Error reading initack on %d\n", PMI_fd);
        perror("Error on readline:");
        SPMI_Abort(-1, "Above error when reading after init");
    }
    SPMIU_parse_keyvals(buf);
    cmd[0] = 0;
    SPMIU_getval("cmd", cmd, SPMIU_MAXLINE);
    if (strncmp(cmd, "response_to_init", SPMIU_MAXLINE) != 0) {
        snprintf(errmsg, SPMIU_MAXLINE, "got unexpected response to init :%.*s: (full line = %.*s)",
                 truncation_length, cmd, truncation_length, buf);
        SPMI_Abort(-1, errmsg);
    } else {
        char buf1[SPMIU_MAXLINE];
        SPMIU_getval("rc", buf, SPMIU_MAXLINE);
        if (strncmp(buf, "0", SPMIU_MAXLINE) != 0) {
            SPMIU_getval("pmi_version", buf, SPMIU_MAXLINE);
            SPMIU_getval("pmi_subversion", buf1, SPMIU_MAXLINE);
            snprintf(errmsg, SPMIU_MAXLINE, "pmi_version mismatch; client=%d.%d mgr=%.*s.%.*s",
                     PMI_VERSION, PMI_SUBVERSION, truncation_length, buf, truncation_length, buf1);
            SPMI_Abort(-1, errmsg);
        }
    }
    err = GetResponse("cmd=get_maxes\n", "maxes", 0);
    if (err == SPMI_SUCCESS) {
        SPMIU_getval("kvsname_max", buf, SPMIU_MAXLINE);
        *kvsname_max = atoi(buf);
        SPMIU_getval("keylen_max", buf, SPMIU_MAXLINE);
        *keylen_max = atoi(buf);
        SPMIU_getval("vallen_max", buf, SPMIU_MAXLINE);
        *vallen_max = atoi(buf);
    }
    return err;
}

/* ----------------------------------------------------------------------- */
/*
 * This function is used to request information from the server and check
 * that the response uses the expected command name.  On a successful
 * return from this routine, additional SPMIU_getval calls may be used
 * to access information about the returned value.
 *
 * If checkRc is true, this routine also checks that the rc value returned
 * was 0.  If not, it uses the "msg" value to report on the reason for
 * the failure.
 */
static int GetResponse(const char request[], const char expectedCmd[], int checkRc) {
    int err, n;
    char *p;
    char recvbuf[SPMIU_MAXLINE];
    char cmdName[SPMIU_MAXLINE];

    /* FIXME: This is an example of an incorrect fix - writeline can change
       the second argument in some cases, and that will break the const'ness
       of request.  Instead, writeline should take a const item and return
       an error in the case in which it currently truncates the data. */
    err = SPMIU_writeline(PMI_fd, (char *)request);
    if (err) {
        return err;
    }
    n = SPMIU_readline(PMI_fd, recvbuf, sizeof(recvbuf));
    if (n <= 0) {
        SPMIU_printf(1, "readline failed\n");
        return SPMI_FAIL;
    }
    err = SPMIU_parse_keyvals(recvbuf);
    if (err) {
        SPMIU_printf(1, "parse_kevals failed %d\n", err);
        return err;
    }
    p = SPMIU_getval("cmd", cmdName, sizeof(cmdName));
    if (!p) {
        SPMIU_printf(1, "getval cmd failed\n");
        return SPMI_FAIL;
    }
    if (strcmp(expectedCmd, cmdName) != 0) {
        SPMIU_printf(1, "expecting cmd=%s, got %s\n", expectedCmd, cmdName);
        return SPMI_FAIL;
    }
    if (checkRc) {
        p = SPMIU_getval("rc", cmdName, SPMIU_MAXLINE);
        if (p && strcmp(cmdName, "0") != 0) {
            SPMIU_getval("msg", cmdName, SPMIU_MAXLINE);
            SPMIU_printf(1, "Command %s failed, reason='%s'\n", request, cmdName);
            return SPMI_FAIL;
        }
    }

    return err;
}

static int getPMIFD(int *notset) {
    char *p;

    /* Set the default */
    PMI_fd = -1;

    p = getenv("PMI_FD");
    if (p) {
        PMI_fd = atoi(p);
        return 0;
    }

    p = getenv("PMI_PORT");
    if (p) {
        int portnum;
        char hostname[MAXHOSTNAME + 1];
        char *pn, *ph;
        int id = 0;

        /* Connect to the indicated port (in format hostname:portnumber)
           and get the fd for the socket */

        /* Split p into host and port */
        pn = p;
        ph = hostname;
        while (*pn && *pn != ':' && (ph - hostname) < MAXHOSTNAME) {
            *ph++ = *pn++;
        }
        *ph = 0;

        if (PMI_debug) {
            SPMIU_printf(1, "Connecting to %s\n", p);
        }
        if (*pn == ':') {
            portnum = atoi(pn + 1);
            /* FIXME: Check for valid integer after : */
            /* This routine only gets the fd to use to talk to
               the process manager. The handshake below is used
               to setup the initial values */
            PMI_fd = PMII_Connect_to_pm(hostname, portnum);
            if (PMI_fd < 0) {
                SPMIU_printf(1, "Unable to connect to %s on %d\n", hostname, portnum);
                return -1;
            }
        } else {
            SPMIU_printf(1, "unable to decode hostport from %s\n", p);
            return SPMI_FAIL;
        }

        /* We should first handshake to get size, rank, debug. */
        p = getenv("PMI_ID");
        if (p) {
            id = atoi(p);
            /* PMII_Set_from_port sets up the values that are delivered
               by enviroment variables when a separate port is not used */
            PMII_Set_from_port(PMI_fd, id);
            *notset = 0;
        }
        return 0;
    }

    /* Singleton init case - its ok to return success with no fd set */
    return 0;
}

static int PMII_Set_from_port(int fd, int id) {
    char buf[SPMIU_MAXLINE], cmd[SPMIU_MAXLINE];
    int err, rc;

    /* We start by sending a startup message to the server */

    /* Handshake and initialize from a port */

    rc = snprintf(buf, SPMIU_MAXLINE, "cmd=initack pmiid=%d\n", id);
    if (rc < 0) {
        return SPMI_FAIL;
    }
    SPMIU_printf(1, "writing on fd %d line :%s:\n", fd, buf);
    err = SPMIU_writeline(fd, buf);
    if (err) {
        SPMIU_printf(1, "Error in writeline initack\n");
        return -1;
    }

    /* cmd=initack */
    buf[0] = 0;
    SPMIU_printf(PMI_debug, "reading initack\n");
    err = SPMIU_readline(fd, buf, SPMIU_MAXLINE);
    if (err < 0) {
        SPMIU_printf(1, "Error reading initack on %d\n", fd);
        perror("Error on readline:");
        return -1;
    }
    SPMIU_parse_keyvals(buf);
    SPMIU_getval("cmd", cmd, SPMIU_MAXLINE);
    if (strcmp(cmd, "initack")) {
        SPMIU_printf(1, "got unexpected input %s\n", buf);
        return -1;
    }

    /* Read, in order, size, rank, and debug.  Eventually, we'll want
     *        the handshake to include a version number */

    /* size */
    SPMIU_printf(PMI_debug, "reading size\n");
    err = SPMIU_readline(fd, buf, SPMIU_MAXLINE);
    if (err < 0) {
        SPMIU_printf(1, "Error reading size on %d\n", fd);
        perror("Error on readline:");
        return -1;
    }
    SPMIU_parse_keyvals(buf);
    SPMIU_getval("cmd", cmd, SPMIU_MAXLINE);
    if (strcmp(cmd, "set")) {
        SPMIU_printf(1, "got unexpected command %s in %s\n", cmd, buf);
        return -1;
    }
    /* cmd=set size=n */
    SPMIU_getval("size", cmd, SPMIU_MAXLINE);
    PMI_size = atoi(cmd);

    /* rank */
    SPMIU_printf(PMI_debug, "reading rank\n");
    err = SPMIU_readline(fd, buf, SPMIU_MAXLINE);
    if (err < 0) {
        SPMIU_printf(1, "Error reading rank on %d\n", fd);
        perror("Error on readline:");
        return -1;
    }
    SPMIU_parse_keyvals(buf);
    SPMIU_getval("cmd", cmd, SPMIU_MAXLINE);
    if (strcmp(cmd, "set")) {
        SPMIU_printf(1, "got unexpected command %s in %s\n", cmd, buf);
        return -1;
    }
    /* cmd=set rank=n */
    SPMIU_getval("rank", cmd, SPMIU_MAXLINE);
    PMI_rank = atoi(cmd);
    SPMIU_Set_rank(PMI_rank);

    /* debug flag */
    err = SPMIU_readline(fd, buf, SPMIU_MAXLINE);
    if (err < 0) {
        SPMIU_printf(1, "Error reading debug on %d\n", fd);
        return -1;
    }
    SPMIU_parse_keyvals(buf);
    SPMIU_getval("cmd", cmd, SPMIU_MAXLINE);
    if (strcmp(cmd, "set")) {
        SPMIU_printf(1, "got unexpected command %s in %s\n", cmd, buf);
        return -1;
    }
    /* cmd=set debug=n */
    SPMIU_getval("debug", cmd, SPMIU_MAXLINE);
    PMI_debug = atoi(cmd);

    return 0;
}

/* stub for connecting to a specified host/port instead of using a
 *    specified fd inherited from a parent process */
static int PMII_Connect_to_pm(char *hostname, int portnum) {
    struct hostent *hp;
    struct sockaddr_in sa;
    int fd;
    int optval = 1;

    hp = gethostbyname(hostname);
    if (!hp) {
        SPMIU_printf(1, "Unable to get host entry for %s\n", hostname);
        return -1;
    }

    memset((void *)&sa, 0, sizeof(sa));
    /* POSIX might define h_addr_list only and node define h_addr */
#ifdef HAVE_H_ADDR_LIST
    memcpy((void *)&sa.sin_addr, (void *)hp->h_addr_list[0], hp->h_length);
#else
    memcpy((void *)&sa.sin_addr, (void *)hp->h_addr, hp->h_length);
#endif
    sa.sin_family = hp->h_addrtype;
    sa.sin_port = htons((unsigned short)portnum);

    fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (fd < 0) {
        SPMIU_printf(1, "Unable to get AF_INET socket\n");
        return -1;
    }

    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (char *)&optval, sizeof(optval))) {
        perror("Error calling setsockopt:");
    }

    /* We wait here for the connection to succeed */
    if (connect(fd, (struct sockaddr *)&sa, sizeof(sa)) < 0) {
        switch (errno) {
            case ECONNREFUSED:
                SPMIU_printf(1, "connect failed with connection refused\n");
                /* (close socket, get new socket, try again) */
                close(fd);
                return -1;

            case EINPROGRESS: /*  (nonblocking) - select for writing. */
                break;

            case EISCONN: /*  (already connected) */
                break;

            case ETIMEDOUT: /* timed out */
                SPMIU_printf(1, "connect failed with timeout\n");
                close(fd);
                return -1;

            default:
                SPMIU_printf(1, "connect failed with errno %d\n", errno);
                close(fd);
                return -1;
        }
    }

    return fd;
}
