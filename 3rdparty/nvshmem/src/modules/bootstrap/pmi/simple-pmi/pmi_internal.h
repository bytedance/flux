/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */
/* Copyright (c) 2001-2016, The Ohio State University. All rights
 * reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 */

#ifndef PMI_H_INCLUDED
#define PMI_H_INCLUDED

/* prototypes for the PMI interface in MPICH */

#if defined(__cplusplus)
extern "C" {
#endif

/*D
PMI_CONSTANTS - PMI definitions

Error Codes:
+ SPMI_SUCCESS - operation completed successfully
. SPMI_FAIL - operation failed
. SPMI_ERR_NOMEM - input buffer not large enough
. SPMI_ERR_INIT - SPMI not initialized
. SPMI_ERR_INVALID_ARG - invalid argument
. SPMI_ERR_INVALID_KEY - invalid key argument
. SPMI_ERR_INVALID_KEY_LENGTH - invalid key length argument
. SPMI_ERR_INVALID_VAL - invalid val argument
. SPMI_ERR_INVALID_VAL_LENGTH - invalid val length argument
. SPMI_ERR_INVALID_LENGTH - invalid length argument
. SPMI_ERR_INVALID_NUM_ARGS - invalid number of arguments
. SPMI_ERR_INVALID_ARGS - invalid args argument
. SPMI_ERR_INVALID_NUM_PARSED - invalid num_parsed length argument
. SPMI_ERR_INVALID_KEYVALP - invalid keyvalp argument
- SPMI_ERR_INVALID_SIZE - invalid size argument

Booleans:
+ SPMI_TRUE - true
- SPMI_FALSE - false

D*/
#define SPMI_SUCCESS 0
#define SPMI_FAIL -1
#define SPMI_ERR_INIT 1
#define SPMI_ERR_NOMEM 2
#define SPMI_ERR_INVALID_ARG 3
#define SPMI_ERR_INVALID_KEY 4
#define SPMI_ERR_INVALID_KEY_LENGTH 5
#define SPMI_ERR_INVALID_VAL 6
#define SPMI_ERR_INVALID_VAL_LENGTH 7
#define SPMI_ERR_INVALID_LENGTH 8
#define SPMI_ERR_INVALID_NUM_ARGS 9
#define SPMI_ERR_INVALID_ARGS 10
#define SPMI_ERR_INVALID_NUM_PARSED 11
#define SPMI_ERR_INVALID_KEYVALP 12
#define SPMI_ERR_INVALID_SIZE 13

/* PMI Group functions */

/*@
SPMI_Init - initialize the Process Manager Interface

Output Parameter:
. spawned - spawned flag

Return values:
+ SPMI_SUCCESS - initialization completed successfully
. SPMI_ERR_INVALID_ARG - invalid argument
- SPMI_FAIL - initialization failed

Notes:
Initialize PMI for this process group. The value of spawned indicates whether
this process was created by 'PMI_Spawn_multiple'.  'spawned' will be 'PMI_TRUE' if
this process group has a parent and 'PMI_FALSE' if it does not.

@*/
int SPMI_Init(int *spawned);

/*@
SPMI_Initialized - check if SPMI has been initialized

Output Parameter:
. initialized - boolean value

Return values:
+ SPMI_SUCCESS - initialized successfully set
. SPMI_ERR_INVALID_ARG - invalid argument
- SPMI_FAIL - unable to set the variable

Notes:
On successful output, initialized will either be 'SPMI_TRUE' or 'SPMI_FALSE'.

+ SPMI_TRUE - initialize has been called.
- SPMI_FALSE - initialize has not been called or previously failed.

@*/
int SPMI_Initialized(int *initialized);

/*@
SPMI_Finalize - finalize the Process Manager Interface

Return values:
+ SPMI_SUCCESS - finalization completed successfully
- SPMI_FAIL - finalization failed

Notes:
 Finalize SPMI for this process group.

@*/
int SPMI_Finalize(void);

/*@
SPMI_Get_size - obtain the size of the process group

Output Parameters:
. size - pointer to an integer that receives the size of the process group

Return values:
+ SPMI_SUCCESS - size successfully obtained
. SPMI_ERR_INVALID_ARG - invalid argument
- SPMI_FAIL - unable to return the size

Notes:
This function returns the size of the process group to which the local process
belongs.

@*/
int SPMI_Get_size(int *size);

/*@
SPMI_Get_rank - obtain the rank of the local process in the process group

Output Parameters:
. rank - pointer to an integer that receives the rank in the process group

Return values:
+ SPMI_SUCCESS - rank successfully obtained
. SPMI_ERR_INVALID_ARG - invalid argument
- SPMI_FAIL - unable to return the rank

Notes:
This function returns the rank of the local process in its process group.

@*/
int SPMI_Get_rank(int *rank);

/*@
SPMI_Lookup_name - lookup a service by name

Input parameters:
. service_name - string representing the service being published

Output parameters:
. port - string representing the port on which to contact the service

Return values:
+ SPMI_SUCCESS - port for service successfully obtained
. SPMI_ERR_INVALID_ARG - invalid argument
- SPMI_FAIL - unable to lookup service


@*/
int SPMI_Lookup_name(const char service_name[], char port[]);

/*@
SPMI_Barrier - barrier across the process group

Return values:
+ SPMI_SUCCESS - barrier successfully finished
- SPMI_FAIL - barrier failed

Notes:
This function is a collective call across all processes in the process group
the local process belongs to.  It will not return until all the processes
have called 'SPMI_Barrier()'.

@*/
int SPMI_Barrier(void);

/*@
SPMI_Abort - abort the process group associated with this process

Input Parameters:
+ exit_code - exit code to be returned by this process
- error_msg - error message to be printed

Return values:
. none - this function should not return
@*/
int SPMI_Abort(int exit_code, const char error_msg[]);

/* SPMI Keymap functions */
/*@
SPMI_KVS_Get_my_name - obtain the name of the keyval space the local process group has access to

Input Parameters:
. length - length of the kvsname character array

Output Parameters:
. kvsname - a string that receives the keyval space name

Return values:
+ SPMI_SUCCESS - kvsname successfully obtained
. SPMI_ERR_INVALID_ARG - invalid argument
. SPMI_ERR_INVALID_LENGTH - invalid length argument
- SPMI_FAIL - unable to return the kvsname

Notes:
This function returns the name of the keyval space that this process and all
other processes in the process group have access to.  The output parameter,
kvsname, must be at least as long as the value returned by
'SPMI_KVS_Get_name_length_max()'.

@*/
int SPMI_KVS_Get_my_name(char kvsname[], int length);

/*@
SPMI_KVS_Get_name_length_max - obtain the length necessary to store a kvsname

Output Parameter:
. length - maximum length required to hold a keyval space name

Return values:
+ SPMI_SUCCESS - length successfully set
. SPMI_ERR_INVALID_ARG - invalid argument
- SPMI_FAIL - unable to set the length

Notes:
This function returns the string length required to store a keyval space name.

A routine is used rather than setting a maximum value in 'pmi.h' to allow
different implementations of SPMI to be used with the same executable.  These
different implementations may allow different maximum lengths; by using a
routine here, we can interface with a variety of implementations of SPMI.

@*/
int SPMI_KVS_Get_name_length_max(int *length);

/*@
SPMI_KVS_Get_key_length_max - obtain the length necessary to store a key

Output Parameter:
. length - maximum length required to hold a key string.

Return values:
+ SPMI_SUCCESS - length successfully set
. SPMI_ERR_INVALID_ARG - invalid argument
- SPMI_FAIL - unable to set the length

Notes:
This function returns the string length required to store a key.

@*/
int SPMI_KVS_Get_key_length_max(int *length);

/*@
SPMI_KVS_Get_value_length_max - obtain the length necessary to store a value

Output Parameter:
. length - maximum length required to hold a keyval space value

Return values:
+ SPMI_SUCCESS - length successfully set
. SPMI_ERR_INVALID_ARG - invalid argument
- SPMI_FAIL - unable to set the length

Notes:
This function returns the string length required to store a value from a
keyval space.

@*/
int SPMI_KVS_Get_value_length_max(int *length);

/*@
SPMI_KVS_Put - put a key/value pair in a keyval space

Input Parameters:
+ kvsname - keyval space name
. key - key
- value - value

Return values:
+ SPMI_SUCCESS - keyval pair successfully put in keyval space
. SPMI_ERR_INVALID_KVS - invalid kvsname argument
. SPMI_ERR_INVALID_KEY - invalid key argument
. SPMI_ERR_INVALID_VAL - invalid val argument
- SPMI_FAIL - put failed

Notes:
This function puts the key/value pair in the specified keyval space.  The
value is not visible to other processes until 'SPMI_KVS_Commit()' is called.
The function may complete locally.  After 'SPMI_KVS_Commit()' is called, the
value may be retrieved by calling 'SPMI_KVS_Get()'.  All keys put to a keyval
space must be unique to the keyval space.  You may not put more than once
with the same key.

@*/
int SPMI_KVS_Put(const char kvsname[], const char key[], const char value[]);

/*@
SPMI_KVS_Commit - commit all previous puts to the keyval space

Return values:
+ SPMI_SUCCESS - commit succeeded
. SPMI_ERR_INVALID_ARG - invalid argument
- SPMI_FAIL - commit failed

Notes:
This function commits all previous puts since the last 'SPMI_KVS_Commit()' into
the specified keyval space. It is a process local operation.

@*/
int SPMI_KVS_Commit();

/*@
SPMI_KVS_Get - get a key/value pair from a keyval space

Input Parameters:
+ kvsname - keyval space name
. key - key
- length - length of value character array

Output Parameters:
. value - value

Return values:
+ SPMI_SUCCESS - get succeeded
. SPMI_ERR_INVALID_KVS - invalid kvsname argument
. SPMI_ERR_INVALID_KEY - invalid key argument
. SPMI_ERR_INVALID_VAL - invalid val argument
. SPMI_ERR_INVALID_LENGTH - invalid length argument
- SPMI_FAIL - get failed

Notes:
This function gets the value of the specified key in the keyval space.

@*/
int SPMI_KVS_Get(const char kvsname[], const char key[], char value[], int length);

/* SPMI Process Creation functions */

/*S
SPMI_keyval_t - keyval structure used by SPMI_Spawn_mulitiple

Fields:
+ key - name of the key
- val - value of the key

S*/
typedef struct SPMI_keyval_t {
    const char *key;
    char *val;
} SPMI_keyval_t;

#if defined(__cplusplus)
}
#endif

#endif
