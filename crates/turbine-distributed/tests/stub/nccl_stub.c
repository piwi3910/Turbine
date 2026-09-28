/* Test-only stand-in for an NCCL-API library (librccl.so.1 / libnccl.so.2)
 * used by turbine-distributed's collective::ffi tests. It exports the 13
 * symbols the binding resolves, with the NCCL C signatures (ncclResult_t and
 * the enums are int, ncclUniqueId is 128 bytes), and host semantics for a
 * world of one rank: buffers are host memory, streams are ignored.
 *
 * build.rs compiles it with the host C compiler once per variant.
 * STUB_VERSION is the NCCL_VERSION_CODE that ncclGetVersion reports;
 * STUB_OMIT_GROUP_END leaves ncclGroupEnd out for the missing-symbol test;
 * STUB_INIT_NEVER_COMPLETES makes ncclCommInitRankConfig return
 * ncclInProgress and ncclCommGetAsyncError keep reporting it, for any world
 * size (the init-watchdog test). stub_abort_calls() counts ncclCommAbort. */
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef int ncclResult_t;
typedef int ncclDataType_t;
typedef int ncclRedOp_t;
typedef struct stub_comm *ncclComm_t;
typedef struct {
  char internal[128];
} ncclUniqueId;
typedef struct ncclConfig ncclConfig_t;

enum {
  ncclSuccess = 0,
  ncclInvalidArgument = 4,
  ncclInProgress = 7,
};

static int abort_calls = 0;

/* Test hook: how many times ncclCommAbort was called. */
int stub_abort_calls(void) { return abort_calls; }

struct stub_comm {
  int nranks;
  int rank;
};

static size_t dtype_size(ncclDataType_t t) {
  switch (t) {
  case 0: /* ncclInt8 */
  case 1: /* ncclUint8 */
    return 1;
  case 6: /* ncclFloat16 */
  case 9: /* ncclBfloat16 */
    return 2;
  case 2: /* ncclInt32 */
  case 3: /* ncclUint32 */
  case 7: /* ncclFloat32 */
    return 4;
  default:
    return 8;
  }
}

ncclResult_t ncclGetVersion(int *version) {
  if (version == NULL)
    return ncclInvalidArgument;
  *version = STUB_VERSION;
  return ncclSuccess;
}

ncclResult_t ncclGetUniqueId(ncclUniqueId *id) {
  if (id == NULL)
    return ncclInvalidArgument;
  for (int i = 0; i < 128; i++)
    id->internal[i] = (char)i;
  return ncclSuccess;
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm, int nranks,
                                    ncclUniqueId id, int rank,
                                    ncclConfig_t *config) {
  (void)id;
  (void)config;
#ifdef STUB_INIT_NEVER_COMPLETES
  if (comm == NULL || rank < 0 || rank >= nranks)
    return ncclInvalidArgument;
#else
  if (comm == NULL || nranks != 1 || rank != 0)
    return ncclInvalidArgument;
#endif
  struct stub_comm *c = malloc(sizeof *c);
  if (c == NULL)
    return ncclInvalidArgument;
  c->nranks = nranks;
  c->rank = rank;
  *comm = c;
#ifdef STUB_INIT_NEVER_COMPLETES
  return ncclInProgress;
#else
  return ncclSuccess;
#endif
}

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *async_error) {
  if (comm == NULL || async_error == NULL)
    return ncclInvalidArgument;
#ifdef STUB_INIT_NEVER_COMPLETES
  *async_error = ncclInProgress;
#else
  *async_error = ncclSuccess;
#endif
  return ncclSuccess;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) {
  abort_calls++;
  free(comm);
  return ncclSuccess;
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  free(comm);
  return ncclSuccess;
}

static ncclResult_t copy(const void *send, void *recv, size_t count,
                         ncclDataType_t t, ncclComm_t comm) {
  if (comm == NULL)
    return ncclInvalidArgument;
  if (send != recv)
    memmove(recv, send, count * dtype_size(t));
  return ncclSuccess;
}

ncclResult_t ncclAllReduce(const void *send, void *recv, size_t count,
                           ncclDataType_t t, ncclRedOp_t op, ncclComm_t comm,
                           void *stream) {
  (void)op;
  (void)stream;
  return copy(send, recv, count, t, comm);
}

ncclResult_t ncclAllGather(const void *send, void *recv, size_t sendcount,
                           ncclDataType_t t, ncclComm_t comm, void *stream) {
  (void)stream;
  return copy(send, recv, sendcount, t, comm);
}

ncclResult_t ncclReduceScatter(const void *send, void *recv, size_t recvcount,
                               ncclDataType_t t, ncclRedOp_t op,
                               ncclComm_t comm, void *stream) {
  (void)op;
  (void)stream;
  return copy(send, recv, recvcount, t, comm);
}

ncclResult_t ncclBroadcast(const void *send, void *recv, size_t count,
                           ncclDataType_t t, int root, ncclComm_t comm,
                           void *stream) {
  (void)stream;
  if (root != 0)
    return ncclInvalidArgument;
  return copy(send, recv, count, t, comm);
}

ncclResult_t ncclGroupStart(void) { return ncclSuccess; }

#ifndef STUB_OMIT_GROUP_END
ncclResult_t ncclGroupEnd(void) { return ncclSuccess; }
#endif

const char *ncclGetErrorString(ncclResult_t result) {
  switch (result) {
  case ncclSuccess:
    return "no error";
  case ncclInvalidArgument:
    return "invalid argument";
  default:
    return "unknown result code";
  }
}
