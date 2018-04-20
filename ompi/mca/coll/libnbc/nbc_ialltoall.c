/* -*- Mode: C; c-basic-offset:2 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2006      The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2006      The Technical University of Chemnitz. All
 *                         rights reserved.
 * Copyright (c) 2013-2015 Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2014      NVIDIA Corporation.  All rights reserved.
 * Copyright (c) 2014-2017 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2017      IBM Corporation.  All rights reserved.
 * Copyright (c) 2018      FUJITSU LIMITED.
 * Copyright (c) 2018      EPCC, The University of Edinburgh. All rights reserved.
 * Copyright (c) 2018      Technical University of Vienna. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *
 */
#include "nbc_internal.h"

static inline int a2a_sched_linear(int rank, int p, MPI_Aint sndext, MPI_Aint rcvext, NBC_Schedule *schedule,
                                   const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                                   int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
static inline int a2a_sched_pairwise(int rank, int p, MPI_Aint sndext, MPI_Aint rcvext, NBC_Schedule *schedule,
                                     const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                                     int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
static inline int a2a_sched_diss(int rank, int p, MPI_Aint sndext, MPI_Aint rcvext, NBC_Schedule* schedule,
                                 const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                                 int recvcount, MPI_Datatype recvtype, MPI_Comm comm, void* tmpbuf);
static inline int a2a_sched_inplace(int rank, int p, NBC_Schedule* schedule, void* buf, int count,
                                   MPI_Datatype type, MPI_Aint ext, ptrdiff_t gap, MPI_Comm comm);

#ifdef NBC_CACHE_SCHEDULE
/* tree comparison function for schedule cache */
int NBC_Alltoall_args_compare(NBC_Alltoall_args *a, NBC_Alltoall_args *b, void *param) {
  if ((a->sendbuf == b->sendbuf) &&
      (a->sendcount == b->sendcount) &&
      (a->sendtype == b->sendtype) &&
      (a->recvbuf == b->recvbuf) &&
      (a->recvcount == b->recvcount) &&
      (a->recvtype == b->recvtype)) {
    return 0;
  }

  if( a->sendbuf < b->sendbuf ) {
    return -1;
  }

  return 1;
}
#endif

/* simple linear MPI_Ialltoall the (simple) algorithm just sends to all nodes */
static int nbc_ialltoall(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                         MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                         struct mca_coll_base_module_2_2_0_t *module, bool persistent)
{
  int rank, p, res, datasize;
  size_t a2asize, sndsize;
  NBC_Schedule *schedule;
  MPI_Aint rcvext, sndext;
#ifdef NBC_CACHE_SCHEDULE
  NBC_Alltoall_args *args, *found, search;
#endif
  char *rbuf, *sbuf, inplace;
  enum {NBC_A2A_LINEAR, NBC_A2A_PAIRWISE, NBC_A2A_DISS, NBC_A2A_INPLACE} alg;
  void *tmpbuf = NULL;
  ompi_coll_libnbc_module_t *libnbc_module = (ompi_coll_libnbc_module_t*) module;
  ptrdiff_t span, gap;

  NBC_IN_PLACE(sendbuf, recvbuf, inplace);

  rank = ompi_comm_rank (comm);
  p = ompi_comm_size (comm);

  res = ompi_datatype_type_extent(sendtype, &sndext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  res = ompi_datatype_type_extent(recvtype, &rcvext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  res = ompi_datatype_type_size(sendtype, &sndsize);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_size() (%i)", res);
    return res;
  }

  /* algorithm selection */
  a2asize = sndsize * sendcount * p;
  /* this number is optimized for TCP on odin.cs.indiana.edu */
  if (inplace) {
    alg = NBC_A2A_INPLACE;
  } else if((p <= 8) && ((a2asize < 1<<17) || (sndsize*sendcount < 1<<12))) {
    /* just send as fast as we can if we have less than 8 peers, if the
     * total communicated size is smaller than 1<<17 *and* if we don't
     * have eager messages (msgsize < 1<<13) */
    alg = NBC_A2A_LINEAR;
  } else if(a2asize < (1<<12)*(unsigned int)p) {
    /*alg = NBC_A2A_DISS;*/
    alg = NBC_A2A_LINEAR;
  } else
    alg = NBC_A2A_LINEAR; /*NBC_A2A_PAIRWISE;*/

  /* allocate temp buffer if we need one */
  if (alg == NBC_A2A_INPLACE) {
    span = opal_datatype_span(&recvtype->super, recvcount, &gap);
    tmpbuf = malloc(span);
    if (OPAL_UNLIKELY(NULL == tmpbuf)) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }
  } else if (alg == NBC_A2A_DISS) {
    /* only A2A_DISS needs buffers */
    if(NBC_Type_intrinsic(sendtype)) {
      datasize = sndext * sendcount;
    } else {
      res = PMPI_Pack_size (sendcount, sendtype, comm, &datasize);
      if (MPI_SUCCESS != res) {
        NBC_Error("MPI Error in PMPI_Pack_size() (%i)", res);
        return res;
      }
    }

    /* allocate temporary buffers */
    if ((p & 1) == 0) {
      tmpbuf = malloc (datasize * p * 2);
    } else {
      /* we cannot divide p by two, so alloc more to be safe ... */
      tmpbuf = malloc (datasize * (p / 2 + 1) * 2 * 2);
    }

    if (OPAL_UNLIKELY(NULL == tmpbuf)) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }

    /* phase 1 - rotate n data blocks upwards into the tmpbuffer */
#if OPAL_CUDA_SUPPORT
    if (NBC_Type_intrinsic(sendtype) && !(opal_cuda_check_bufs((char *)sendbuf, (char *)recvbuf))) {
#else
    if (NBC_Type_intrinsic(sendtype)) {
#endif /* OPAL_CUDA_SUPPORT */
      /* contiguous - just copy (1st copy) */
      memcpy (tmpbuf, (char *) sendbuf + datasize * rank, datasize * (p - rank));
      if (rank != 0) {
        memcpy ((char *) tmpbuf + datasize * (p - rank), sendbuf, datasize * rank);
      }
    } else {
      int pos=0;

      /* non-contiguous - pack */
      res = PMPI_Pack ((char *) sendbuf + rank * sendcount * sndext, (p - rank) * sendcount, sendtype, tmpbuf,
                      (p - rank) * datasize, &pos, comm);
      if (OPAL_UNLIKELY(MPI_SUCCESS != res)) {
        NBC_Error("MPI Error in PMPI_Pack() (%i)", res);
        free(tmpbuf);
        return res;
      }

      if (rank != 0) {
        pos = 0;
        res = PMPI_Pack(sendbuf, rank * sendcount, sendtype, (char *) tmpbuf + datasize * (p - rank),
                       rank * datasize, &pos, comm);
        if (OPAL_UNLIKELY(MPI_SUCCESS != res)) {
          NBC_Error("MPI Error in PMPI_Pack() (%i)", res);
          free(tmpbuf);
          return res;
        }
      }
    }
  }

#ifdef NBC_CACHE_SCHEDULE
  /* search schedule in communicator specific tree */
  search.sendbuf = sendbuf;
  search.sendcount = sendcount;
  search.sendtype = sendtype;
  search.recvbuf = recvbuf;
  search.recvcount = recvcount;
  search.recvtype = recvtype;
  found = (NBC_Alltoall_args *) hb_tree_search ((hb_tree *) libnbc_module->NBC_Dict[NBC_ALLTOALL], &search);
  if (NULL == found) {
#endif
    /* not found - generate new schedule */
    schedule = OBJ_NEW(NBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
      free(tmpbuf);
      return OMPI_ERR_OUT_OF_RESOURCE;
    }

    if (!inplace) {
      /* copy my data to receive buffer */
      rbuf = (char *) recvbuf + rank * recvcount * rcvext;
      sbuf = (char *) sendbuf + rank * sendcount * sndext;
      res = NBC_Sched_copy (sbuf, false, sendcount, sendtype,
                            rbuf, false, recvcount, recvtype, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        free(tmpbuf);
        return res;
      }
    }

    switch(alg) {
      case NBC_A2A_INPLACE:
        res = a2a_sched_inplace(rank, p, schedule, recvbuf, recvcount, recvtype, rcvext, gap, comm);
        break;
      case NBC_A2A_LINEAR:
        res = a2a_sched_linear(rank, p, sndext, rcvext, schedule, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
        break;
      case NBC_A2A_DISS:
        res = a2a_sched_diss(rank, p, sndext, rcvext, schedule, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, tmpbuf);
        break;
      case NBC_A2A_PAIRWISE:
        res = a2a_sched_pairwise(rank, p, sndext, rcvext, schedule, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
        break;
    }

    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      free(tmpbuf);
      return res;
    }

    res = NBC_Sched_commit(schedule);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      free(tmpbuf);
      return res;
    }

#ifdef NBC_CACHE_SCHEDULE
    /* save schedule to tree */
    args = (NBC_Alltoall_args *) malloc (sizeof (args));
    if (NULL != args) {
      args->sendbuf = sendbuf;
      args->sendcount = sendcount;
      args->sendtype = sendtype;
      args->recvbuf = recvbuf;
      args->recvcount = recvcount;
      args->recvtype = recvtype;
      args->schedule = schedule;
      res = hb_tree_insert ((hb_tree *) libnbc_module->NBC_Dict[NBC_ALLTOALL], args, args, 0);
      if (0 == res) {
        OBJ_RETAIN(schedule);

        /* increase number of elements for A2A */
        if (++libnbc_module->NBC_Dict_size[NBC_ALLTOALL] > NBC_SCHED_DICT_UPPER) {
          NBC_SchedCache_dictwipe ((hb_tree *) libnbc_module->NBC_Dict[NBC_ALLTOALL],
                                   &libnbc_module->NBC_Dict_size[NBC_ALLTOALL]);
        }
      } else {
        NBC_Error("error in dict_insert() (%i)", res);
        free (args);
      }
    }
  } else {
    /* found schedule */
    schedule = found->schedule;
    OBJ_RETAIN(schedule);
  }
#endif

  res = NBC_Schedule_request(schedule, comm, libnbc_module, persistent, request, tmpbuf);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return res;
  }

  return OMPI_SUCCESS;
}

int ompi_coll_libnbc_ialltoall(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                               MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                               struct mca_coll_base_module_2_2_0_t *module) {
    int res = nbc_ialltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                            comm, request, module, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }
  
    res = NBC_Start(*(ompi_coll_libnbc_request_t **)request);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        NBC_Return_handle ((ompi_coll_libnbc_request_t *)request);
        *request = &ompi_request_null.request;
        return res;
    }

    return OMPI_SUCCESS;
}

static int nbc_ialltoall_inter (const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                struct mca_coll_base_module_2_2_0_t *module, bool persistent)
{
  int res, rsize;
  MPI_Aint sndext, rcvext;
  NBC_Schedule *schedule;
  char *rbuf, *sbuf;
  ompi_coll_libnbc_module_t *libnbc_module = (ompi_coll_libnbc_module_t*) module;

  rsize = ompi_comm_remote_size (comm);

  res = ompi_datatype_type_extent (sendtype, &sndext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  res = ompi_datatype_type_extent (recvtype, &rcvext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  schedule = OBJ_NEW(NBC_Schedule);
  if (OPAL_UNLIKELY(NULL == schedule)) {
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  for (int i = 0; i < rsize; i++) {
    /* post all sends */
    sbuf = (char *) sendbuf + i * sendcount * sndext;
    res = NBC_Sched_send (sbuf, false, sendcount, sendtype, i, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      break;
    }

    /* post all receives */
    rbuf = (char *) recvbuf + i * recvcount * rcvext;
    res = NBC_Sched_recv (rbuf, false, recvcount, recvtype, i, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      break;
    }
  }

  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    return res;
  }

  res = NBC_Sched_commit (schedule);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    return res;
  }

  res = NBC_Schedule_request(schedule, comm, libnbc_module, persistent, request, NULL);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    return res;
  }

  return OMPI_SUCCESS;
}

int ompi_coll_libnbc_ialltoall_inter (const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
				      MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
				      struct mca_coll_base_module_2_2_0_t *module) {
    int res = nbc_ialltoall_inter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                                  comm, request, module, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }
  
    res = NBC_Start(*(ompi_coll_libnbc_request_t **)request);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        NBC_Return_handle ((ompi_coll_libnbc_request_t *)request);
        *request = &ompi_request_null.request;
        return res;
    }

    return OMPI_SUCCESS;
}

static inline int a2a_sched_pairwise(int rank, int p, MPI_Aint sndext, MPI_Aint rcvext, NBC_Schedule* schedule,
                                     const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                     MPI_Datatype recvtype, MPI_Comm comm) {
  int res;

  if (p < 2) {
    return OMPI_SUCCESS;
  }

  for (int r = 1 ; r < p ; ++r) {
    int sndpeer = (rank + r) % p;
    int rcvpeer = (rank - r + p) % p;

    char *rbuf = (char *) recvbuf + rcvpeer * recvcount * rcvext;
    res = NBC_Sched_recv (rbuf, false, recvcount, recvtype, rcvpeer, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }

    char *sbuf = (char *) sendbuf + sndpeer * sendcount * sndext;
    res = NBC_Sched_send (sbuf, false, sendcount, sendtype, sndpeer, schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
  }

  return OMPI_SUCCESS;
}

static inline int a2a_sched_linear(int rank, int p, MPI_Aint sndext, MPI_Aint rcvext, NBC_Schedule* schedule,
                                   const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                   MPI_Datatype recvtype, MPI_Comm comm) {
  int res;

  for (int r = 0 ; r < p ; ++r) {
    /* easy algorithm */
    if (r == rank) {
      continue;
    }

    char *rbuf = (char *) recvbuf + r * recvcount * rcvext;
    res = NBC_Sched_recv (rbuf, false, recvcount, recvtype, r, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }

    char *sbuf = (char *) sendbuf + r * sendcount * sndext;
    res = NBC_Sched_send (sbuf, false, sendcount, sendtype, r, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
  }

  return OMPI_SUCCESS;
}

static inline int a2a_sched_diss(int rank, int p, MPI_Aint sndext, MPI_Aint rcvext, NBC_Schedule* schedule,
                                 const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                 MPI_Datatype recvtype, MPI_Comm comm, void* tmpbuf) {
  int res, speer, rpeer, datasize, offset, virtp;
  char *rbuf, *rtmpbuf, *stmpbuf;

  if (p < 2) {
    return OMPI_SUCCESS;
  }

  if(NBC_Type_intrinsic(sendtype)) {
    datasize = sndext*sendcount;
  } else {
    res = PMPI_Pack_size(sendcount, sendtype, comm, &datasize);
    if (MPI_SUCCESS != res) {
      NBC_Error("MPI Error in PMPI_Pack_size() (%i)", res);
      return res;
    }
  }

  /* allocate temporary buffers */
  if ((p & 1) == 0) {
    rtmpbuf = (char *)tmpbuf + datasize * p;
    stmpbuf = (char *)tmpbuf + datasize * (p + p / 2);
  } else {
    /* we cannot divide p by two, so alloc more to be safe ... */
    virtp = (p / 2 + 1) * 2;
    rtmpbuf = (char *)tmpbuf + datasize * p;
    stmpbuf = (char *)tmpbuf + datasize * (p + virtp / 2);
  }

  /* phase 2 - communicate */
  for (int r = 1; r < p; r <<= 1) {
    offset = 0;
    for (int i = 1 ; i < p; ++i) {
      /* test if bit r is set in rank number i */
      if (i & r) {
        /* copy data to sendbuffer (2nd copy) - could be avoided using iovecs */
        /*printf("[%i] round %i: copying element %i to buffer %lu\n", rank, r, i, (unsigned long)(stmpbuf+offset));*/
        res = NBC_Sched_copy((void *)(intptr_t)(i * datasize), true, datasize, MPI_BYTE, stmpbuf + offset -
                             (intptr_t)tmpbuf, true, datasize, MPI_BYTE, schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
          return res;
        }
        offset += datasize;
      }
    }

    speer = (rank + r) % p;
    /* add p because modulo does not work with negative values */
    rpeer = ((rank - r) + p) % p;

    res = NBC_Sched_recv (rtmpbuf - (intptr_t)tmpbuf, true, offset, MPI_BYTE, rpeer, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }

    res = NBC_Sched_send (stmpbuf - (intptr_t)tmpbuf, true, offset, MPI_BYTE, speer, schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }

    /* unpack from buffer */
    offset = 0;
    for (int i = 1; i < p; ++i) {
      /* test if bit r is set in rank number i */
      if (i & r) {
        /* copy data to tmpbuffer (3rd copy) - could be avoided using iovecs */
        res = NBC_Sched_copy (rtmpbuf + offset - (intptr_t)tmpbuf, true, datasize, MPI_BYTE,
                              (void *)(intptr_t)(i * datasize), true, datasize, MPI_BYTE, schedule,
                              false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
          return res;
        }

        offset += datasize;
      }
    }
  }

  /* phase 3 - reorder - data is now in wrong order in tmpbuf - reorder it into recvbuf */
  for (int i = 0 ; i < p; ++i) {
    rbuf = (char *) recvbuf + ((rank - i + p) % p) * recvcount * rcvext;
    res = NBC_Sched_unpack ((void *)(intptr_t) (i * datasize), true, recvcount, recvtype, rbuf, false, schedule,
                            false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
  }

  return OMPI_SUCCESS;
}

static inline int a2a_sched_inplace(int rank, int p, NBC_Schedule* schedule, void* buf, int count,
                                   MPI_Datatype type, MPI_Aint ext, ptrdiff_t gap, MPI_Comm comm) {
  int res;

  for (int i = 1 ; i < (p+1)/2 ; i++) {
    int speer = (rank + i) % p;
    int rpeer = (rank + p - i) % p;
    char *sbuf = (char *) buf + speer * count * ext;
    char *rbuf = (char *) buf + rpeer * count * ext;

    res = NBC_Sched_copy (rbuf, false, count, type,
                          (void *)(-gap), true, count, type,
                          schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
    res = NBC_Sched_send (sbuf, false , count, type, speer, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
    res = NBC_Sched_recv (rbuf, false , count, type, rpeer, schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }

    res = NBC_Sched_send ((void *)(-gap), true, count, type, rpeer, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
    res = NBC_Sched_recv (sbuf, false, count, type, speer, schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
  }
  if (0 == (p%2)) {
    int peer = (rank + p/2) % p;

    char *tbuf = (char *) buf + peer * count * ext;
    res = NBC_Sched_copy (tbuf, false, count, type,
                          (void *)(-gap), true, count, type,
                          schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
    res = NBC_Sched_send ((void *)(-gap), true , count, type, peer, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
    res = NBC_Sched_recv (tbuf, false , count, type, peer, schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
  }

  return OMPI_SUCCESS;
}

int ompi_coll_libnbc_alltoall_init (const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                    MPI_Datatype recvtype, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                    struct mca_coll_base_module_2_2_0_t *module) {

  // information derived from input parameters (used only in this function)
  int comm_size, rank;
  MPI_Aint sendext, recvext;
  size_t recvtypesize;

  rank = ompi_comm_rank (comm);
  comm_size = ompi_comm_size (comm);

  // general purpose temporary stack variables (used only in this function)
  int res, pof2, nrounds, count, dst, src, round, bits[comm_size];
  unsigned int j;
  char inplace;
  size_t tmpbufsize;
  unsigned int mask = 0xFFFFFFFF;
  ompi_coll_libnbc_module_t *libnbc_module = (ompi_coll_libnbc_module_t*) module;

  // local pointers to long-term data that will be attached to the request
  NBC_Schedule *schedule;
  int sendranks[comm_size];          // needed for schedule
  int recvranks[comm_size];          // needed for schedule
  int sendblocks[comm_size];         // needed for schedule
  int recvblocks[comm_size];         // needed for schedule
  MPI_Aint recvindex[comm_size];     // needed for schedule
  MPI_Aint sendindex[comm_size];     // needed for schedule
  MPI_Datatype sendtypes[comm_size]; // needed for schedule
  MPI_Datatype recvtypes[comm_size]; // needed for schedule
  void *tmpbuf = NULL;
  MPI_Datatype *sendblocktype;  // points to part of tmpbuf
  MPI_Datatype *recvblocktype;  // points to part of tmpbuf
  char *interbuf;               // points to part of tmpbuf

  // derive information directly from input parameters

  res = ompi_datatype_type_extent(sendtype, &sendext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  res = ompi_datatype_type_extent(recvtype, &recvext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  res = ompi_datatype_type_size(recvtype, &recvtypesize);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_size() (%i)", res);
    return res;
  }
  recvtypesize *= recvcount;

  NBC_IN_PLACE(sendbuf, recvbuf, inplace);

  // calculate additional information needed to create long-term data storage

  // calculate number of rounds in log_2 loop
  nrounds = 0;
  for (pof2 = 1; pof2 < comm_size; pof2 <<= 1)
    nrounds++;

  // compute number of 1-bits for all j, 0<=j<size
  bits[0] = 0;
  for (j = 1; j < comm_size; ++j)
    bits[j] = bits[j>>1]+(j&0x1);

  // calculate total size of tmpbuf
  tmpbufsize = sizeof(MPI_Datatype)*nrounds*2 + comm_size*recvtypesize;

  // create long-term data storage objects
  // * tmpbuf contains datatype handles and intermediate data buffer

  tmpbuf = (char *)malloc(tmpbufsize);
  if (OPAL_UNLIKELY(NULL == tmpbuf)) {
    return OMPI_ERR_OUT_OF_RESOURCE;
  }
  sendblocktype = (MPI_Datatype*)tmpbuf;
  recvblocktype = (MPI_Datatype*)((char*)tmpbuf + sizeof(MPI_Datatype)*nrounds);
  interbuf      = (void*)((char*)tmpbuf + sizeof(MPI_Datatype)*nrounds*2);

  // calculate data for long-term data storage in tmpbuf

  // calculate data for each round of the log_2 loop
  for (pof2 = 1, round = 0; pof2 < comm_size; pof2 <<= 1, ++round) {
    count = 0;
    j = pof2;

    do {
      // bit Set
      dst = (rank - j + comm_size) % comm_size;
      src = (rank + j) % comm_size;

      if ((bits[j&mask]&0x1) == 0x1) {
        // to recvbuf
        recvblocks[count] = recvcount;
        recvindex[count] = (MPI_Aint)((char *)recvbuf + src*recvcount*recvext);
        recvtypes[count] = recvtype;

        if ((j&mask) == j) {
          // from sendbuf
          sendblocks[count] = sendcount;
          sendindex[count] = (MPI_Aint)((char *)sendbuf + dst*sendcount*sendext);
          sendtypes[count] = sendtype;
        } else {
          // from inter
          sendblocks[count] = recvtypesize;
          sendindex[count] = (MPI_Aint)(interbuf + j*recvtypesize);

          sendtypes[count] = MPI_BYTE;
        }
      } else {
        // to inter
        recvblocks[count] = recvtypesize;
        recvindex[count] = (MPI_Aint)(interbuf + j*recvtypesize);
        recvtypes[count] = MPI_BYTE;

        if ((j&mask) == j) {
          // from sendbuf
          sendblocks[count] = sendcount;
          sendindex[count] =
                  (MPI_Aint)((char *)sendbuf + dst*sendcount*sendext);
          sendtypes[count] = sendtype;
        } else {
          // from recv
          sendblocks[count] = recvcount;
          sendindex[count] =
                  (MPI_Aint)((char *)recvbuf + src*recvcount*recvext);
          sendtypes[count] = recvtype;
        }
      }
      count++;
      j++;

      if ((j&pof2) != pof2)
        j += pof2;

    } while(j < comm_size);

    MPI_Type_create_struct(count, sendblocks, sendindex, sendtypes, &sendblocktype[round]);
    MPI_Type_commit(&sendblocktype[round]);

    MPI_Type_create_struct(count, recvblocks, recvindex, recvtypes, &recvblocktype[round]);
    MPI_Type_commit(&recvblocktype[round]);

    sendranks[round] = (rank - pof2 + comm_size) % comm_size;
    recvranks[round] = (rank + pof2) % comm_size;

    mask <<= 1; // shift in zero bit

  } // end of for loop to calculate data for each round of the log_2 loop

  // create long-term data storage objects
  // * schedule contains all scheduled steps for this operation

  schedule = OBJ_NEW(NBC_Schedule);
  if (OPAL_UNLIKELY(NULL == schedule)) {
    free(tmpbuf);
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  // Modified first step
  if(!inplace) {
    // MPI_Sendrecv((char *)sendbuf + rank*sendcount*sndext, sendcount, sendtype, rank, BRUCK,
    //              (char *)recvbuf + rank*recvcount*rcvext, recvcount, recvtype, rank, BRUCK,
    //              comm, MPI_STATUS_IGNORE);
    res = NBC_Sched_copy ((char *) sendbuf + rank * sendcount * sendext, false, sendcount, sendtype,
                          (char *) recvbuf + rank * recvcount * recvext, false, recvcount, recvtype,
                          schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      free(tmpbuf);
      return res;
    }
  }

  // Modified second step
  for (round = 0; round < nrounds; ++round) {
    // MPI_Sendrecv(MPI_BOTTOM, 1, sendblocktype[round], sendrank[round], BRUCK,
    //             MPI_BOTTOM, 1, recvblocktype[round], recvrank[round], BRUCK,
    //             comm, MPI_STATUS_IGNORE);

    res = NBC_Sched_recv (MPI_BOTTOM, false, 1, recvblocktype[round], recvranks[round], schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      free(tmpbuf);
      return res;
    }

    res = NBC_Sched_send (MPI_BOTTOM, false, 1, sendblocktype[round], sendranks[round], schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      free(tmpbuf);
      return res;
    }
  }

  res = NBC_Sched_commit(schedule);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return res;
  }

  res = NBC_Schedule_request(schedule, comm, libnbc_module, true, request, tmpbuf);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return res;
  }

  return OMPI_SUCCESS;
}

int ompi_coll_libnbc_alltoall_inter_init (const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                          MPI_Datatype recvtype, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                          struct mca_coll_base_module_2_2_0_t *module) {
    int res = nbc_ialltoall_inter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                                  comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}
