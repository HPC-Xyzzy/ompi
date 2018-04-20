/* -*- Mode: C; c-basic-offset:2 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2006      The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2006      The Technical University of Chemnitz. All
 *                         rights reserved.
 * Copyright (c) 2014-2017 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2015-2017 Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2017      IBM Corporation.  All rights reserved.
 * Copyright (c) 2018      FUJITSU LIMITED.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *
 */
#include "nbc_internal.h"

static inline int a2av_sched_linear(int rank, int p, NBC_Schedule *schedule,
                                    const void *sendbuf, const int *sendcounts,
                                    const int *sdispls, MPI_Aint sndext, MPI_Datatype sendtype,
                                    void *recvbuf, const int *recvcounts,
                                    const int *rdispls, MPI_Aint rcvext, MPI_Datatype recvtype);

static inline int a2av_sched_pairwise(int rank, int p, NBC_Schedule *schedule,
                                      const void *sendbuf, const int *sendcounts, const int *sdispls,
                                      MPI_Aint sndext, MPI_Datatype sendtype,
                                      void *recvbuf, const int *recvcounts, const int *rdispls,
                                      MPI_Aint rcvext, MPI_Datatype recvtype);

static inline int a2av_sched_inplace(int rank, int p, NBC_Schedule *schedule,
                                    void *buf, const int *counts, const int *displs,
                                    MPI_Aint ext, MPI_Datatype type, ptrdiff_t gap);

/* an alltoallv schedule can not be cached easily because the contents
 * ot the recvcounts array may change, so a comparison of the address
 * would not be sufficient ... we simply do not cache it */

/* simple linear Alltoallv */
static int nbc_ialltoallv(const void* sendbuf, const int *sendcounts, const int *sdispls,
                          MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *rdispls,
                          MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                          struct mca_coll_base_module_2_2_0_t *module, bool persistent)
{
  int rank, p, res;
  MPI_Aint sndext, rcvext;
  NBC_Schedule *schedule;
  char *rbuf, *sbuf, inplace;
  ptrdiff_t gap, span;
  void * tmpbuf = NULL;
  ompi_coll_libnbc_module_t *libnbc_module = (ompi_coll_libnbc_module_t*) module;

  NBC_IN_PLACE(sendbuf, recvbuf, inplace);

  rank = ompi_comm_rank (comm);
  p = ompi_comm_size (comm);

  res = ompi_datatype_type_extent (recvtype, &rcvext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  /* copy data to receivbuffer */
  if (inplace) {
    int count = 0;
    for (int i = 0; i < p; i++) {
      if (recvcounts[i] > count) {
        count = recvcounts[i];
      }
    }
    span = opal_datatype_span(&recvtype->super, count, &gap);
    if (OPAL_UNLIKELY(0 == span)) {
      return nbc_get_noop_request(persistent, request);
    }
    tmpbuf = malloc(span);
    if (OPAL_UNLIKELY(NULL == tmpbuf)) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }
    sendcounts = recvcounts;
    sdispls = rdispls;
  } else {
    res = ompi_datatype_type_extent (sendtype, &sndext);
    if (MPI_SUCCESS != res) {
      NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
      return res;
    }
  }

  schedule = OBJ_NEW(NBC_Schedule);
  if (OPAL_UNLIKELY(NULL == schedule)) {
    free(tmpbuf);
    return OMPI_ERR_OUT_OF_RESOURCE;
  }


  if (!inplace && sendcounts[rank] != 0) {
    rbuf = (char *) recvbuf + rdispls[rank] * rcvext;
    sbuf = (char *) sendbuf + sdispls[rank] * sndext;
    res = NBC_Sched_copy (sbuf, false, sendcounts[rank], sendtype,
                          rbuf, false, recvcounts[rank], recvtype, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      return res;
    }
  }

  if (inplace) {
    res = a2av_sched_inplace(rank, p, schedule, recvbuf, recvcounts,
                                 rdispls, rcvext, recvtype, gap);
  } else {
    res = a2av_sched_linear(rank, p, schedule,
                            sendbuf, sendcounts, sdispls, sndext, sendtype,
                            recvbuf, recvcounts, rdispls, rcvext, recvtype);
  }
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return res;
  }

  res = NBC_Sched_commit (schedule);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return res;
  }

  res = NBC_Schedule_request(schedule, comm, libnbc_module, persistent, request, tmpbuf);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return res;
  }

  return OMPI_SUCCESS;
}

int ompi_coll_libnbc_ialltoallv(const void* sendbuf, const int *sendcounts, const int *sdispls,
                                MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *rdispls,
                                MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                struct mca_coll_base_module_2_2_0_t *module) {
    int res = nbc_ialltoallv(sendbuf, sendcounts, sdispls, sendtype,
                             recvbuf, recvcounts, rdispls, recvtype,
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

/* simple linear Alltoallv */
static int nbc_ialltoallv_inter (const void* sendbuf, const int *sendcounts, const int *sdispls,
                                 MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *rdispls,
                                 MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                 struct mca_coll_base_module_2_2_0_t *module, bool persistent)
{
  int res, rsize;
  MPI_Aint sndext, rcvext;
  NBC_Schedule *schedule;
  ompi_coll_libnbc_module_t *libnbc_module = (ompi_coll_libnbc_module_t*) module;


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

  rsize = ompi_comm_remote_size (comm);

  schedule = OBJ_NEW(NBC_Schedule);
  if (OPAL_UNLIKELY(NULL == schedule)) {
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  for (int i = 0; i < rsize; i++) {
    /* post all sends */
    if (sendcounts[i] != 0) {
      char *sbuf = (char *) sendbuf + sdispls[i] * sndext;
      res = NBC_Sched_send (sbuf, false, sendcounts[i], sendtype, i, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
      }
    }
    /* post all receives */
    if (recvcounts[i] != 0) {
      char *rbuf = (char *) recvbuf + rdispls[i] * rcvext;
      res = NBC_Sched_recv (rbuf, false, recvcounts[i], recvtype, i, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
      }
    }
  }

  res = NBC_Sched_commit(schedule);
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

int ompi_coll_libnbc_ialltoallv_inter (const void* sendbuf, const int *sendcounts, const int *sdispls,
				       MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *rdispls,
				       MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
				       struct mca_coll_base_module_2_2_0_t *module) {
    int res = nbc_ialltoallv_inter(sendbuf, sendcounts, sdispls, sendtype,
                                   recvbuf, recvcounts, rdispls, recvtype,
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

__opal_attribute_unused__
static inline int a2av_sched_linear(int rank, int p, NBC_Schedule *schedule,
                                    const void *sendbuf, const int *sendcounts, const int *sdispls,
                                    MPI_Aint sndext, MPI_Datatype sendtype,
                                    void *recvbuf, const int *recvcounts, const int *rdispls,
                                    MPI_Aint rcvext, MPI_Datatype recvtype) {
  int res;

  for (int i = 0 ; i < p ; ++i) {
    if (i == rank) {
      continue;
    }

    /* post send */
    if (sendcounts[i] != 0) {
      char *sbuf = ((char *) sendbuf) + (sdispls[i] * sndext);
      res = NBC_Sched_send(sbuf, false, sendcounts[i], sendtype, i, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }

    /* post receive */
    if (recvcounts[i] != 0) {
      char *rbuf = ((char *) recvbuf) + (rdispls[i] * rcvext);
      res = NBC_Sched_recv(rbuf, false, recvcounts[i], recvtype, i, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
  }

  return OMPI_SUCCESS;
}

__opal_attribute_unused__
static inline int a2av_sched_pairwise(int rank, int p, NBC_Schedule *schedule,
                                      const void *sendbuf, const int *sendcounts, const int *sdispls,
                                      MPI_Aint sndext, MPI_Datatype sendtype,
                                      void *recvbuf, const int *recvcounts, const int *rdispls,
                                      MPI_Aint rcvext, MPI_Datatype recvtype) {
  int res;

  for (int i = 1 ; i < p ; ++i) {
    int sndpeer = (rank + i) % p;
    int rcvpeer = (rank + p - i) %p;

    /* post send */
    if (sendcounts[sndpeer] != 0) {
      char *sbuf = ((char *) sendbuf) + (sdispls[sndpeer] * sndext);
      res = NBC_Sched_send(sbuf, false, sendcounts[sndpeer], sendtype, sndpeer, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }

    /* post receive */
    if (recvcounts[rcvpeer] != 0) {
      char *rbuf = ((char *) recvbuf) + (rdispls[rcvpeer] * rcvext);
      res = NBC_Sched_recv(rbuf, false, recvcounts[rcvpeer], recvtype, rcvpeer, schedule, true);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
  }

  return OMPI_SUCCESS;
}

static inline int a2av_sched_inplace(int rank, int p, NBC_Schedule *schedule,
                                    void *buf, const int *counts, const int *displs,
                                    MPI_Aint ext, MPI_Datatype type, ptrdiff_t gap) {
  int res;

  for (int i = 1; i < (p+1)/2; i++) {
    int speer = (rank + i) % p;
    int rpeer = (rank + p - i) % p;
    char *sbuf = (char *) buf + displs[speer] * ext;
    char *rbuf = (char *) buf + displs[rpeer] * ext;

    if (0 != counts[rpeer]) {
      res = NBC_Sched_copy (rbuf, false, counts[rpeer], type,
                            (void *)(-gap), true, counts[rpeer], type,
                            schedule, true);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
    if (0 != counts[speer]) {
      res = NBC_Sched_send (sbuf, false , counts[speer], type, speer, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
    if (0 != counts[rpeer]) {
      res = NBC_Sched_recv (rbuf, false , counts[rpeer], type, rpeer, schedule, true);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }

    if (0 != counts[rpeer]) {
      res = NBC_Sched_send ((void *)(-gap), true, counts[rpeer], type, rpeer, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
    if (0 != counts[speer]) {
      res = NBC_Sched_recv (sbuf, false, counts[speer], type, speer, schedule, true);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
  }
  if (0 == (p%2)) {
    int peer = (rank + p/2) % p;

    char *tbuf = (char *) buf + displs[peer] * ext;
    res = NBC_Sched_copy (tbuf, false, counts[peer], type,
                          (void *)(-gap), true, counts[peer], type,
                          schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
    if (0 != counts[peer]) {
      res = NBC_Sched_send ((void *)(-gap), true , counts[peer], type, peer, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
      res = NBC_Sched_recv (tbuf, false , counts[peer], type, peer, schedule, true);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
  }

  return OMPI_SUCCESS;
}

/* Helper type constructor: create a vector inside a bounded buffer */
int MPIX_Type_create_vector_bounded(int bound, int block, int stride, MPI_Datatype oldtype, MPI_Datatype *newtype) {
  int c, l;

  c = bound/stride;
  l = bound%stride;

  if (l>=block) {
    c++;
    l = 0;
  }

  if (c>0) {
    if (l>0) {
      MPI_Datatype dt[2];
      int bl[2];
      MPI_Aint di[2], lb;

      MPI_Type_get_extent(oldtype, &lb, &di[1]);

      bl[0] = 1;
      bl[1] = l;

      di[0] = 0;
      di[1] *= c*stride;

      MPI_Type_vector(c, block, stride, oldtype, &dt[0]);
      dt[1] = oldtype;

      MPI_Type_create_struct(2, bl, di, dt, newtype);
    } else {
      MPI_Type_vector(c, block, stride, oldtype, newtype);
    }
  } else {
    MPI_Type_contiguous(l, oldtype, newtype);
  }

  return OMPI_SUCCESS;
}

const int BRUCK = -333;

int ompi_coll_libnbc_alltoallv_init(const void* sendbuf, const int *sendcounts, const int *sdispls, MPI_Datatype sendtype,
                                          void* recvbuf, const int *recvcounts, const int *rdispls, MPI_Datatype recvtype,
                                    struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                    struct mca_coll_base_module_2_2_0_t *module) {
//    int res = nbc_ialltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
//                             comm, request, module, true);
//  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
//    return res;
//  }

/////////////////////////////

  // information derived from input parameters (used only in this function)
  int comm_size, rank;
  MPI_Aint sendext, recvext;
  size_t recvtypesize, sendtypesize;
  int packsize;
  int minmaxcount[2], sendcount, recvcount; // total counts

  rank = ompi_comm_rank (comm);
  comm_size = ompi_comm_size (comm);

  // general purpose temporary stack variables (used only in this function)
  int res, pof2, nrounds, count, dst, src, round, bits[comm_size];
  unsigned int j, w;
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
  char *interbuf[2];            // points to part of tmpbuf

	// counts for intermediate blocks
	int intercount[comm_size];
	int ic[comm_size];

	// type for counts to be received in round k
	MPI_Datatype counttype;

  // derive information directly from input parameters

  res = ompi_datatype_type_extent(sendtype, &sendext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  res = ompi_datatype_type_extent(recvtype, &recvext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
  }

  res = ompi_datatype_type_size(sendtype, &sendtypesize);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_size() (%i)", res);
    return res;
  }

  res = ompi_datatype_type_size(recvtype, &recvtypesize);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_size() (%i)", res);
    return res;
  }

  sendcount = 0;
  recvcount = 0;
  minmaxcount[0] = 0;
  minmaxcount[1] = 0;
  for (j = 0; j < comm_size; ++j) {
    // find total send count and total recv count
    sendcount += sendcounts[j];
    recvcount += recvcounts[j];

    // find local min and max send count
    if (minmaxcount[0] < -sendcounts[j])
      minmaxcount[0] = -sendcounts[j];
    if (minmaxcount[1] < sendcounts[j])
      minmaxcount[1] = sendcounts[j];
  }

/*  request->zero = 0;
  if (sendcount==0&&recvcount==0) {
    request->zero = 1;
    return OMPI_SUCCESS;
  }*/

  MPI_Allreduce(MPI_IN_PLACE,minmaxcount,2,MPI_INT,MPI_MAX,comm);
  recvtypesize *= minmaxcount[1];

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
  tmpbufsize = sizeof(MPI_Datatype)*nrounds*2 + comm_size*recvtypesize*2;

  // create long-term data storage objects
  // * tmpbuf contains datatype handles and intermediate data buffer

  tmpbuf = (char *)malloc(tmpbufsize);
  if (OPAL_UNLIKELY(NULL == tmpbuf)) {
    return OMPI_ERR_OUT_OF_RESOURCE;
  }
  sendblocktype = (MPI_Datatype*)        tmpbuf;
  recvblocktype = (MPI_Datatype*)((char*)tmpbuf      + sizeof(MPI_Datatype)*nrounds);
  interbuf[0]   = (void*)        ((char*)tmpbuf      + sizeof(MPI_Datatype)*nrounds*2);
  interbuf[1]   = (void*)        ((char*)interbuf[0] + comm_size*recvtypesize);

/*	request->sendbuf = sendbuf;
	request->recvbuf = recvbuf;
	// two intermediate buffers (recvbuf cannot be used)
	request->interbuf[0] = (char *)malloc(2*comm_size*recvtypesize);
	request->interbuf[1] = request->interbuf[0] + comm_size*recvtypesize;

	request->sendcount = sendcounts[rank];
	request->recvcount = recvcounts[rank];
	request->senddispl = sdispls[rank];
	request->recvdispl = rdispls[rank];

	request->sendextent = sendext;
	request->recvextent = recvext;
	request->sendtype = sendtype;
	request->recvtype = recvtype;

	request->comm = comm;

	request->sendblocktype = (MPI_Datatype *)malloc(sizeof(MPI_Datatype)*nrounds);
	request->recvblocktype = (MPI_Datatype *)malloc(sizeof(MPI_Datatype)*nrounds);
	request->sendrank = (int *)malloc(sizeof(int)*nrounds);
	request->recvrank = (int *)malloc(sizeof(int)*nrounds);
*/

  // calculate data for long-term data storage in tmpbuf

  // calculate data for each round of the log_2 loop
  for (pof2 = 1, round = 0; pof2 < comm_size; pof2 <<= 1, ++round) {
    count = 0;
    j = pof2;

    // prepare send information for this round

    do {
      if ((j&mask) == j) {
        // init: from sendbuf
        dst = (rank - j + comm_size) % comm_size;
        sendblocks[count] = sendcounts[dst];
        sendindex[count] = (MPI_Aint)((char*)sendbuf + sdispls[dst]*sendext);
        sendtypes[count] = sendtype;
        intercount[j] = sendcounts[dst]*sendtypesize;
      } else {
        // from inter
        w = bits[j&mask]%2;
        sendblocks[count] = intercount[j];
        sendindex[count] = (MPI_Aint)(interbuf[w] + j*recvtypesize);
        sendtypes[count] = MPI_BYTE;
      }
      count++;
      j++;
      if ((j&pof2) != pof2)
        j += pof2;
    } while (j < comm_size);

    // prepare receive information for this round

    MPIX_Type_create_vector_bounded((comm_size-pof2), pof2, (pof2<<1), MPI_INT, &counttype);
    MPI_Type_commit(&counttype);
    MPI_Pack_size(1, counttype, comm, &packsize);

    dst = (rank - pof2 + comm_size) % comm_size;
    src = (rank + pof2) % comm_size;

    MPI_Sendrecv(intercount + pof2, 1,        counttype,  dst, BRUCK,
                 ic,                packsize, MPI_PACKED, src, BRUCK,
                 comm, MPI_STATUS_IGNORE);

    int pos = 0;
    MPI_Unpack(ic, packsize, &pos, intercount + pof2, 1, counttype, comm);

    MPI_Type_free(&counttype);

    count = 0;
    j = pof2;

    int nextmask = mask<<1;
    do {
      // bit set
      if ((j&(~nextmask)) == j) {
        // done: to recvbuf
        dst = (rank + j) % comm_size;
        recvblocks[count] = recvcounts[dst];
        recvindex[count] = (MPI_Aint)((char*)recvbuf + rdispls[dst]*recvext);
        recvtypes[count] = recvtype;
      } else {
        // to inter
        w = bits[j&mask] % 2;
        recvblocks[count] = intercount[j];
        recvindex[count] = (MPI_Aint)(interbuf[1-w]+j*recvtypesize);
        recvtypes[count] = MPI_BYTE;
      }

      count++;
      j++;

      if ((j&pof2) != pof2)
        j += pof2;

    } while (j < comm_size);

    // types creation
    MPI_Type_create_struct(count, sendblocks, sendindex, sendtypes, &sendblocktype[round]);
    MPI_Type_commit(&sendblocktype[round]);

    MPI_Type_create_struct(count, recvblocks, recvindex, recvtypes, &recvblocktype[round]);
    MPI_Type_commit(&recvblocktype[round]);

    sendranks[round] = (rank - pof2 + comm_size) % comm_size;
    recvranks[round] = (rank + pof2) % comm_size;

    mask = nextmask; // shift in zero bit

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
    //MPI_Sendrecv((char *)sendbuf + senddispl*sendextent, sendcount, sendtype, rank, BRUCK,
    //             (char *)recvbuf + recvdispl*recvextent, recvcount, recvtype, rank, BRUCK,
    //             comm, MPI_STATUS_IGNORE);
    res = NBC_Sched_copy ((char *) sendbuf + sdispls[rank] * sendext, false, sendcount, sendtype,
                          (char *) recvbuf + rdispls[rank] * recvext, false, recvcount, recvtype,
                          schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      free(tmpbuf);
      return res;
    }
  }

  // Modified second step
  for (round = 0; round < nrounds; ++round) {
    //MPI_Sendrecv(MPI_BOTTOM, 1, sendblocktype[round], sendrank[round], BRUCK,
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

  return MPI_SUCCESS;
}

int ompi_coll_libnbc_alltoallv_inter_init(const void* sendbuf, const int *sendcounts, const int *sdispls,
                                          MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *rdispls,
                                          MPI_Datatype recvtype, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                          struct mca_coll_base_module_2_2_0_t *module) {
    int res = nbc_ialltoallv_inter(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
                                   comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}
