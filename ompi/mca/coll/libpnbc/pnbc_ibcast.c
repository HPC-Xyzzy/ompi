/* -*- Mode: C; c-basic-offset:2 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2006      The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2006      The Technical University of Chemnitz. All
 *                         rights reserved.
 * Copyright (c) 2014-2017 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2016-2017 IBM Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *
 */
#include "pnbc_internal.h"

static inline int bcast_sched_binomial(int rank, int p, int root, PNBC_Schedule *schedule, void *buffer, int count,
                                       MPI_Datatype datatype);
static inline int bcast_sched_linear(int rank, int p, int root, PNBC_Schedule *schedule, void *buffer, int count,
                                     MPI_Datatype datatype);
static inline int bcast_sched_chain(int rank, int p, int root, PNBC_Schedule *schedule, void *buffer, int count,
                                    MPI_Datatype datatype, int fragsize, size_t size);

#ifdef PNBC_CACHE_SCHEDULE
/* tree comparison function for schedule cache */
int PNBC_Bcast_args_compare(PNBC_Bcast_args *a, PNBC_Bcast_args *b, void *param) {
  if ((a->buffer == b->buffer) &&
      (a->count == b->count) &&
      (a->datatype == b->datatype) &&
      (a->root == b->root) ) {
    return 0;
  }

  if( a->buffer < b->buffer ) {
    return -1;
  }

  return 1;
}
#endif

int ompi_coll_libpnbc_ibcast(void *buffer, int count, MPI_Datatype datatype, int root,
                            struct ompi_communicator_t *comm, ompi_request_t ** request,
                            struct mca_coll_base_module_2_2_0_t *module)
{
  int rank, p, res, segsize;
  size_t size;
  PNBC_Schedule *schedule;
#ifdef PNBC_CACHE_SCHEDULE
  PNBC_Bcast_args *args, *found, search;
#endif
  enum { PNBC_BCAST_LINEAR, PNBC_BCAST_BINOMIAL, PNBC_BCAST_CHAIN } alg;
  PNBC_Handle *handle;
  ompi_coll_libpnbc_module_t *libpnbc_module = (ompi_coll_libpnbc_module_t*) module;

  rank = ompi_comm_rank (comm);
  p = ompi_comm_size (comm);

  if (1 == p) {
    *request = &ompi_request_empty;
    return OMPI_SUCCESS;
  }

  res = ompi_datatype_type_size(datatype, &size);
  if (MPI_SUCCESS != res) {
    PNBC_Error("MPI Error in ompi_datatype_type_size() (%i)", res);
    return res;
  }

  segsize = 16384;
  /* algorithm selection */
  if( libpnbc_ibcast_skip_dt_decision ) {
    if (p <= 4) {
      alg = PNBC_BCAST_LINEAR;
    }
    else {
      alg = PNBC_BCAST_BINOMIAL;
    }
  }
  else {
    if (p <= 4) {
      alg = PNBC_BCAST_LINEAR;
    } else if (size * count < 65536) {
      alg = PNBC_BCAST_BINOMIAL;
    } else if (size * count < 524288) {
      alg = PNBC_BCAST_CHAIN;
      segsize = 8192;
    } else {
      alg = PNBC_BCAST_CHAIN;
      segsize = 32768;
    }
  }

#ifdef PNBC_CACHE_SCHEDULE
  /* search schedule in communicator specific tree */
  search.buffer = buffer;
  search.count = count;
  search.datatype = datatype;
  search.root = root;
  found = (PNBC_Bcast_args *) hb_tree_search ((hb_tree *) libpnbc_module->PNBC_Dict[PNBC_BCAST], &search);
  if (NULL == found) {
#endif
    schedule = OBJ_NEW(PNBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }

    switch(alg) {
      case PNBC_BCAST_LINEAR:
        res = bcast_sched_linear(rank, p, root, schedule, buffer, count, datatype);
        break;
      case PNBC_BCAST_BINOMIAL:
        res = bcast_sched_binomial(rank, p, root, schedule, buffer, count, datatype);
        break;
      case PNBC_BCAST_CHAIN:
        res = bcast_sched_chain(rank, p, root, schedule, buffer, count, datatype, segsize, size);
        break;
    }

    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      return res;
    }

    res = PNBC_Sched_commit (schedule);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      return res;
    }

#ifdef PNBC_CACHE_SCHEDULE
    /* save schedule to tree */
    args = (PNBC_Bcast_args *) malloc (sizeof (args));
    if (NULL != args) {
      args->buffer = buffer;
      args->count = count;
      args->datatype = datatype;
      args->root = root;
      args->schedule = schedule;
      res = hb_tree_insert ((hb_tree *) libpnbc_module->PNBC_Dict[PNBC_BCAST], args, args, 0);
      if (0 == res) {
        OBJ_RETAIN (schedule);

        /* increase number of elements for A2A */
        if (++libpnbc_module->PNBC_Dict_size[PNBC_BCAST] > PNBC_SCHED_DICT_UPPER) {
          PNBC_SchedCache_dictwipe ((hb_tree *) libpnbc_module->PNBC_Dict[PNBC_BCAST],
                                   &libpnbc_module->PNBC_Dict_size[PNBC_BCAST]);
        }
      } else {
        PNBC_Error("error in dict_insert() (%i)", res);
        free (args);
      }
    }
  } else {
    /* found schedule */
    schedule = found->schedule;
    OBJ_RETAIN(schedule);
  }
#endif

  res = PNBC_Init_handle (comm, &handle, libpnbc_module);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    return res;
  }

  res = PNBC_Start (handle, schedule);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    PNBC_Return_handle (handle);
    return res;
  }

  *request = (ompi_request_t *) handle;

  return OMPI_SUCCESS;
}

/* better binomial bcast
 * working principle:
 * - each node gets a virtual rank vrank
 * - the 'root' node get vrank 0
 * - node 0 gets the vrank of the 'root'
 * - all other ranks stay identical (they do not matter)
 *
 * Algorithm:
 * - each node with vrank > 2^r and vrank < 2^r+1 receives from node
 *   vrank - 2^r (vrank=1 receives from 0, vrank 0 receives never)
 * - each node sends each round r to node vrank + 2^r
 * - a node stops to send if 2^r > commsize
 */
#define RANK2VRANK(rank, vrank, root) \
{ \
  vrank = rank; \
  if (rank == 0) vrank = root; \
  if (rank == root) vrank = 0; \
}
#define VRANK2RANK(rank, vrank, root) \
{ \
  rank = vrank; \
  if (vrank == 0) rank = root; \
  if (vrank == root) rank = 0; \
}
static inline int bcast_sched_binomial(int rank, int p, int root, PNBC_Schedule *schedule, void *buffer, int count, MPI_Datatype datatype) {
  int maxr, vrank, peer, res;

  maxr = (int)ceil((log((double)p)/LOG2));

  RANK2VRANK(rank, vrank, root);

  /* receive from the right hosts  */
  if (vrank != 0) {
    for (int r = 0 ; r < maxr ; ++r) {
      if ((vrank >= (1 << r)) && (vrank < (1 << (r + 1)))) {
        VRANK2RANK(peer, vrank - (1 << r), root);
        res = PNBC_Sched_recv (buffer, false, count, datatype, peer, schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
          return res;
        }
      }
    }

    res = PNBC_Sched_barrier (schedule);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
  }

  /* now send to the right hosts */
  for (int r = 0 ; r < maxr ; ++r) {
    if (((vrank + (1 << r) < p) && (vrank < (1 << r))) || (vrank == 0)) {
      VRANK2RANK(peer, vrank + (1 << r), root);
      res = PNBC_Sched_send (buffer, false, count, datatype, peer, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
  }

  return OMPI_SUCCESS;
}

/* simple linear MPI_Ibcast */
static inline int bcast_sched_linear(int rank, int p, int root, PNBC_Schedule *schedule, void *buffer, int count, MPI_Datatype datatype) {
  int res;

  /* send to all others */
  if(rank == root) {
    for (int peer = 0 ; peer < p ; ++peer) {
      if (peer != root) {
        /* send msg to peer */
        res = PNBC_Sched_send (buffer, false, count, datatype, peer, schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
          return res;
        }
      }
    }
  } else {
    /* recv msg from root */
    res = PNBC_Sched_recv (buffer, false, count, datatype, root, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
  }

  return OMPI_SUCCESS;
}

/* simple chained MPI_Ibcast */
static inline int bcast_sched_chain(int rank, int p, int root, PNBC_Schedule *schedule, void *buffer, int count, MPI_Datatype datatype, int fragsize, size_t size) {
  int res, vrank, rpeer, speer, numfrag, fragcount, thiscount;
  MPI_Aint ext;
  char *buf;

  RANK2VRANK(rank, vrank, root);
  VRANK2RANK(rpeer, vrank-1, root);
  VRANK2RANK(speer, vrank+1, root);
  res = ompi_datatype_type_extent(datatype, &ext);
  if (MPI_SUCCESS != res) {
    PNBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  if (count == 0) {
    return OMPI_SUCCESS;
  }

  numfrag = count * size/fragsize;
  if ((count * size) % fragsize != 0) {
    numfrag++;
  }

  fragcount = count/numfrag;

  for (int fragnum = 0 ; fragnum < numfrag ; ++fragnum) {
    buf = (char *) buffer + fragnum * fragcount * ext;
    thiscount = fragcount;
    if (fragnum == numfrag-1) {
      /* last fragment may not be full */
      thiscount = count - fragcount * fragnum;
    }

    /* root does not receive */
    if (vrank != 0) {
      res = PNBC_Sched_recv (buf, false, thiscount, datatype, rpeer, schedule, true);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }

    /* last rank does not send */
    if (vrank != p-1) {
      res = PNBC_Sched_send (buf, false, thiscount, datatype, speer, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }

      /* this barrier here seems awaward but isn't!!!! */
      if (vrank == 0)  {
        res = PNBC_Sched_barrier (schedule);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
          return res;
        }
      }
    }
  }

  return OMPI_SUCCESS;
}
