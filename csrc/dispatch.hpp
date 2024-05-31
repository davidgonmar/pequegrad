#pragma once
#include "dtype.hpp"

#define PG_DISPATCH_CASE(TYPE, CTYPE, ...)                                     \
  case TYPE: {                                                                 \
    using scalar_t = CTYPE;                                                    \
    __VA_ARGS__                                                                \
    break;                                                                     \
  }

#define PG_DISPATCH_DEFAULT(...)                                               \
  default: {                                                                   \
    __VA_ARGS__                                                                \
    break;                                                                     \
  }

#define PG_DISPATCH_SWITCH(TYPE, ...)                                          \
  switch (TYPE) { __VA_ARGS__ }

#define THROW_NOT_IMPLEMENTED_ERROR(NAME, TYPE)                                \
  throw std::runtime_error(std::string(NAME) + " not implemented for " +       \
                           dtype_to_string(TYPE));

#define THROW_NOT_IMPLEMENTED_ERROR_PAIRS(NAME, TYPE1, TYPE2)                  \
  throw std::runtime_error(std::string(NAME) + " not implemented for " +       \
                           dtype_to_string(TYPE1) + " and " +                  \
                           dtype_to_string(TYPE2));

#define PG_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                            \
  PG_DISPATCH_SWITCH(                                                          \
      TYPE,                                                                    \
      PG_DISPATCH_CASE(DType::Float32, float, __VA_ARGS__();)                  \
          PG_DISPATCH_CASE(DType::Float64, double, __VA_ARGS__();)             \
              PG_DISPATCH_DEFAULT(THROW_NOT_IMPLEMENTED_ERROR(NAME, TYPE)))

#define PG_DISPATCH_ALL_TYPES(TYPE, NAME, ...)                                 \
  PG_DISPATCH_SWITCH(                                                          \
      TYPE, PG_DISPATCH_CASE(DType::Float32, float, __VA_ARGS__();)            \
                PG_DISPATCH_CASE(DType::Float64, double, __VA_ARGS__();)       \
                    PG_DISPATCH_CASE(DType::Int32, int, __VA_ARGS__();)        \
                        PG_DISPATCH_DEFAULT(                                   \
                            THROW_NOT_IMPLEMENTED_ERROR(NAME, TYPE)))

#define PG_DISPATCH_CASE_TYPE1(TYPE1, CTYPE1, ...)                             \
  case TYPE1: {                                                                \
    using scalar_t1 = CTYPE1;                                                  \
    __VA_ARGS__                                                                \
    break;                                                                     \
  }

#define PG_DISPATCH_CASE_TYPE2(TYPE2, CTYPE2, ...)                             \
  case TYPE2: {                                                                \
    using scalar_t2 = CTYPE2;                                                  \
    __VA_ARGS__                                                                \
    break;                                                                     \
  }

#define PG_DISPATCH_CASE_TYPE2_ALL_TYPES(...)                                  \
  PG_DISPATCH_CASE(DType::Float32, float, __VA_ARGS__();)                      \
  PG_DISPATCH_CASE(DType::Float64, double, __VA_ARGS__();)                     \
  PG_DISPATCH_CASE(DType::Int32, int, __VA_ARGS__();)                          \
  PG_DISPATCH_DEFAULT(THROW_NOT_IMPLEMENTED_ERROR_PAIRS(NAME, TYPE1, TYPE2))

#define PG_DISPATCH_ALL_TYPES_TWO_TYPES(TYPE1, TYPE2, NAME, ...)               \
  PG_DISPATCH_SWITCH(                                                          \
      TYPE1,                                                                   \
      PG_DISPATCH_CASE_TYPE1(                                                  \
          DType::Float32, float,                                               \
          PG_DISPATCH_SWITCH(                                                  \
              TYPE2,                                                           \
              PG_DISPATCH_CASE_TYPE2(DType::Float32, float, __VA_ARGS__();)    \
                  PG_DISPATCH_CASE_TYPE2(DType::Float64, double,               \
                                         __VA_ARGS__();)                       \
                      PG_DISPATCH_CASE_TYPE2(DType::Int32, int,                \
                                             __VA_ARGS__();)                   \
                          PG_DISPATCH_DEFAULT(                                 \
                              THROW_NOT_IMPLEMENTED_ERROR_PAIRS(NAME, TYPE1,   \
                                                                TYPE2))))      \
          PG_DISPATCH_CASE_TYPE1(                                              \
              DType::Float64, double,                                          \
              PG_DISPATCH_SWITCH(                                              \
                  TYPE2, PG_DISPATCH_CASE_TYPE2(DType::Float32, float,         \
                                                __VA_ARGS__();)                \
                             PG_DISPATCH_CASE_TYPE2(DType::Float64, double,    \
                                                    __VA_ARGS__();)            \
                                 PG_DISPATCH_CASE_TYPE2(DType::Int32, int,     \
                                                        __VA_ARGS__();)        \
                                     PG_DISPATCH_DEFAULT(                      \
                                         THROW_NOT_IMPLEMENTED_ERROR_PAIRS(    \
                                             NAME, TYPE1, TYPE2))))            \
              PG_DISPATCH_CASE_TYPE1(                                          \
                  DType::Int32, int,                                           \
                  PG_DISPATCH_SWITCH(                                          \
                      TYPE2,                                                   \
                      PG_DISPATCH_CASE_TYPE2(DType::Float32, float,            \
                                             __VA_ARGS__();)                   \
                          PG_DISPATCH_CASE_TYPE2(DType::Float64, double,       \
                                                 __VA_ARGS__();)               \
                              PG_DISPATCH_CASE_TYPE2(DType::Int32, int,        \
                                                     __VA_ARGS__();)           \
                                  PG_DISPATCH_DEFAULT(                         \
                                      THROW_NOT_IMPLEMENTED_ERROR_PAIRS(       \
                                          NAME, TYPE1, TYPE2))))               \
                  PG_DISPATCH_DEFAULT(                                         \
                      THROW_NOT_IMPLEMENTED_ERROR_PAIRS(NAME, TYPE1, TYPE2)))
