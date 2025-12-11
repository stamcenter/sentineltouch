cmake_minimum_required (VERSION 3.5.1)

project(SecureCon)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(BUILD_STATIC "Set to ON to include static versions of the library" OFF)

# Find OpenFHE
find_package(OpenFHE CONFIG REQUIRED)
if (OpenFHE_FOUND)
    message(STATUS "FOUND PACKAGE OpenFHE")
    message(STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
    message(STATUS "OpenFHE installed as shared libraries: ${OpenFHE_SHARED}")
    message(STATUS "OpenFHE include files location: ${OpenFHE_INCLUDE}")
    message(STATUS "OpenFHE lib files location: ${OpenFHE_LIBDIR}")
    message(STATUS "OpenFHE Native Backend size: ${OpenFHE_NATIVE_SIZE}")
else()
    message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif ()

set(CMAKE_CXX_FLAGS ${OpenFHE_CXX_FLAGS})

# Include directories
include_directories(${OPENMP_INCLUDES})
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)

# Link directories
link_directories(${OpenFHE_LIBDIR})
link_directories(${OPENMP_LIBRARIES})

# GCC filesystem flag
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lstdc++fs")
endif()


# Build type (static/shared)
if(BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries(${OpenFHE_STATIC_LIBRARIES} stdc++fs)
else()
    set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
    link_libraries(${OpenFHE_SHARED_LIBRARIES} stdc++fs)
endif()


add_executable(hybrid_16
     src/Utils.h src/UtilsData.h src/LeNet5Hybrid.cpp 
        src/FHEController.h src/FHEController.cpp 
        src/ANNController.h src/ANNController.cpp
)
target_compile_definitions(hybrid_16 PRIVATE DATASET="polyu" OPERATION="register" INDEX_VALUE=0 NUM_USERS=16 NUM_TO_READ=16  EMBEDDING_SPACE=16)

add_executable(hybrid_32
        src/Utils.h src/UtilsData.h src/LeNet5Hybrid.cpp 
        src/FHEController.h src/FHEController.cpp 
        src/ANNController.h src/ANNController.cpp
)
target_compile_definitions(hybrid_32 PRIVATE DATASET="polyu" OPERATION="register" INDEX_VALUE=0 NUM_USERS=16 NUM_TO_READ=16  EMBEDDING_SPACE=32)


add_executable(lenet5_16
    src/LeNet5.cpp
    src/Utils.h src/UtilsData.h
    src/FHEController.h src/FHEController.cpp
    src/ANNController.h src/ANNController.cpp
)
target_compile_definitions(lenet5_16 PRIVATE DATASET="polyu" OPERATION="register"  INDEX_VALUE=0 NUM_USERS=16 NUM_TO_READ=16  EMBEDDING_SPACE=16)

add_executable(lenet5_32
        src/Utils.h src/UtilsData.h src/LeNet5.cpp 
        src/FHEController.h src/FHEController.cpp 
        src/ANNController.h src/ANNController.cpp
)
target_compile_definitions(lenet5_32 PRIVATE DATASET="polyu" OPERATION="register" INDEX_VALUE=0 NUM_USERS=16 NUM_TO_READ=16 EMBEDDING_SPACE=32)

