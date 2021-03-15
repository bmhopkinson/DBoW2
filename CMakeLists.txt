cmake_minimum_required(VERSION 3.0)
project(DBoW2)
include(ExternalProject)



option(BUILD_DBoW2   "Build DBoW2"            ON)
option(BUILD_Demo    "Build demo application" ON)
option(BUILD_Creator    "Build vocab creator application" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) #will include non-standard opencv location in installed rpath 

set(INSTALL_LIBDIR "/usr/local/lib")
set(INSTALL_INCLUDEDIR "/usr/local/include")

set(HDRS
  include/DBoW2/BowVector.h           include/DBoW2/FBrief.h     include/DBoW2/FSurf64.h
  include/DBoW2/QueryResults.h        include/DBoW2/TemplatedDatabase.h   include/DBoW2/FORB.h
  include/DBoW2/DBoW2.h               include/DBoW2/FClass.h              include/DBoW2/FeatureVector.h
  include/DBoW2/ScoringObject.h       include/DBoW2/TemplatedVocabulary.h	include/DBoW2/Random.h
  include/DBoW2/Timestamp.h
  )
set(SRCS 
  src/BowVector.cpp     src/FBrief.cpp        src/FORB.cpp	src/FSurf64.cpp    src/Timestamp.cpp
  src/FeatureVector.cpp src/QueryResults.cpp  src/ScoringObject.cpp	src/Random.cpp)

set(DEPENDENCY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dependencies)
set(DEPENDENCY_INSTALL_DIR ${DEPENDENCY_DIR}/install)

#find_package(OpenCV 3.1.0 REQUIRED PATHS /home/nvidia/Documents/opencv_build/share/OpenCV NO_DEFAULT_PATH)
find_package(OpenCV 3.1.0 REQUIRED PATHS /home/cv-bhlab/Documents/Libraries/opencv/opencv_3_4_nonfree/install/share/OpenCV NO_DEFAULT_PATH)
#find_package(OpenCV 3.1.0 REQUIRED PATHS /home/brian/Documents/Libraries/opencv/3_4_13/install/share/OpenCV NO_DEFAULT_PATH)

include_directories(${OpenCV_INCLUDE_DIRS})

if(BUILD_DBoW2)
  set(LIB_SHARED "SHARED")
  if(WIN32)
    set(LIB_SHARED "STATIC")
  endif(WIN32)
  add_library(${PROJECT_NAME} ${LIB_SHARED} ${SRCS})
  target_include_directories(${PROJECT_NAME} PUBLIC include/DBoW2/ include/)
  target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
  set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)
endif(BUILD_DBoW2)

if(BUILD_Demo)
  add_executable(demo demo/demo.cpp)
  target_link_libraries(demo ${PROJECT_NAME} ${OpenCV_LIBS})
  set_target_properties(demo PROPERTIES CXX_STANDARD 11)
  file(COPY demo/images DESTINATION ${CMAKE_BINARY_DIR}/)
endif(BUILD_Demo)

if(BUILD_Creator)
  add_executable(vocab_creator demo/create_test_vocab.cpp)
  set_property(TARGET vocab_creator PROPERTY CXX_STANDARD 17 )
  target_link_libraries(vocab_creator ${PROJECT_NAME} ${OpenCV_LIBS} stdc++fs)
  set_target_properties(vocab_creator PROPERTIES CXX_STANDARD 11)
endif(BUILD_Creator)

configure_file(src/DBoW2.cmake.in
  "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" @ONLY)
  

install(TARGETS ${PROJECT_NAME} DESTINATION ${INSTALL_LIBDIR})
if(BUILD_DBoW2)
  install(DIRECTORY include/DBoW2 DESTINATION ${INSTALL_INCLUDEDIR})
endif()
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DBoW2Config.cmake"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(FILES "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" 
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/DBoW2/)
install(DIRECTORY ${DEPENDENCY_INSTALL_DIR}/ DESTINATION ${CMAKE_INSTALL_PREFIX} OPTIONAL)

