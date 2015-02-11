# -*-cmake-*-
PROJECT(CSIRO-FaceAnalysis-SDK)

cmake_minimum_required(VERSION 2.8)

#set(CMAKE_VERBOSE_MAKEFILE true)


find_package( PythonLibs 2.7 REQUIRED )
include_directories( ${PYTHON_INCLUDE_DIRS} )

find_package( Boost COMPONENTS python REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIR} )

# print out test message for python version
foreach(REQUIRED_PYTHON_VERSION 2.7)
 message(STATUS "======== LOOKING FOR ${REQUIRED_PYTHON_VERSION} ========================")
 find_package ( PythonInterp ${REQUIRED_PYTHON_VERSION} REQUIRED)
 find_package ( PythonLibs ${PYTHON_VERSION_STRING} EXACT)
 #find_package ( PythonLibs ${PYTHON_VERSION_STRING} EXACT REQUIRED )


  message(STATUS "PYTHON_EXECUTABLE:FILEPATH=${PYTHON_EXECUTABLE}")
  message(STATUS "PYTHON_LIBRARY:FILEPATH=${PYTHON_LIBRARY}")
  message(STATUS "PYTHON_INCLUDE_DIR:FILEPATH=${PYTHON_INCLUDE_DIR}")
  message(STATUS "PYTHON_FRAMEWORK_INCLUDES=${PYTHON_FRAMEWORK_INCLUDES}")
  message(STATUS "PYTHONLIBS_VERSION_STRING=${PYTHONLIBS_VERSION_STRING}")
  message(STATUS "Python_FRAMEWORKS=${Python_FRAMEWORKS}")

  unset(PYTHON_EXECUTABLE CACHE)
  unset(PYTHON_LIBRARY CACHE)
  unset(PYTHON_INCLUDE_DIR CACHE)
endforeach()

# Default values for options
if(NOT DEFINED OpenCV_PREFIX)
  set(OpenCV_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

# Configurable options
OPTION(WITH_GUI "Build the GUI" OFF)

# Third party libraries
find_package(OpenCV REQUIRED core highgui imgproc objdetect
  PATHS ${OpenCV_PREFIX}/lib/cmake/
        ${OpenCV_PREFIX}/share/OpenCV/
  NO_DEFAULT_PATH) # For some reason CMake uses its defaults before the above paths.

IF(APPLE)
find_library(FOUNDATION Foundation)
SET(EXTRA_LIBS ${FOUNDATION})
ENDIF()

# Third party programs
if(NOT DEFINED FFMPEG)
  find_program(FFMPEG ffmpeg)
endif()

if (NOT DEFINED BASH)
  find_program(BASH bash)
endif()

# C++ Preprocessor definitions
ADD_DEFINITIONS(-D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE)
ADD_DEFINITIONS(-D_WITH_AVATAR_)

# Compiling and linking
# - type
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
if(DEFINED CMAKE_BUILD_TYPE AND CMAKE_VERSION VERSION_GREATER "2.8")
   set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
endif()
SET(CMAKE_BUILD_TYPE Release CACHE STRING "")

# - flags
SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -g -O0  -Wall -Wextra -DDEBUG")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g  -Wall -Wextra -DDEBUG")

SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3  -Wall -Wextra")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -Wall -O3  -Wextra")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

SET(LIBS ${OpenCV_LIBS} ${EXTRA_LIBS})

# Build paths
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
#SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")

# Include directories
INCLUDE_DIRECTORIES("src/")
INCLUDE_DIRECTORIES("src/avatar/")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_BINARY_DIR}/src/")



# Subdirectories with CMakeLists.txt
ADD_SUBDIRECTORY(src/utils)
ADD_SUBDIRECTORY(src/tracker)
ADD_SUBDIRECTORY(src/avatar)
ADD_SUBDIRECTORY(src/face-fit)
ADD_SUBDIRECTORY(src/test)
ADD_SUBDIRECTORY(src/scripts)
ADD_SUBDIRECTORY(src/map-list)
ADD_SUBDIRECTORY(src/change-pathnames)
ADD_SUBDIRECTORY(src/expression-transfer)
ADD_SUBDIRECTORY(src/display-avatar)
ADD_SUBDIRECTORY(src/display-tracking)
ADD_SUBDIRECTORY(src/create-avatar-model)

IF(WITH_GUI)
  SET(CPACK_PACKAGE_NAME "clm")
  SET(CPACK_PACKAGE_VENDOR "CI2CV-CSIRO ci2cv.net")
  SET(CPACK_PACKAGE_VERSION "0.9.1")
  SET(CPACK_PACKAGE_VERSION_MAJOR "0")
  SET(CPACK_PACKAGE_VERSION_MINOR "9")
  SET(CPACK_PACKAGE_VERSION_PATCH "1")

  set(CPACK_BINARY_DRAGNDROP ON)
  SET(CPACK_BINARY_PACKAGEMAKER OFF)
  SET(CPACK_SOURCE_TGZ OFF)
  ADD_SUBDIRECTORY(src/qt-gui)
endif()

#cpack stuff goes here

