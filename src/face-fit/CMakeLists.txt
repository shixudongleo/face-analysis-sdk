# for boost.python
#find_package( PythonLibs 2.7 REQUIRED )
#include_directories( ${PYTHON_INCLUDE_DIRS} )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

# Build ndarray cv::Mat conversion library
add_library( conversion SHARED conversion.cpp )
target_link_libraries( conversion ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${LIBS} ${OpenCV_LIBRARIES} )

# Build face landmark generator warpper
add_library(face_fit SHARED face_fit.cpp )
target_link_libraries(face_fit ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${LIBS} ${OpenCV_LIBRARIES} face_fit utilities clmTracker conversion)
# don't prepend wrapper library name with lib
set_target_properties(face_fit PROPERTIES PREFIX "" )
if(WIN32)
  set_target_properties(face_fit PROPERTIES SUFFIX ".pyd")
elseif (APPLE)
  set_target_properties(face_fit PROPERTIES SUFFIX ".so")
endif()

# Define the wrapper library that wraps our library
add_library(examples SHARED examples.cpp )
target_link_libraries(examples ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${OpenCV_LIBRARIES} examples conversion)
# don't prepend wrapper library name with lib
set_target_properties(examples PROPERTIES PREFIX "" )
if(WIN32)
  set_target_properties(examples PROPERTIES SUFFIX ".pyd")
elseif (APPLE)
  set_target_properties(examples PROPERTIES SUFFIX ".so")
endif()

# -*-cmake-*-
add_executable(face-fit main.cpp)
target_link_libraries(face-fit utilities clmTracker ${LIBS})
