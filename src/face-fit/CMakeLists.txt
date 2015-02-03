# for boost.python
find_package( PythonLibs 2.7 REQUIRED )
include_directories( ${PYTHON_INCLUDE_DIRS} )

find_package( Boost COMPONENTS python REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIR} )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

# Build our library
add_library( conversion SHARED conversion.cpp )
target_link_libraries( conversion ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${LIBS} ${OpenCV_LIBRARIES} )

# Define the wrapper library that wraps our library
add_library(face_fit SHARED main.cpp )
target_link_libraries(face_fit ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${LIBS} ${OpenCV_LIBRARIES} face_fit utilities clmTracker conversion)
# don't prepend wrapper library name with lib
set_target_properties(face_fit PROPERTIES PREFIX "" )

# Define the wrapper library that wraps our library
add_library(examples SHARED examples.cpp )
target_link_libraries(examples ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${OpenCV_LIBRARIES} examples conversion)
# don't prepend wrapper library name with lib
set_target_properties(examples PROPERTIES PREFIX "" )

# -*-cmake-*-
#add_executable(face-fit main.cpp)
#target_link_libraries(face-fit utilities clmTracker ${LIBS})
