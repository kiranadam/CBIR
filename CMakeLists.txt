cmake_minimum_required(VERSION 2.8) 
PROJECT (CBIR)
find_package(OpenCV REQUIRED )
set( NAME_SRC
    src/CBIR.cpp
	src/CBIR_test.cpp
)

set( NAME_HEADERS       
     include/CBIR.h
     
)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )
link_directories( ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable( CBIR ${NAME_SRC} ${NAME_HEADERS} )

target_link_libraries( CBIR ${OpenCV_LIBS} )
