project(3dview)

set(CXX_SOURCES src/main.cc)
add_executable(3dview ${CXX_SOURCES})
target_link_libraries(3dview glut GL GLU fmt)
set_property(TARGET 3dview PROPERTY CXX_STANDARD 17)
