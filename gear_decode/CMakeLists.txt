# Build shared library for Python binding

# Shared library version for Python binding
set(TARGET_LIB gear_decode)
add_library(${TARGET_LIB} SHARED gear_generate.cpp)
target_link_libraries(${TARGET_LIB} PRIVATE llama common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_LIB} PRIVATE cxx_std_17)
target_include_directories(${TARGET_LIB} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
install(TARGETS ${TARGET_LIB} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Set output directory for the shared library
set_target_properties(${TARGET_LIB} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# On Windows, also set the DLL output directory
if(WIN32)
    set_target_properties(${TARGET_LIB} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()
