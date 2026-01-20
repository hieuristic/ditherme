cmake_minimum_required(VERSION 3.10)
project(ditherme C)

set(CMAKE_C_STANDARD 11)

# Set binary output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Find Vulkan
find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)

# Find slangc for shader compilation
find_program(SLANG_COMPILER slangc)
if(NOT SLANG_COMPILER)
    message(FATAL_ERROR "slangc not found. Please install Slang.")
endif()

# Shader compilation function using slangc
function(compile_shader SHADER_SOURCE SHADER_OUTPUT)
    add_custom_command(
        OUTPUT ${SHADER_OUTPUT}
        COMMAND ${SLANG_COMPILER} ${SHADER_SOURCE} -target spirv -o ${SHADER_OUTPUT}
        DEPENDS ${SHADER_SOURCE}
        COMMENT "Compiling shader ${SHADER_SOURCE} with Slang"
    )
endfunction()

# Compile the shaders
set(DITHER_SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/dither.slang")
set(DITHER_SHADER_SPV "${CMAKE_CURRENT_BINARY_DIR}/dither.spv")
compile_shader(${DITHER_SHADER_SRC} ${DITHER_SHADER_SPV})

# Executable
add_executable(ditherme src/main.c ${DITHER_SHADER_SPV})

target_include_directories(ditherme PRIVATE 
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/ext
)

target_link_libraries(ditherme PRIVATE ${Vulkan_LIBRARIES} Threads::Threads)

# Ensure the shaders are accessible
target_compile_definitions(ditherme PRIVATE 
    DITHER_SHADER_PATH="${DITHER_SHADER_SPV}"
)
