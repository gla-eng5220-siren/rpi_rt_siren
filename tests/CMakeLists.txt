Include(FetchContent)
include(CTest)

option(USE_SYSTEM_CATCH2 "Use system-installed Catch2" ON)

if(USE_SYSTEM_CATCH2)
  find_package(Catch2 3.7.1 QUIET)

  if(NOT Catch2_FOUND)
    message(WARNING "System Catch2 not found, falling back to FetchContent")
    set(USE_SYSTEM_CATCH2 OFF)
  endif()
endif()

if(NOT USE_SYSTEM_CATCH2)
  include(FetchContent)

  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1 # or newer
  )

  FetchContent_MakeAvailable(Catch2)
endif()

include(Catch)

set(TEST_SRCS
    test_shufflenet.cpp
    test_system.cpp
)

foreach(test_src IN ITEMS ${TEST_SRCS})
	get_filename_component(test_target_prefix ${test_src} NAME_WE)
  set(test_target ${test_target_prefix}_test)
	add_executable(${test_target} ${test_src})
  target_include_directories(${test_target} PRIVATE
    ${PROJECT_SOURCE_DIR}
    "${PROJECT_SOURCE_DIR}/include"
  )
  target_compile_definitions(${test_target} PRIVATE
    TESTDATA_PATH="${PROJECT_SOURCE_DIR}/testdata"
  )
  target_link_libraries(${test_target} PUBLIC Catch2::Catch2WithMain
    flame_iris_core
  )
  catch_discover_tests(${test_target})
endforeach()


