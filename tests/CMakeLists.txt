Include(FetchContent)
include(CTest)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)

include(Catch)

set(TEST_SRCS
    test_shufflenet.cpp
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
    Threads::Threads
    XNNPACK
  )
  catch_discover_tests(${test_target})
endforeach()


