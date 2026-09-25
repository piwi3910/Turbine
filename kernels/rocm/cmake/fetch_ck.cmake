# FetchContent download step for Composable Kernel (script mode).
#
# rocm-libraries is a multi-gigabyte monorepo; only projects/composablekernel is
# needed, so this fetches exactly the pinned commit with a shallow, blob-less,
# sparse checkout instead of a full clone.
#
#   cmake -DREPOSITORY=<url> -DCOMMIT=<sha> -DSUBDIR=<path> -DDEST=<dir> -P fetch_ck.cmake
foreach(var REPOSITORY COMMIT SUBDIR DEST)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "fetch_ck.cmake: ${var} is not set")
  endif()
endforeach()

find_package(Git REQUIRED)

set(stamp "${DEST}/.turbine-ck-${COMMIT}")
if(EXISTS "${stamp}")
  return()
endif()

file(REMOVE_RECURSE "${DEST}")
file(MAKE_DIRECTORY "${DEST}")

function(git_step)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" ${ARGN}
    WORKING_DIRECTORY "${DEST}"
    RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "fetch_ck.cmake: git ${ARGN} failed (${rc})")
  endif()
endfunction()

git_step(init -q)
git_step(remote add origin "${REPOSITORY}")
git_step(sparse-checkout set --cone "${SUBDIR}")
git_step(fetch -q --depth 1 --filter=blob:none origin "${COMMIT}")
git_step(checkout -q FETCH_HEAD)

execute_process(
  COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
  WORKING_DIRECTORY "${DEST}"
  OUTPUT_VARIABLE head
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT head STREQUAL COMMIT)
  message(FATAL_ERROR "fetch_ck.cmake: checked out ${head}, expected ${COMMIT}")
endif()
file(TOUCH "${stamp}")
