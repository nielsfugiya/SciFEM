# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.1

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /u/fac/bangerth/bin/bin/cmake

# The command to remove a file.
RM = /u/fac/bangerth/bin/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /u/g/ga/galileo2012/bin/deal.II/examples/step-29

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /u/g/ga/galileo2012/bin/deal.II/examples/step-29

# Utility rule file for runclean.

# Include the progress variables for this target.
include CMakeFiles/runclean.dir/progress.make

CMakeFiles/runclean:
	$(CMAKE_COMMAND) -E cmake_progress_report /u/g/ga/galileo2012/bin/deal.II/examples/step-29/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "runclean invoked"
	/u/fac/bangerth/bin/bin/cmake -E remove *.log *.gmv *.gnuplot *.gpl *.eps *.pov *.vtk *.ucd *.d2

runclean: CMakeFiles/runclean
runclean: CMakeFiles/runclean.dir/build.make
.PHONY : runclean

# Rule to build all files generated by this target.
CMakeFiles/runclean.dir/build: runclean
.PHONY : CMakeFiles/runclean.dir/build

CMakeFiles/runclean.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/runclean.dir/cmake_clean.cmake
.PHONY : CMakeFiles/runclean.dir/clean

CMakeFiles/runclean.dir/depend:
	cd /u/g/ga/galileo2012/bin/deal.II/examples/step-29 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /u/g/ga/galileo2012/bin/deal.II/examples/step-29 /u/g/ga/galileo2012/bin/deal.II/examples/step-29 /u/g/ga/galileo2012/bin/deal.II/examples/step-29 /u/g/ga/galileo2012/bin/deal.II/examples/step-29 /u/g/ga/galileo2012/bin/deal.II/examples/step-29/CMakeFiles/runclean.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/runclean.dir/depend

