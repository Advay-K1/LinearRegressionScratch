# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.29.3/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.29.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/advaykadam/LinearRegressionScratch

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/advaykadam/LinearRegressionScratch/build

# Include any dependencies generated for this target.
include CMakeFiles/LinearRegression.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/LinearRegression.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/LinearRegression.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/LinearRegression.dir/flags.make

CMakeFiles/LinearRegression.dir/src/main.cpp.o: CMakeFiles/LinearRegression.dir/flags.make
CMakeFiles/LinearRegression.dir/src/main.cpp.o: /Users/advaykadam/LinearRegressionScratch/src/main.cpp
CMakeFiles/LinearRegression.dir/src/main.cpp.o: CMakeFiles/LinearRegression.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/advaykadam/LinearRegressionScratch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/LinearRegression.dir/src/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/LinearRegression.dir/src/main.cpp.o -MF CMakeFiles/LinearRegression.dir/src/main.cpp.o.d -o CMakeFiles/LinearRegression.dir/src/main.cpp.o -c /Users/advaykadam/LinearRegressionScratch/src/main.cpp

CMakeFiles/LinearRegression.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/LinearRegression.dir/src/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/advaykadam/LinearRegressionScratch/src/main.cpp > CMakeFiles/LinearRegression.dir/src/main.cpp.i

CMakeFiles/LinearRegression.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/LinearRegression.dir/src/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/advaykadam/LinearRegressionScratch/src/main.cpp -o CMakeFiles/LinearRegression.dir/src/main.cpp.s

CMakeFiles/LinearRegression.dir/src/LinearRegression.cpp.o: CMakeFiles/LinearRegression.dir/flags.make
CMakeFiles/LinearRegression.dir/src/LinearRegression.cpp.o: /Users/advaykadam/LinearRegressionScratch/src/LinearRegression.cpp
CMakeFiles/LinearRegression.dir/src/LinearRegression.cpp.o: CMakeFiles/LinearRegression.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/advaykadam/LinearRegressionScratch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/LinearRegression.dir/src/LinearRegression.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/LinearRegression.dir/src/LinearRegression.cpp.o -MF CMakeFiles/LinearRegression.dir/src/LinearRegression.cpp.o.d -o CMakeFiles/LinearRegression.dir/src/LinearRegression.cpp.o -c /Users/advaykadam/LinearRegressionScratch/src/LinearRegression.cpp

CMakeFiles/LinearRegression.dir/src/LinearRegression.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/LinearRegression.dir/src/LinearRegression.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/advaykadam/LinearRegressionScratch/src/LinearRegression.cpp > CMakeFiles/LinearRegression.dir/src/LinearRegression.cpp.i

CMakeFiles/LinearRegression.dir/src/LinearRegression.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/LinearRegression.dir/src/LinearRegression.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/advaykadam/LinearRegressionScratch/src/LinearRegression.cpp -o CMakeFiles/LinearRegression.dir/src/LinearRegression.cpp.s

# Object files for target LinearRegression
LinearRegression_OBJECTS = \
"CMakeFiles/LinearRegression.dir/src/main.cpp.o" \
"CMakeFiles/LinearRegression.dir/src/LinearRegression.cpp.o"

# External object files for target LinearRegression
LinearRegression_EXTERNAL_OBJECTS =

LinearRegression: CMakeFiles/LinearRegression.dir/src/main.cpp.o
LinearRegression: CMakeFiles/LinearRegression.dir/src/LinearRegression.cpp.o
LinearRegression: CMakeFiles/LinearRegression.dir/build.make
LinearRegression: CMakeFiles/LinearRegression.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/advaykadam/LinearRegressionScratch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable LinearRegression"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LinearRegression.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/LinearRegression.dir/build: LinearRegression
.PHONY : CMakeFiles/LinearRegression.dir/build

CMakeFiles/LinearRegression.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/LinearRegression.dir/cmake_clean.cmake
.PHONY : CMakeFiles/LinearRegression.dir/clean

CMakeFiles/LinearRegression.dir/depend:
	cd /Users/advaykadam/LinearRegressionScratch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/advaykadam/LinearRegressionScratch /Users/advaykadam/LinearRegressionScratch /Users/advaykadam/LinearRegressionScratch/build /Users/advaykadam/LinearRegressionScratch/build /Users/advaykadam/LinearRegressionScratch/build/CMakeFiles/LinearRegression.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/LinearRegression.dir/depend

